import copy
import json
import random
import sys
import os
import imageio
import numpy as np
import torch
from tqdm import tqdm
from mmengine import Config as MMConfig
import time 
import torch.distributed as dist


from kairos.modules.utils import save_video, save_image
from kairos.modules.utils import load_state_dict
from kairos.pipelines.kairos_embodied_pipeline import KairosEmbodiedPipeline
from kairos.apis.builder import PIPELINES_API, KAIROS_PROCESSOR, DITS

@PIPELINES_API.register_module()
class KairosEmbodiedAPI(torch.nn.Module):

    def __init__(
        self,
        config=MMConfig(dict(
            pipeline_type='KairosEmbodiedPipeline',
            pretrained_dit=None,
            vae_path=None,
            text_encoder_path=None,
            pipeline_args=None,
            tea_cache_l1_thresh=None,
            tea_cache_model_id="",
        )),
        torch_dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        
        self._init_config = config

        self.tea_cache_l1_thresh = config.get('tea_cache_l1_thresh',None)
        self.tea_cache_model_id = config.get('tea_cache_model_id',"")
        self.parallel_mode = config.get('parallel_mode',None)

        pretrained_dit = config.get('pretrained_dit',None)
        pipeline_type = config.get('pipeline_type','KairosEmbodiedPipeline')
        pipeline_args = config.get('pipeline_args',dict())
        
        if pipeline_args:
            dit_config = pipeline_args.pop('dit_config', None)
            load_dit_fn=pipeline_args.pop('load_dit_fn', None)
            pipeline_args["parallel_mode"] = config.get('parallel_mode',None)
        else:
            dit_config = None
            load_dit_fn=None

        if dit_config:
            print('Init KairosDiT model with config: ', dit_config)
            dit_type = dit_config.pop('dit_type')
            dit_cls = DITS.get(dit_type)
            dit = dit_cls(**dit_config)
            total_params = sum(p.numel() for p in dit.parameters()) / 1e9
            print(f"Total parameters of DiT: {total_params:.3f} B")
            if pretrained_dit:
                if load_dit_fn == 'strict_load':
                    print(f'using strict_load || Loading DiT from {pretrained_dit}')
                    state_dict = load_state_dict(pretrained_dit)
                    dit.load_state_dict(state_dict, strict=False)
                else:
                    raise NotImplementedError()
                
            dit = dit.bfloat16().cuda()
            pipeline_args['dit'] = dit

        pipeline_cls = KAIROS_PROCESSOR.get(pipeline_type)

        self.pipe = pipeline_cls.from_pretrained(
            torch_dtype=torch_dtype, 
            device=device,
            **pipeline_args,
        )
        total_params = sum(p.numel() for p in self.pipe.parameters()) / 1e9
        print(f"Total parameters of the whole model: {total_params:.3f} B")

    def __call__(self, **kwargs):
        # Provide TeaCache configuration here
        kwargs["tea_cache_l1_thresh"] = self.tea_cache_l1_thresh
        kwargs["tea_cache_model_id"] = self.tea_cache_model_id
        kwargs["parallel_mode"] = self.parallel_mode
    
        return self.pipe(**kwargs)
