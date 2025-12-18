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
from PIL import Image


from .utils import save_video, save_image
from .dits import KairosDiT
from .pipelines import KairosVideoPipeline
from .utils import load_state_dict

from kairos.pipelines.builder import PIPELINES, KAIROS_PIPELINES, DITS



@PIPELINES.register_module()
class KairosPipeline(torch.nn.Module):

    def __init__(
        self,
        config=MMConfig(dict(
            exec_mode='infer',
            pipeline_type='KairosVideoPipeline',
            pretrained_dit=None,
            vae_path=None,
            text_encoder_path=None,
            pipeline_args=None,
        )),
        torch_dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        
        self._init_config = config

        exec_mode = config.exec_mode

        pretrained_dit = config.get('pretrained_dit',None)
        pipeline_type = config.get('pipeline_type','KairosVideoPipeline')
        pipeline_args = config.get('pipeline_args',dict())
        
        if exec_mode == 'train':
            raise NotImplementedError()
        else:
            if pipeline_args:
                dit_config = pipeline_args.pop('dit_config', None)
                load_dit_fn=pipeline_args.pop('load_dit_fn', None)
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
                        dit.load_state_dict(state_dict, strict=True)
                    else:
                        raise NotImplementedError()
                    
                dit = dit.bfloat16().cuda()
                pipeline_args['dit'] = dit

            pipeline_cls = KAIROS_PIPELINES.get(pipeline_type)

            self.pipe = pipeline_cls.from_pretrained(
                torch_dtype=torch_dtype, 
                device=device,
                **pipeline_args,
            )
            total_params = sum(p.numel() for p in self.pipe.parameters()) / 1e9
            print(f"Total parameters of the whole model: {total_params:.3f} B")

    def __call__(self, **kwargs):

        save_path = kwargs.pop('save_path', '~/tmp.mp4')
        save_fps = kwargs.pop('save_fps', 16)

        if 'input_image' in kwargs and isinstance(kwargs['input_image'], str):
            if not kwargs['input_image']:
                kwargs.pop('input_image')
            else:
                image = Image.open(kwargs['input_image'])
                kwargs['input_image'] = [image]

        video = self.pipe(**kwargs)

        base_dir = os.path.dirname(save_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            
        if len(video) == 1:
            save_path = save_path.replace('.mp4', '.jpg')
            save_image(video[0], save_path)
        else:
            save_video(video, save_path, fps=save_fps, quality=5)
        
        return save_path



