import copy
import torch.nn as nn
from mmengine.registry import Registry


PIPELINES = Registry('model_pipeline')

KAIROS_PIPELINES = Registry('kairos_pipeline')

DITS = Registry('dits')


def build_model_pipeline(cfg):
    
    model_cls = PIPELINES.get(cfg['type'])

    _cfg = copy.deepcopy(cfg)
    _cfg.pop('type')


    return model_cls(config=_cfg)

