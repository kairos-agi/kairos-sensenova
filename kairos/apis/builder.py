import copy
import torch.nn as nn
from mmengine.registry import Registry


PIPELINES_API = Registry('model_pipeline')
KAIROS_PROCESSOR = Registry('kairos_processor')
DITS = Registry('dits')

def build_model_pipeline(cfg):
    model_cls = PIPELINES_API.get(cfg['type'])
    _cfg = copy.deepcopy(cfg)
    _cfg.pop('type')
    return model_cls(config=_cfg)


