import os
import random
import sys
import threading
from typing import Any, Dict, Optional, Tuple

import torch
from mmengine.config import Config
from mmengine.dist import get_dist_info, init_dist

_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_KAIROS_WAM_ROOT = os.path.abspath(os.path.join(_BENCHMARK_ROOT, '..', '..'))
if _KAIROS_WAM_ROOT not in sys.path:
    sys.path.insert(0, _KAIROS_WAM_ROOT)

from kairos.apis.builder import build_model_pipeline


def _ensure_dist_init() -> None:
    dist_keys = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    dist_available = all(key in os.environ for key in dist_keys)
    if not dist_available:
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(random.randint(20000, 65535))

    init_dist(launcher='pytorch')
    rank, world_size = get_dist_info()
    print(f'[wam_service] dist initialized: rank={rank}, world_size={world_size}')


class WamInferEngine:
    def __init__(self, cfg_path: str, device: Optional[str] = None):
        self.cfg_path = cfg_path
        self.device = device
        self._lock = threading.Lock()

        _ensure_dist_init()
        cfg = Config.fromfile(self.cfg_path)
        self.pipeline = build_model_pipeline(cfg.pipeline)
        self.pipeline.eval()
        print(f'[wam_service] pipeline loaded from: {self.cfg_path}')

    def infer(
        self,
        *,
        robot_state: Any,
        robot_state_mask: Any,
        robot_action_horizon: int,
        input_image: Any,
        prompt: str,
        negative_prompt: str = '',
        wam_infer_mode: str = 'action',
        seed: int = 0,
        tiled: bool = True,
        height: int = 256,
        width: int = 448,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 20,
        cfg_scale: float = 8.0,
        save_fps: int = 16,
        save_path: str = '',
    ) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        if num_frames is None:
            num_frames = 1 if wam_infer_mode == 'action' else 81

        input_args: Dict[str, Any] = dict(
            robot_state=robot_state,
            robot_state_mask=robot_state_mask,
            robot_action_horizon=robot_action_horizon,
            wam_infer_mode=wam_infer_mode,
            input_image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            tiled=tiled,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
        )

        if wam_infer_mode == 'video':
            input_args.pop('robot_state', None)
            input_args.pop('robot_state_mask', None)
            input_args.pop('robot_action_horizon', None)

        with self._lock:
            with torch.inference_mode():
                out = self.pipeline(**input_args)

        out_video = None
        out_action = None
        if isinstance(out, dict):
            out_video = out.get('video', None)
            out_action = out.get('action', None)
        elif isinstance(out, (tuple, list)):
            if len(out) >= 1:
                out_video = out[0]
            if len(out) >= 2:
                out_action = out[1]
        else:
            out_video = out

        video_path = save_path if out_video is not None and save_path else None
        if out_action is not None:
            out_action = out_action.detach().to(torch.float32).cpu()

        return video_path, out_action
