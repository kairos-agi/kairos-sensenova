from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Any, Optional

import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Per-endpoint: avoid duplicate /load_engine in one Python process (e.g. retries).
_LOAD_ENGINE_DONE: set[str] = set()


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _ensure_wam_service_ready(
    endpoint: str,
    timeout_s: float,
    *,
    skip_load_engine: bool = False,
) -> None:
    """Ping WAM service; call /load_engine only if the server pool is not loaded yet."""
    base = endpoint.rstrip("/")
    if _env_truthy("WAM_SKIP_CLIENT_INIT"):
        logger.debug("WAM_SKIP_CLIENT_INIT=1: skip health/load_engine for %s", base)
        return

    resp = requests.get(f"{base}/health", timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"WAM service health check failed at {base}: HTTP {resp.status_code}")
    try:
        health = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {base}/health") from exc

    skip_load = skip_load_engine or _env_truthy("WAM_SKIP_LOAD_ENGINE", default=True)
    if skip_load:
        logger.debug("skip_load_engine enabled; health ok at %s", base)
        return

    if health.get("workers_loaded") or health.get("already_loaded"):
        logger.debug("WAM workers already loaded at %s; skip /load_engine", base)
        return

    if base in _LOAD_ENGINE_DONE:
        logger.debug("load_engine already called in this process for %s", base)
        return

    resp = requests.get(f"{base}/load_engine", timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"WAM load_engine failed at {base}: HTTP {resp.status_code}")
    try:
        body = resp.json()
    except json.JSONDecodeError:
        body = {}
    _LOAD_ENGINE_DONE.add(base)
    if body.get("already_loaded"):
        logger.info("WAM engine pool already warm at %s", base)
    else:
        logger.info("WAM engine pool loaded via %s/load_engine", base)


class ExternalModelAdapter(torch.nn.Module):
    """Adapter interface for external/inference-service models."""

    is_external_model = True

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000",
        timeout_s: float = 30.0,
        model_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        skip_load_engine: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.torch_dtype = model_dtype
        self.device = torch.device(device)

        _ensure_wam_service_ready(
            self.endpoint,
            self.timeout_s,
            skip_load_engine=bool(skip_load_engine),
        )

    def to(self, device: Any = None, *args: Any, **kwargs: Any) -> "ExternalModelAdapter":
        if device is not None:
            self.device = torch.device(device)
        return self

    def eval(self) -> "ExternalModelAdapter":
        return self

    def load_checkpoint(self, path: str, optimizer: Any = None) -> None:
        _ = path
        _ = optimizer
        return None

    def infer_action(self, **kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["prompt"]
        negative_prompt = kwargs["negative_prompt"]
        input_image = kwargs["input_image"]
        action_horizon = kwargs["action_horizon"]
        num_inference_steps = kwargs["num_inference_steps"]
        proprio = kwargs["proprio"]
        seed = kwargs["seed"]

        if isinstance(input_image, Image.Image):
            width = input_image.size[0]
            height = input_image.size[1]
        elif isinstance(input_image, torch.Tensor):
            assert input_image.ndim == 4
            assert input_image.shape[0] == 1

            input_image = input_image.to(torch.float32)
            input_image = (input_image * 0.5 + 0.5).clamp(0, 1)
            input_image = input_image.permute(0, 2, 3, 1).cpu().numpy()
            input_image = Image.fromarray((input_image[0] * 255).astype(np.uint8))

            width = input_image.size[0]
            height = input_image.size[1]
        else:
            width = 448
            height = 256

        _project_root = os.environ.get("PROJECT_ROOT")
        if not _project_root:
            _project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        _tmp_dir = os.path.join(_project_root, "tmp")
        os.makedirs(_tmp_dir, exist_ok=True)

        payload = dict(
            save_path=os.path.join(_tmp_dir, "wam_infer_tmp_out.mp4"),
            robot_state=proprio,
            robot_state_mask=None,
            robot_action_horizon=action_horizon,
            wam_infer_mode="action",
            input_image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            tiled=True,
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=num_inference_steps,
            cfg_scale=8.0,
            save_fps=16,
        )

        max_retries = int(os.environ.get("WAM_INFER_MAX_RETRIES", "3"))
        retry_sleep_s = float(os.environ.get("WAM_INFER_RETRY_SLEEP_SEC", "2.0"))
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.endpoint}/infer",
                    data=pickle.dumps(payload),
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=3600,
                )
                break
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "WAM /infer connection failed (attempt %d/%d), retry in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    retry_sleep_s,
                    exc,
                )
                time.sleep(retry_sleep_s)
        else:
            assert last_exc is not None
            raise last_exc

        if resp.status_code != 200:
            body = resp.text[:4000]
            try:
                body_json = resp.json()
                body = json.dumps(body_json, ensure_ascii=False)[:4000]
            except Exception:
                pass
            raise RuntimeError(
                f"WAM /infer failed: HTTP {resp.status_code}, "
                f"content_type={resp.headers.get('content-type')}, body={body}"
            )

        try:
            result = pickle.loads(resp.content)
        except Exception as exc:
            body = resp.content[:4000]
            content_type = resp.headers.get("content-type")
            try:
                decoded = resp.text[:4000]
            except Exception:
                decoded = repr(body)
            raise RuntimeError(
                f"WAM /infer returned non-pickle response: "
                f"HTTP {resp.status_code}, content_type={content_type}, body={decoded}"
            ) from exc

        actions = result["output"][1]

        actions = actions.detach().to(torch.float32).cpu()
        if actions.ndim == 3:
            actions = actions.squeeze(0)

        return {"action": actions}

    def infer_joint(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        raise NotImplementedError("ExternalModelAdapter.infer_joint is not implemented yet.")
