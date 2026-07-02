from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any

import requests
import torch


@dataclass
class WAMServiceClient:
    endpoint: str
    timeout_s: float = 3600.0
    load_engine_on_init: bool = True

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")
        self._check_health()
        if self.load_engine_on_init:
            self.load_engine()

    def _check_health(self) -> None:
        resp = requests.get(f"{self.endpoint}/health", timeout=min(self.timeout_s, 30.0))
        if resp.status_code != 200:
            raise RuntimeError(f"WAM service is unhealthy at {self.endpoint}: {resp.status_code}")
        try:
            payload = json.loads(resp.content.decode())
        except Exception:
            payload = resp.text
        print(f"[WAMServiceClient] health ok: {payload}", flush=True)

    def load_engine(self) -> None:
        resp = requests.get(f"{self.endpoint}/load_engine", timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to load WAM service at {self.endpoint}: {resp.text}")
        print(f"[WAMServiceClient] engine loaded at {self.endpoint}", flush=True)

    def infer_action(
        self,
        *,
        input_image: Any,
        robot_state: torch.Tensor,
        action_horizon: int,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        seed: int | None = None,
        tiled: bool = False,
        num_frames: int = 1,
        num_inference_steps: int = 10,
        cfg_scale: float = 8.0,
        save_path: str | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        reset_rtc: bool | None = None,
    ) -> torch.Tensor:
        if save_path is None:
            save_path = os.environ.get("QWORLD_WAM_SAVE_PATH", "/tmp/qworld_wam_robotwin_tmp_out.mp4")

        if hasattr(input_image, "size"):
            width, height = input_image.size
        elif isinstance(input_image, list) and len(input_image) > 0:
            first = input_image[0][0] if isinstance(input_image[0], list) else input_image[0]
            width, height = first.size
        else:
            raise TypeError(f"Unsupported input_image payload type: {type(input_image)}")

        payload: dict[str, Any] = {
            "save_path": save_path,
            "robot_state": robot_state.to(dtype=torch.float32, device="cpu"),
            "robot_state_mask": None,
            "robot_action_horizon": int(action_horizon),
            "wam_infer_mode": "action",
            "input_image": input_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "tiled": bool(tiled),
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
            "save_fps": 16,
        }
        if prev_chunk_left_over is not None:
            payload["prev_chunk_left_over"] = prev_chunk_left_over.to(dtype=torch.float32, device="cpu")
        if reset_rtc is not None:
            payload["reset_rtc"] = bool(reset_rtc)

        resp = requests.post(
            f"{self.endpoint}/infer",
            data=pickle.dumps(payload),
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout_s,
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"WAM service returned {resp.status_code}: {detail}")

        result = pickle.loads(resp.content)
        action = result["output"][1]
        if action is None:
            raise RuntimeError("WAM service returned no action tensor.")
        return action.detach().to(dtype=torch.float32, device="cpu")

