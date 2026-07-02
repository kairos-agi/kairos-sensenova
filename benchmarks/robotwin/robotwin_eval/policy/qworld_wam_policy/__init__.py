from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image

from .http_client import WAMServiceClient
from .normalizer import RobotWinStatsNormalizer


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _is_none_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _parse_optional_int(value: Any, default: int | None = None) -> int | None:
    if _is_none_like(value):
        return default
    return int(value)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode="RGB").resize(size_wh, Image.BILINEAR)


class QWorldWAMRobotTwinPolicy:
    def __init__(
        self,
        *,
        endpoint: str,
        dataset_stats_path: str,
        action_horizon: int = 32,
        replan_steps: int = 24,
        num_inference_steps: int = 10,
        seed: int | None = None,
        cfg_scale: float = 8.0,
        negative_prompt: str = "",
        tiled: bool = False,
        timeout_s: float = 3600.0,
        timing_enabled: bool = False,
    ) -> None:
        self.client = WAMServiceClient(endpoint=endpoint, timeout_s=timeout_s)
        self.normalizer = RobotWinStatsNormalizer(dataset_stats_path)
        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed
        self.cfg_scale = float(cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self.pending_actions: deque[np.ndarray] = deque()
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        print(
            "[QWorldWAMRobotTwinPolicy] "
            f"endpoint={endpoint} stats={dataset_stats_path} "
            f"action_horizon={self.action_horizon} replan_steps={self.replan_steps} "
            f"num_inference_steps={self.num_inference_steps} cfg_scale={self.cfg_scale}",
            flush=True,
        )

    def _build_robotwin_image(self, observation: dict[str, Any]) -> Image.Image:
        obs_data = observation["observation"]
        head = np.asarray(_resize_rgb(obs_data["head_camera"]["rgb"], (320, 256)))
        left = np.asarray(_resize_rgb(obs_data["left_camera"]["rgb"], (160, 128)))
        right = np.asarray(_resize_rgb(obs_data["right_camera"]["rgb"], (160, 128)))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        return Image.fromarray(image.astype(np.uint8), mode="RGB")

    def _infer_action_chunk(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        image = self._build_robotwin_image(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        robot_state = self.normalizer.normalize_state(state_vector)
        prompt = DEFAULT_PROMPT.format(task=instruction)

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        action_tensor = self.client.infer_action(
            input_image=image,
            robot_state=robot_state,
            action_horizon=self.action_horizon,
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            tiled=self.tiled,
            num_frames=1,
            num_inference_steps=self.num_inference_steps,
            cfg_scale=self.cfg_scale,
        )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_chunk = self.normalizer.denormalize_action(action_tensor)[0]
        return np.asarray(action_chunk, dtype=np.float32)

    def _enqueue_action_chunk(self, action_chunk: np.ndarray) -> None:
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk [T,D], got {action_chunk.shape}")
        for idx in range(min(self.replan_steps, action_chunk.shape[0])):
            self.pending_actions.append(np.asarray(action_chunk[idx], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env: Any, observation: dict[str, Any] | None) -> None:
        instruction = task_env.get_instruction()
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required at qworld WAM replan step.")
            self._enqueue_action_chunk(self._infer_action_chunk(observation, instruction))

        if not self.pending_actions:
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0

    def reset(self) -> None:
        self.pending_actions.clear()

    def get_timing_rollout(self) -> dict[str, float]:
        return dict(self._timing_rollout)

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0


def get_model(usr_args: dict[str, Any]) -> QWorldWAMRobotTwinPolicy:
    endpoint = usr_args.get("qworld_endpoint") or usr_args.get("endpoint") or usr_args.get("external_model_endpoint")
    if _is_none_like(endpoint):
        raise ValueError("`qworld_endpoint` is required.")

    dataset_stats_path = usr_args.get("dataset_stats_path")
    if _is_none_like(dataset_stats_path):
        raise ValueError("`dataset_stats_path` is required.")

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"), 32)
    replan_steps = _parse_optional_int(usr_args.get("replan_steps"), 24)
    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"), 10)
    seed = _parse_optional_int(usr_args.get("seed"), None)

    return QWorldWAMRobotTwinPolicy(
        endpoint=str(endpoint),
        dataset_stats_path=str(dataset_stats_path),
        action_horizon=int(action_horizon or 32),
        replan_steps=int(replan_steps or 24),
        num_inference_steps=int(num_inference_steps or 10),
        seed=seed,
        cfg_scale=float(usr_args.get("cfg_scale", 8.0)),
        negative_prompt=str(usr_args.get("negative_prompt", "")),
        tiled=_parse_bool(usr_args.get("tiled", False)),
        timeout_s=float(usr_args.get("timeout_s", 3600.0)),
        timing_enabled=_parse_bool(usr_args.get("timing_enabled", False)),
    )


def eval(TASK_ENV: Any, model: QWorldWAMRobotTwinPolicy, observation: dict[str, Any] | None) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: QWorldWAMRobotTwinPolicy) -> None:
    model.reset()

