from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _numeric_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return True
    first = value[0]
    if isinstance(first, (int, float)):
        return all(isinstance(x, (int, float)) for x in value)
    if isinstance(first, list):
        return all(_numeric_list(x) for x in value)
    return False


def _to_tensor_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_tensor_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        if _numeric_list(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32)
        return [_to_tensor_tree(v) for v in value]
    return value


def load_dataset_stats(path: str | Path) -> dict[str, Any]:
    stats_path = Path(path).expanduser().resolve()
    with stats_path.open("r", encoding="utf-8") as f:
        return _to_tensor_tree(json.load(f))


class FieldZScoreNormalizer:
    std_reg = 1e-8

    def __init__(self, stats: dict[str, torch.Tensor]) -> None:
        self.mean = self._squeeze_step(stats["mean"])
        self.std = self._squeeze_step(stats["std"])

    @staticmethod
    def _squeeze_step(value: torch.Tensor) -> torch.Tensor:
        value = value.to(dtype=torch.float32, device="cpu")
        if value.ndim >= 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        return value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        out = (value.to(dtype=torch.float32, device="cpu") - self.mean) / (self.std + self.std_reg)
        return torch.clamp(out, -5.0, 5.0)

    def backward(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=torch.float32, device="cpu") * (self.std + self.std_reg) + self.mean


class RobotWinStatsNormalizer:
    """Minimal normalizer used by the RoboTwin qworld-WAM policy.

    The training stats follow the FASTWAM/RoboTwin schema:
    stats["state"]["default"]["global_mean"], stats["action"]["default"]["global_std"], ...
    Only the z-score/global path is needed for the exported qworld client.
    """

    def __init__(self, stats_path: str | Path, key: str = "default") -> None:
        stats = load_dataset_stats(stats_path)
        self.state = FieldZScoreNormalizer(
            {
                "mean": stats["state"][key]["global_mean"],
                "std": stats["state"][key]["global_std"],
            }
        )
        self.action = FieldZScoreNormalizer(
            {
                "mean": stats["action"][key]["global_mean"],
                "std": stats["action"][key]["global_std"],
            }
        )

    def normalize_state(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device="cpu")
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return self.state.forward(state_tensor)

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D] or [T,D], got {tuple(action.shape)}")
        out = self.action.backward(action)
        return out.numpy()

