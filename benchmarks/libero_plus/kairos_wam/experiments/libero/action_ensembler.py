from collections import defaultdict
import numpy as np
import torch

class ActionEnsembler:
    def __init__(self):
        self.action_cache = defaultdict(list)

    def reset(self):
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int):
        if action_chunk.ndim == 3:
            action_chunk = action_chunk.squeeze(0)
        horizon, action_dim = action_chunk.shape

        for i in range(horizon):
            target_ts = start_timestamp + i
            self.action_cache[target_ts].append(action_chunk[i, :])

    def get_action(self, timestamp: int) -> np.ndarray:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        preds = self.action_cache[timestamp]
        stacked_preds = np.stack(preds, axis=0)
        averaged_action = np.mean(stacked_preds, axis=0)
        return averaged_action

    def _cleanup(self, current_timestamp: int):
        keys_to_delete = [ts for ts in self.action_cache.keys() if ts < current_timestamp]
        for ts in keys_to_delete:
            del self.action_cache[ts]