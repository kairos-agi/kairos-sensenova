from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import torch
import json
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hashlib
from pathlib import Path
from git import Repo
from kairoswam.utils.logging_config import get_logger

from kairoswam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score"], ConstConstStr]

class LinearNormalizer:
    def __init__(
            self, 
            shape_meta, 
            use_stepwise_action_norm,
            default_mode: NormMode, 
            exception_mode: Dict[str, Dict[str, NormMode]], 
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

        for meta in shape_meta["action"]:
            key = meta["key"]
            
            if use_stepwise_action_norm:
                cur_stats = {k.removeprefix("stepwise_"): v for k, v in stats["action"][key].items() if k.startswith("stepwise_")}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in stats["action"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats, 
                mode=cur_mode,
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            cur_stats = {k.removeprefix("global_"): v for k, v in stats["state"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats, 
                mode=cur_mode,
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()}
        }
        return stats
                
    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])

        return batch
    
    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for key, norm in self.normalizers["action"].items():
            batch["action"][key] = norm.backward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])
        
        return batch


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0
    def __init__(self, stats, mode: NormMode="min/max"):
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = - input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            else: 
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset
    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x

def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_stats = convert_tensor(dataset_stats)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)

def load_dataset_stats_from_json(file_path: str, 
                                 try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True  
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data


def search_dataset_stats_cache_json(cache_dir: str | Path, data_config: DictConfig) -> Tuple[bool, str | None]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_git_hash() -> Optional[str]:
        repo = Repo(__file__, search_parent_directories=True)
        return repo.head.commit.hexsha

    def to_plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def normalize_str_list(value: Any) -> List[str]:
        value = to_plain(value)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in value if item is not None]

    def normalize_transforms(value: Any) -> Any:
        value = to_plain(value)
        if isinstance(value, dict):
            return [value]
        return value

    def normalize_dataset_dirs(cfg: DictConfig) -> Any:
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            return None
        embodiment_datasets = dataset_cfg.get("embodiment_datasets")
        if embodiment_datasets is not None:
            emb_dirs: Dict[str, List[str]] = {}
            for emb, emb_cfg in embodiment_datasets.items():
                dataset_groups = emb_cfg.get("dataset_groups")
                if dataset_groups is None:
                    emb_dirs[emb] = []
                    continue
                dirs: List[str] = []
                for group in dataset_groups:
                    group_dirs = group.get("dataset_dirs")
                    if group_dirs is None:
                        continue
                    dirs.extend(normalize_str_list(group_dirs))
                emb_dirs[emb] = sorted(dirs)
            return emb_dirs

        dataset_dirs = dataset_cfg.get("dataset_dirs")
        return sorted(normalize_str_list(dataset_dirs))

    def normalize_action_state_transforms(cfg: DictConfig) -> Any:
        processor_cfg = cfg.get("processor")
        if processor_cfg is None:
            return None
        embodiment_processors = processor_cfg.get("embodiment_processors")
        if embodiment_processors is not None:
            emb_transforms: Dict[str, Any] = {}
            for emb, emb_cfg in embodiment_processors.items():
                transforms = emb_cfg.get("action_state_transforms")
                emb_transforms[emb] = normalize_transforms(transforms)
            return emb_transforms

        transforms = processor_cfg.get("action_state_transforms")
        return normalize_transforms(transforms)

    signature = {
        "action_size": data_config.dataset.action_size, 
        "dataset_dirs": normalize_dataset_dirs(data_config),
        "action_state_transforms": normalize_action_state_transforms(data_config),
    }
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    dataset_hash = hashlib.sha256(signature_json.encode("utf-8"), usedforsecurity=False).hexdigest()

    git_hash = get_git_hash()
    precise_name = f"dataset_stats_{dataset_hash}_{git_hash}.json"
    precise = cache_dir / precise_name
    if precise.exists():
        logger.info(f"Found dataset stats cache with precisely matching dataset and git hash: {precise_name}.")
        return True, str(precise)
    
    candidates = sorted(cache_dir.glob(f"dataset_stats_{dataset_hash}_*.json"))
    if not candidates:
        logger.info(f"No dataset stats cache found for dataset hash {dataset_hash}")
        return False, str(precise) # return precise cache path for saving cache

    picked = candidates[0]
    prefix = f"dataset_stats_{dataset_hash}_"
    picked_git_hash = picked.name[len(prefix):-5]
    assert picked_git_hash != git_hash
    logger.warning(f"Found substitute dataset stats cache {picked.name} which mismatch current git hash {git_hash}.")
    return True, str(picked)
