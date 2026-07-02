from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "robotwin_eval" / "policy" / "qworld_wam_policy" / "deploy_policy.yml"
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "robotwin_eval" / "robotwin_scripts" / "eval_policy_chunked.py"


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RoboTwin task/config/chunk against qworld WAM.")
    parser.add_argument("--robotwin-root", default=str(DEFAULT_ROBOTWIN_ROOT))
    parser.add_argument("--policy-config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument("--eval-script", default=str(DEFAULT_EVAL_SCRIPT))
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True, choices=("demo_clean", "demo_randomized"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-num-episodes", type=int, default=100)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--ckpt-setting", default="/nothing/to/load")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=8.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--timing-enabled", action="store_true")
    parser.add_argument("--skip-get-obs-within-replan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)

    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-seed-start", type=int, default=None)
    parser.add_argument("--chunk-seed-candidate-count", type=int, default=None)
    parser.add_argument("--global-episode-start", type=int, default=0)
    parser.add_argument("--total-task-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robotwin_root = _resolve_path(args.robotwin_root)
    policy_config = _resolve_path(args.policy_config)
    eval_script = _resolve_path(args.eval_script)
    output_dir = _resolve_path(args.output_dir)
    dataset_stats_path = _resolve_path(args.dataset_stats_path)

    for required_path, label in (
        (robotwin_root, "robotwin root"),
        (policy_config, "policy config"),
        (eval_script, "RoboTwin eval script"),
        (dataset_stats_path, "dataset stats"),
    ):
        if label == "robotwin root":
            exists = required_path.is_dir()
        else:
            exists = required_path.is_file()
        if not exists:
            raise FileNotFoundError(f"Missing {label}: {required_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / (
        f"eval_{args.task_name}_{args.task_config}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    )

    overrides: list[str] = []
    _append_override(overrides, "task_name", args.task_name)
    _append_override(overrides, "task_config", args.task_config)
    _append_override(overrides, "ckpt_setting", args.ckpt_setting)
    _append_override(overrides, "seed", args.seed)
    _append_override(overrides, "policy_name", "robotwin_eval.policy.qworld_wam_policy")
    _append_override(overrides, "instruction_type", args.instruction_type)
    _append_override(overrides, "eval_num_episodes", args.eval_num_episodes)
    _append_override(overrides, "eval_output_dir", str(output_dir / args.task_name))
    _append_override(overrides, "qworld_endpoint", args.endpoint)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", args.action_horizon)
    _append_override(overrides, "replan_steps", args.replan_steps)
    _append_override(overrides, "num_inference_steps", args.num_inference_steps)
    _append_override(overrides, "cfg_scale", args.cfg_scale)
    _append_override(overrides, "negative_prompt", args.negative_prompt)
    _append_override(overrides, "tiled", args.tiled)
    _append_override(overrides, "timing_enabled", args.timing_enabled)
    _append_override(overrides, "skip_get_obs_within_replan", args.skip_get_obs_within_replan)
    _append_override(overrides, "timeout_s", args.timeout_s)
    _append_override(overrides, "chunk_idx", args.chunk_idx)
    _append_override(overrides, "chunk_size", args.chunk_size or args.eval_num_episodes)
    _append_override(overrides, "chunk_seed_start", args.chunk_seed_start)
    _append_override(overrides, "chunk_seed_candidate_count", args.chunk_seed_candidate_count)
    _append_override(overrides, "global_episode_start", args.global_episode_start)
    _append_override(overrides, "total_task_episodes", args.total_task_episodes or args.eval_num_episodes)

    cmd = [
        sys.executable,
        "-u",
        str(eval_script),
        "--config",
        str(policy_config),
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["ROBOTWIN_ROOT"] = str(robotwin_root)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    print("[robotwin_eval.single] " + " ".join(cmd), flush=True)
    with log_file.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"[robotwin_eval.single] finished. Log: {log_file}", flush=True)


if __name__ == "__main__":
    main()

