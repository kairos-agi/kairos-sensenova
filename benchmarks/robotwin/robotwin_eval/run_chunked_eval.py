from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
DEFAULT_EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "robotwin_eval" / "config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


@dataclass
class WorkItem:
    task_name: str
    phase: str
    chunk_idx: int
    chunk_size: int
    global_episode_start: int
    seed_start: int
    seed_candidate_count: int


@dataclass
class RunningState:
    item: WorkItem
    gpu_id: int
    process: subprocess.Popen[str]
    start_time: float
    stdout_thread: threading.Thread | None = None


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_gpu_ids(value: str | None, num_gpus: int | None) -> list[int]:
    if value is None or str(value).strip() == "":
        if num_gpus is None:
            raise ValueError("Either --gpu-ids or --num-gpus must be provided.")
        return list(range(int(num_gpus)))
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    gpu_ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(gpu_ids) == 0:
        raise ValueError(f"Empty GPU list: {value}")
    return gpu_ids


def _load_all_tasks(eval_step_limit_file: Path) -> list[str]:
    with eval_step_limit_file.open("r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        task_map = yaml.safe_load(text)
    except ModuleNotFoundError:
        task_map = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            task_map[key.strip()] = value.strip()
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {eval_step_limit_file}")
    return list(dict.fromkeys(str(task_name) for task_name in task_map.keys()))


def _parse_chunk_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("task_name", "phase", "chunk_idx", "episode_count", "success_count", "success_rate"):
        if key not in payload:
            raise ValueError(f"Missing key `{key}` in chunk result: {path}")
    return payload


def _to_jsonable(value: float | None) -> float | None:
    return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunked RoboTwin evaluation against qworld WAM service.")
    parser.add_argument("--robotwin-root", default=str(DEFAULT_ROBOTWIN_ROOT))
    parser.add_argument("--eval-step-limit-file", default=str(DEFAULT_EVAL_STEP_LIMIT_FILE))
    parser.add_argument("--task-name", default=None, help="Optional single task. Defaults to all tasks in eval-step file.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-num-episodes", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--chunk-seed-candidate-multiplier", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-ids", default=None, help="Comma list or bracket list, e.g. 0,1,2 or [0,1,2].")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=8.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--timing-enabled", action="store_true")
    parser.add_argument("--skip-get-obs-within-replan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robotwin_root = _resolve_path(args.robotwin_root)
    eval_step_limit_file = _resolve_path(args.eval_step_limit_file)
    dataset_stats_path = _resolve_path(args.dataset_stats_path)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for required_path, label in (
        (robotwin_root, "robotwin root"),
        (eval_step_limit_file, "eval step limit file"),
        (dataset_stats_path, "dataset stats"),
    ):
        exists = required_path.is_dir() if label == "robotwin root" else required_path.is_file()
        if not exists:
            raise FileNotFoundError(f"Missing {label}: {required_path}")

    gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_gpus)
    if args.max_tasks_per_gpu <= 0:
        raise ValueError("--max-tasks-per-gpu must be positive.")
    if args.eval_num_episodes <= 0:
        raise ValueError("--eval-num-episodes must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.chunk_seed_candidate_multiplier <= 0:
        raise ValueError("--chunk-seed-candidate-multiplier must be positive.")

    manager_log = output_dir / "manager_chunked.log"
    failed_chunks_file = output_dir / "failed_chunks.txt"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    chunk_summary_csv = output_dir / "chunk_summary.csv"
    chunk_summary_json = output_dir / "chunk_summary.json"

    tasks = [args.task_name] if args.task_name else _load_all_tasks(eval_step_limit_file)
    phases = ["clean", "random"]
    phase_to_task_config = {"clean": "demo_clean", "random": "demo_randomized"}
    seed_base = 100000 * (1 + int(args.seed))

    running_states: list[RunningState] = []
    failed_records: list[dict[str, Any]] = []
    chunk_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    stdout_print_lock = threading.Lock()

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def chunk_dir(item: WorkItem) -> Path:
        return output_dir / item.task_name / item.phase / f"chunk_{item.chunk_idx:03d}"

    def chunk_result_file(item: WorkItem) -> Path:
        return chunk_dir(item) / item.task_name / "chunk_result.json"

    def build_work_items() -> list[WorkItem]:
        items: list[WorkItem] = []
        num_chunks = (args.eval_num_episodes + args.chunk_size - 1) // args.chunk_size
        candidate_count_default = args.chunk_size * args.chunk_seed_candidate_multiplier
        for phase in phases:
            for task_name in tasks:
                for chunk_idx in range(num_chunks):
                    global_episode_start = chunk_idx * args.chunk_size
                    remaining = args.eval_num_episodes - global_episode_start
                    actual_chunk_size = min(args.chunk_size, remaining)
                    if actual_chunk_size <= 0:
                        continue
                    items.append(
                        WorkItem(
                            task_name=task_name,
                            phase=phase,
                            chunk_idx=chunk_idx,
                            chunk_size=actual_chunk_size,
                            global_episode_start=global_episode_start,
                            seed_start=seed_base + chunk_idx * candidate_count_default,
                            seed_candidate_count=max(
                                candidate_count_default,
                                actual_chunk_size * args.chunk_seed_candidate_multiplier,
                            ),
                        )
                    )
        return items

    def try_resume_item(item: WorkItem) -> bool:
        path = chunk_result_file(item)
        if not path.exists():
            return False
        try:
            payload = _parse_chunk_result(path)
            if str(payload["task_name"]) != item.task_name:
                return False
            if str(payload["phase"]) != item.phase:
                return False
            if int(payload["chunk_idx"]) != item.chunk_idx:
                return False
            if int(payload["episode_count"]) != item.chunk_size:
                return False
            success_count = int(payload["success_count"])
            if success_count < 0 or success_count > item.chunk_size:
                return False
        except Exception as exc:
            log(f"resume ignore invalid chunk result: file={path} error={repr(exc)}")
            return False
        chunk_records[(item.task_name, item.phase, item.chunk_idx)] = payload
        return True

    all_items = build_work_items()
    pending_items: deque[WorkItem] = deque()
    resumed_count = 0
    for item in all_items:
        if args.resume and try_resume_item(item):
            resumed_count += 1
            continue
        pending_items.append(item)

    def build_cmd(item: WorkItem, gpu_id: int) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "robotwin_eval.eval_single",
            "--robotwin-root",
            str(robotwin_root),
            "--task-name",
            item.task_name,
            "--task-config",
            phase_to_task_config[item.phase],
            "--output-dir",
            str(chunk_dir(item)),
            "--dataset-stats-path",
            str(dataset_stats_path),
            "--endpoint",
            args.endpoint,
            "--gpu-id",
            str(gpu_id),
            "--seed",
            str(args.seed),
            "--eval-num-episodes",
            str(item.chunk_size),
            "--instruction-type",
            args.instruction_type,
            "--action-horizon",
            str(args.action_horizon),
            "--replan-steps",
            str(args.replan_steps),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--cfg-scale",
            str(args.cfg_scale),
            "--negative-prompt",
            args.negative_prompt,
            "--timeout-s",
            str(args.timeout_s),
            "--chunk-idx",
            str(item.chunk_idx),
            "--chunk-size",
            str(item.chunk_size),
            "--chunk-seed-start",
            str(item.seed_start),
            "--chunk-seed-candidate-count",
            str(item.seed_candidate_count),
            "--global-episode-start",
            str(item.global_episode_start),
            "--total-task-episodes",
            str(args.eval_num_episodes),
        ]
        if args.tiled:
            cmd.append("--tiled")
        if args.timing_enabled:
            cmd.append("--timing-enabled")
        if not args.skip_get_obs_within_replan:
            cmd.append("--no-skip-get-obs-within-replan")
        return cmd

    def stdout_reader(state: RunningState) -> None:
        if state.process.stdout is None:
            return
        for line in state.process.stdout:
            with stdout_print_lock:
                sys.stdout.write(line)
                sys.stdout.flush()

    def launch_item(item: WorkItem, gpu_id: int) -> RunningState:
        cmd = build_cmd(item, gpu_id)
        log(
            f"launch task={item.task_name} phase={item.phase} chunk={item.chunk_idx} "
            f"episodes={item.chunk_size} seed_start={item.seed_start} "
            f"seed_candidates={item.seed_candidate_count} gpu={gpu_id}"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state = RunningState(item=item, gpu_id=gpu_id, process=process, start_time=time.time())
        state.stdout_thread = threading.Thread(target=stdout_reader, args=(state,), daemon=True)
        state.stdout_thread.start()
        return state

    def gpu_running_count(gpu_id: int) -> int:
        return sum(1 for state in running_states if state.gpu_id == gpu_id and state.process.poll() is None)

    def try_launch_pending(gpu_id: int) -> None:
        while pending_items and gpu_running_count(gpu_id) < args.max_tasks_per_gpu:
            running_states.append(launch_item(pending_items.popleft(), gpu_id))

    def terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=TERMINATE_TIMEOUT_SEC)

    def write_summaries() -> None:
        rows: list[dict[str, Any]] = []
        for key in sorted(chunk_records.keys()):
            task_name, phase, chunk_idx = key
            payload = chunk_records[key]
            rows.append(
                {
                    "task_name": task_name,
                    "phase": phase,
                    "chunk_idx": chunk_idx,
                    "episode_count": int(payload["episode_count"]),
                    "success_count": int(payload["success_count"]),
                    "success_rate": float(payload["success_rate"]),
                    "path": str(chunk_result_file(
                        WorkItem(task_name, phase, chunk_idx, 0, 0, 0, 0)
                    )),
                }
            )

        with chunk_summary_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["task_name", "phase", "chunk_idx", "episode_count", "success_count", "success_rate", "path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        chunk_summary_json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_payload: dict[str, dict[str, dict[str, float | int | None]]] = {}
        for task_name in tasks:
            summary_payload[task_name] = {}
            for phase in phases:
                phase_records = [
                    payload
                    for (record_task, record_phase, _), payload in chunk_records.items()
                    if record_task == task_name and record_phase == phase
                ]
                episode_count = sum(int(payload["episode_count"]) for payload in phase_records)
                success_count = sum(int(payload["success_count"]) for payload in phase_records)
                success_rate = None if episode_count == 0 else success_count / episode_count
                summary_payload[task_name][phase] = {
                    "episode_count": episode_count,
                    "success_count": success_count,
                    "success_rate": _to_jsonable(success_rate),
                }

        summary_json.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "task_name",
                "clean_success_rate",
                "random_success_rate",
                "clean_success_count",
                "clean_episode_count",
                "random_success_count",
                "random_episode_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for task_name in tasks:
                clean = summary_payload[task_name]["clean"]
                random_phase = summary_payload[task_name]["random"]
                writer.writerow(
                    {
                        "task_name": task_name,
                        "clean_success_rate": "" if clean["success_rate"] is None else clean["success_rate"],
                        "random_success_rate": "" if random_phase["success_rate"] is None else random_phase["success_rate"],
                        "clean_success_count": clean["success_count"],
                        "clean_episode_count": clean["episode_count"],
                        "random_success_count": random_phase["success_count"],
                        "random_episode_count": random_phase["episode_count"],
                    }
                )

    log(
        f"start chunked eval tasks={len(tasks)} total_items={len(all_items)} "
        f"pending={len(pending_items)} resumed={resumed_count} gpu_ids={gpu_ids}"
    )

    try:
        for gpu_id in gpu_ids:
            try_launch_pending(gpu_id)

        while running_states:
            for state in list(running_states):
                return_code = state.process.poll()
                if return_code is None:
                    continue
                running_states.remove(state)
                if state.stdout_thread is not None:
                    state.stdout_thread.join(timeout=2)

                result_path = chunk_result_file(state.item)
                if return_code == 0 and result_path.exists():
                    try:
                        payload = _parse_chunk_result(result_path)
                        chunk_records[(state.item.task_name, state.item.phase, state.item.chunk_idx)] = payload
                        log(
                            f"done task={state.item.task_name} phase={state.item.phase} "
                            f"chunk={state.item.chunk_idx} success={payload['success_count']}/"
                            f"{payload['episode_count']} rate={payload['success_rate']}"
                        )
                    except Exception as exc:
                        failed_records.append(
                            {
                                "task_name": state.item.task_name,
                                "phase": state.item.phase,
                                "chunk_idx": state.item.chunk_idx,
                                "gpu_id": state.gpu_id,
                                "return_code": return_code,
                                "error": f"invalid result file: {exc}",
                            }
                        )
                else:
                    failed_records.append(
                        {
                            "task_name": state.item.task_name,
                            "phase": state.item.phase,
                            "chunk_idx": state.item.chunk_idx,
                            "gpu_id": state.gpu_id,
                            "return_code": return_code,
                            "error": f"missing result file: {result_path}",
                        }
                    )
                    log(
                        f"failed task={state.item.task_name} phase={state.item.phase} "
                        f"chunk={state.item.chunk_idx} return_code={return_code}"
                    )

                try_launch_pending(state.gpu_id)

            write_summaries()
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        log("received KeyboardInterrupt; terminating running workers")
        for state in running_states:
            terminate_process(state.process)
        raise
    finally:
        write_summaries()

    if failed_records:
        with failed_chunks_file.open("w", encoding="utf-8") as f:
            for record in failed_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        raise RuntimeError(f"{len(failed_records)} chunks failed. See {failed_chunks_file}")

    log(f"all chunks finished. summary={summary_csv}")


if __name__ == "__main__":
    main()
