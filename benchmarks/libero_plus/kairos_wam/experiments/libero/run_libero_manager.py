import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf

from libero_plus_eval_utils import LIBERO_PLUS_SUITES, resolve_category_values


def _resolve_suite_names(task_suite_names: list[str], include_libero_mix: bool) -> tuple[list[str], bool]:
    """Resolve which benchmarks to run.

    - ``task_suite_names: [libero_mix]`` → mix only (no fallback to the four base suites).
    - ``task_suite_names: []`` with no mix → all four base suites (legacy default).
    - ``include_libero_mix: true`` adds ``libero_mix`` on top of selected base suites.
    """
    suites: list[str] = []
    has_mix = False
    for name in task_suite_names:
        if name == "libero_mix":
            has_mix = True
            continue
        if name in LIBERO_PLUS_SUITES:
            suites.append(name)
    # Fallback only when nothing was explicitly selected (not for mix-only runs).
    if not suites and not has_mix:
        suites = list(LIBERO_PLUS_SUITES)
    if include_libero_mix:
        has_mix = True
    return suites, has_mix


def create_task_file(
    output_file: Path,
    task_suite_names: list[str],
    *,
    category_value: str | None = None,
    category_values: list[str] | None = None,
    run_all_categories: bool = False,
    include_libero_mix: bool = True,
) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if run_all_categories:
        categories = resolve_category_values(category_values)
        suites, use_mix = _resolve_suite_names(task_suite_names, include_libero_mix)
    else:
        if category_value is None or str(category_value).strip() == "":
            raise ValueError(
                "EVALUATION.category_value is required unless MULTIRUN.run_all_categories=true."
            )
        categories = [str(category_value)]
        suites, use_mix = _resolve_suite_names(task_suite_names, include_libero_mix)
        if not suites and not use_mix:
            raise ValueError(
                "No evaluation suites selected. Set MULTIRUN.task_suite_names "
                "(e.g. libero_spatial or libero_mix) or include_libero_mix=true."
            )

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for cat in categories:
            print(f"\n[category] {cat}")
            for suite_name in suites:
                task_suite = benchmark_dict[suite_name](category_value=str(cat))
                n_tasks = int(task_suite.n_tasks)
                print(f"  {suite_name}: {n_tasks} tasks")
                for task_id in range(n_tasks):
                    f.write(f"{suite_name},{task_id},{cat}\n")
                    total_tasks += 1
            if use_mix:
                task_suite = benchmark_dict["libero_mix"](category_value=str(cat))
                n_tasks = int(task_suite.n_tasks)
                print(f"  libero_mix: {n_tasks} tasks")
                for task_id in range(n_tasks):
                    f.write(f"libero_mix,{task_id},{cat}\n")
                    total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Categories: {len(categories)}")
    print(f"Suites: {list(suites)} + {'libero_mix' if use_mix else '(no mix)'}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
    category_value: str | None = None,
) -> None:
    script_path = Path("experiments/libero/run_libero_parallel_test.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = os.getcwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
        }
    )
    if category_value is not None and str(category_value).strip() != "":
        env["LIBERO_CATEGORY_VALUE"] = str(category_value)

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    try:
        subprocess.run(
            ["bash", str(script_path), str(task_file)],
            env=env,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding="utf-8"))
        raise


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN
    run_all_categories = bool(manager.get("run_all_categories", False))

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:
        task_file = output_dir / "tasks.txt"

    category_values_cfg = manager.get("category_values")
    if category_values_cfg is not None:
        category_values = list(category_values_cfg)
    else:
        category_values = None

    task_file = create_task_file(
        task_file,
        list(manager.task_suite_names),
        category_value=cfg.EVALUATION.get("category_value"),
        category_values=category_values,
        run_all_categories=run_all_categories,
        include_libero_mix=bool(manager.get("include_libero_mix", True)),
    )

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
        category_value=None if run_all_categories else cfg.EVALUATION.get("category_value"),
    )


if __name__ == "__main__":
    main()
