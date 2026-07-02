#!/usr/bin/env python3
"""Run LIBERO-Plus evaluation in batches, one category_value per batch.

Each batch writes a small ``tasks_<category_dir>.txt`` and runs the parallel scheduler
on it. All batches share one ``EVALUATION.output_dir`` so results accumulate as::

  <output_dir>/libero_mix/<Category_Dir>/gpu*_task*_results.json

After all categories finish, runs ``summarize_results.py`` on the shared output dir.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_KAIROS_WAM_BENCH_ROOT = Path(__file__).resolve().parents[2]
_LIBERO_DIR = _KAIROS_WAM_BENCH_ROOT / "experiments" / "libero"
if str(_LIBERO_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBERO_DIR))

from libero_plus_eval_utils import (  # noqa: E402
    get_all_category_values,
    resolve_category_values,
    sanitize_category_value,
)

from run_libero_manager import create_task_file  # noqa: E402


def _parse_extra_overrides(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    return list(raw)


def generate_category_task_lists(
    *,
    lists_dir: Path,
    task_suite_names: list[str],
    categories: list[str],
    include_libero_mix: bool,
) -> dict[str, Path]:
    lists_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for cat in categories:
        cat_dir = sanitize_category_value(cat)
        out_file = lists_dir / f"tasks_{cat_dir}.txt"
        create_task_file(
            out_file,
            task_suite_names,
            category_values=[cat],
            run_all_categories=True,
            include_libero_mix=include_libero_mix,
        )
        paths[cat] = out_file
    return paths


def run_manager_batch(
    *,
    task_file: Path,
    batch_output_dir: Path,
    category: str,
    manager_args: list[str],
    cwd: Path,
    create_only: bool,
) -> None:
    escaped_cat = str(category).replace("'", "\\'")
    cmd = [
        sys.executable,
        str(_LIBERO_DIR / "run_libero_manager.py"),
        f"EVALUATION.output_dir={batch_output_dir}",
        f"MULTIRUN.task_file={task_file}",
        "MULTIRUN.run_all_categories=false",
        f"EVALUATION.category_value='{escaped_cat}'",
        f"MULTIRUN.create_only={'true' if create_only else 'false'}",
        *manager_args,
    ]
    print("\n" + "=" * 72)
    print(f"[batch] category={category!r}")
    print(f"[batch] task_file={task_file}")
    print(f"[batch] output_dir={batch_output_dir}")
    print(f"[batch] cmd: {' '.join(cmd)}")
    print("=" * 72)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO-Plus eval: one category per batch (small tasks.txt each)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Shared EVALUATION.output_dir for all category batches.",
    )
    parser.add_argument(
        "--lists-dir",
        type=str,
        default=None,
        help="Where to write tasks_<category>.txt (default: <output-dir>/category_task_lists).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Subset of category names; default = all 7 from task_classification.json.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        action="append",
        default=["libero_mix"],
        help="MULTIRUN.task_suite_names entry (repeatable). Default: libero_mix only.",
    )
    parser.add_argument(
        "--include-libero-mix",
        action="store_true",
        help="Also append libero_mix when base suites are listed (usually off for mix-only).",
    )
    parser.add_argument(
        "--create-lists-only",
        action="store_true",
        help="Only generate per-category tasks_*.txt under lists-dir, do not run eval.",
    )
    parser.add_argument(
        "--skip-summarize",
        action="store_true",
        help="Do not run summarize_results.py after all batches.",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Resume: skip categories before this name (exact match, e.g. 'Camera Viewpoints').",
    )
    parser.add_argument(
        "manager_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to run_libero_manager.py (e.g. task=... model.endpoint=...).",
    )
    args = parser.parse_args()
    manager_args = _parse_extra_overrides(args.manager_args)

    batch_output_dir = Path(os.path.expanduser(os.path.expandvars(args.output_dir))).resolve()
    lists_dir = (
        Path(os.path.expanduser(os.path.expandvars(args.lists_dir))).resolve()
        if args.lists_dir
        else batch_output_dir / "category_task_lists"
    )
    categories = resolve_category_values(args.categories)
    task_suite_names = list(args.suite)

    cwd = _KAIROS_WAM_BENCH_ROOT
    print(f"kairos_wam bench root: {cwd}")
    print(f"Categories ({len(categories)}): {categories}")
    print(f"Suites: {task_suite_names}")
    print(f"Lists dir: {lists_dir}")
    print(f"Batch output dir: {batch_output_dir}")

    task_paths = generate_category_task_lists(
        lists_dir=lists_dir,
        task_suite_names=task_suite_names,
        categories=categories,
        include_libero_mix=bool(args.include_libero_mix),
    )

    if args.create_lists_only:
        print("\nDone (create-lists-only). Task files:")
        for cat, path in task_paths.items():
            n_lines = sum(1 for _ in path.open(encoding="utf-8"))
            print(f"  {sanitize_category_value(cat)}: {path} ({n_lines} lines)")
        return

    started = args.start_from is None
    for cat in categories:
        if not started:
            if cat == args.start_from:
                started = True
            else:
                print(f"[batch] skip (resume): {cat}")
                continue

        run_manager_batch(
            task_file=task_paths[cat],
            batch_output_dir=batch_output_dir,
            category=cat,
            manager_args=manager_args,
            cwd=cwd,
            create_only=False,
        )

    if not args.skip_summarize:
        summarize_script = _LIBERO_DIR / "summarize_results.py"
        print(f"\n[batch] summarize -> {batch_output_dir}")
        subprocess.run(
            [sys.executable, str(summarize_script), f"--output_dir={batch_output_dir}"],
            cwd=str(cwd),
            check=True,
        )

    print("\n[batch] All category batches finished.")


if __name__ == "__main__":
    main()
