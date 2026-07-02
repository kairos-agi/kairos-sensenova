import argparse
import json
import math
import os
from collections import defaultdict

import pandas as pd

from libero_plus_eval_utils import (
    LIBERO_PLUS_SUITES,
    get_all_category_values,
    sanitize_category_value,
)


def format_time(seconds):
    """Format seconds as a human-readable duration string."""
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds:02d}s"
    if seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"
    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    remaining_seconds = remaining % 60
    return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"


def _rate(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    return successes / trials * 100.0


def _collect_result_files(output_dir: str):
    """Yield (suite, category_dir, category_raw, result_path)."""
    for suite in os.listdir(output_dir):
        suite_path = os.path.join(output_dir, suite)
        if not os.path.isdir(suite_path):
            continue
        if suite in {"task_logs", "task_status", "traces"}:
            continue
        has_category_subdirs = False
        for entry in os.listdir(suite_path):
            entry_path = os.path.join(suite_path, entry)
            if not os.path.isdir(entry_path):
                continue
            result_files = [
                f
                for f in os.listdir(entry_path)
                if f.startswith("gpu") and f.endswith("_results.json")
            ]
            if not result_files:
                continue
            has_category_subdirs = True
            category_dir = entry
            category_raw = category_dir.replace("_", " ")
            for fname in result_files:
                yield suite, category_dir, category_raw, os.path.join(entry_path, fname)

        if has_category_subdirs:
            continue

        result_files = [
            f
            for f in os.listdir(suite_path)
            if f.startswith("gpu") and f.endswith("_results.json")
        ]
        for fname in result_files:
            yield suite, "", "", os.path.join(suite_path, fname)


def _is_libero_plus_layout(output_dir: str) -> bool:
    for suite in LIBERO_PLUS_SUITES + ("libero_mix",):
        suite_path = os.path.join(output_dir, suite)
        if not os.path.isdir(suite_path):
            continue
        for entry in os.listdir(suite_path):
            if os.path.isdir(os.path.join(suite_path, entry)):
                return True
    return False


def summarize_libero_plus_results(output_dir: str) -> dict:
    """Build hierarchical success-rate report for LIBERO-Plus matrix eval."""
    categories = get_all_category_values()
    cat_dirs = {sanitize_category_value(c): c for c in categories}

    per_suite_category = {
        suite: {cat: {"successes": 0, "trials": 0, "tasks": 0} for cat in categories}
        for suite in LIBERO_PLUS_SUITES
    }
    mix_category = {cat: {"successes": 0, "trials": 0, "tasks": 0} for cat in categories}
    grand = {"successes": 0, "trials": 0, "tasks": 0}

    task_results = {}

    for suite, category_dir, category_raw, result_path in _collect_result_files(output_dir):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        cat = result.get("category_value") or cat_dirs.get(category_dir, category_raw)
        if not cat:
            cat = category_dir.replace("_", " ")

        successes = int(result.get("successes", 0))
        trials = int(result.get("total_episodes", 0))
        task_id = int(result.get("task_id", -1))

        if suite in per_suite_category and cat in per_suite_category[suite]:
            bucket = per_suite_category[suite][cat]
            bucket["successes"] += successes
            bucket["trials"] += trials
            bucket["tasks"] += 1

        if suite == "libero_mix" and cat in mix_category:
            bucket = mix_category[cat]
            bucket["successes"] += successes
            bucket["trials"] += trials
            bucket["tasks"] += 1

        grand["successes"] += successes
        grand["trials"] += trials
        grand["tasks"] += 1

        task_key = f"{suite}/{sanitize_category_value(cat)}_task{task_id}"
        task_results[task_key] = {
            "success_rate": _rate(successes, trials),
            "duration": result.get("duration"),
            "total_episodes": trials,
            "successes": successes,
            "task_description": result.get("task_description", ""),
            "category_value": cat,
        }

    per_category_overall = {}
    for cat in categories:
        mix = mix_category[cat]
        per_category_overall[cat] = {
            "successes": mix["successes"],
            "trials": mix["trials"],
            "tasks": mix["tasks"],
            "success_rate": _rate(mix["successes"], mix["trials"]),
        }

    per_suite_summary = {}
    for suite in LIBERO_PLUS_SUITES:
        suite_successes = sum(per_suite_category[suite][c]["successes"] for c in categories)
        suite_trials = sum(per_suite_category[suite][c]["trials"] for c in categories)
        suite_tasks = sum(per_suite_category[suite][c]["tasks"] for c in categories)
        per_suite_summary[suite] = {
            "successes": suite_successes,
            "trials": suite_trials,
            "tasks": suite_tasks,
            "success_rate": _rate(suite_successes, suite_trials),
            "by_category": {
                cat: {
                    **per_suite_category[suite][cat],
                    "success_rate": _rate(
                        per_suite_category[suite][cat]["successes"],
                        per_suite_category[suite][cat]["trials"],
                    ),
                }
                for cat in categories
            },
        }

    report = {
        "categories": categories,
        "suites": list(LIBERO_PLUS_SUITES),
        "per_suite_per_category": per_suite_summary,
        "per_category_overall_libero_mix": per_category_overall,
        "grand_overall": {
            **grand,
            "success_rate": _rate(grand["successes"], grand["trials"]),
        },
        "task_results": task_results,
    }

    _print_libero_plus_report(report)
    _save_libero_plus_artifacts(output_dir, report)
    return report


def _print_libero_plus_report(report: dict) -> None:
    categories = report["categories"]
    print("\n=== LIBERO-Plus Evaluation Summary ===")

    print("\n[1] Per suite × category success rate (%)")
    header = ["suite"] + categories + ["suite_overall"]
    print("\t".join(header))
    for suite in report["suites"]:
        row = [suite]
        suite_stats = report["per_suite_per_category"][suite]
        for cat in categories:
            rate = suite_stats["by_category"][cat]["success_rate"]
            row.append(f"{rate:.2f}")
        row.append(f"{suite_stats['success_rate']:.2f}")
        print("\t".join(row))

    print("\n[2] Overall per category (libero_mix pooled)")
    for cat in categories:
        stats = report["per_category_overall_libero_mix"][cat]
        print(
            f"- {cat}: {stats['success_rate']:.2f}% "
            f"({stats['successes']}/{stats['trials']}, tasks={stats['tasks']})"
        )

    grand = report["grand_overall"]
    print("\n[3] Grand overall success rate (all suites × all categories)")
    print(
        f"- {grand['success_rate']:.2f}% "
        f"({grand['successes']}/{grand['trials']}, tasks={grand['tasks']})"
    )


def _save_libero_plus_artifacts(output_dir: str, report: dict) -> None:
    summary_path = os.path.join(output_dir, "libero_plus_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    categories = report["categories"]
    suites = list(report["suites"]) + ["libero_mix", "grand_overall"]

    rows = []
    for suite in report["suites"]:
        suite_stats = report["per_suite_per_category"][suite]
        row = {"row": suite}
        for cat in categories:
            row[cat] = f"{suite_stats['by_category'][cat]['success_rate']:.2f}"
        row["overall"] = f"{suite_stats['success_rate']:.2f}"
        rows.append(row)

    mix_row = {"row": "libero_mix (per category overall)"}
    for cat in categories:
        mix_row[cat] = (
            f"{report['per_category_overall_libero_mix'][cat]['success_rate']:.2f}"
        )
    mix_row["overall"] = ""
    rows.append(mix_row)

    grand = report["grand_overall"]
    rows.append(
        {
            "row": "grand_overall",
            **{cat: "" for cat in categories},
            "overall": f"{grand['success_rate']:.2f}",
        }
    )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "libero_plus_summary_matrix.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {csv_path}")


def summarize_results(output_dir):
    """Summarize evaluation results (legacy + LIBERO-Plus matrix)."""
    if _is_libero_plus_layout(output_dir):
        summarize_libero_plus_results(output_dir)
        return

    suite_stats = defaultdict(
        lambda: {
            "total_tasks": 0,
            "total_trials": 0,
            "total_successes": 0,
            "total_time": 0,
            "max_time": 0,
            "psnr_sum": 0.0,
            "psnr_count": 0,
        }
    )
    task_results = {}
    has_psnr_metric = False

    for suite, _, _, result_path in _collect_result_files(output_dir):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        parts = os.path.basename(result_path).split("_")
        task_id = int(parts[1].replace("task", ""))
        suite_key = suite

        stats = suite_stats[suite_key]
        stats["total_tasks"] += 1
        stats["total_trials"] += result["total_episodes"]
        stats["total_successes"] += result["successes"]
        stats["total_time"] += result["duration"]
        stats["max_time"] = max(stats["max_time"], result["duration"])
        if "future_video_psnr_mean" in result:
            has_psnr_metric = True
            if result["future_video_psnr_mean"] is not None:
                stats["psnr_sum"] += float(result["future_video_psnr_mean"])
                stats["psnr_count"] += 1

        task_key = f"{suite_key}_{task_id}"
        task_results[task_key] = {
            "success_rate": result["successes"] / result["total_episodes"] * 100,
            "duration": result["duration"],
            "total_episodes": result["total_episodes"],
            "successes": result["successes"],
            "task_description": result.get("task_description", ""),
        }

    print("\n=== Evaluation Results Summary ===")
    if not suite_stats:
        print("No result files found.")
        return

    for suite, stats in suite_stats.items():
        if stats["total_trials"] <= 0:
            continue
        success_rate = stats["total_successes"] / stats["total_trials"] * 100
        print(f"\n{suite}: {success_rate:.2f}% ({stats['total_successes']}/{stats['total_trials']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory containing evaluation results",
    )
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
