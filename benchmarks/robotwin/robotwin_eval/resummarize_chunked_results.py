#!/usr/bin/env python3
"""Re-aggregate RoboTwin chunk25 results and write summary.csv in sort_result.TARGET_ORDER."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Keep in sync with sort_result.py
TARGET_ORDER = [
    "AdjustBottle",
    "BeatBlockHammer",
    "BlocksRankingRGB",
    "BlocksRankingSize",
    "ClickAlarmclock",
    "ClickBell",
    "DumpBinBigbin",
    "GrabRoller",
    "HandoverBlock",
    "HandoverMic",
    "HangingMug",
    "LiftPot",
    "MoveCanPot",
    "MovePillbottlePad",
    "MovePlayingcardAway",
    "MoveStaplerPad",
    "OpenLaptop",
    "OpenMicrowave",
    "PickDiverseBottles",
    "PickDualBottles",
    "PlaceA2BLeft",
    "PlaceA2BRight",
    "PlaceBreadBasket",
    "PlaceBreadSkillet",
    "PlaceBurgerFries",
    "PlaceCanBasket",
    "PlaceCansPlasticbox",
    "PlaceContainerPlate",
    "PlaceDualShoes",
    "PlaceEmptyCup",
    "PlaceFan",
    "PlaceMousePad",
    "PlaceObjectBasket",
    "PlaceObjectScale",
    "PlaceObjectStand",
    "PlacePhoneStand",
    "PlaceShoe",
    "PressStapler",
    "PutBottlesDustbin",
    "PutObjectCabinet",
    "RotateQRcode",
    "ScanObject",
    "ShakeBottle",
    "ShakeBottleHorizontally",
    "StackBlocksThree",
    "StackBlocksTwo",
    "StackBowlsThree",
    "StackBowlsTwo",
    "StampSeal",
    "TurnSwitch",
]

PHASES = ("clean", "random")


def camel_to_snake(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def find_chunk_jsons(root: Path, snake_task: str, phase: str) -> list[Path]:
    phase_dir = root / snake_task / phase
    if not phase_dir.is_dir():
        return []
    return sorted(phase_dir.glob(f"chunk_*/{snake_task}/chunk_result.json"))


def aggregate_from_chunks(root: Path) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for camel in TARGET_ORDER:
        snake = camel_to_snake(camel)
        out[snake] = {
            phase: {"success_count": 0, "episode_count": 0, "success_rate": None}
            for phase in PHASES
        }
        for phase in PHASES:
            for jpath in find_chunk_jsons(root, snake, phase):
                try:
                    payload = json.loads(jpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                ep = int(payload.get("episode_count", 0) or 0)
                suc = int(payload.get("success_count", 0) or 0)
                out[snake][phase]["episode_count"] = int(out[snake][phase]["episode_count"]) + ep
                out[snake][phase]["success_count"] = int(out[snake][phase]["success_count"]) + suc
            ep_total = int(out[snake][phase]["episode_count"])
            suc_total = int(out[snake][phase]["success_count"])
            if ep_total > 0:
                out[snake][phase]["success_rate"] = suc_total / ep_total
    return out


def mean_rate(values: list[float | None]) -> str:
    xs = [v for v in values if v is not None]
    return "" if not xs else str(sum(xs) / len(xs))


def write_summary(
    root: Path,
    output: Path,
    *,
    task_name_style: str = "camel",
    include_counts: bool = False,
) -> None:
    agg = aggregate_from_chunks(root)
    rows: list[dict[str, str]] = []
    clean_rates: list[float | None] = []
    random_rates: list[float | None] = []

    for camel in TARGET_ORDER:
        snake = camel_to_snake(camel)
        clean = agg[snake]["clean"]
        random_phase = agg[snake]["random"]
        cr = clean["success_rate"]
        rr = random_phase["success_rate"]
        clean_rates.append(cr if isinstance(cr, float) else None)
        random_rates.append(rr if isinstance(rr, float) else None)

        display_name = camel if task_name_style == "camel" else snake
        row = {
            "task_name": display_name,
            "clean_success_rate": "" if cr is None else str(cr),
            "random_success_rate": "" if rr is None else str(rr),
        }
        if include_counts:
            row.update(
                {
                    "clean_success_count": str(clean["success_count"]),
                    "clean_episode_count": str(clean["episode_count"]),
                    "random_success_count": str(random_phase["success_count"]),
                    "random_episode_count": str(random_phase["episode_count"]),
                }
            )
        rows.append(row)

    overall_row: dict[str, str] = {
        "task_name": "__overall__",
        "clean_success_rate": mean_rate(clean_rates),
        "random_success_rate": mean_rate(random_rates),
    }
    if include_counts:
        overall_row.update(
            {
                "clean_success_count": "",
                "clean_episode_count": "",
                "random_success_count": "",
                "random_episode_count": "",
            }
        )
    rows.append(overall_row)

    fieldnames = ["task_name", "clean_success_rate", "random_success_rate"]
    if include_counts:
        fieldnames += [
            "clean_success_count",
            "clean_episode_count",
            "random_success_count",
            "random_episode_count",
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output} ({len(rows) - 1} tasks + overall)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-aggregate chunk25 RoboTwin results into summary.csv (TARGET_ORDER)."
    )
    parser.add_argument(
        "root_dir",
        help="RoboTwin benchmark output directory (e.g. .../external_model/RUN_NAME)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output csv path (default: <root_dir>/summary.csv)",
    )
    parser.add_argument(
        "--task-name-style",
        choices=("camel", "snake"),
        default="camel",
        help="Task name column format (default: camel, matches sort_result.py / Feishu)",
    )
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Add success/episode count columns (manager-style 7 columns)",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "summary.csv"
    )

    write_summary(
        root,
        output,
        task_name_style=args.task_name_style,
        include_counts=args.include_counts,
    )


if __name__ == "__main__":
    main()
