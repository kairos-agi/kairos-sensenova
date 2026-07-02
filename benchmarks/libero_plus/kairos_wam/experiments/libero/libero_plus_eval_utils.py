"""Helpers for LIBERO-Plus multi-suite / multi-category evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

LIBERO_PLUS_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

_DEFAULT_LIBERO_PKG_ROOT = Path(
    "/path/to/LIBERO-plus"
)
LIBERO_PKG_ROOT = Path(os.environ.get("LIBERO_PKG_ROOT", _DEFAULT_LIBERO_PKG_ROOT))

TASK_CLASSIFICATION_PATH = (
    LIBERO_PKG_ROOT
    / "libero"
    / "libero"
    / "benchmark"
    / "task_classification.json"
)


def sanitize_category_value(category_value: str) -> str:
    return str(category_value).strip().replace(" ", "_")


def get_all_category_values(
    classification_path: Path | None = None,
) -> list[str]:
    path = classification_path or TASK_CLASSIFICATION_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories: set[str] = set()
    for tasks in data.values():
        for task in tasks:
            cat = task.get("category")
            if cat:
                categories.add(str(cat))
    return sorted(categories)


def resolve_category_values(
    category_values: Iterable[str] | None,
) -> list[str]:
    if category_values is None:
        return get_all_category_values()
    resolved = [str(v).strip() for v in category_values if str(v).strip()]
    if not resolved:
        return get_all_category_values()
    return resolved
