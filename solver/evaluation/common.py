"""Shared manifest and output helpers for solver evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


FAMILIES = ("ca", "sr", "ps")
SPLITS = ("valid", "test")
LABELS = ("sat", "unsat")
MANIFEST_FIELDS = (
    "family",
    "split",
    "label",
    "group",
    "solver_seed",
    "relative_path",
)


def load_manifest(data_dir: Path) -> list[dict[str, str]]:
    path = data_dir / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"unexpected manifest columns in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def select_group(
    rows: Iterable[dict[str, str]], family: str, split: str, group: int
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and row["split"] == split
        and int(row["group"]) == group
    ]
    return sorted(selected, key=lambda row: (row["label"], row["relative_path"]))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
