#!/usr/bin/env python3
"""Prepare grouped G4SATBench Medium instances for solver evaluation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.evaluation.common import (
    FAMILIES,
    LABELS,
    MANIFEST_FIELDS,
    SPLITS,
)  # noqa: E402


def build_manifest(
    sources: dict[str, Path],
    output_dir: Path,
    seed: int,
    groups: int,
    instances_per_label: int,
) -> list[dict[str, object]]:
    master = random.Random(seed)
    rows: list[dict[str, object]] = []
    required = groups * instances_per_label
    for family in FAMILIES:
        for split in SPLITS:
            for label in LABELS:
                source_dir = sources[family] / split / label
                names = sorted(
                    path.name
                    for path in source_dir.iterdir()
                    if path.is_file() and path.suffix == ".cnf"
                )
                if len(names) < required:
                    raise ValueError(
                        f"{source_dir} contains {len(names)} CNFs; {required} required"
                    )
                rng = random.Random(master.getrandbits(64))
                selected = rng.sample(names, required)
                rng.shuffle(selected)
                for index, name in enumerate(selected):
                    group = index // instances_per_label
                    source = (source_dir / name).resolve()
                    relative = Path(family) / split / f"group_{group}" / label / name
                    destination = output_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.is_symlink():
                        if destination.resolve() != source:
                            raise ValueError(f"existing link has a different target: {destination}")
                    elif destination.exists():
                        raise ValueError(f"expected a symbolic link: {destination}")
                    else:
                        destination.symlink_to(source)
                    rows.append(
                        {
                            "family": family,
                            "split": split,
                            "label": label,
                            "group": group,
                            "solver_seed": group,
                            "relative_path": relative.as_posix(),
                        }
                    )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = output_dir / "manifest.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-source", type=Path, required=True)
    parser.add_argument("--sr-source", type=Path, required=True)
    parser.add_argument("--ps-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--instances-per-label", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.groups < 1 or args.instances_per_label < 1:
        raise ValueError("groups and instances-per-label must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(
        {"ca": args.ca_source, "sr": args.sr_source, "ps": args.ps_source},
        args.output_dir,
        args.seed,
        args.groups,
        args.instances_per_label,
    )
    path = write_manifest(args.output_dir, rows)
    print(f"prepared {len(rows)} instances")
    print(f"manifest={path}")


if __name__ == "__main__":
    main()
