#!/usr/bin/env python3
"""Build SAT-ACT core indexes and diagnostics sidecars before training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NEURO_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = NEURO_ROOT.parent
for search_path in (REPOSITORY_ROOT, NEURO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from satact.dataset.action_preferences import (
    ActionEvalShardDatasetSATACT,
    SATACT_VARIANT,
    canonicalize_satact_variant,
)
from satact.dataset.heuristic_supervision import (
    ActionEvalShardDatasetSATACTMultiNative,
    SATACT_INDEX_CONTRACTS,
    canonicalize_satact_index_contract,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=SATACT_VARIANT, choices=(SATACT_VARIANT,))
    parser.add_argument(
        "--index-contract",
        dest="index_contract",
        choices=SATACT_INDEX_CONTRACTS,
        default="preference-only",
    )
    for split in ("train", "valid", "test"):
        parser.add_argument(f"--{split}_dir", required=split == "train")
        parser.add_argument(f"--{split}_index_path", required=split == "train")
        parser.add_argument(f"--max_{split}_states", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify", choices=("none", "metadata"), default="metadata")
    parser.add_argument("--max_open_shards", type=int, default=2)
    parser.add_argument("--max_header_mb", type=int, default=1024)
    parser.add_argument("--max_array_mb", type=int, default=512)
    parser.add_argument("--progress_every_parts", type=int, default=1)
    parser.add_argument("--summary_json", default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    canonicalize_satact_variant(args.variant)
    canonicalize_satact_index_contract(args.index_contract)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for name in ("max_open_shards", "max_header_mb", "max_array_mb"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    for split in ("train", "valid", "test"):
        data_dir = getattr(args, f"{split}_dir")
        index_path = getattr(args, f"{split}_index_path")
        if (data_dir is None) != (index_path is None):
            raise ValueError(f"--{split}_dir and --{split}_index_path must be supplied together")
        if getattr(args, f"max_{split}_states") < 0:
            raise ValueError(f"--max_{split}_states must be non-negative")
    if args.progress_every_parts < 0:
        raise ValueError("--progress_every_parts must be non-negative")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("SATACT preprocessing runs directly, before torchrun")


def _progress(split: str, every: int):
    def report(event: Any) -> None:
        if every and (
            event.part_index == event.part_count or event.part_index % every == 0
        ):
            print(
                f"[{split}] INDEX part={event.part_index}/{event.part_count} "
                f"stem={event.part_stem} states={event.states}",
                flush=True,
            )

    return report


def preprocess(args: argparse.Namespace) -> list[dict[str, Any]]:
    validate_args(args)
    results: list[dict[str, Any]] = []
    dataset_class = (
        ActionEvalShardDatasetSATACT
        if args.index_contract == "preference-only"
        else ActionEvalShardDatasetSATACTMultiNative
    )
    for split in ("train", "valid", "test"):
        if getattr(args, f"{split}_dir") is None:
            continue
        started = time.perf_counter()
        limit_value = int(getattr(args, f"max_{split}_states"))
        dataset = dataset_class(
            getattr(args, f"{split}_dir"),
            index_cache_path=getattr(args, f"{split}_index_path"),
            expected_split=split,
            verify_metadata=args.verify != "none",
            verify_checksums=False,
            preprocess_workers=args.workers,
            max_states=limit_value or None,
            max_open_shards=args.max_open_shards,
            max_header_bytes=args.max_header_mb * (1 << 20),
            max_array_bytes=args.max_array_mb * (1 << 20),
            progress_callback=_progress(split, args.progress_every_parts),
        )
        try:
            row = {
                "split": split,
                "variant": SATACT_VARIANT,
                "index_contract": args.index_contract,
                "data_dir": str(dataset.split_dir),
                "index_path": str(dataset.index_cache_path),
                "diagnostics_path": str(
                    Path(dataset.index_cache_path).with_name(
                        Path(dataset.index_cache_path).name.removesuffix(".index.pt")
                        + ".diagnostics.pt"
                    )
                ),
                "cache_status": "hit" if dataset.index_cache_hit else "built",
                "states": len(dataset),
                "instances": dataset.instance_count,
                "parts": dataset.part_count,
                "workers": args.workers,
                "elapsed_seconds": time.perf_counter() - started,
            }
            results.append(row)
            print(
                f"[{split}] COMPLETE cache={row['cache_status']} states={row['states']} "
                f"instances={row['instances']} elapsed={row['elapsed_seconds']:.3f}s",
                flush=True,
            )
        finally:
            dataset.close()
    return results


def main() -> int:
    args = build_arg_parser().parse_args()
    results = preprocess(args)
    if args.summary_json:
        destination = Path(args.summary_json).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "satact-preprocess-summary",
                    "variant": SATACT_VARIANT,
                    "index_contract": args.index_contract,
                    "splits": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
