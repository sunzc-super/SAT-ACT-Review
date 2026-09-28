#!/usr/bin/env python3
"""Launch SAT-ACT training from an anonymous experiment configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocess-workers", type=int, default=1)
    parser.add_argument("--max-train-states", type=int, default=0)
    parser.add_argument("--max-valid-states", type=int, default=0)
    parser.add_argument("--max-test-states", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    public_id = str(config["public_id"])
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    split_dirs = {
        "train": args.train_dir,
        "valid": args.valid_dir,
        "test": args.test_dir,
    }
    index_paths = {
        split: args.cache_dir / f"{public_id}_{split}.pt"
        for split, data_dir in split_dirs.items()
        if data_dir is not None
    }
    preprocess = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "preprocess.py"),
        "--index-contract",
        "heuristic-supervision",
        "--workers",
        str(args.preprocess_workers),
    ]
    for split, data_dir in split_dirs.items():
        if data_dir is not None:
            preprocess.extend(
                [
                    f"--{split}_dir",
                    str(data_dir),
                    f"--{split}_index_path",
                    str(index_paths[split]),
                    f"--max_{split}_states",
                    str(getattr(args, f"max_{split}_states")),
                ]
            )
    subprocess.run(preprocess, check=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "train.py"),
        "--train_dir",
        str(args.train_dir),
        "--train_index_cache_path",
        str(index_paths["train"]),
        "--run_root",
        str(args.output_dir),
        "--experiment_key",
        public_id,
        "--device",
        args.device,
        "--index_contract",
        "heuristic-supervision",
        "--heuristic_action_source",
        "replay",
        "--heuristic_supervision_weight",
        str(config["heuristic_supervision_weight"]),
        "--heuristic_supervision_mode",
        str(config["heuristic_supervision_mode"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--num_rounds",
        str(config["num_rounds"]),
        "--embedding_dim",
        str(config["embedding_dim"]),
        "--batch_size",
        str(config["batch_size"]),
        "--epochs",
        str(args.epochs if args.epochs is not None else config["epochs"]),
        "--lr",
        str(config["lr"]),
        "--seed",
        str(config["seed"]),
        "--max_train_states",
        str(args.max_train_states),
        "--max_valid_states",
        str(args.max_valid_states),
        "--max_test_states",
        str(args.max_test_states),
    ]
    for split in ("valid", "test"):
        data_dir = split_dirs[split]
        if data_dir is not None:
            command.extend(
                [
                    f"--{split}_dir",
                    str(data_dir),
                    f"--{split}_index_cache_path",
                    str(index_paths[split]),
                ]
            )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
