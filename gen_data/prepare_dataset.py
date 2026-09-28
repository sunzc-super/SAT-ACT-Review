#!/usr/bin/env python3
"""Run the SAT-ACT action-rollout and preference-dataset pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROPAGATION_HORIZON = {"ca": 1000, "sr": 5000, "ps": 15000}


def run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("prepare", "run", "shards", "all"))
    result.add_argument("--family", required=True, choices=tuple(PROPAGATION_HORIZON))
    result.add_argument("--split", required=True)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    python = sys.executable
    manifest = args.output_dir / "manifests" / "instances.jsonl"
    policy = args.output_dir / "manifests" / "pipeline.policy.json"
    plan = args.output_dir / "manifests" / "pipeline.parts.jsonl"

    if args.stage in {"prepare", "all"}:
        run(
            [
                python,
                "-m",
                "gen_data.action_rollouts",
                "manifest-satbench",
                "--split",
                f"{args.split}={args.input_dir}",
                "--difficulty",
                "medium",
                "--family",
                args.family,
                "--dataset-release-id",
                "external-satbench",
                "--output-tag",
                "satact",
                "--output-path",
                str(args.output_dir),
                "--workers",
                str(args.workers),
                "--part-size",
                "512",
                "--max-input-nodes",
                "100000",
                "--max-literal-occurrences",
                "4000000",
                "--max-input-cells",
                "none",
                "--max-cnf-bytes",
                "none",
                "--max-clause-length",
                "none",
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                python,
                "-m",
                "gen_data.action_rollouts",
                "pipeline-prepare",
                "--preset",
                "satbench",
                "--plain",
                "--output-tag",
                "satact",
                "--binary",
                str(args.binary),
                "--build-root",
                str(args.output_dir),
                "--manifest",
                str(manifest),
                "--policy-path",
                str(policy),
                "--plan-path",
                str(plan),
                "--target-initial-keep-count",
                "2",
                "--target-early-window-end",
                "8",
                "--target-early-sample-count",
                "2",
                "--target-post-restart-keep-count",
                "2",
                "--target-late-fallback-window-end",
                "20",
                "--max-state-nodes",
                "100000",
                "--max-state-literal-occurrences",
                "4000000",
                "--max-snapshot-bytes",
                "268435456",
                "--max-state-clause-length",
                "0",
                "--max-prefix-conflicts",
                "10000",
                "--max-prefix-decisions",
                "0",
                "--max-prefix-propagations",
                "10000000",
                "--max-prefix-callbacks",
                "1000",
                "--activity-top-k",
                "8",
                "--jw-top-k",
                "8",
                "--random-top-k",
                "8",
                "--max-actions",
                "8",
                "--max-actions-mode",
                "variable",
                "--eval-max-conflicts",
                "0",
                "--eval-max-decisions",
                "0",
                "--eval-max-propagations",
                str(PROPAGATION_HORIZON[args.family]),
                "--eval-max-timeout",
                "300",
                "--seed",
                str(args.seed),
                "--solver-set",
                "stabilize=1",
                "--solver-set",
                "stabilizeonly=1",
                "--solver-set",
                "score=1",
                "--solver-set",
                "shuffle=0",
                "--cnf-cache-root",
                str(args.cache_dir),
                "--timeout-seconds",
                "500",
                "--raw-codec",
                "gzip",
            ],
            dry_run=args.dry_run,
        )

    if args.stage in {"run", "all"}:
        run(
            [
                python,
                "-m",
                "gen_data.action_rollouts",
                "pipeline-run-all",
                "--policy",
                str(policy),
                "--plan",
                str(plan),
                "--workers",
                str(args.workers),
                "--fail-fast",
            ],
            dry_run=args.dry_run,
        )

    if args.stage in {"shards", "all"}:
        run(
            [
                python,
                "-m",
                "gen_data.rollout_dataset",
                "build-shards",
                "--source-root",
                str(args.output_dir),
                "--parts-per-shard",
                "10",
                "--workers",
                str(args.workers),
                "--fail-fast",
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
