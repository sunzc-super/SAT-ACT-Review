#!/usr/bin/env python3
"""Summarize grouped SAT-ACT results against the CaDiCaL baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.evaluation.common import FAMILIES, SPLITS, write_csv  # noqa: E402
from solver.evaluation.run_evaluation import BASELINE_NAME  # noqa: E402


SUBSETS = ("ALL", "SAT", "UNSAT")
METRICS = (
    "delta_conflicts",
    "delta_decisions",
    "delta_propagations",
    "median_propagation_ratio",
    "win_rate_1pct",
    "wins",
    "ties",
    "losses",
    "sat",
    "unsat",
    "unknown",
    "mean_wall_time",
    "mean_inference_time",
    "mean_request_time",
    "mean_model_calls",
)


def read_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or not all(row.get("complete") for row in rows):
        raise RuntimeError(f"incomplete results: {path}")
    return sorted(rows, key=lambda row: row["instance"])


def subset(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "ALL":
        return rows
    return [row for row in rows if row["label"] == name.lower()]


def calculate(
    current: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> dict[str, float]:
    if [row["instance"] for row in current] != [row["instance"] for row in baseline]:
        raise RuntimeError("model and baseline instance sets differ")
    result: dict[str, float] = {}
    for metric, field in (
        ("delta_conflicts", "conflicts"),
        ("delta_decisions", "decisions"),
        ("delta_propagations", "propagations"),
    ):
        model_total = sum(row["result"][field] for row in current)
        baseline_total = sum(row["result"][field] for row in baseline)
        result[metric] = (
            100.0 * (model_total / baseline_total - 1.0)
            if baseline_total
            else float("nan")
        )
    model_propagations = [row["result"]["propagations"] for row in current]
    baseline_propagations = [row["result"]["propagations"] for row in baseline]
    ratios = [
        model / control
        for model, control in zip(model_propagations, baseline_propagations)
        if control > 0
    ]
    result["median_propagation_ratio"] = (
        statistics.median(ratios) if ratios else float("nan")
    )
    result["win_rate_1pct"] = 100.0 * statistics.mean(
        (model + 1) / (control + 1) < 0.99
        for model, control in zip(model_propagations, baseline_propagations)
    )
    result["wins"] = float(
        sum(model < control for model, control in zip(model_propagations, baseline_propagations))
    )
    result["ties"] = float(
        sum(model == control for model, control in zip(model_propagations, baseline_propagations))
    )
    result["losses"] = float(
        sum(model > control for model, control in zip(model_propagations, baseline_propagations))
    )
    for status in ("SAT", "UNSAT", "UNKNOWN"):
        result[status.lower()] = float(
            sum(row["result"]["status"] == status for row in current)
        )
    result["mean_wall_time"] = statistics.mean(
        row["result"].get("wall_time", 0.0) for row in current
    )
    result["mean_inference_time"] = statistics.mean(
        row["result"].get("inference_time", 0.0) for row in current
    )
    result["mean_request_time"] = statistics.mean(
        row["result"].get("request_time", 0.0) for row in current
    )
    result["mean_model_calls"] = statistics.mean(
        row["result"].get("model_calls", 0) for row in current
    )
    return result


def format_value(mean: float, standard_deviation: float, percent: bool = False) -> str:
    suffix = "%" if percent else ""
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.3f} ± {standard_deviation:.3f}{suffix}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--splits", choices=SPLITS, nargs="+", default=list(SPLITS))
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--neural-calls", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    per_group: list[dict[str, object]] = []
    for split in args.splits:
        for group in args.groups:
            root = args.results_dir / args.family / split / f"group_{group}"
            baseline = read_results(root / BASELINE_NAME / "results.jsonl")
            for calls in args.neural_calls:
                current = read_results(
                    root / args.model_name / f"calls_{calls}" / "results.jsonl"
                )
                for subset_name in SUBSETS:
                    metrics = calculate(
                        subset(current, subset_name), subset(baseline, subset_name)
                    )
                    per_group.append(
                        {
                            "family": args.family,
                            "split": split,
                            "group": group,
                            "solver_seed": group,
                            "model": args.model_name,
                            "neural_calls": calls,
                            "subset": subset_name,
                            **metrics,
                        }
                    )

    per_group_fields = (
        "family",
        "split",
        "group",
        "solver_seed",
        "model",
        "neural_calls",
        "subset",
        *METRICS,
    )
    write_csv(args.output_dir / "per_group_metrics.csv", per_group, per_group_fields)

    aggregate: list[dict[str, object]] = []
    for split in args.splits:
        for calls in args.neural_calls:
            for subset_name in SUBSETS:
                selected = [
                    row
                    for row in per_group
                    if row["split"] == split
                    and row["neural_calls"] == calls
                    and row["subset"] == subset_name
                ]
                if len(selected) != len(args.groups):
                    raise RuntimeError("group coverage mismatch")
                for metric in METRICS:
                    values = [float(row[metric]) for row in selected]
                    aggregate.append(
                        {
                            "family": args.family,
                            "split": split,
                            "model": args.model_name,
                            "neural_calls": calls,
                            "subset": subset_name,
                            "metric": metric,
                            "mean": statistics.mean(values),
                            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                        }
                    )
    aggregate_fields = (
        "family",
        "split",
        "model",
        "neural_calls",
        "subset",
        "metric",
        "mean",
        "sd",
    )
    write_csv(args.output_dir / "mean_sd.csv", aggregate, aggregate_fields)

    lookup = {
        (row["split"], row["neural_calls"], row["subset"], row["metric"]): row
        for row in aggregate
    }
    lines = [
        f"# {args.family.upper()} solver evaluation",
        "",
        "Results are measured relative to the CaDiCaL baseline.",
        "",
    ]
    for split in args.splits:
        lines.extend(
            [
                f"## {split.title()}",
                "",
                "| Calls | Subset | Conflicts | Decisions | Propagations | Median propagation ratio | Win rate |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for calls in args.neural_calls:
            for subset_name in SUBSETS:
                values = []
                for metric in (
                    "delta_conflicts",
                    "delta_decisions",
                    "delta_propagations",
                    "median_propagation_ratio",
                    "win_rate_1pct",
                ):
                    row = lookup[(split, calls, subset_name, metric)]
                    values.append(
                        format_value(
                            float(row["mean"]),
                            float(row["sd"]),
                            metric.startswith("delta_") or metric == "win_rate_1pct",
                        )
                    )
                lines.append(
                    f"| {calls} | {subset_name} | " + " | ".join(values) + " |"
                )
        lines.append("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"summary={args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
