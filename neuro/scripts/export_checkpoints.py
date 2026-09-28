#!/usr/bin/env python3
"""Export trained weights into the anonymous SAT-ACT checkpoint schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ARCHITECTURE_KEYS = (
    "model",
    "embedding_dim",
    "num_rounds",
    "mlp_layers",
    "readout",
    "shared_updates",
    "use_layer_norm",
    "use_residual",
)
TRAINING_KEYS = (
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "seed",
)


def parse_item(value: str) -> tuple[str, Path, Path]:
    fields = value.split("|", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("item must be PUBLIC_ID|SOURCE|DESTINATION")
    return fields[0], Path(fields[1]), Path(fields[2])


def export(public_id: str, source_path: Path, destination: Path) -> dict[str, object]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or not isinstance(source.get("model"), dict):
        raise ValueError(f"invalid source checkpoint for {public_id}")
    old_opts = dict(source.get("opts") or {})
    old_objective = dict(source.get("objective") or {})
    weight = float(old_objective.get("native_weight", old_opts.get("native_weight", 0.0)))
    mode = str(
        old_objective.get("native_supervision", old_opts.get("native_supervision", "always"))
    )
    opts = {key: old_opts[key] for key in (*ARCHITECTURE_KEYS, *TRAINING_KEYS) if key in old_opts}
    opts.update(
        variant="satact",
        heuristic_supervision_weight=weight,
        heuristic_supervision_mode=mode,
    )
    payload = {
        "schema": "satact-checkpoint",
        "model_family": "SAT-ACT",
        "public_id": public_id,
        "epoch": int(source.get("epoch", -1)),
        "global_step": int(source.get("global_step", 0)),
        "feature_schema": "satact-portable-features",
        "feature_whitelist": list(source.get("feature_whitelist") or ()),
        "cnf_view": "solver_active",
        "index_contract": "heuristic-supervision",
        "heuristic_action_source": str(source.get("native_outcome", "replay")),
        "variant": "satact",
        "wire_variant": "satact",
        "readout": str(source.get("readout", old_opts.get("readout", "contextual"))),
        "objective": {
            "heuristic_supervision_weight": weight,
            "heuristic_supervision_mode": mode,
            "top_set_weight": float(
                old_objective.get("top_set_weight", old_opts.get("top_set_weight", 0.0))
            ),
        },
        "opts": opts,
        "model": source["model"],
        "selector": str(source.get("selector", "best_pair_accuracy")),
        "metrics": dict(source.get("metrics") or {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=False)
    if set(restored["model"]) != set(source["model"]):
        raise RuntimeError(f"parameter names changed while exporting {public_id}")
    for name, tensor in source["model"].items():
        if not torch.equal(tensor, restored["model"][name]):
            raise RuntimeError(f"parameter changed while exporting {public_id}: {name}")
    return {
        "public_id": public_id,
        "epoch": payload["epoch"],
        "parameter_tensors": len(payload["model"]),
        "bytes": destination.stat().st_size,
        "weights_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", type=parse_item, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    records = [export(*item) for item in args.item]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
