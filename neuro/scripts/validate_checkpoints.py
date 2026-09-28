#!/usr/bin/env python3
"""Load selected archived checkpoints and run deterministic branch queries."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

NEURO_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = NEURO_ROOT.parent
for search_path in (REPOSITORY_ROOT, NEURO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from satact.serving.inference import SATActInference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path)
    source.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = (
        sorted(args.checkpoint_dir.glob("*.pt"))
        if args.checkpoint_dir is not None
        else list(args.checkpoint)
    )
    if not checkpoints:
        raise ValueError("no checkpoints selected")
    results = []
    for checkpoint in checkpoints:
        inference = SATActInference(checkpoint, args.device)
        prediction = inference.predict_branch(
            model_variant="satact",
            full=True,
            n_vars=2,
            n_clauses=2,
            c_idxs=[0, 0, 1, 1],
            l_idxs=[0, 3, 1, 2],
            assignment_value=[0, 0],
            assignment_level=[-1, -1],
            decision_level=0,
            candidate_variable=[True, True],
        )
        results.append(
            {
                "checkpoint": checkpoint.name,
                "selected_literal_index": prediction.selected_literal_index,
                "selected_log_probability": prediction.selected_log_probability,
                "logits": prediction.action_logits,
            }
        )
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
