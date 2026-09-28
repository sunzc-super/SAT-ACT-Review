"""Run directories and an isolated checkpoint schema for ActionEval SAT-ACT."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from satact.dataset.action_preferences import SATACT_VARIANT, ObjectiveConfigSATACT
from satact.models.model_factory import canonicalize_readout
from satact.models.satact_factory import (
    FEATURE_SCHEMA_SATACT,
    FEATURE_WHITELIST_SATACT,
    MODEL_FAMILY_SATACT,
)


CHECKPOINT_SCHEMA_SATACT = "satact-checkpoint"
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MODEL_STRUCTURAL_OPTIONS = (
    "variant",
    "model",
    "embedding_dim",
    "num_rounds",
    "mlp_layers",
    "readout",
    "shared_updates",
    "use_layer_norm",
    "use_residual",
)
_RESUME_STRUCTURAL_OPTIONS = (
    *_MODEL_STRUCTURAL_OPTIONS,
    "amp",
)


@dataclass(frozen=True)
class RunSATACT:
    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    log_path: Path
    metrics_path: Path
    config_path: Path


def _component(name: str, value: str) -> str:
    if not value or _SAFE.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported path characters: {value!r}")
    return value


def create_run_satact(args: Mapping[str, Any] | object, *, run_id: str | None = None) -> RunSATACT:
    opts = dict(args) if isinstance(args, Mapping) else dict(vars(args))
    resolved_id = _component(
        "run_id", run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    experiment = _component("experiment_key", str(opts.get("experiment_key", "default")))
    root = Path(str(opts["run_root"])).expanduser().resolve()
    run_dir = root / SATACT_VARIANT / "train" / "solver_active" / experiment / f"run_{resolved_id}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "config").mkdir()
    run = RunSATACT(
        resolved_id,
        run_dir,
        checkpoint_dir,
        run_dir / "logs" / "train.log",
        run_dir / "metrics" / "metrics.jsonl",
        run_dir / "config" / "run_config.json",
    )
    run.config_path.write_text(
        json.dumps(
            {
                "run_id": resolved_id,
                "command": " ".join([sys.executable, *sys.argv]),
                "args": opts,
                "model_family": MODEL_FAMILY_SATACT,
                "feature_schema": FEATURE_SCHEMA_SATACT,
                "feature_whitelist": list(FEATURE_WHITELIST_SATACT),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def append_metrics_satact(
    run: RunSATACT, *, epoch: int, stage: str, metrics: Mapping[str, Any]
) -> None:
    with run.metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "epoch": int(epoch),
                    "stage": stage,
                    "metrics": dict(metrics),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def capture_rng_satact() -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        values["cuda"] = torch.cuda.get_rng_state_all()
    return values


def restore_rng_satact(values: Mapping[str, Any]) -> None:
    random.setstate(values["python"])
    torch.set_rng_state(values["torch"])
    if "cuda" in values and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(values["cuda"])


def checkpoint_payload_satact(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: Mapping[str, Any] | object,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    split_fingerprints: Mapping[str, str],
    best_scores: Mapping[str, float],
    rng_states: list[Mapping[str, Any]],
    selector: str,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    opts = dict(args) if isinstance(args, Mapping) else dict(vars(args))
    opts["readout"] = canonicalize_readout(opts.get("readout"))
    index_contract = str(
        opts.get("satact_index_contract", "preference-only")
    )
    native_outcome = str(opts.get("satact_native_outcome", "replay"))
    return {
        "schema": CHECKPOINT_SCHEMA_SATACT,
        "model_family": MODEL_FAMILY_SATACT,
        "feature_schema": FEATURE_SCHEMA_SATACT,
        "feature_whitelist": list(FEATURE_WHITELIST_SATACT),
        "cnf_view": "solver_active",
        "index_contract": (
            "satact-preference-index"
            if index_contract == "preference-only"
            else "satact-heuristic-index"
        ),
        "heuristic_action_source": native_outcome,
        "variant": SATACT_VARIANT,
        "wire_variant": "satact",
        "readout": opts["readout"],
        "objective": {
            "heuristic_supervision_weight": float(opts.get("native_weight", 0.0)),
            "heuristic_supervision_mode": str(opts.get("native_supervision", "always")),
            "top_set_weight": float(opts.get("top_set_weight", 0.0)),
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "split_fingerprints": dict(split_fingerprints),
        "best_scores": dict(best_scores),
        "rng_states": list(rng_states),
        "selector": selector,
        "initialization": dict(opts.get("initialization") or {}) or None,
        "opts": opts,
    }


def save_checkpoint_satact(path: Path, **kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint_payload_satact(**kwargs), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint_satact(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_dir():
        for candidate in (source / "latest.pt", source / "checkpoints" / "latest.pt"):
            if candidate.is_file():
                source = candidate
                break
    source = source.resolve(strict=True)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_SATACT:
        raise ValueError(f"not a SAT-ACT checkpoint: {source}")
    return payload


def objective_config_from_checkpoint_satact(
    checkpoint: Mapping[str, Any],
) -> ObjectiveConfigSATACT:
    objective = dict(checkpoint.get("objective") or {})
    opts = dict(checkpoint.get("opts") or {})
    return ObjectiveConfigSATACT(
        native_weight=float(
            objective.get(
                "heuristic_supervision_weight", opts.get("native_weight", 0.1)
            )
        ),
        native_supervision=str(
            objective.get(
                "heuristic_supervision_mode", opts.get("native_supervision", "always")
            )
        ),
        top_set_weight=float(
            objective.get("top_set_weight", opts.get("top_set_weight", 0.0))
        ),
    )


def validate_resume_satact(
    checkpoint: Mapping[str, Any], args: Mapping[str, Any] | object
) -> None:
    opts = dict(args) if isinstance(args, Mapping) else dict(vars(args))
    if checkpoint.get("model_family") != MODEL_FAMILY_SATACT:
        raise ValueError("resume checkpoint is not model_family=SAT-ACT")
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA_SATACT:
        raise ValueError("resume checkpoint uses a different SATACT feature schema")
    old = dict(checkpoint.get("opts") or {})
    old["readout"] = canonicalize_readout(old.get("readout"))
    opts["readout"] = canonicalize_readout(opts.get("readout"))
    mismatches = [
        name
        for name in _RESUME_STRUCTURAL_OPTIONS
        if old.get(name) != opts.get(name)
    ]
    if mismatches:
        raise ValueError("resume structural option mismatch: " + ", ".join(mismatches))


def validate_init_checkpoint_satact(
    checkpoint: Mapping[str, Any], args: Mapping[str, Any] | object
) -> None:
    """Validate a strict full-model warm start without resume semantics."""

    opts = dict(args) if isinstance(args, Mapping) else dict(vars(args))
    if checkpoint.get("model_family") != MODEL_FAMILY_SATACT:
        raise ValueError("init checkpoint is not model_family=SAT-ACT")
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA_SATACT:
        raise ValueError("init checkpoint uses a different SATACT feature schema")
    if tuple(checkpoint.get("feature_whitelist") or ()) != tuple(
        FEATURE_WHITELIST_SATACT
    ):
        raise ValueError("init checkpoint uses a different SATACT feature whitelist")
    if checkpoint.get("cnf_view") != "solver_active":
        raise ValueError("init checkpoint is not solver_active")
    if checkpoint.get("wire_variant") != "satact":
        raise ValueError("init checkpoint does not use wire_variant=satact")

    old = dict(checkpoint.get("opts") or {})
    old["readout"] = canonicalize_readout(old.get("readout"))
    opts["readout"] = canonicalize_readout(opts.get("readout"))
    mismatches = [
        name
        for name in _MODEL_STRUCTURAL_OPTIONS
        if old.get(name) != opts.get(name)
    ]
    if mismatches:
        raise ValueError(
            "init checkpoint structural option mismatch: " + ", ".join(mismatches)
        )


__all__ = [
    "CHECKPOINT_SCHEMA_SATACT",
    "RunSATACT",
    "append_metrics_satact",
    "capture_rng_satact",
    "create_run_satact",
    "load_checkpoint_satact",
    "objective_config_from_checkpoint_satact",
    "restore_rng_satact",
    "save_checkpoint_satact",
    "validate_init_checkpoint_satact",
    "validate_resume_satact",
]
