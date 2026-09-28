from __future__ import annotations

from typing import Any, Final, Mapping

import torch

from .policy import ActionEvalPolicy, DEFER_VARIANTS, MODEL_VARIANTS
from .literal_policy import ActionEvalLiteralMLPPolicy
from .neurocore import NeuroCoreBackbone
from .neurosat import NeuroSATBackbone


MODEL_FAMILY: Final[str] = "SAT-ACT"
READOUT_CONTEXTUAL: Final[str] = "contextual"
READOUT_LITERAL_MLP: Final[str] = "literal-mlp"
READOUTS: Final[tuple[str, ...]] = (
    READOUT_CONTEXTUAL,
    READOUT_LITERAL_MLP,
)
FEATURE_SCHEMA: Final[str] = "satact-portable-features"
FEATURE_WHITELIST: Final[tuple[str, ...]] = (
    "solver_active_cnf",
    "assignment_value",
    "assignment_level",
    "decision_level",
    "literal_occurrence",
    "derived_graph_statistics",
)


def _value(opts: Mapping[str, Any] | object, name: str, default: Any) -> Any:
    return opts.get(name, default) if isinstance(opts, Mapping) else getattr(opts, name, default)


def canonicalize_readout(value: Any = None) -> str:
    readout = READOUT_CONTEXTUAL if value is None else str(value).lower()
    if readout not in READOUTS:
        raise ValueError(f"unsupported SAT-ACT readout: {readout!r}")
    return readout


def build_actioneval_model(opts: Mapping[str, Any] | object) -> torch.nn.Module:
    """Build a SAT-ACT branching policy from CLI or checkpoint options."""

    variant = str(_value(opts, "variant", "basic-action"))
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unsupported SAT-ACT variant: {variant!r}")
    embedding_dim = int(_value(opts, "embedding_dim", 64))
    num_rounds = int(_value(opts, "num_rounds", 4))
    mlp_layers = int(_value(opts, "mlp_layers", 2))
    readout = canonicalize_readout(_value(opts, "readout", None))
    if readout == READOUT_LITERAL_MLP and variant in DEFER_VARIANTS:
        raise ValueError(
            f"literal-mlp readout does not support DEFER variant {variant!r}"
        )
    backbone_name = str(_value(opts, "backbone", _value(opts, "model", "neurocore"))).lower()
    common = dict(
        embedding_dim=embedding_dim,
        num_rounds=num_rounds,
        mlp_layers=mlp_layers,
        use_layer_norm=bool(_value(opts, "use_layer_norm", True)),
    )
    if backbone_name == "neurocore":
        backbone = NeuroCoreBackbone(
            shared_updates=bool(_value(opts, "shared_updates", False)),
            use_residual=bool(_value(opts, "use_residual", True)),
            **common,
        )
    elif backbone_name == "neurosat":
        backbone = NeuroSATBackbone(**common)
    else:
        raise ValueError(f"unsupported graph backbone: {backbone_name!r}")
    policy_type = (
        ActionEvalPolicy
        if readout == READOUT_CONTEXTUAL
        else ActionEvalLiteralMLPPolicy
    )
    return policy_type(
        backbone,
        embedding_dim=embedding_dim,
        variant=variant,
        mlp_layers=mlp_layers,
    )


__all__ = [
    "FEATURE_SCHEMA",
    "FEATURE_WHITELIST",
    "MODEL_FAMILY",
    "READOUT_CONTEXTUAL",
    "READOUT_LITERAL_MLP",
    "READOUTS",
    "MODEL_VARIANTS",
    "build_actioneval_model",
    "canonicalize_readout",
]
