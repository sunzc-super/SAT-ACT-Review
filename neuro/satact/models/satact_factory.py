"""Model factory owned by SAT-ACT while reusing the stable satact network."""

from __future__ import annotations

from typing import Any, Final, Mapping

import torch

from satact.models.model_factory import (
    FEATURE_SCHEMA,
    FEATURE_WHITELIST,
    build_actioneval_model,
)


MODEL_FAMILY_SATACT: Final[str] = "SAT-ACT"
FEATURE_SCHEMA_SATACT: Final[str] = FEATURE_SCHEMA
FEATURE_WHITELIST_SATACT: Final[tuple[str, ...]] = FEATURE_WHITELIST


def _options(opts: Mapping[str, Any] | object) -> dict[str, Any]:
    values = dict(opts) if isinstance(opts, Mapping) else dict(vars(opts))
    values["variant"] = "satact"
    return values


def build_actioneval_model_satact(opts: Mapping[str, Any] | object) -> torch.nn.Module:
    return build_actioneval_model(_options(opts))


__all__ = [
    "FEATURE_SCHEMA_SATACT",
    "FEATURE_WHITELIST_SATACT",
    "MODEL_FAMILY_SATACT",
    "build_actioneval_model_satact",
]
