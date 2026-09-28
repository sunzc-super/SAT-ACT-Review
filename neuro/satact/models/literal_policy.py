from __future__ import annotations

import torch
from torch import nn
from torch.autograd.profiler import record_function

from satact.dataset.lcg import LCG

from .policy import (
    ActionEvalPolicyOutput,
    DEFER_VARIANTS,
    MODEL_VARIANTS,
)
from .common import LCGBackboneOutput, MLP


class ActionEvalLiteralMLPPolicy(nn.Module):
    """Score each backbone literal embedding with one shared MLP."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embedding_dim: int = 64,
        variant: str = "basic-action",
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unsupported SAT-ACT variant: {variant!r}")
        if variant in DEFER_VARIANTS:
            raise ValueError(
                f"literal-mlp readout does not support DEFER variant {variant!r}"
            )
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.backbone = backbone
        self.embedding_dim = int(embedding_dim)
        self.variant = variant
        self.literal_readout = MLP(
            self.embedding_dim, self.embedding_dim, 1, mlp_layers
        )

    def forward(self, data: LCG) -> ActionEvalPolicyOutput:
        with record_function("satact_backbone"):
            encoded = self.backbone(data)
        if not isinstance(encoded, LCGBackboneOutput):
            raise TypeError("SAT-ACT backbone must return LCGBackboneOutput")
        n_vars = data.assignment_value.numel()
        if encoded.literal_embeddings.shape != (2 * n_vars, self.embedding_dim):
            raise ValueError("backbone literal embedding shape mismatch")

        with record_function("satact_literal_mlp_readout"):
            literal_logits = self.literal_readout(encoded.literal_embeddings).squeeze(-1)
        counts = (
            data.n_vars.long().reshape(-1)
            if isinstance(data.n_vars, torch.Tensor)
            else torch.tensor(
                [int(data.n_vars)], dtype=torch.long, device=literal_logits.device
            )
        ) * 2
        graph_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=counts.device),
                torch.cumsum(counts, dim=0),
            )
        )
        return ActionEvalPolicyOutput(literal_logits, graph_offsets)


__all__ = ["ActionEvalLiteralMLPPolicy"]
