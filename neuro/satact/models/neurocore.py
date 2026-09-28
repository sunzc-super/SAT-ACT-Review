from __future__ import annotations

import torch
from torch import nn

from satact.dataset.lcg import LCG

from .common import (
    LCGBackboneOutput,
    MLP,
    PortableFeatureEncoder,
    flip_literal_embeddings,
    scatter_sum,
)


class NeuroCoreBackbone(nn.Module):
    """Independent NeuroCore-style encoder for the portable feature contract."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_rounds: int = 4,
        mlp_layers: int = 2,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        shared_updates: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or num_rounds < 1:
            raise ValueError("embedding_dim and num_rounds must be positive")
        self.embedding_dim = int(embedding_dim)
        self.num_rounds = int(num_rounds)
        self.use_residual = bool(use_residual)
        self.feature_encoder = PortableFeatureEncoder(self.embedding_dim)

        update_count = 1 if shared_updates else self.num_rounds
        self.clause_updates = nn.ModuleList(
            [MLP(2 * self.embedding_dim, self.embedding_dim, self.embedding_dim, mlp_layers) for _ in range(update_count)]
        )
        self.literal_updates = nn.ModuleList(
            [MLP(3 * self.embedding_dim, self.embedding_dim, self.embedding_dim, mlp_layers) for _ in range(update_count)]
        )
        self.clause_norms = (
            nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(update_count)])
            if use_layer_norm
            else None
        )
        self.literal_norms = (
            nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(update_count)])
            if use_layer_norm
            else None
        )
        self.literal_to_clause_scale = nn.Parameter(torch.tensor(1.0))
        self.clause_to_literal_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, data: LCG) -> LCGBackboneOutput:
        n_vars = data.assignment_value.numel()
        n_clauses = data.c_batch.numel()
        literal_index = data.l_edge_index.long()
        clause_index = data.c_edge_index.long()
        literals, clauses = self.feature_encoder(data, n_vars, n_clauses)

        for round_index in range(self.num_rounds):
            module_index = round_index % len(self.clause_updates)
            literal_messages = scatter_sum(literals[literal_index], clause_index, n_clauses)
            next_clauses = self.clause_updates[module_index](
                torch.cat((clauses, literal_messages * self.literal_to_clause_scale), dim=-1)
            )
            if self.use_residual:
                next_clauses = next_clauses + clauses
            if self.clause_norms is not None:
                next_clauses = self.clause_norms[module_index](next_clauses)
            clauses = next_clauses

            clause_messages = scatter_sum(clauses[clause_index], literal_index, 2 * n_vars)
            next_literals = self.literal_updates[module_index](
                torch.cat(
                    (literals, clause_messages * self.clause_to_literal_scale, flip_literal_embeddings(literals)),
                    dim=-1,
                )
            )
            if self.use_residual:
                next_literals = next_literals + literals
            if self.literal_norms is not None:
                next_literals = self.literal_norms[module_index](next_literals)
            literals = next_literals

        return LCGBackboneOutput(literals, clauses)
