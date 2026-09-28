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


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.gate_norms = (
            nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)]) if use_layer_norm else None
        )
        self.cell_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None

    def forward(
        self, values: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        input_gate, update_gate, forget_gate, output_gate = self.linear(
            torch.cat((values, hidden), dim=-1)
        ).chunk(4, dim=-1)
        if self.gate_norms is not None:
            input_gate, update_gate, forget_gate, output_gate = (
                norm(gate)
                for norm, gate in zip(
                    self.gate_norms, (input_gate, update_gate, forget_gate, output_gate), strict=True
                )
            )
        next_cell = torch.sigmoid(input_gate) * torch.tanh(update_gate) + torch.sigmoid(forget_gate) * cell
        if self.cell_norm is not None:
            next_cell = self.cell_norm(next_cell)
        next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
        return next_hidden, next_cell


class NeuroSATBackbone(nn.Module):
    """Independent NeuroSAT-style encoder for the portable feature contract."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_rounds: int = 4,
        mlp_layers: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or num_rounds < 1:
            raise ValueError("embedding_dim and num_rounds must be positive")
        self.embedding_dim = int(embedding_dim)
        self.num_rounds = int(num_rounds)
        self.feature_encoder = PortableFeatureEncoder(self.embedding_dim)
        self.literal_to_clause = MLP(self.embedding_dim, self.embedding_dim, self.embedding_dim, mlp_layers)
        self.clause_to_literal = MLP(self.embedding_dim, self.embedding_dim, self.embedding_dim, mlp_layers)
        self.clause_update = LayerNormLSTMCell(
            self.embedding_dim, self.embedding_dim, use_layer_norm=use_layer_norm
        )
        self.literal_update = LayerNormLSTMCell(
            2 * self.embedding_dim, self.embedding_dim, use_layer_norm=use_layer_norm
        )

    def forward(self, data: LCG) -> LCGBackboneOutput:
        n_vars = data.assignment_value.numel()
        n_clauses = data.c_batch.numel()
        literal_index = data.l_edge_index.long()
        clause_index = data.c_edge_index.long()
        literals, clauses = self.feature_encoder(data, n_vars, n_clauses)
        literal_cell = torch.zeros_like(literals)
        clause_cell = torch.zeros_like(clauses)

        for _ in range(self.num_rounds):
            literal_messages = self.literal_to_clause(literals)
            literal_aggregate = scatter_sum(literal_messages[literal_index], clause_index, n_clauses)
            clauses, clause_cell = self.clause_update(literal_aggregate, (clauses, clause_cell))

            clause_messages = self.clause_to_literal(clauses)
            clause_aggregate = scatter_sum(clause_messages[clause_index], literal_index, 2 * n_vars)
            literal_input = torch.cat((clause_aggregate, flip_literal_embeddings(literals)), dim=-1)
            literals, literal_cell = self.literal_update(literal_input, (literals, literal_cell))

        return LCGBackboneOutput(literals, clauses)
