from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
from torch.autograd.profiler import record_function

from satact.dataset.lcg import LCG

from .common import LCGBackboneOutput, MLP, scatter_sum, variable_batch


MODEL_VARIANTS: Final[tuple[str, ...]] = (
    "basic-action",
    "satact",
    "signed-action",
    "preference-action",
    "list-action",
    "signed-list-action",
    "defer-action",
    "signed-defer-action",
)
DEFER_VARIANTS: Final[frozenset[str]] = frozenset({"defer-action", "signed-defer-action"})


@dataclass(frozen=True)
class ActionEvalPolicyOutput:
    """Ragged, direct signed-literal scores for a batch of graphs.

    ``literal_logits[graph_offsets[g]:graph_offsets[g+1]]`` contains exactly
    ``2*n_vars[g]`` entries.  Even local indices are positive literals and odd
    local indices are negative literals.  The defer variants supply one DEFER
    logit per graph.
    """

    literal_logits: torch.Tensor
    graph_offsets: torch.Tensor
    defer_logits: torch.Tensor | None = None

    @property
    def num_graphs(self) -> int:
        return int(self.graph_offsets.numel() - 1)

    def literal_slice(self, graph_index: int) -> slice:
        if not 0 <= graph_index < self.num_graphs:
            raise IndexError("graph_index is outside the batch")
        return slice(int(self.graph_offsets[graph_index]), int(self.graph_offsets[graph_index + 1]))


class ActionEvalPolicy(nn.Module):
    """Direct 2N signed-literal policy shared by all offline SAT-ACT objectives."""

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
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.backbone = backbone
        self.embedding_dim = int(embedding_dim)
        self.variant = variant

        # Positive/negative memories are merged symmetrically.  This prevents
        # the variable representation itself from leaking an arbitrary sign.
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )
        self.graph_token = nn.Parameter(torch.zeros(self.embedding_dim))
        self.context_init = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.graph_gru = nn.GRUCell(self.embedding_dim, self.embedding_dim)
        self.variable_gru = nn.GRUCell(self.embedding_dim, self.embedding_dim)
        # One shared head scores both signs.  It sees the sign-specific literal
        # memory, the sign-symmetric variable memory, and graph context.
        self.literal_head = MLP(
            3 * self.embedding_dim, self.embedding_dim, 1, mlp_layers
        )
        self.defer_head = (
            MLP(self.embedding_dim, self.embedding_dim, 1, mlp_layers)
            if variant in DEFER_VARIANTS
            else None
        )

    def _pair_variables(self, literals: torch.Tensor) -> torch.Tensor:
        if literals.shape[0] % 2:
            raise ValueError("backbone must emit two literal embeddings per variable")
        pairs = literals.reshape(-1, 2, self.embedding_dim)
        return self.pair_encoder(
            torch.cat((pairs[:, 0] + pairs[:, 1], torch.abs(pairs[:, 0] - pairs[:, 1])), dim=-1)
        )

    def _decode_batch(
        self,
        variables: torch.Tensor,
        literals: torch.Tensor,
        clauses: torch.Tensor,
        variable_graph: torch.Tensor,
        clause_graph: torch.Tensor,
        graph_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = torch.cat((variables, clauses), dim=0)
        memory_graph = torch.cat((variable_graph, clause_graph), dim=0)
        context_sum = scatter_sum(memory, memory_graph, graph_count)
        context_count = scatter_sum(
            torch.ones((memory.size(0), 1), dtype=memory.dtype, device=memory.device),
            memory_graph,
            graph_count,
        )
        context = context_sum / context_count
        graph_hidden = self.graph_gru(
            self.graph_token.unsqueeze(0).expand(graph_count, -1),
            torch.tanh(self.context_init(context)),
        )
        conditioned_variables = self.variable_gru(
            variables, graph_hidden[variable_graph]
        )
        variable_memory = conditioned_variables.repeat_interleave(2, dim=0)
        literal_graph = variable_graph.repeat_interleave(2)
        graph_memory = graph_hidden[literal_graph]
        literal_logits = self.literal_head(
            torch.cat((literals, variable_memory, graph_memory), dim=-1)
        ).squeeze(-1)
        return literal_logits, graph_hidden

    def forward(self, data: LCG) -> ActionEvalPolicyOutput:
        with record_function("satact_backbone"):
            encoded = self.backbone(data)
        if not isinstance(encoded, LCGBackboneOutput):
            raise TypeError("SAT-ACT backbone must return LCGBackboneOutput")
        n_vars = data.assignment_value.numel()
        if encoded.literal_embeddings.shape != (2 * n_vars, self.embedding_dim):
            raise ValueError("backbone literal embedding shape mismatch")
        if encoded.clause_embeddings.shape != (data.c_batch.numel(), self.embedding_dim):
            raise ValueError("backbone clause embedding shape mismatch")

        variables = self._pair_variables(encoded.literal_embeddings)
        variable_graph = variable_batch(data, n_vars)
        clause_graph = (
            data.c_batch.long()
            if data.c_batch is not None
            else torch.zeros(
                encoded.clause_embeddings.size(0),
                dtype=torch.long,
                device=encoded.clause_embeddings.device,
            )
        )
        graph_count = data.n_vars.numel() if isinstance(data.n_vars, torch.Tensor) else 1
        with record_function("satact_policy_decoder"):
            literal_logits, graph_hidden = self._decode_batch(
                variables,
                encoded.literal_embeddings,
                encoded.clause_embeddings,
                variable_graph,
                clause_graph,
                graph_count,
            )
            defer_logits = (
                self.defer_head(graph_hidden).squeeze(-1)
                if self.defer_head is not None
                else None
            )
        counts = (
            data.n_vars.long().reshape(-1)
            if isinstance(data.n_vars, torch.Tensor)
            else torch.tensor([int(data.n_vars)], dtype=torch.long, device=literal_logits.device)
        ) * 2
        graph_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=counts.device), torch.cumsum(counts, dim=0))
        )
        return ActionEvalPolicyOutput(literal_logits, graph_offsets, defer_logits)


__all__ = [
    "ActionEvalPolicyOutput",
    "ActionEvalPolicy",
    "DEFER_VARIANTS",
    "MODEL_VARIANTS",
]
