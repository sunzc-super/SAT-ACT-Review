from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

from satact.dataset.lcg import LCG


@dataclass(frozen=True)
class LCGBackboneOutput:
    """Literal/clause memories produced by a graph backbone."""

    literal_embeddings: torch.Tensor
    clause_embeddings: torch.Tensor


def num_graphs(data: LCG) -> int:
    """Return the number of graphs in an LCG batch."""

    return int(data.n_vars.numel()) if isinstance(data.n_vars, torch.Tensor) else 1


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    return scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    result = scatter(src, index, dim=0, dim_size=dim_size, reduce="max")
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def flip_literal_embeddings(literals: torch.Tensor) -> torch.Tensor:
    """Swap the positive/negative row of every variable."""

    if literals.ndim != 2 or literals.size(0) % 2:
        raise ValueError("literal embeddings must have shape [2*N, D]")
    return literals.reshape(-1, 2, literals.size(-1)).flip(1).reshape_as(literals)


def variable_batch(data: LCG, n_vars: int) -> torch.Tensor:
    """Return a graph id for every variable in a batched LCG."""

    if data.l_batch is None:
        return torch.zeros(n_vars, dtype=torch.long, device=data.l_edge_index.device)
    if data.l_batch.numel() != 2 * n_vars:
        raise ValueError("l_batch length must equal 2 * total_n_vars")
    literal_batch = data.l_batch.long().reshape(n_vars, 2)
    return literal_batch[:, 0]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(current, hidden_dim), nn.GELU()))
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(frozen=True)
class ClauseState:
    size: torch.Tensor
    true_count: torch.Tensor
    false_count: torch.Tensor
    unassigned_count: torch.Tensor


def _literal_truth_masks(data: LCG, n_vars: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignment = data.assignment_value.long().reshape(-1)
    if assignment.numel() != n_vars:
        raise ValueError("assignment_value length must equal total_n_vars")
    positive_true = assignment > 0
    negative_true = assignment < 0
    literal_true = torch.stack((positive_true, negative_true), dim=1).reshape(-1)
    literal_false = torch.stack((negative_true, positive_true), dim=1).reshape(-1)
    literal_unassigned = (assignment == 0).unsqueeze(1).expand(-1, 2).reshape(-1)
    return literal_true, literal_false, literal_unassigned


def _clause_state(data: LCG, n_vars: int, n_clauses: int) -> ClauseState:
    literal_true, literal_false, literal_unassigned = _literal_truth_masks(data, n_vars)
    literal_index = data.l_edge_index.long()
    clause_index = data.c_edge_index.long()
    ones = torch.ones(clause_index.numel(), dtype=torch.float32, device=clause_index.device)
    return ClauseState(
        size=scatter_sum(ones, clause_index, n_clauses),
        true_count=scatter_sum(literal_true[literal_index].float(), clause_index, n_clauses),
        false_count=scatter_sum(literal_false[literal_index].float(), clause_index, n_clauses),
        unassigned_count=scatter_sum(literal_unassigned[literal_index].float(), clause_index, n_clauses),
    )


class PortableFeatureEncoder(nn.Module):
    """Encode only the explicitly portable input whitelist.

    The implementation is a positive whitelist: it reads only the active CNF,
    assignment values/levels, decision level, and statistics derived from those
    values.  Candidate eligibility is not a graph feature; callers use it only
    to mask supervision and decoding.
    """

    LITERAL_FEATURE_DIM = 10
    CLAUSE_FEATURE_DIM = 7

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.literal_proj = nn.Linear(self.LITERAL_FEATURE_DIM, embedding_dim)
        self.clause_proj = nn.Linear(self.CLAUSE_FEATURE_DIM, embedding_dim)

    def _literal_features(self, data: LCG, n_vars: int) -> torch.Tensor:
        literal_true, literal_false, literal_unassigned = _literal_truth_masks(data, n_vars)
        graph_count = num_graphs(data)
        var_batch = variable_batch(data, n_vars)
        literal_batch = var_batch.repeat_interleave(2)

        assignment_level = data.assignment_level.float().reshape(-1)
        if assignment_level.numel() != n_vars:
            raise ValueError("assignment_level length must equal total_n_vars")
        decision_level = data.decision_level.float().reshape(-1)
        if decision_level.numel() != graph_count:
            raise ValueError("decision_level must contain one value per graph")
        level_scale = decision_level.clamp_min(1.0)[var_batch]
        normalized_level = assignment_level.clamp_min(0.0) / level_scale
        normalized_level_lit = normalized_level.repeat_interleave(2)

        occurrences = scatter_sum(
            torch.ones(data.l_edge_index.numel(), dtype=torch.float32, device=data.l_edge_index.device),
            data.l_edge_index.long(),
            2 * n_vars,
        )
        occurrence_max = scatter_max(occurrences, literal_batch, graph_count).clamp_min(1.0)
        normalized_occurrence = occurrences / occurrence_max[literal_batch]

        n_vars_per_graph = data.n_vars.float().reshape(-1).clamp_min(1.0)
        n_clauses_per_graph = data.n_clauses.float().reshape(-1).clamp_min(0.0)
        assigned_per_graph = scatter_sum(
            (data.assignment_value.reshape(-1) != 0).float(), var_batch, graph_count
        )
        assigned_fraction = assigned_per_graph / n_vars_per_graph
        normalized_decision_level = decision_level.clamp_min(0.0) / n_vars_per_graph

        return torch.stack(
            (
                literal_unassigned.float(),
                literal_true.float(),
                literal_false.float(),
                normalized_level_lit,
                normalized_occurrence,
                torch.log1p(occurrences),
                torch.log1p(n_vars_per_graph)[literal_batch],
                torch.log1p(n_clauses_per_graph)[literal_batch],
                assigned_fraction[literal_batch],
                normalized_decision_level[literal_batch],
            ),
            dim=-1,
        )

    def _clause_features(self, data: LCG, n_vars: int, n_clauses: int) -> torch.Tensor:
        state = _clause_state(data, n_vars, n_clauses)
        safe_size = state.size.clamp_min(1.0)
        satisfied = state.true_count > 0
        unsatisfied = ~satisfied
        return torch.stack(
            (
                torch.log1p(state.size),
                state.true_count / safe_size,
                state.false_count / safe_size,
                state.unassigned_count / safe_size,
                satisfied.float(),
                (unsatisfied & (state.unassigned_count == 1)).float(),
                (unsatisfied & (state.unassigned_count == 2)).float(),
            ),
            dim=-1,
        )

    def forward(self, data: LCG, n_vars: int, n_clauses: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.literal_proj(self._literal_features(data, n_vars)),
            self.clause_proj(self._clause_features(data, n_vars, n_clauses)),
        )
