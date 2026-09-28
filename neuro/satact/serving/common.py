from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from satact.dataset.lcg import LCG


BranchAction = Literal["literal", "defer"]

@dataclass(frozen=True)
class BranchPrediction:
    model_variant: str
    action: BranchAction
    selected_literal_index: int
    selected_log_probability: float
    n_secs_inference: float
    action_logits: list[float]


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def build_online_lcg(
    *,
    n_vars: int,
    n_clauses: int,
    c_idxs: list[int],
    l_idxs: list[int],
    assignment_value: list[int],
    assignment_level: list[int],
    decision_level: int,
    candidate_variable: list[bool],
) -> tuple[LCG, torch.Tensor]:
    """Validate one RPC snapshot and construct a feature-whitelisted LCG.

    Candidate eligibility is returned separately and is never attached to the
    graph passed to the model.  Every eligible variable enables both signed
    literal indices: ``2*i`` is positive and ``2*i+1`` is negative.
    """

    if n_vars <= 0:
        raise ValueError("n_vars must be positive")
    if n_clauses < 0:
        raise ValueError("n_clauses must be non-negative")
    if len(c_idxs) != len(l_idxs):
        raise ValueError("c_idxs and l_idxs must have equal length")
    if n_clauses == 0 and c_idxs:
        raise ValueError("a zero-clause graph cannot contain edges")
    if any(index < 0 or index >= n_clauses for index in c_idxs):
        raise ValueError("clause index is outside the declared range")
    if any(index < 0 or index >= 2 * n_vars for index in l_idxs):
        raise ValueError("literal index is outside the declared range")
    if len(assignment_value) != n_vars:
        raise ValueError("assignment_value length must equal n_vars")
    if len(assignment_level) != n_vars:
        raise ValueError("assignment_level length must equal n_vars")
    if len(candidate_variable) != n_vars:
        raise ValueError("candidate_variable length must equal n_vars")
    if any(value not in {-1, 0, 1} for value in assignment_value):
        raise ValueError("assignment values must be in {-1, 0, 1}")
    if any(value < -1 for value in assignment_level):
        raise ValueError("assignment levels must be at least -1")
    if decision_level < 0:
        raise ValueError("decision_level must be non-negative")

    candidates = torch.tensor(candidate_variable, dtype=torch.bool)
    assignments = torch.tensor(assignment_value, dtype=torch.long)
    if not bool(candidates.any()):
        raise ValueError("request has no candidate variable")
    if bool((candidates & (assignments != 0)).any()):
        raise ValueError("assigned variables cannot be candidates")

    sample = LCG(
        n_vars=torch.tensor([n_vars], dtype=torch.long),
        n_clauses=torch.tensor([n_clauses], dtype=torch.long),
        l_edge_index=torch.tensor(l_idxs, dtype=torch.long),
        c_edge_index=torch.tensor(c_idxs, dtype=torch.long),
        l_batch=torch.zeros(2 * n_vars, dtype=torch.long),
        c_batch=torch.zeros(n_clauses, dtype=torch.long),
        assignment_value=assignments,
        assignment_level=torch.tensor(assignment_level, dtype=torch.long),
        decision_level=torch.tensor([decision_level], dtype=torch.long),
    )
    sample.validate_lcg()
    eligibility = candidates.repeat_interleave(2)
    return sample, eligibility


__all__ = [
    "BranchPrediction",
    "build_online_lcg",
    "resolve_device",
]
