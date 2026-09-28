"""DecisionTrace-ActionEval reader used by SAT-ACT.

The reader is deliberately plan driven.  It reads
``manifests/pipeline.parts.jsonl`` and resolves the three committed shard
files for every named part.  The plan may name either a normal one-part shard
or a grouped shard produced by the raw-to-shard packer.  It never discovers
data with glob/rglob/walk.

Only portable graph inputs are materialized: the solver-active CNF,
assignment value/level, decision level, and statistics derived from those
arrays.  Exact eligibility and rollout outcomes live in the separate
``ActionEvalTargetStorage`` object and are never attached to the model graph.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch_geometric.data import Batch

from gen_data.rollout_dataset.compact import CompactArrayReader as CompactArrayReaderBase

CompactArrayReader = CompactArrayReaderBase

from .lcg import LCG


STORAGE_VARIANTS = ("basic-action", "satact", "signed-action", "preference-action", "list-action", "signed-list-action", "defer-action", "signed-defer-action")
STORAGE_DEFER_VARIANTS = frozenset({"defer-action", "signed-defer-action"})
_STORAGE_SCALAR_PAIR_INDEX_VARIANTS = frozenset({"satact", "signed-action", "defer-action", "signed-defer-action"})
_STORAGE_LIST_INDEX_VARIANTS = frozenset({"list-action", "signed-list-action"})
PLAN_SCHEMA = "decisiontrace-actioneval-pipeline-plan-v1"
SHARD_POLICY_SCHEMA_V2 = "decisiontrace-actioneval-shard-policy-v2"
SHARD_PLAN_SCHEMA_V2 = "decisiontrace-actioneval-shard-plan-v2"
SHARD_PAYLOAD_V2 = "shard-plan-v2"
LEGACY_PAYLOAD_V2 = "legacy-source-provenance"
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SOLVED = {10, 20}


class ActionEvalDatasetError(ValueError):
    """The uploaded split or a committed compact shard is inconsistent."""


@dataclass(frozen=True)
class IndexBuildProgressStorage:
    """One plan-part progress event emitted while preparing an index."""

    phase: str
    part_index: int
    part_count: int
    part_stem: str
    states: int
    limit_reached: bool = False


@dataclass(frozen=True)
class ObjectiveConfigStorage:
    """All label semantics that can change which states are usable."""

    tie_margin: float = 0.05
    listwise_temperature: float = 0.25
    censored_penalty: float = 1.0
    native_weight: float = 0.1
    cost_conflicts_weight: float = 1.0
    cost_propagations_weight: float = 0.1
    cost_decisions_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "tie_margin",
            "listwise_temperature",
            "censored_penalty",
            "native_weight",
            "cost_conflicts_weight",
            "cost_propagations_weight",
            "cost_decisions_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.listwise_temperature <= 0:
            raise ValueError("listwise_temperature must be positive")


def _index_config_storage(
    *,
    reader_schema: int,
    parts: Sequence[str],
    variant: str,
    objective: ObjectiveConfigStorage | Mapping[str, Any],
    max_states: int | None,
) -> dict[str, Any]:
    """Return only values that can change the materialized index entries."""

    def value(name: str) -> float:
        if isinstance(objective, ObjectiveConfigStorage):
            return float(getattr(objective, name))
        return float(objective[name])

    index_objective = {
        "cost_conflicts_weight": value("cost_conflicts_weight"),
        "cost_propagations_weight": value("cost_propagations_weight"),
        "cost_decisions_weight": value("cost_decisions_weight"),
        "censored_penalty": value("censored_penalty"),
    }
    if variant in _STORAGE_SCALAR_PAIR_INDEX_VARIANTS:
        index_objective["tie_margin"] = value("tie_margin")
    if variant in _STORAGE_LIST_INDEX_VARIANTS:
        index_objective["listwise_temperature"] = value("listwise_temperature")
    return {
        "reader_schema": int(reader_schema),
        "parts": [str(part) for part in parts],
        "variant": str(variant),
        "objective": index_objective,
        "max_states": max_states,
    }


def _legacy_index_config_storage(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Project a legacy cache description onto the reusable index contract."""

    return _index_config_storage(
        reader_schema=int(fingerprint["reader_schema"]),
        parts=[str(row["stem"]) for row in fingerprint["inventory"]],
        variant=str(fingerprint["variant"]),
        objective=fingerprint["objective"],
        max_states=fingerprint.get("max_states"),
    )


@dataclass(frozen=True)
class EvalOutcomeStorage:
    literal_index: int | None  # None is the native/DEFER baseline.
    external_literal: int
    status: int
    censored: bool
    complete: bool
    cost: float


@dataclass(frozen=True)
class ActionEvalTargetStorage:
    """Supervision for one state; no field in this object enters the GNN."""

    instance_id: str
    variant: str
    eligible_state: int
    record_id: str
    shard_stem: str
    n_literals: int
    eligibility_mask: torch.Tensor
    native_literal_index: int | None
    pair_better: torch.Tensor  # -1 denotes DEFER.
    pair_worse: torch.Tensor
    listwise_literals: torch.Tensor
    listwise_target: torch.Tensor
    evaluated_literals: torch.Tensor
    evaluated_costs: torch.Tensor
    evaluated_complete: torch.Tensor
    baseline_cost: float | None
    baseline_complete: bool
    instance_state_count: int
    instance_weight: float

    def validate(self) -> None:
        if self.eligibility_mask.dtype != torch.bool or self.eligibility_mask.shape != (
            self.n_literals,
        ):
            raise ActionEvalDatasetError("invalid exact signed-literal eligibility mask")
        if self.native_literal_index is not None:
            if not 0 <= self.native_literal_index < self.n_literals:
                raise ActionEvalDatasetError("native literal index is out of range")
            if not bool(self.eligibility_mask[self.native_literal_index]):
                raise ActionEvalDatasetError("native target is outside exact eligibility")
        if self.pair_better.shape != self.pair_worse.shape:
            raise ActionEvalDatasetError("pair arrays have different lengths")
        for values in (self.pair_better, self.pair_worse):
            if values.dtype != torch.long:
                raise ActionEvalDatasetError("pair indices must be int64")
            if values.numel() and (
                int(values.min()) < -1 or int(values.max()) >= self.n_literals
            ):
                raise ActionEvalDatasetError("pair action index is out of range")
        if self.listwise_literals.numel() != self.listwise_target.numel():
            raise ActionEvalDatasetError("listwise action/target lengths differ")
        if self.listwise_target.numel() and not torch.isclose(
            self.listwise_target.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ActionEvalDatasetError("listwise target must sum to one")
        if self.instance_state_count < 1 or self.instance_weight <= 0:
            raise ActionEvalDatasetError("invalid instance equalization weight")


class ActionEvalLCGStorage(LCG):
    """A storage graph containing only the explicitly approved feature whitelist."""

    _ALLOWED_EXTRA_FIELDS = frozenset(
        {
            "assignment_value",
            "assignment_level",
            "decision_level",
            "literal_occurrence",
            "variable_occurrence",
            "clause_size",
        }
    )

    def validate_actioneval_storage(self) -> None:
        self.validate_lcg()
        n_vars, n_clauses = self.num_variables, self.num_clauses_lcg
        if n_vars < 1:
            raise ActionEvalDatasetError("SAT-ACT states must contain at least one variable")
        if self.assignment_value.shape != (n_vars,):
            raise ActionEvalDatasetError("assignment_value length mismatch")
        if self.assignment_level.shape != (n_vars,):
            raise ActionEvalDatasetError("assignment_level length mismatch")
        if self.decision_level.numel() != 1:
            raise ActionEvalDatasetError("decision_level must be scalar per graph")
        if self.literal_occurrence.shape != (2 * n_vars,):
            raise ActionEvalDatasetError("literal_occurrence length mismatch")
        if self.variable_occurrence.shape != (n_vars,):
            raise ActionEvalDatasetError("variable_occurrence length mismatch")
        if self.clause_size.shape != (n_clauses,):
            raise ActionEvalDatasetError("clause_size length mismatch")
        forbidden = {
            "activity",
            "saved_phase",
            "assignment_source",
            "trail_position",
            "conflicts",
            "decisions",
            "propagations",
            "restarts",
            "native_literal",
            "outcomes",
        }
        present = forbidden.intersection(self.to_dict())
        if present:
            raise ActionEvalDatasetError(
                "solver-policy/outcome fields leaked into graph: " + ",".join(sorted(present))
            )


@dataclass(frozen=True)
class ActionEvalSampleStorage:
    graph: ActionEvalLCGStorage
    supervision: ActionEvalTargetStorage


@dataclass(frozen=True)
class ActionEvalBatchStorage:
    graph: Batch
    supervision: tuple[ActionEvalTargetStorage, ...]
    variant: str

    @property
    def batch_size(self) -> int:
        return len(self.supervision)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ActionEvalBatchStorage":
        # Supervision is intentionally kept separate.  The objective copies
        # only the small masks/indices it needs to the logits' device.
        return ActionEvalBatchStorage(
            graph=self.graph.to(device, non_blocking=non_blocking),
            supervision=self.supervision,
            variant=self.variant,
        )


def collate_actioneval_storage(samples: Sequence[ActionEvalSampleStorage]) -> ActionEvalBatchStorage:
    if not samples:
        raise ValueError("cannot collate an empty SAT-ACT batch")
    variants = {sample.supervision.variant for sample in samples}
    if len(variants) != 1:
        raise ValueError("one SAT-ACT batch cannot mix objective variants")
    supervision = _pack_supervision_storage_storage(
        tuple(sample.supervision for sample in samples)
    )
    return ActionEvalBatchStorage(
        graph=Batch.from_data_list([sample.graph for sample in samples]),
        supervision=supervision,
        variant=next(iter(variants)),
    )


def _pack_supervision_storage_storage(
    targets: tuple[ActionEvalTargetStorage, ...],
) -> tuple[ActionEvalTargetStorage, ...]:
    """Pack ragged target tensors into three shared CPU storages.

    A target owns eight small tensors.  Returning 128 independent targets from
    a multiprocessing DataLoader would otherwise require roughly 1024 shared
    memory file descriptors for supervision alone.  Views into one storage per
    dtype preserve the existing per-state API while reducing that count to
    three, independent of batch size.
    """

    groups = (
        (
            torch.long,
            ("pair_better", "pair_worse", "listwise_literals", "evaluated_literals"),
        ),
        (torch.float32, ("listwise_target", "evaluated_costs")),
        (torch.bool, ("eligibility_mask", "evaluated_complete")),
    )
    replacements: list[dict[str, torch.Tensor]] = [dict() for _ in targets]
    for expected_dtype, names in groups:
        parts: list[torch.Tensor] = []
        descriptors: list[tuple[int, str, int]] = []
        for target_index, target in enumerate(targets):
            for name in names:
                values = getattr(target, name)
                if (
                    not isinstance(values, torch.Tensor)
                    or values.device.type != "cpu"
                    or values.dtype != expected_dtype
                    or values.ndim != 1
                ):
                    raise ActionEvalDatasetError(
                        f"{name} must be a 1-D CPU {expected_dtype} tensor before collation"
                    )
                flat = values.reshape(-1)
                parts.append(flat)
                descriptors.append((target_index, name, flat.numel()))
        storage = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=expected_dtype)
        offset = 0
        for target_index, name, length in descriptors:
            replacements[target_index][name] = storage.narrow(0, offset, length)
            offset += length
    packed = tuple(
        replace(target, **replacement)
        for target, replacement in zip(targets, replacements, strict=True)
    )
    return packed


@dataclass(frozen=True)
class _RawOutcome:
    requested: int
    status: int
    censored: bool
    horizon: bool
    conflicts: int
    decisions: int
    propagations: int
    restarts: int
    reliable: bool
    complete: bool


def signed_external_to_internal_index(
    external_literal: int, i2e: Sequence[int]
) -> int | None:
    """Map a signed external literal through a possibly signed i2e map."""

    if not external_literal:
        return None
    wanted = abs(int(external_literal))
    found: int | None = None
    for internal_index, mapped in enumerate(i2e):
        if abs(int(mapped)) != wanted:
            continue
        if found is not None:
            raise ActionEvalDatasetError(f"ambiguous i2e mapping for external variable {wanted}")
        internal_positive_is_external_positive = int(mapped) > 0
        internal_positive = (external_literal > 0) == internal_positive_is_external_positive
        found = 2 * internal_index + (0 if internal_positive else 1)
    return found


def exact_internal_literal_mask(i2e: Sequence[int], eligibility_bits: Sequence[int]) -> torch.Tensor:
    """Expand the external LSB-first variable bitset to an internal 2N mask."""

    result = torch.zeros(2 * len(i2e), dtype=torch.bool)
    for internal_index, mapped_value in enumerate(i2e):
        mapped = abs(int(mapped_value))
        if not mapped:
            continue
        bit_index = mapped - 1
        byte_index = bit_index >> 3
        if byte_index >= len(eligibility_bits):
            raise ActionEvalDatasetError("i2e mapping exceeds external eligibility bitset")
        if int(eligibility_bits[byte_index]) & (1 << (bit_index & 7)):
            result[2 * internal_index : 2 * internal_index + 2] = True
    return result


def action_cost_storage(conflicts: int, propagations: int, decisions: int, config: ObjectiveConfigStorage) -> float:
    if min(conflicts, propagations, decisions) < 0:
        raise ActionEvalDatasetError("negative rollout delta counter")
    return (
        config.cost_conflicts_weight * math.log1p(conflicts)
        + config.cost_propagations_weight * math.log1p(propagations)
        + config.cost_decisions_weight * math.log1p(decisions)
    )


def _compare_outcomes(left: _RawOutcome, right: _RawOutcome, margin: float) -> int:
    """Return -1 when left wins, +1 when right wins, and 0 for no/tie pair."""

    if not left.reliable or not right.reliable:
        return 0
    if left.complete != right.complete:
        return -1 if left.complete else 1
    if not left.complete:  # Two censored/UNKNOWN observations are incomparable.
        return 0
    left_cost = (left.conflicts, left.propagations, left.decisions)
    right_cost = (right.conflicts, right.propagations, right.decisions)
    # The caller replaces these tuples with configured scalar costs.
    del left_cost, right_cost
    return 2  # sentinel: compare configured scalar costs in _derive_plan


def _derive_plan(
    *,
    variant: str,
    config: ObjectiveConfigStorage,
    state: Mapping[str, Any],
    i2e: Sequence[int],
    eligibility_bits: Sequence[int],
    arrays: Mapping[str, Sequence[int]],
) -> dict[str, Any] | None:
    if variant not in STORAGE_VARIANTS:
        raise ValueError(f"unsupported SAT-ACT variant: {variant!r}")
    mask = exact_internal_literal_mask(i2e, eligibility_bits)
    native_external = int(state.get("native_external", 0))
    native_index = signed_external_to_internal_index(native_external, i2e)
    if native_index is not None and not bool(mask[native_index]):
        native_index = None

    names = (
        "eval_requested_literal",
        "eval_target_reached",
        "eval_locator_match",
        "eval_action_eligible",
        "eval_action_applied",
        "eval_native_literal",
        "eval_applied_literal",
        "eval_status",
        "eval_censored",
        "eval_horizon_hit",
        "eval_delta_conflicts",
        "eval_delta_decisions",
        "eval_delta_propagations",
    )
    lengths = {len(arrays[name]) for name in names}
    if variant == "preference-action":
        lengths.add(len(arrays["eval_delta_restarts"]))
    if len(lengths) != 1:
        raise ActionEvalDatasetError("rollout array lengths disagree")
    outcomes: list[tuple[_RawOutcome, int | None, float]] = []
    solved_statuses: set[int] = set()
    restart_values = (
        arrays["eval_delta_restarts"]
        if variant == "preference-action"
        else [0] * len(arrays[names[0]])
    )
    for row in zip(*(arrays[name] for name in names), restart_values):
        requested, reached, locator, eligible, applied, eval_native, applied_lit, status, censored, horizon, dc, dd, dp, dr = (
            int(value) for value in row
        )
        native_matches = eval_native == native_external
        applied_correct = bool(applied) and native_matches and (
            (requested == 0 and applied_lit == eval_native)
            or (requested != 0 and applied_lit == requested)
        )
        reliable = bool(reached and locator and eligible and applied_correct)
        complete = bool(reliable and not censored and not horizon and status in _SOLVED)
        if complete:
            solved_statuses.add(status)
        outcome = _RawOutcome(
            requested=requested,
            status=status,
            censored=bool(censored),
            horizon=bool(horizon),
            conflicts=dc,
            decisions=dd,
            propagations=dp,
            restarts=dr,
            reliable=reliable,
            complete=complete,
        )
        literal_index = None if requested == 0 else signed_external_to_internal_index(requested, i2e)
        if literal_index is not None and not bool(mask[literal_index]):
            # A replay claiming eligibility for a literal outside the captured
            # exact mask is corrupt, not a hard negative.
            if reliable:
                raise ActionEvalDatasetError("reliable rollout lies outside exact eligibility")
            literal_index = None
        cost = action_cost_storage(dc, dp, dd, config)
        outcomes.append((outcome, literal_index, cost))
    if len(solved_statuses) > 1:
        raise ActionEvalDatasetError("SAT/UNSAT rollout inconsistency within one state")

    baseline_rows = [item for item in outcomes if item[0].requested == 0]
    if len(baseline_rows) > 1:
        raise ActionEvalDatasetError("state contains multiple native baseline evaluations")
    baseline = baseline_rows[0] if baseline_rows and baseline_rows[0][0].reliable else None

    forced: list[tuple[_RawOutcome, int, float]] = []
    seen_literals: set[int] = set()
    for outcome, literal_index, cost in outcomes:
        if outcome.requested == 0 or not outcome.reliable or literal_index is None:
            continue
        if literal_index in seen_literals:
            raise ActionEvalDatasetError("duplicate forced evaluation literal")
        seen_literals.add(literal_index)
        forced.append((outcome, literal_index, cost))

    def compare(a: tuple[_RawOutcome, int, float], b: tuple[_RawOutcome, int, float]) -> int:
        base = _compare_outcomes(a[0], b[0], config.tie_margin)
        if base != 2:
            return base
        if variant == "preference-action":
            left = (
                a[0].conflicts,
                a[0].propagations,
                a[0].decisions,
                a[0].restarts,
            )
            right = (
                b[0].conflicts,
                b[0].propagations,
                b[0].decisions,
                b[0].restarts,
            )
            if all(x <= y for x, y in zip(left, right)) and left != right:
                return -1
            if all(y <= x for x, y in zip(left, right)) and left != right:
                return 1
            return 0
        gap = a[2] - b[2]
        if abs(gap) < config.tie_margin:
            return 0
        return -1 if gap < 0 else 1

    pairs: list[tuple[int, int]] = []
    if variant in {"satact", "preference-action", "defer-action"}:
        for left_pos in range(len(forced)):
            for right_pos in range(left_pos + 1, len(forced)):
                order = compare(forced[left_pos], forced[right_pos])
                if order < 0:
                    pairs.append((forced[left_pos][1], forced[right_pos][1]))
                elif order > 0:
                    pairs.append((forced[right_pos][1], forced[left_pos][1]))
    elif variant in {"signed-action", "signed-defer-action"} and forced:
        complete = [item for item in forced if item[0].complete]
        best = min(complete, key=lambda item: (item[2], item[1])) if complete else None
        incomplete = [item for item in forced if not item[0].complete]
        worst = (
            min(incomplete, key=lambda item: item[1])
            if incomplete
            else max(complete, key=lambda item: (item[2], -item[1]))
            if complete
            else None
        )
        native_forced = next(
            (item for item in forced if item[0].requested == native_external), None
        )
        role_pairs = ((best, native_forced), (native_forced, worst), (best, worst))
        seen_pairs: set[tuple[int, int]] = set()
        for left, right in role_pairs:
            if left is None or right is None or left[1] == right[1]:
                continue
            order = compare(left, right)
            pair = (left[1], right[1]) if order < 0 else (right[1], left[1]) if order > 0 else None
            if pair is not None and pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)

    if variant in STORAGE_DEFER_VARIANTS and baseline is not None:
        baseline_outcome, _, baseline_cost = baseline
        for outcome, literal_index, cost in forced:
            if outcome.complete and not baseline_outcome.complete:
                pairs.append((literal_index, -1))
            elif baseline_outcome.complete and not outcome.complete:
                pairs.append((-1, literal_index))
            elif outcome.complete and baseline_outcome.complete:
                if baseline_cost - cost > config.tie_margin:
                    pairs.append((literal_index, -1))
                else:  # DEFER wins every comparable tie or worse literal.
                    pairs.append((-1, literal_index))

    list_literals: list[int] = []
    list_target: list[float] = []
    if variant == "list-action":
        usable = [item for item in forced if item[0].complete]
        if len(usable) >= 2:
            list_literals = [item[1] for item in usable]
            logits = torch.tensor([-item[2] / config.listwise_temperature for item in usable])
            list_target = torch.softmax(logits, dim=0).tolist()
    elif variant == "signed-list-action":
        solved = [item for item in forced if item[0].complete]
        if solved and len(forced) >= 2:
            penalty = max(item[2] for item in solved) + config.censored_penalty
            list_literals = [item[1] for item in forced]
            effective = [item[2] if item[0].complete else penalty for item in forced]
            logits = torch.tensor([-cost / config.listwise_temperature for cost in effective])
            list_target = torch.softmax(logits, dim=0).tolist()

    usable = (
        native_index is not None
        if variant == "basic-action"
        else bool(pairs)
        if variant in {"satact", "signed-action", "preference-action", "defer-action", "signed-defer-action"}
        else bool(list_literals)
    )
    if not usable:
        return None

    solved_costs = [item[2] for item in forced if item[0].complete]
    metric_penalty = (max(solved_costs) + config.censored_penalty) if solved_costs else math.nan
    evaluated = sorted(forced, key=lambda item: item[1])
    return {
        "n_literals": 2 * len(i2e),
        "native_literal_index": native_index,
        "pair_better": [item[0] for item in pairs],
        "pair_worse": [item[1] for item in pairs],
        "listwise_literals": list_literals,
        "listwise_target": list_target,
        "evaluated_literals": [item[1] for item in evaluated],
        "evaluated_costs": [
            item[2]
            if item[0].complete
            else metric_penalty
            if math.isfinite(metric_penalty)
            else None
            for item in evaluated
        ],
        "evaluated_complete": [item[0].complete for item in evaluated],
        "baseline_cost": baseline[2] if baseline is not None and baseline[0].complete else None,
        "baseline_complete": bool(baseline is not None and baseline[0].complete),
    }


@dataclass(frozen=True)
class _IndexEntry:
    shard_index: int
    record_index: int
    record_id: str
    instance_id: str
    eligible_state: int
    target: Mapping[str, Any]
    instance_state_count: int = 0


@dataclass(frozen=True)
class _PlanPartStorage:
    stem: str
    instance_ids: frozenset[str]
    source_parts: tuple[str, ...] | None
    source_part_count: int | None
    parts_per_shard: int | None
    source_policy_sha256: str
    split: str | None
    dataset_groups: tuple[str, ...]
    shard_build_id: str | None = None
    source_generations: tuple[str, ...] = ()
    metadata_version: int = 1


class ActionEvalShardDatasetStorage(torch.utils.data.Dataset[ActionEvalSampleStorage]):
    """Random-access, variant-filtered compact shard dataset."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        variant: str,
        objective_config: ObjectiveConfigStorage | None = None,
        expected_split: str | None = None,
        verify_metadata: bool = True,
        verify_checksums: bool = False,
        index_cache_dir: str | Path | None = None,
        index_cache_path: str | Path | None = None,
        index_cache_read_only: bool = False,
        max_states: int | None = None,
        max_open_shards: int = 2,
        max_header_bytes: int = 1024 * 1024 * 1024,
        max_array_bytes: int = 512 * 1024 * 1024,
        progress_callback: Callable[[IndexBuildProgressStorage], None] | None = None,
    ) -> None:
        if variant not in STORAGE_VARIANTS:
            raise ValueError(f"unsupported SAT-ACT variant: {variant!r}")
        if max_states is not None and max_states < 1:
            raise ValueError("max_states must be positive when supplied")
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self.split_dir = Path(split_dir).expanduser().resolve(strict=True)
        self.variant = variant
        self.objective_config = objective_config or ObjectiveConfigStorage()
        self.verify_checksums = bool(verify_checksums)
        self.verify_metadata = bool(verify_metadata or verify_checksums)
        self.max_open_shards = int(max_open_shards)
        self.max_header_bytes = int(max_header_bytes)
        self.max_array_bytes = int(max_array_bytes)
        self._progress_callback = progress_callback
        self._readers: OrderedDict[int, CompactArrayReader] = OrderedDict()
        # DataLoader uses ``fork`` by default on Linux.  Remember which
        # process owns these readers so a child never uses the parent's open
        # file descriptions (and, consequently, its shared seek position).
        self._reader_pid = os.getpid()
        self._checksum_verified_shards: set[int] = set()
        shard_manifests = self.split_dir / "manifests" / "shards"
        v2_policy = shard_manifests / "policy.json"
        v2_plan = shard_manifests / "plan.jsonl"
        if v2_policy.exists() != v2_plan.exists():
            raise ActionEvalDatasetError(
                "incomplete shard v2 metadata: policy.json and plan.jsonl must both exist"
            )
        self._metadata_version = 2 if v2_policy.is_file() else 1
        self._reader_class = CompactArrayReaderBase
        if self._metadata_version == 2:
            self.plan_path = v2_plan
            self.policy_path = v2_policy
        else:
            self.plan_path = self.split_dir / "manifests" / "pipeline.parts.jsonl"
            self.policy_path = self.split_dir / "manifests" / "pipeline.policy.json"
        if not self.plan_path.is_file():
            raise FileNotFoundError(f"missing plan manifest: {self.plan_path}")
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"missing pipeline policy: {self.policy_path}")
        try:
            policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ActionEvalDatasetError(f"invalid pipeline policy: {self.policy_path}") from exc
        if not isinstance(policy, dict):
            raise ActionEvalDatasetError("pipeline policy must be a JSON object")
        if self._metadata_version == 2 and policy.get("payload_binding") == SHARD_PAYLOAD_V2:
            # Keep the v1-only path independent of the optional v2 package.
            # Spawned DataLoader workers can pickle this class normally after
            # the dataset has selected the v2 on-disk contract.
            from gen_data.rollout_dataset.compact import CompactArrayReader as CompactArrayReaderCurrent

            self._reader_class = CompactArrayReaderCurrent
        self.policy_fingerprint = hashlib.sha256(
            json.dumps(
                policy,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        plan_bytes = self.plan_path.read_bytes()
        self.manifest_fingerprint = hashlib.sha256(plan_bytes).hexdigest()
        if self._metadata_version == 2:
            self._plan_parts, instance_rows = self._parse_v2_plan(
                plan_bytes, expected_split, policy
            )
            self._parts = [part.stem for part in self._plan_parts]
        else:
            self._plan_parts = []
            self._parts, instance_rows = self._parse_plan(
                plan_bytes, expected_split, self.policy_fingerprint
            )
        self.part_count = len(self._parts)
        self.manifest_split = self._validate_manifest_split(instance_rows, expected_split)
        explicit_index_config: Mapping[str, Any] | None = None
        legacy_fingerprint_payload: Mapping[str, Any] | None = None
        if index_cache_path is not None:
            self.index_cache_path = Path(index_cache_path).expanduser().absolute()
            index_config = _index_config_storage(
                reader_schema=self._metadata_version,
                parts=self._parts,
                variant=variant,
                objective=self.objective_config,
                max_states=max_states,
            )
            self.split_fingerprint = f"index-cache-path:{self.index_cache_path}"
            entries = self._load_explicit_cached_index(
                self.index_cache_path, index_config
            )
            self.index_cache_hit = entries is not None
            if entries is None and index_cache_read_only:
                raise ActionEvalDatasetError(
                    f"explicit index cache is missing or incompatible: {self.index_cache_path}"
                )
            if entries is None:
                entries = self._build_index(max_states=max_states)
                explicit_index_config = index_config
        else:
            inventory = (
                self._validate_commits()
                if self.verify_metadata
                else self._commit_inventory(validate_shard_metadata=False)
            )
            fingerprint_payload = {
                "reader_schema": self._metadata_version,
                "root": str(self.split_dir),
                "manifest": self.manifest_fingerprint,
                "policy": self.policy_fingerprint,
                "inventory": inventory,
                "variant": variant,
                "objective": asdict(self.objective_config),
                "max_states": max_states,
            }
            self.split_fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            self.index_cache_path = self._cache_path(index_cache_dir)
            entries = self._load_cached_index(self.index_cache_path, fingerprint_payload)
            self.index_cache_hit = entries is not None
            if not self.index_cache_hit:
                entries = self._build_index(max_states=max_states)
                legacy_fingerprint_payload = fingerprint_payload
        if self.verify_checksums:
            # A cached index must never bypass the requested payload audit.
            # This also checks all remaining parts when max_states stopped the
            # index build before the end of the plan.
            self._audit_all_checksums(states=len(entries))
        if explicit_index_config is not None:
            self._write_explicit_cached_index(
                self.index_cache_path, explicit_index_config, entries
            )
        if legacy_fingerprint_payload is not None:
            self._write_cached_index(
                self.index_cache_path, legacy_fingerprint_payload, entries
            )
        counts: dict[str, int] = defaultdict(int)
        for entry in entries:
            counts[entry.instance_id] += 1
        self._entries = [replace(entry, instance_state_count=counts[entry.instance_id]) for entry in entries]
        self.instance_count = len(counts)
        self.shard_ids = [entry.shard_index for entry in self._entries]
        # Index construction can leave the last shards open.  DataLoader
        # workers are started lazily, so close them before they can be
        # inherited.  Readers reopen on demand in the process that consumes
        # samples.
        self.close()

    def _parse_v2_plan(
        self,
        payload: bytes,
        expected_split: str | None,
        policy: Mapping[str, Any],
    ) -> tuple[list[_PlanPartStorage], dict[str, Mapping[str, Any]]]:
        if policy.get("schema") != SHARD_POLICY_SCHEMA_V2:
            raise ActionEvalDatasetError("unexpected shard v2 policy schema")
        build_id = policy.get("shard_build_id")
        source_policy_id = policy.get("source_policy_id")
        payload_binding = policy.get("payload_binding")
        if not isinstance(build_id, str) or not build_id:
            raise ActionEvalDatasetError("shard v2 policy has no shard_build_id")
        if not isinstance(source_policy_id, str) or not source_policy_id:
            raise ActionEvalDatasetError("shard v2 policy has no source_policy_id")
        if payload_binding not in {SHARD_PAYLOAD_V2, LEGACY_PAYLOAD_V2}:
            raise ActionEvalDatasetError("unsupported shard v2 payload binding")
        shard_count = policy.get("shard_count")
        source_part_total = policy.get("source_part_count")
        if (
            isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count < 1
            or isinstance(source_part_total, bool)
            or not isinstance(source_part_total, int)
            or source_part_total < 1
        ):
            raise ActionEvalDatasetError("invalid shard v2 policy counts")
        policy_parts_per_shard = policy.get("parts_per_shard")
        if payload_binding == SHARD_PAYLOAD_V2:
            source_policy_version = policy.get("source_policy_version")
            if (
                isinstance(source_policy_version, bool)
                or not isinstance(source_policy_version, int)
                or source_policy_version not in {1, 2}
            ):
                raise ActionEvalDatasetError("invalid shard v2 source policy version")
            if (
                isinstance(policy_parts_per_shard, bool)
                or not isinstance(policy_parts_per_shard, int)
                or policy_parts_per_shard < 1
            ):
                raise ActionEvalDatasetError("invalid shard v2 parts_per_shard")
        elif policy_parts_per_shard is not None:
            raise ActionEvalDatasetError(
                "legacy-provenance shard policy must not declare one global grouping"
            )
        for path_field in ("source_root", "output_root"):
            value = policy.get(path_field)
            if not isinstance(value, str) or not value:
                raise ActionEvalDatasetError(
                    f"shard v2 policy has invalid {path_field}"
                )

        parts: list[_PlanPartStorage] = []
        instances: dict[str, Mapping[str, Any]] = {}
        seen_stems: set[str] = set()
        seen_source_parts: set[str] = set()
        try:
            plan_lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ActionEvalDatasetError("shard v2 plan is not valid UTF-8") from exc
        for line_number, line in enumerate(plan_lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ActionEvalDatasetError(
                    f"invalid shard v2 plan JSON at line {line_number}"
                ) from exc
            if (
                not isinstance(row, dict)
                or row.get("schema") != SHARD_PLAN_SCHEMA_V2
                or row.get("shard_build_id") != build_id
            ):
                raise ActionEvalDatasetError(
                    f"unexpected shard v2 plan row at line {line_number}"
                )
            stem = row.get("shard_stem")
            source_parts = row.get("source_parts")
            source_generations = row.get("source_generations")
            instance_ids = row.get("instance_ids")
            dataset_group = row.get("dataset_group")
            split = row.get("split")
            source_count = row.get("source_part_count")
            parts_per_shard = row.get("parts_per_shard")
            shard_index = row.get("shard_index")
            instance_count = row.get("instance_count")
            if (
                not isinstance(stem, str)
                or not _SAFE_STEM.fullmatch(stem)
                or stem in seen_stems
                or isinstance(shard_index, bool)
                or not isinstance(shard_index, int)
                or shard_index != len(parts)
                or not isinstance(source_parts, list)
                or not source_parts
                or not all(
                    isinstance(item, str) and _SAFE_STEM.fullmatch(item)
                    for item in source_parts
                )
                or len(source_parts) != len(set(source_parts))
                or any(item in seen_source_parts for item in source_parts)
                or isinstance(source_count, bool)
                or not isinstance(source_count, int)
                or source_count < 1
                or source_count != len(source_parts)
                or isinstance(parts_per_shard, bool)
                or not isinstance(parts_per_shard, int)
                or parts_per_shard < 1
                or source_count > parts_per_shard
                or not isinstance(instance_ids, list)
                or not instance_ids
                or not all(isinstance(item, str) and item for item in instance_ids)
                or len(instance_ids) != len(set(instance_ids))
                or isinstance(instance_count, bool)
                or not isinstance(instance_count, int)
                or instance_count != len(instance_ids)
                or not isinstance(dataset_group, str)
                or not dataset_group
                or not isinstance(split, str)
                or not split
            ):
                raise ActionEvalDatasetError(
                    f"invalid shard v2 plan fields at line {line_number}"
                )
            if payload_binding == SHARD_PAYLOAD_V2 and (
                parts_per_shard != policy_parts_per_shard
                or not isinstance(source_generations, list)
                or len(source_generations) != len(source_parts)
                or not all(
                    isinstance(item, str) and item for item in source_generations
                )
            ):
                raise ActionEvalDatasetError(
                    f"invalid shard v2 source generations at line {line_number}"
                )
            if payload_binding == LEGACY_PAYLOAD_V2 and source_generations != []:
                raise ActionEvalDatasetError(
                    f"legacy shard provenance has source generations at line {line_number}"
                )
            for instance_id in instance_ids:
                if instance_id in instances:
                    raise ActionEvalDatasetError("duplicate instance_id in shard v2 plan")
                instances[instance_id] = {
                    "instance_id": instance_id,
                    "split": split,
                    "dataset_group": dataset_group,
                }
            legacy_payload = payload_binding == LEGACY_PAYLOAD_V2
            legacy_single = legacy_payload and len(source_parts) == 1 and source_parts[0] == stem
            parts.append(
                _PlanPartStorage(
                    stem=stem,
                    instance_ids=frozenset(instance_ids),
                    source_parts=None if legacy_single else tuple(source_parts),
                    source_part_count=None if legacy_single else source_count,
                    parts_per_shard=None if legacy_single else parts_per_shard,
                    source_policy_sha256=source_policy_id,
                    split=split,
                    dataset_groups=(dataset_group,),
                    shard_build_id=build_id,
                    source_generations=tuple(source_generations or ()),
                    metadata_version=1 if legacy_payload else 2,
                )
            )
            seen_stems.add(stem)
            seen_source_parts.update(source_parts)
        if not parts:
            raise ActionEvalDatasetError("shard v2 plan contains no rows")
        if len(parts) != shard_count:
            raise ActionEvalDatasetError("shard v2 policy/plan shard count mismatch")
        if len(seen_source_parts) != source_part_total:
            raise ActionEvalDatasetError("shard v2 policy/plan source part count mismatch")
        if expected_split is not None and any(part.split != expected_split for part in parts):
            raise ActionEvalDatasetError(
                f"shard v2 plan does not match expected split {expected_split!r}"
            )
        return parts, instances

    def _parse_plan(
        self,
        payload: bytes,
        expected_split: str | None,
        expected_policy_sha256: str,
    ) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
        """Parse the original v1 plan without imposing v2 provenance fields."""

        parts: list[str] = []
        instances: dict[str, Mapping[str, Any]] = {}
        seen_stems: set[str] = set()
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ActionEvalDatasetError(f"invalid plan JSON at line {line_number}") from exc
            if not isinstance(row, dict) or row.get("schema") != PLAN_SCHEMA:
                raise ActionEvalDatasetError(f"unexpected plan schema at line {line_number}")
            if row.get("policy_sha256") != expected_policy_sha256:
                raise ActionEvalDatasetError(f"plan/policy SHA-256 mismatch at line {line_number}")
            stem = row.get("part_stem")
            if not isinstance(stem, str) or not _SAFE_STEM.fullmatch(stem) or stem in seen_stems:
                raise ActionEvalDatasetError(f"invalid/duplicate part_stem at line {line_number}")
            rows = row.get("instances")
            if not isinstance(rows, list) or not rows:
                raise ActionEvalDatasetError(f"plan part has no instances at line {line_number}")
            for instance in rows:
                if not isinstance(instance, dict):
                    raise ActionEvalDatasetError("plan instance must be an object")
                instance_id = instance.get("instance_id")
                if not isinstance(instance_id, str) or not instance_id or instance_id in instances:
                    raise ActionEvalDatasetError("invalid/duplicate instance_id in plan")
                instances[instance_id] = instance
            parts.append(stem)
            seen_stems.add(stem)
        if not parts:
            raise ActionEvalDatasetError("plan manifest contains no parts")
        return parts, instances

    @staticmethod
    def _validate_manifest_split(instances: Mapping[str, Mapping[str, Any]], expected: str | None) -> str | None:
        values = {row.get("split") for row in instances.values() if row.get("split") is not None}
        if len(values) > 1:
            raise ActionEvalDatasetError("one build root contains multiple manifest splits")
        actual = next(iter(values)) if values else None
        if expected is not None and actual != expected:
            raise ActionEvalDatasetError(f"manifest split {actual!r} does not match {expected!r}")
        return str(actual) if actual is not None else None

    def _validate_shard_plan_binding(
        self, plan_part: _PlanPartStorage, reader: CompactArrayReader
    ) -> None:
        metadata = reader.header.get("part_metadata")
        if not isinstance(metadata, dict):
            raise ActionEvalDatasetError(
                f"shard part metadata mismatch for {plan_part.stem}"
            )
        if plan_part.metadata_version == 2:
            merge = metadata.get("merge")
            source_parts = metadata.get("source_parts")
            if (
                metadata.get("group_stem") != plan_part.stem
                or metadata.get("shard_build_id") != plan_part.shard_build_id
                or metadata.get("source_policy_id") != plan_part.source_policy_sha256
                or metadata.get("split") != plan_part.split
                or metadata.get("dataset_groups") != list(plan_part.dataset_groups)
                or not isinstance(merge, dict)
                or merge.get("source_part_count") != plan_part.source_part_count
                or merge.get("parts_per_shard") != plan_part.parts_per_shard
                or not isinstance(source_parts, list)
                or len(source_parts) != plan_part.source_part_count
                or not all(isinstance(item, Mapping) for item in source_parts)
                or [item.get("source_part") for item in source_parts]
                != list(plan_part.source_parts or ())
            ):
                raise ActionEvalDatasetError(
                    f"shard v2 metadata mismatch for {plan_part.stem}"
                )
            if plan_part.source_generations and (
                [item.get("source_generation_id") for item in source_parts]
                != list(plan_part.source_generations)
            ):
                raise ActionEvalDatasetError(
                    f"shard v2 source generation mismatch for {plan_part.stem}"
                )
        elif plan_part.source_parts is None:
            if (
                metadata.get("source_part") != plan_part.stem
                or metadata.get("source_policy_sha256")
                != plan_part.source_policy_sha256
            ):
                raise ActionEvalDatasetError(
                    f"shard part metadata mismatch for {plan_part.stem}"
                )
        else:
            merge = metadata.get("merge")
            source_parts = metadata.get("source_parts")
            if (
                metadata.get("source_policy_sha256")
                != plan_part.source_policy_sha256
                or not isinstance(merge, dict)
                or merge.get("source_part_count")
                != plan_part.source_part_count
                or merge.get("parts_per_shard") != plan_part.parts_per_shard
                or not isinstance(source_parts, list)
                or len(source_parts) != plan_part.source_part_count
            ):
                raise ActionEvalDatasetError(
                    f"grouped shard metadata mismatch for {plan_part.stem}"
                )
            actual_stems: list[str] = []
            for source in source_parts:
                if (
                    not isinstance(source, dict)
                    or not isinstance(source.get("source_part"), str)
                    or source.get("source_policy_sha256")
                    != plan_part.source_policy_sha256
                ):
                    raise ActionEvalDatasetError(
                        f"invalid grouped source provenance for {plan_part.stem}"
                    )
                actual_stems.append(source["source_part"])
            if tuple(actual_stems) != plan_part.source_parts:
                raise ActionEvalDatasetError(
                    f"grouped shard/source plan mismatch for {plan_part.stem}"
                )

        seen_record_ids: set[str] = set()
        seen_state_keys: set[tuple[str, int]] = set()
        for record in reader.header["records"]:
            record_id = record.get("record_id")
            record_metadata = record.get("metadata")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in seen_record_ids
                or not isinstance(record_metadata, dict)
            ):
                raise ActionEvalDatasetError(
                    f"invalid/duplicate shard record for {plan_part.stem}"
                )
            seen_record_ids.add(record_id)
            kind = record_metadata.get("kind")
            if kind == "state":
                state = record_metadata.get("state")
                if not isinstance(state, dict):
                    raise ActionEvalDatasetError("state record has no scalar metadata")
                instance_id = state.get("instance")
                eligible_state = state.get("eligible_state")
            elif kind == "capture_failure":
                instance_id = record_metadata.get("instance")
                eligible_state = record_metadata.get("eligible_state")
            else:
                raise ActionEvalDatasetError("unknown compact record kind")
            if (
                not isinstance(instance_id, str)
                or instance_id not in plan_part.instance_ids
                or isinstance(eligible_state, bool)
                or not isinstance(eligible_state, int)
                or eligible_state < 1
            ):
                raise ActionEvalDatasetError(
                    f"shard record is absent from plan part {plan_part.stem}"
                )
            state_key = (instance_id, eligible_state)
            if (
                state_key in seen_state_keys
                or record_id != f"{instance_id}:{eligible_state}"
            ):
                raise ActionEvalDatasetError(
                    f"duplicate/mismatched state identity in {plan_part.stem}"
                )
            seen_state_keys.add(state_key)

    def _commit_inventory(
        self, *, validate_shard_metadata: bool
    ) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for shard_index, stem in enumerate(self._parts):
            complete_path = self.split_dir / "shards" / f"{stem}.complete.json"
            if not complete_path.is_file():
                raise ActionEvalDatasetError(f"upload incomplete: missing {complete_path}")
            try:
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ActionEvalDatasetError(f"invalid completion marker: {complete_path}") from exc
            files = complete.get("files", {}) if isinstance(complete, dict) else {}
            row: dict[str, Any] = {
                "stem": stem,
                "complete_bytes": complete_path.stat().st_size,
                "complete_mtime_ns": complete_path.stat().st_mtime_ns,
            }
            for kind, suffix in (("header", "header.json"), ("arrays", "arrays.bin")):
                descriptor = files.get(kind, {}) if isinstance(files, dict) else {}
                expected_name = f"{stem}.{suffix}"
                if descriptor.get("name") != expected_name or not isinstance(descriptor.get("bytes"), int):
                    raise ActionEvalDatasetError(f"invalid {kind} descriptor for shard {stem}")
                path = self.split_dir / "shards" / expected_name
                if not path.is_file() or path.stat().st_size != descriptor["bytes"]:
                    raise ActionEvalDatasetError(f"upload incomplete: {kind} size mismatch for {stem}")
                row[f"{kind}_bytes"] = descriptor["bytes"]
                row[f"{kind}_mtime_ns"] = path.stat().st_mtime_ns
                row[f"{kind}_sha256"] = descriptor.get("sha256")
            if self._metadata_version == 2 and validate_shard_metadata:
                with self._reader_class(
                    complete_path,
                    verify=False,
                    max_header_bytes=self.max_header_bytes,
                    max_array_bytes=self.max_array_bytes,
                ) as reader:
                    self._validate_shard_plan_binding(
                        self._plan_parts[shard_index], reader
                    )
            inventory.append(row)
        return inventory

    def _validate_commits(self) -> list[dict[str, Any]]:
        return self._commit_inventory(validate_shard_metadata=True)

    def _cache_path(self, root: str | Path | None) -> Path | None:
        if root is False:  # type: ignore[comparison-overlap]
            return None
        if root is None:
            root_path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "satact"
        else:
            root_path = Path(root).expanduser()
        return root_path / f"{self.split_fingerprint}.index.json"

    @staticmethod
    def _load_explicit_cached_index(
        path: Path, index_config: Mapping[str, Any]
    ) -> list[_IndexEntry] | None:
        return ActionEvalShardDatasetStorage._load_cached_index(path, index_config)

    @staticmethod
    def _write_explicit_cached_index(
        path: Path,
        index_config: Mapping[str, Any],
        entries: Sequence[_IndexEntry],
    ) -> None:
        ActionEvalShardDatasetStorage._write_cached_index(path, index_config, entries)

    @staticmethod
    def _load_cached_index(path: Path | None, fingerprint: Mapping[str, Any]) -> list[_IndexEntry] | None:
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != fingerprint:
                return None
            return [_IndexEntry(**item) for item in payload["entries"]]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cached_index(path: Path | None, fingerprint: Mapping[str, Any], entries: Sequence[_IndexEntry]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {"fingerprint": fingerprint, "entries": [asdict(item) for item in entries]}
        try:
            temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    def _open_reader(self, shard_index: int) -> CompactArrayReader:
        current_pid = os.getpid()
        if current_pid != self._reader_pid:
            # ``__getstate__`` protects spawn/pickle, while this branch is the
            # corresponding protection for Linux fork.  Closing the child's
            # inherited descriptors does not close the parent's descriptors.
            self.close()
            self._reader_pid = current_pid
        reader = self._readers.pop(shard_index, None)
        if reader is None:
            stem = self._parts[shard_index]
            verify_checksum = (
                self.verify_checksums and shard_index not in self._checksum_verified_shards
            )
            reader = self._reader_class(
                self.split_dir / "shards" / f"{stem}.complete.json",
                verify=verify_checksum,
                max_header_bytes=self.max_header_bytes,
                max_array_bytes=self.max_array_bytes,
            )
            if verify_checksum:
                self._checksum_verified_shards.add(shard_index)
        self._readers[shard_index] = reader
        while len(self._readers) > self.max_open_shards:
            _, old = self._readers.popitem(last=False)
            old.close()
        return reader

    def _report_progress(
        self,
        *,
        phase: str,
        shard_index: int,
        states: int,
        limit_reached: bool = False,
    ) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(
            IndexBuildProgressStorage(
                phase=phase,
                part_index=shard_index + 1,
                part_count=len(self._parts),
                part_stem=self._parts[shard_index],
                states=states,
                limit_reached=limit_reached,
            )
        )

    def _audit_all_checksums(self, *, states: int) -> None:
        for shard_index in range(len(self._parts)):
            self._open_reader(shard_index)
            self._report_progress(
                phase="checksum",
                shard_index=shard_index,
                states=states,
            )

    @staticmethod
    def _read_arrays(reader: CompactArrayReader, record: Mapping[str, Any], names: Sequence[str]) -> dict[str, list[int]]:
        read_many = getattr(reader, "read_arrays", None)
        if read_many is not None:
            return {
                name: list(values)
                for name, values in read_many(record, names).items()
            }
        return {name: list(reader.read_array(record, name)) for name in names}

    def _build_index(self, *, max_states: int | None) -> list[_IndexEntry]:
        entries: list[_IndexEntry] = []
        eval_names = (
            "eval_requested_literal", "eval_target_reached", "eval_locator_match",
            "eval_action_eligible", "eval_action_applied", "eval_applied_literal",
            "eval_native_literal",
            "eval_status", "eval_censored", "eval_horizon_hit", "eval_delta_conflicts",
            "eval_delta_decisions", "eval_delta_propagations",
        )
        if self.variant == "preference-action":
            eval_names += ("eval_delta_restarts",)
        for shard_index, stem in enumerate(self._parts):
            reader = self._open_reader(shard_index)
            if self._metadata_version == 1:
                # Preserve the original v1 contract exactly.  In particular,
                # grouped v1 plans and headers predate all v2 split/group
                # provenance fields and remain valid without them.
                metadata = reader.header.get("part_metadata", {})
                if not isinstance(metadata, dict):
                    raise ActionEvalDatasetError(
                        f"shard part metadata mismatch for {stem}"
                    )
                merge = metadata.get("merge")
                if merge is None:
                    if metadata.get("source_part") != stem:
                        raise ActionEvalDatasetError(
                            f"shard part metadata mismatch for {stem}"
                        )
                elif (
                    not isinstance(merge, dict)
                    or not isinstance(merge.get("source_part_count"), int)
                    or merge.get("source_part_count", 0) < 1
                    or not isinstance(metadata.get("source_parts"), list)
                    or len(metadata["source_parts"])
                    != merge["source_part_count"]
                ):
                    raise ActionEvalDatasetError(
                        f"grouped shard metadata mismatch for {stem}"
                    )
            for record_index, record in enumerate(reader.header["records"]):
                record_metadata = record.get("metadata", {})
                if record_metadata.get("kind") == "capture_failure":
                    continue
                if record_metadata.get("kind") != "state":
                    raise ActionEvalDatasetError("unknown compact record kind")
                state = record_metadata.get("state")
                if not isinstance(state, dict):
                    raise ActionEvalDatasetError("state record has no scalar metadata")
                state_arrays = self._read_arrays(
                    reader,
                    record,
                    ("i2e", "eligibility_bits", *eval_names),
                )
                i2e = state_arrays["i2e"]
                eligibility = state_arrays["eligibility_bits"]
                arrays = {name: state_arrays[name] for name in eval_names}
                target = _derive_plan(
                    variant=self.variant,
                    config=self.objective_config,
                    state=state,
                    i2e=i2e,
                    eligibility_bits=eligibility,
                    arrays=arrays,
                )
                if target is None:
                    continue
                instance_id = state.get("instance")
                if not isinstance(instance_id, str):
                    raise ActionEvalDatasetError("state instance id is invalid")
                entries.append(
                    _IndexEntry(
                        shard_index=shard_index,
                        record_index=record_index,
                        record_id=str(record.get("record_id")),
                        instance_id=instance_id,
                        eligible_state=int(state.get("eligible_state")),
                        target=target,
                    )
                )
                if max_states is not None and len(entries) >= max_states:
                    self._report_progress(
                        phase="index",
                        shard_index=shard_index,
                        states=len(entries),
                        limit_reached=True,
                    )
                    return entries
            self._report_progress(
                phase="index",
                shard_index=shard_index,
                states=len(entries),
            )
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> ActionEvalSampleStorage:
        entry = self._entries[index]
        reader = self._open_reader(entry.shard_index)
        record = reader.header["records"][entry.record_index]
        if record.get("record_id") != entry.record_id:
            raise ActionEvalDatasetError("cached record identity no longer matches shard")
        state = record["metadata"]["state"]
        arrays = self._read_arrays(
            reader,
            record,
            (
                "i2e",
                "eligibility_bits",
                "assignment_value",
                "assignment_level",
                "clause_offsets",
                "clause_literals_internal",
            ),
        )
        i2e = arrays["i2e"]
        eligibility_bits = arrays["eligibility_bits"]
        assignment_value = torch.tensor(arrays["assignment_value"], dtype=torch.long)
        assignment_level = torch.tensor(arrays["assignment_level"], dtype=torch.long)
        offsets = arrays["clause_offsets"]
        literals = arrays["clause_literals_internal"]
        n_vars = len(i2e)
        n_clauses = len(offsets) - 1
        l_edge: list[int] = []
        c_edge: list[int] = []
        for clause_index in range(n_clauses):
            begin, end = int(offsets[clause_index]), int(offsets[clause_index + 1])
            for literal in literals[begin:end]:
                literal = int(literal)
                if literal == 0 or abs(literal) > n_vars:
                    raise ActionEvalDatasetError("invalid internal literal in solver-active CNF")
                l_edge.append(2 * (abs(literal) - 1) + (literal < 0))
                c_edge.append(clause_index)
        l_edge_tensor = torch.tensor(l_edge, dtype=torch.long)
        c_edge_tensor = torch.tensor(c_edge, dtype=torch.long)
        literal_occurrence = torch.bincount(l_edge_tensor, minlength=2 * n_vars).float()
        variable_occurrence = literal_occurrence.reshape(n_vars, 2).sum(dim=1)
        clause_size = torch.bincount(c_edge_tensor, minlength=n_clauses).float()
        graph = ActionEvalLCGStorage(
            n_vars=torch.tensor([n_vars], dtype=torch.long),
            n_clauses=torch.tensor([n_clauses], dtype=torch.long),
            l_edge_index=l_edge_tensor,
            c_edge_index=c_edge_tensor,
            l_batch=torch.zeros(2 * n_vars, dtype=torch.long),
            c_batch=torch.zeros(n_clauses, dtype=torch.long),
            label_var_mask=None,
            label_clause_mask=None,
            assignment_value=assignment_value,
            assignment_level=assignment_level,
            decision_level=torch.tensor([int(state["level"])], dtype=torch.long),
            literal_occurrence=literal_occurrence,
            variable_occurrence=variable_occurrence,
            clause_size=clause_size,
        )
        target_values = dict(entry.target)
        target = ActionEvalTargetStorage(
            instance_id=entry.instance_id,
            variant=self.variant,
            eligible_state=entry.eligible_state,
            record_id=entry.record_id,
            shard_stem=self._parts[entry.shard_index],
            n_literals=int(target_values["n_literals"]),
            eligibility_mask=exact_internal_literal_mask(i2e, eligibility_bits),
            native_literal_index=target_values["native_literal_index"],
            pair_better=torch.tensor(target_values["pair_better"], dtype=torch.long),
            pair_worse=torch.tensor(target_values["pair_worse"], dtype=torch.long),
            listwise_literals=torch.tensor(target_values["listwise_literals"], dtype=torch.long),
            listwise_target=torch.tensor(target_values["listwise_target"], dtype=torch.float32),
            evaluated_literals=torch.tensor(target_values["evaluated_literals"], dtype=torch.long),
            evaluated_costs=torch.tensor(
                [math.nan if value is None else value for value in target_values["evaluated_costs"]],
                dtype=torch.float32,
            ),
            evaluated_complete=torch.tensor(target_values["evaluated_complete"], dtype=torch.bool),
            baseline_cost=target_values["baseline_cost"],
            baseline_complete=bool(target_values["baseline_complete"]),
            instance_state_count=entry.instance_state_count,
            instance_weight=1.0 / entry.instance_state_count,
        )
        return ActionEvalSampleStorage(graph=graph, supervision=target)

    def close(self) -> None:
        while self._readers:
            _, reader = self._readers.popitem()
            reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # File descriptors must never be inherited/pickled into DataLoader workers.
        state["_readers"] = OrderedDict()
        state["_progress_callback"] = None
        return state


# Backward-friendly aliases used by the training/evaluation entrypoints.
DecisionTraceActionEvalDatasetStorage = ActionEvalShardDatasetStorage
derive_target_storage = _derive_plan


__all__ = [
    "ActionEvalBatchStorage",
    "ActionEvalDatasetError",
    "ActionEvalSampleStorage",
    "ActionEvalShardDatasetStorage",
    "IndexBuildProgressStorage",
    "ActionEvalTargetStorage",
    "DecisionTraceActionEvalDatasetStorage",
    "EvalOutcomeStorage",
    "ActionEvalLCGStorage",
    "ObjectiveConfigStorage",
    "STORAGE_VARIANTS",
    "STORAGE_DEFER_VARIANTS",
    "action_cost_storage",
    "collate_actioneval_storage",
    "derive_target_storage",
    "exact_internal_literal_mask",
    "signed_external_to_internal_index",
]


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate/index one plan-driven DecisionTrace-ActionEval-v1 split"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--variant", required=True, choices=STORAGE_VARIANTS)
    parser.add_argument("--expected_split", default=None)
    parser.add_argument(
        "--verify", choices=("none", "metadata", "checksum"), default="metadata"
    )
    parser.add_argument("--index_cache_dir", default=None)
    parser.add_argument("--max_states", type=int, default=None)
    parser.add_argument("--max_open_shards", type=int, default=2)
    parser.add_argument("--max_header_mb", type=int, default=1024)
    parser.add_argument("--max_array_mb", type=int, default=512)
    args = parser.parse_args()
    dataset = ActionEvalShardDatasetStorage(
        args.data_dir,
        variant=args.variant,
        expected_split=args.expected_split,
        verify_metadata=args.verify != "none",
        verify_checksums=args.verify == "checksum",
        index_cache_dir=args.index_cache_dir,
        max_states=args.max_states,
        max_open_shards=args.max_open_shards,
        max_header_bytes=args.max_header_mb * 1024 * 1024,
        max_array_bytes=args.max_array_mb * 1024 * 1024,
    )
    try:
        print(
            json.dumps(
                {
                    "data_dir": str(dataset.split_dir),
                    "variant": dataset.variant,
                    "verify": args.verify,
                    "split": dataset.manifest_split,
                    "states": len(dataset),
                    "instances": dataset.instance_count,
                    "parts": len(dataset._parts),
                    "manifest_fingerprint": dataset.manifest_fingerprint,
                    "policy_fingerprint": dataset.policy_fingerprint,
                    "split_fingerprint": dataset.split_fingerprint,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    finally:
        dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
