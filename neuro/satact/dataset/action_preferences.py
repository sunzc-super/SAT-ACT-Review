"""Independent data contract for the ActionEval SAT-ACT objective.

SATACT keeps the portable graph/model feature contract, but owns its pair rules and
materialized index. Existing unrelated indexes are neither accepted nor changed.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import IntFlag
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence

import torch
from torch_geometric.data import Batch

from satact.dataset.action_records import (
    ActionEvalDatasetError,
    ObjectiveConfigStorage as ObjectiveConfig,
    _IndexEntry,
    exact_internal_literal_mask,
    signed_external_to_internal_index,
)
from satact.dataset.compact_records import ActionEvalCompactShardDatasetStorage
from satact.dataset.action_graph import ActionEvalLCG


SATACT_VARIANT: Final[str] = "satact"
SATACT_WIRE_VARIANT: Final[str] = "satact"
SATACT_INDEX_SCHEMA: Final[str] = "decisiontrace-satact-preference-index"
SATACT_DIAGNOSTICS_SCHEMA: Final[str] = "decisiontrace-actioneval-satact-diagnostics"
HEURISTIC_SUPERVISION_MODES: Final[tuple[str, ...]] = ("always", "top-set")
SATACT_NATIVE_SUPERVISION = HEURISTIC_SUPERVISION_MODES
NativeSupervisionSATACT = Literal["always", "top-set"]
SATACT_FILTERS: Final[tuple[str, ...]] = (
    "all",
    "r1",
    "r2",
    "r3",
    "r1-r2",
    "r1-r3",
    "r2-r3",
    "r1-only",
    "r2-only",
    "r3-only",
    "r2-r3-overlap",
)


def canonicalize_satact_variant(value: str) -> str:
    if value in {SATACT_VARIANT, "satact"}:
        return SATACT_VARIANT
    raise ValueError(f"unsupported SATACT variant: {value!r}")


class SATACTPairRule(IntFlag):
    R1_COMPLETE_SOLVED = 1
    R2_FULL_TRAJECTORY_PARETO = 2
    R3_TERMINAL_CLOSURE = 4


@dataclass(frozen=True)
class ObjectiveConfigSATACT:
    native_weight: float = 0.0
    native_supervision: NativeSupervisionSATACT = "always"
    top_set_weight: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.native_weight) or self.native_weight < 0:
            raise ValueError("native_weight must be finite and non-negative")
        if self.native_supervision not in HEURISTIC_SUPERVISION_MODES:
            raise ValueError(
                "heuristic supervision mode must be one of "
                f"{HEURISTIC_SUPERVISION_MODES!r}"
            )
        if not math.isfinite(self.top_set_weight) or self.top_set_weight < 0:
            raise ValueError("top_set_weight must be finite and non-negative")


@dataclass(frozen=True)
class _SATACTOutcome:
    requested: int
    status: int
    conflicts: int
    propagations: int
    decisions: int
    restarts: int
    reliable: bool
    complete: bool


def _sat_terminal_level(outcome: _SATACTOutcome) -> int:
    if outcome.conflicts > 0 or outcome.restarts > 0:
        return 0
    return 2 if outcome.decisions == 1 else 1


def _unsat_terminal_level(outcome: _SATACTOutcome) -> int:
    return 1 if outcome.decisions == 1 else 0


def _terminal_level(outcome: _SATACTOutcome) -> int:
    if outcome.status == 10:
        return _sat_terminal_level(outcome)
    if outcome.status == 20:
        return _unsat_terminal_level(outcome)
    raise ActionEvalDatasetError("terminal closure requires a complete SAT/UNSAT action")


def _pareto_order(left: _SATACTOutcome, right: _SATACTOutcome) -> int:
    left_work = (
        left.conflicts,
        left.propagations,
        left.decisions,
        left.restarts,
    )
    right_work = (
        right.conflicts,
        right.propagations,
        right.decisions,
        right.restarts,
    )
    if left_work != right_work and all(a <= b for a, b in zip(left_work, right_work)):
        return -1
    if left_work != right_work and all(b <= a for a, b in zip(left_work, right_work)):
        return 1
    return 0


def compare_satact_actions(left: _SATACTOutcome, right: _SATACTOutcome) -> tuple[int, int]:
    """Return (order, rule mask); -1 means left wins and +1 right wins."""

    if not left.reliable or not right.reliable:
        return 0, 0
    if left.complete != right.complete:
        return (-1 if left.complete else 1), int(SATACTPairRule.R1_COMPLETE_SOLVED)
    if not left.complete:
        return 0, 0

    order = 0
    mask = SATACTPairRule(0)
    pareto = _pareto_order(left, right)
    if pareto:
        order = pareto
        mask |= SATACTPairRule.R2_FULL_TRAJECTORY_PARETO

    left_level = _terminal_level(left)
    right_level = _terminal_level(right)
    terminal = -1 if left_level > right_level else 1 if right_level > left_level else 0
    if terminal:
        if order == 0:
            order = terminal
        elif order != terminal:
            raise ActionEvalDatasetError("SATACT Pareto and terminal-closure directions disagree")
        mask |= SATACTPairRule.R3_TERMINAL_CLOSURE
    return order, int(mask)


_EVAL_ARRAYS: Final[tuple[str, ...]] = (
    "eval_requested_literal",
    "eval_target_reached",
    "eval_locator_match",
    "eval_action_eligible",
    "eval_action_applied",
    "eval_applied_literal",
    "eval_native_literal",
    "eval_status",
    "eval_censored",
    "eval_horizon_hit",
    "eval_delta_conflicts",
    "eval_delta_decisions",
    "eval_delta_propagations",
    "eval_delta_restarts",
)


def derive_target_satact(
    *,
    state: Mapping[str, Any],
    i2e: Sequence[int],
    eligibility_bits: Sequence[int],
    arrays: Mapping[str, Sequence[int]],
) -> dict[str, Any] | None:
    lengths = {len(arrays[name]) for name in _EVAL_ARRAYS}
    if len(lengths) != 1:
        raise ActionEvalDatasetError("SATACT rollout array lengths disagree")
    native_external = int(state.get("native_external", 0))
    eligibility = exact_internal_literal_mask(i2e, eligibility_bits)
    native_index = signed_external_to_internal_index(native_external, i2e)
    if native_index is not None and not bool(eligibility[native_index]):
        native_index = None

    forced: list[tuple[_SATACTOutcome, int]] = []
    seen_literals: set[int] = set()
    solved_statuses: set[int] = set()
    rows = zip(*(arrays[name] for name in _EVAL_ARRAYS))
    for values in rows:
        (
            requested,
            reached,
            locator,
            eligible,
            applied,
            applied_literal,
            eval_native,
            status,
            censored,
            horizon,
            conflicts,
            decisions,
            propagations,
            restarts,
        ) = (int(value) for value in values)
        native_matches = eval_native == native_external
        applied_correct = bool(applied) and native_matches and (
            (requested == 0 and applied_literal == eval_native)
            or (requested != 0 and applied_literal == requested)
        )
        reliable = bool(reached and locator and eligible and applied_correct)
        complete = bool(reliable and not censored and not horizon and status in {10, 20})
        if requested == 0 or not reliable:
            continue
        literal_index = signed_external_to_internal_index(requested, i2e)
        if literal_index is None or not bool(eligibility[literal_index]):
            raise ActionEvalDatasetError("reliable SATACT rollout lies outside exact eligibility")
        if literal_index in seen_literals:
            raise ActionEvalDatasetError("duplicate SATACT forced evaluation literal")
        seen_literals.add(literal_index)
        if complete:
            solved_statuses.add(status)
        forced.append(
            (
                _SATACTOutcome(
                    requested=requested,
                    status=status,
                    conflicts=conflicts,
                    propagations=propagations,
                    decisions=decisions,
                    restarts=restarts,
                    reliable=reliable,
                    complete=complete,
                ),
                literal_index,
            )
        )
    if len(solved_statuses) > 1:
        raise ActionEvalDatasetError("SAT/UNSAT rollout inconsistency within one SATACT state")

    better: list[int] = []
    worse: list[int] = []
    rule_mask: list[int] = []
    for left_index in range(len(forced)):
        for right_index in range(left_index + 1, len(forced)):
            order, mask = compare_satact_actions(forced[left_index][0], forced[right_index][0])
            if order < 0:
                better.append(forced[left_index][1])
                worse.append(forced[right_index][1])
                rule_mask.append(mask)
            elif order > 0:
                better.append(forced[right_index][1])
                worse.append(forced[left_index][1])
                rule_mask.append(mask)
    if not better:
        return None

    evaluated = sorted(forced, key=lambda item: item[1])
    return {
        "n_literals": 2 * len(i2e),
        "native_literal_index": native_index,
        "pair_better": better,
        "pair_worse": worse,
        "pair_rule_mask": rule_mask,
        "evaluated_literals": [literal for _, literal in evaluated],
        "evaluated_complete": [outcome.complete for outcome, _ in evaluated],
        "evaluated_conflicts": [outcome.conflicts for outcome, _ in evaluated],
        "evaluated_propagations": [outcome.propagations for outcome, _ in evaluated],
        "evaluated_decisions": [outcome.decisions for outcome, _ in evaluated],
        "evaluated_restarts": [outcome.restarts for outcome, _ in evaluated],
        # Fields consumed by the shared graph materializer only.
        "listwise_literals": [],
        "listwise_target": [],
        "evaluated_costs": [None for _ in evaluated],
        "baseline_cost": None,
        "baseline_complete": False,
    }


def _flatten(
    entries: Sequence[_IndexEntry], name: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for entry in entries:
        offsets.append(offsets[-1] + len(entry.target[name]))
    values = torch.empty(offsets[-1], dtype=dtype)
    for index, entry in enumerate(entries):
        begin, end = offsets[index], offsets[index + 1]
        if begin != end:
            values[begin:end] = torch.tensor(entry.target[name], dtype=dtype)
    return torch.tensor(offsets, dtype=torch.int64), values


def _entry_identity(entries: Sequence[_IndexEntry]) -> tuple[list[str], list[str], list[int]]:
    instance_ids: list[str] = []
    lookup: dict[str, int] = {}
    codes: list[int] = []
    for entry in entries:
        code = lookup.get(entry.instance_id)
        if code is None:
            code = len(instance_ids)
            lookup[entry.instance_id] = code
            instance_ids.append(entry.instance_id)
        codes.append(code)
    return [entry.record_id for entry in entries], instance_ids, codes


def _pack_satact_index(entries: Sequence[_IndexEntry], config: Mapping[str, Any]) -> dict[str, Any]:
    record_ids, instance_ids, instance_codes = _entry_identity(entries)
    pair_offsets, pair_better = _flatten(entries, "pair_better", torch.int32)
    _, pair_worse = _flatten(entries, "pair_worse", torch.int32)
    _, pair_rule_mask = _flatten(entries, "pair_rule_mask", torch.uint8)
    evaluated_offsets, evaluated_literals = _flatten(entries, "evaluated_literals", torch.int32)
    _, evaluated_complete = _flatten(entries, "evaluated_complete", torch.bool)
    return {
        "schema": SATACT_INDEX_SCHEMA,
        "index_config": dict(config),
        "record_ids": record_ids,
        "instance_ids": instance_ids,
        "columns": {
            "shard_index": torch.tensor([entry.shard_index for entry in entries], dtype=torch.int32),
            "record_index": torch.tensor([entry.record_index for entry in entries], dtype=torch.int32),
            "instance_code": torch.tensor(instance_codes, dtype=torch.int32),
            "eligible_state": torch.tensor([entry.eligible_state for entry in entries], dtype=torch.int64),
            "n_literals": torch.tensor([entry.target["n_literals"] for entry in entries], dtype=torch.int32),
            "native_literal_index": torch.tensor([
                -1 if entry.target["native_literal_index"] is None else entry.target["native_literal_index"]
                for entry in entries
            ], dtype=torch.int32),
            "pair_offsets": pair_offsets,
            "pair_better": pair_better,
            "pair_worse": pair_worse,
            "pair_rule_mask": pair_rule_mask,
            "evaluated_offsets": evaluated_offsets,
            "evaluated_literals": evaluated_literals,
            "evaluated_complete": evaluated_complete,
        },
    }


def _pack_satact_diagnostics(entries: Sequence[_IndexEntry], config: Mapping[str, Any]) -> dict[str, Any]:
    # Preserve the raw rollout counters' int64 range (SATCOMP can exceed int32).
    offsets, conflicts = _flatten(entries, "evaluated_conflicts", torch.int64)
    _, propagations = _flatten(entries, "evaluated_propagations", torch.int64)
    _, decisions = _flatten(entries, "evaluated_decisions", torch.int64)
    _, restarts = _flatten(entries, "evaluated_restarts", torch.int64)
    return {
        "schema": SATACT_DIAGNOSTICS_SCHEMA,
        "index_config": dict(config),
        "record_ids": [entry.record_id for entry in entries],
        "columns": {
            "evaluated_offsets": offsets,
            "evaluated_conflicts": conflicts,
            "evaluated_propagations": propagations,
            "evaluated_decisions": decisions,
            "evaluated_restarts": restarts,
        },
    }


def _slice(values: torch.Tensor, offsets: torch.Tensor, index: int) -> list[Any]:
    begin, end = int(offsets[index]), int(offsets[index + 1])
    return values[begin:end].tolist()


def _unpack_satact_index(payload: Mapping[str, Any]) -> list[_IndexEntry]:
    columns = payload["columns"]
    record_ids = payload["record_ids"]
    instance_ids = payload["instance_ids"]
    entries: list[_IndexEntry] = []
    for index, record_id in enumerate(record_ids):
        native = int(columns["native_literal_index"][index])
        target = {
            "n_literals": int(columns["n_literals"][index]),
            "native_literal_index": None if native < 0 else native,
            "pair_better": _slice(columns["pair_better"], columns["pair_offsets"], index),
            "pair_worse": _slice(columns["pair_worse"], columns["pair_offsets"], index),
            "pair_rule_mask": _slice(columns["pair_rule_mask"], columns["pair_offsets"], index),
            "evaluated_literals": _slice(columns["evaluated_literals"], columns["evaluated_offsets"], index),
            "evaluated_complete": _slice(columns["evaluated_complete"], columns["evaluated_offsets"], index),
            "listwise_literals": [],
            "listwise_target": [],
            "evaluated_costs": [None] * (
                int(columns["evaluated_offsets"][index + 1])
                - int(columns["evaluated_offsets"][index])
            ),
            "baseline_cost": None,
            "baseline_complete": False,
        }
        entries.append(_IndexEntry(
            shard_index=int(columns["shard_index"][index]),
            record_index=int(columns["record_index"][index]),
            record_id=str(record_id),
            instance_id=str(instance_ids[int(columns["instance_code"][index])]),
            eligible_state=int(columns["eligible_state"][index]),
            target=target,
        ))
    return entries


def _diagnostics_path(index_path: Path) -> Path:
    name = index_path.name
    if name.endswith(".index.pt"):
        return index_path.with_name(name[:-9] + ".diagnostics.pt")
    return index_path.with_name(index_path.stem + ".diagnostics.pt")


def _attach_diagnostics(entries: Sequence[_IndexEntry], payload: Mapping[str, Any]) -> list[_IndexEntry]:
    columns = payload["columns"]
    if list(payload["record_ids"]) != [entry.record_id for entry in entries]:
        raise ActionEvalDatasetError("SATACT diagnostics record order does not match the core index")
    result: list[_IndexEntry] = []
    for index, entry in enumerate(entries):
        updates = {
            name: _slice(columns[name], columns["evaluated_offsets"], index)
            for name in (
                "evaluated_conflicts",
                "evaluated_propagations",
                "evaluated_decisions",
                "evaluated_restarts",
            )
        }
        result.append(replace(entry, target={**entry.target, **updates}))
    return result


def _filter_matches(mask: int, preset: str) -> bool:
    r1, r2, r3 = 1, 2, 4
    if preset == "all":
        return True
    inclusive = {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r1-r2": r1 | r2,
        "r1-r3": r1 | r3,
        "r2-r3": r2 | r3,
    }
    exact = {
        "r1-only": r1,
        "r2-only": r2,
        "r3-only": r3,
        "r2-r3-overlap": r2 | r3,
    }
    if preset in inclusive:
        return bool(mask & inclusive[preset])
    if preset in exact:
        return mask == exact[preset]
    raise ValueError(f"unsupported SATACT pair filter: {preset!r}")


def _part_worker(task: tuple[Any, ...]) -> tuple[int, list[_IndexEntry]]:
    (
        shard_index,
        stem,
        split_dir,
        reader_kind,
        verify,
        max_header_bytes,
        max_array_bytes,
    ) = task
    if reader_kind == "v2":
        from gen_data.rollout_dataset.compact import CompactArrayReader
    else:
        from gen_data.rollout_dataset.compact import CompactArrayReader
    complete = Path(split_dir) / "shards" / f"{stem}.complete.json"
    entries: list[_IndexEntry] = []
    with CompactArrayReader(
        complete,
        verify=bool(verify),
        max_header_bytes=int(max_header_bytes),
        max_array_bytes=int(max_array_bytes),
    ) as reader:
        for record_index, record in enumerate(reader.header["records"]):
            metadata = record.get("metadata", {})
            if metadata.get("kind") == "capture_failure":
                continue
            if metadata.get("kind") != "state":
                raise ActionEvalDatasetError("unknown compact record kind")
            state = metadata.get("state")
            names = ("i2e", "eligibility_bits", *_EVAL_ARRAYS)
            read_many = getattr(reader, "read_arrays", None)
            if read_many is not None:
                loaded = read_many(record, names)
                values = {name: list(loaded[name]) for name in names}
            else:
                values = {name: list(reader.read_array(record, name)) for name in names}
            target = derive_target_satact(
                state=state,
                i2e=values["i2e"],
                eligibility_bits=values["eligibility_bits"],
                arrays={name: values[name] for name in _EVAL_ARRAYS},
            )
            if target is None:
                continue
            entries.append(_IndexEntry(
                shard_index=int(shard_index),
                record_index=record_index,
                record_id=str(record.get("record_id")),
                instance_id=str(state["instance"]),
                eligible_state=int(state["eligible_state"]),
                target=target,
            ))
    return int(shard_index), entries


@dataclass(frozen=True)
class ActionEvalTargetSATACT:
    instance_id: str
    eligible_state: int
    n_literals: int
    eligibility_mask: torch.Tensor
    native_literal_index: int | None
    pair_better: torch.Tensor
    pair_worse: torch.Tensor
    pair_rule_mask: torch.Tensor
    evaluated_literals: torch.Tensor
    evaluated_complete: torch.Tensor
    evaluated_conflicts: torch.Tensor
    evaluated_propagations: torch.Tensor
    evaluated_decisions: torch.Tensor
    evaluated_restarts: torch.Tensor
    instance_weight: float


@dataclass(frozen=True)
class ActionEvalSampleSATACT:
    graph: ActionEvalLCG
    supervision: ActionEvalTargetSATACT


@dataclass(frozen=True)
class FlatSupervisionSATACT:
    fields: Mapping[str, torch.Tensor]
    batch_size: int

    def tensor(self, name: str) -> torch.Tensor:
        return self.fields[name]

    def has_tensor(self, name: str) -> bool:
        return name in self.fields

    @property
    def state_weights(self) -> torch.Tensor:
        return self.fields["state_weights"]

    def pin_memory(self) -> "FlatSupervisionSATACT":
        return FlatSupervisionSATACT(
            {name: value.pin_memory() for name, value in self.fields.items()},
            self.batch_size,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "FlatSupervisionSATACT":
        return FlatSupervisionSATACT(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.fields.items()
            },
            self.batch_size,
        )


@dataclass(frozen=True)
class ActionEvalBatchSATACT:
    graph: Batch
    supervision: FlatSupervisionSATACT
    variant: str = SATACT_VARIANT

    @property
    def batch_size(self) -> int:
        return self.supervision.batch_size

    def pin_memory(self) -> "ActionEvalBatchSATACT":
        return ActionEvalBatchSATACT(self.graph.pin_memory(), self.supervision.pin_memory())

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ActionEvalBatchSATACT":
        return ActionEvalBatchSATACT(
            self.graph.to(device, non_blocking=non_blocking),
            self.supervision.to(device, non_blocking=non_blocking),
        )


def collate_actioneval_satact(samples: Sequence[ActionEvalSampleSATACT]) -> ActionEvalBatchSATACT:
    if not samples:
        raise ValueError("cannot collate an empty SATACT batch")
    targets = [sample.supervision for sample in samples]
    counts = torch.tensor([target.n_literals for target in targets], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    long_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    bool_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    byte_parts: list[torch.Tensor] = []
    native: list[int] = []
    top_set_available: list[bool] = []
    native_evaluated: list[bool] = []
    native_in_top: list[bool] = []
    weights: list[float] = []
    for state_index, target in enumerate(targets):
        offset = int(offsets[state_index])
        long_parts["literal_state"].append(torch.full((target.n_literals,), state_index, dtype=torch.long))
        bool_parts["eligibility"].append(target.eligibility_mask)
        native.append(-1 if target.native_literal_index is None else offset + target.native_literal_index)
        long_parts["pair_better"].append(target.pair_better + offset)
        long_parts["pair_worse"].append(target.pair_worse + offset)
        long_parts["pair_state"].append(torch.full((target.pair_better.numel(),), state_index, dtype=torch.long))
        byte_parts.append(target.pair_rule_mask)
        long_parts["evaluated_literals"].append(target.evaluated_literals + offset)
        long_parts["evaluated_state"].append(torch.full((target.evaluated_literals.numel(),), state_index, dtype=torch.long))
        dominated = torch.zeros(target.n_literals, dtype=torch.bool)
        if target.pair_worse.numel():
            dominated[target.pair_worse] = True
        top_literals = target.evaluated_literals[
            ~dominated[target.evaluated_literals]
        ]
        top_set_available.append(bool(target.evaluated_literals.numel()))
        long_parts["top_literals"].append(top_literals + offset)
        long_parts["top_state"].append(
            torch.full((top_literals.numel(),), state_index, dtype=torch.long)
        )
        native_is_evaluated = bool(
            target.native_literal_index is not None
            and torch.any(target.evaluated_literals == target.native_literal_index)
        )
        native_evaluated.append(native_is_evaluated)
        native_in_top.append(
            bool(
                native_is_evaluated
                and torch.any(top_literals == target.native_literal_index)
            )
        )
        bool_parts["evaluated_complete"].append(target.evaluated_complete)
        for name in (
            "evaluated_conflicts",
            "evaluated_propagations",
            "evaluated_decisions",
            "evaluated_restarts",
        ):
            long_parts[name].append(getattr(target, name))
        weights.append(target.instance_weight)
    cat = lambda parts, dtype: torch.cat(parts) if parts else torch.empty(0, dtype=dtype)
    fields: dict[str, torch.Tensor] = {
        "literal_offsets": offsets,
        "native_literal": torch.tensor(native, dtype=torch.long),
        "top_set_available": torch.tensor(top_set_available, dtype=torch.bool),
        "native_evaluated": torch.tensor(native_evaluated, dtype=torch.bool),
        "native_in_top": torch.tensor(native_in_top, dtype=torch.bool),
        "state_weights": torch.tensor(weights, dtype=torch.float32),
        "pair_rule_mask": cat(byte_parts, torch.uint8),
    }
    fields.update({name: cat(parts, torch.long) for name, parts in long_parts.items()})
    fields.update({name: cat(parts, torch.bool) for name, parts in bool_parts.items()})
    return ActionEvalBatchSATACT(
        graph=Batch.from_data_list([sample.graph for sample in samples]),
        supervision=FlatSupervisionSATACT(fields, len(samples)),
    )


class ActionEvalShardDatasetSATACT(ActionEvalCompactShardDatasetStorage):
    """SATACT view over the plan-driven shard reader with an independent cache."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        index_cache_path: str | Path,
        pair_filter: str = "all",
        include_diagnostics: bool = False,
        include_rule_metrics: bool = False,
        include_top_set_supervision: bool = False,
        preprocess_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        if pair_filter not in SATACT_FILTERS:
            raise ValueError(f"unsupported SATACT pair filter: {pair_filter!r}")
        if preprocess_workers < 1:
            raise ValueError("preprocess_workers must be positive")
        self.pair_filter = pair_filter
        self.include_diagnostics = bool(include_diagnostics)
        self.include_rule_metrics = bool(include_rule_metrics)
        self.include_top_set_supervision = bool(include_top_set_supervision)
        self.preprocess_workers = int(preprocess_workers)
        self._satact_index_path = Path(index_cache_path).expanduser().absolute()
        super().__init__(
            split_dir,
            variant="preference-action",
            objective_config=ObjectiveConfig(),
            index_cache_path=self._satact_index_path,
            **kwargs,
        )
        filtered: list[_IndexEntry] = []
        for entry in self._entries:
            keep = [
                index
                for index, mask in enumerate(entry.target["pair_rule_mask"])
                if _filter_matches(int(mask), pair_filter)
            ]
            if not keep:
                continue
            target = dict(entry.target)
            for name in ("pair_better", "pair_worse", "pair_rule_mask"):
                target[name] = [target[name][index] for index in keep]
            filtered.append(replace(entry, target=target))
        counts: dict[str, int] = defaultdict(int)
        for entry in filtered:
            counts[entry.instance_id] += 1
        self._entries = [
            replace(entry, instance_state_count=counts[entry.instance_id])
            for entry in filtered
        ]
        self.instance_count = len(counts)
        self.shard_ids = [entry.shard_index for entry in self._entries]
        self.variant = SATACT_VARIANT
        self.split_fingerprint = f"{self.split_fingerprint}:satact-filter={pair_filter}"

    def _satact_config(self, max_states: int | None) -> dict[str, Any]:
        return {
            "schema": SATACT_INDEX_SCHEMA,
            "split_dir": str(self.split_dir),
            "manifest_split": self.manifest_split,
            "parts": list(self._parts),
            "max_states": max_states,
        }

    def _load_explicit_cached_index(
        self, path: Path, index_config: Mapping[str, Any]
    ) -> list[_IndexEntry] | None:
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        config = self._satact_config(index_config.get("max_states"))
        if payload.get("schema") != SATACT_INDEX_SCHEMA or payload.get("index_config") != config:
            return None
        entries = _unpack_satact_index(payload)
        if self.include_diagnostics:
            diagnostics_path = _diagnostics_path(path)
            if not diagnostics_path.is_file():
                return None
            diagnostics = torch.load(
                diagnostics_path, map_location="cpu", weights_only=False, mmap=True
            )
            if (
                diagnostics.get("schema") != SATACT_DIAGNOSTICS_SCHEMA
                or diagnostics.get("index_config") != config
            ):
                return None
            entries = _attach_diagnostics(entries, diagnostics)
        return entries

    def _write_explicit_cached_index(
        self,
        path: Path,
        index_config: Mapping[str, Any],
        entries: Sequence[_IndexEntry],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self._satact_config(index_config.get("max_states"))
        diagnostics_path = _diagnostics_path(path)
        for destination, payload in (
            (path, _pack_satact_index(entries, config)),
            (diagnostics_path, _pack_satact_diagnostics(entries, config)),
        ):
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _build_index(self, *, max_states: int | None) -> list[_IndexEntry]:
        reader_kind = (
            "v2"
            if self._reader_class.__module__.startswith(
                "gen_data.rollout_dataset"
            )
            else "v1"
        )
        tasks = [
            (
                shard_index,
                stem,
                str(self.split_dir),
                reader_kind,
                False,
                self.max_header_bytes,
                self.max_array_bytes,
            )
            for shard_index, stem in enumerate(self._parts)
        ]
        ordered: list[_IndexEntry] = []

        def append_part(shard_index: int, entries: Sequence[_IndexEntry]) -> bool:
            for entry in entries:
                if max_states is not None and len(ordered) >= max_states:
                    return True
                ordered.append(entry)
            self._report_progress(
                phase="index", shard_index=shard_index, states=len(ordered)
            )
            return max_states is not None and len(ordered) >= max_states

        if self.preprocess_workers == 1:
            for task in tasks:
                shard_index, entries = _part_worker(task)
                if append_part(shard_index, entries):
                    break
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.preprocess_workers
            ) as executor:
                pending: dict[concurrent.futures.Future[tuple[int, list[_IndexEntry]]], int] = {}
                ready: dict[int, list[_IndexEntry]] = {}
                next_task = 0
                next_part = 0
                limit_reached = False
                while next_task < len(tasks) or pending:
                    while (
                        next_task < len(tasks)
                        and len(pending) + len(ready) < 2 * self.preprocess_workers
                    ):
                        future = executor.submit(_part_worker, tasks[next_task])
                        pending[future] = next_task
                        next_task += 1
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        pending.pop(future)
                        shard_index, entries = future.result()
                        ready[shard_index] = entries
                    while next_part in ready:
                        entries = ready.pop(next_part)
                        if append_part(next_part, entries):
                            limit_reached = True
                            break
                        next_part += 1
                    if limit_reached:
                        for future in pending:
                            future.cancel()
                        break
        return ordered

    def __getitem__(self, index: int) -> ActionEvalSampleSATACT:
        base = super().__getitem__(index)
        entry = self._entries[index]
        target = entry.target
        count = entry.instance_state_count
        evaluated_literals = (
            target["evaluated_literals"]
            if self.include_diagnostics or self.include_top_set_supervision
            else []
        )
        evaluated_complete = (
            target["evaluated_complete"] if self.include_diagnostics else []
        )
        empty_resources: list[int] = []
        supervision = ActionEvalTargetSATACT(
            instance_id=entry.instance_id,
            eligible_state=entry.eligible_state,
            n_literals=base.supervision.n_literals,
            eligibility_mask=base.supervision.eligibility_mask,
            native_literal_index=base.supervision.native_literal_index,
            pair_better=torch.tensor(target["pair_better"], dtype=torch.long),
            pair_worse=torch.tensor(target["pair_worse"], dtype=torch.long),
            pair_rule_mask=torch.tensor(
                target["pair_rule_mask"] if self.include_rule_metrics else [],
                dtype=torch.uint8,
            ),
            evaluated_literals=torch.tensor(evaluated_literals, dtype=torch.long),
            evaluated_complete=torch.tensor(evaluated_complete, dtype=torch.bool),
            evaluated_conflicts=torch.tensor(target.get("evaluated_conflicts", empty_resources), dtype=torch.long),
            evaluated_propagations=torch.tensor(target.get("evaluated_propagations", empty_resources), dtype=torch.long),
            evaluated_decisions=torch.tensor(target.get("evaluated_decisions", empty_resources), dtype=torch.long),
            evaluated_restarts=torch.tensor(target.get("evaluated_restarts", empty_resources), dtype=torch.long),
            instance_weight=1.0 / count,
        )
        graph = ActionEvalLCG(**base.graph.to_dict())
        return ActionEvalSampleSATACT(graph=graph, supervision=supervision)


__all__ = [
    "ActionEvalBatchSATACT",
    "ActionEvalSampleSATACT",
    "ActionEvalShardDatasetSATACT",
    "ActionEvalTargetSATACT",
    "SATACTPairRule",
    "SATACT_FILTERS",
    "HEURISTIC_SUPERVISION_MODES",
    "SATACT_NATIVE_SUPERVISION",
    "SATACT_VARIANT",
    "SATACT_WIRE_VARIANT",
    "NativeSupervisionSATACT",
    "ObjectiveConfigSATACT",
    "canonicalize_satact_variant",
    "collate_actioneval_satact",
    "compare_satact_actions",
    "derive_target_satact",
]
