"""ActionEval SAT-ACT data types with a materialized-index bridge.

The expensive shard/index contract is intentionally reused from the storage layer.  Public
SAT-ACT samples and batches are independent, and supervision is flattened into one
backing tensor per dtype so pinning and host-to-device transfer stay coarse.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import torch
from torch_geometric.data import Batch

from satact.dataset.action_records import (
    ActionEvalDatasetError,
    ActionEvalLCGStorage,
    ActionEvalSampleStorage,
    ActionEvalTargetStorage,
    IndexBuildProgressStorage,
    ObjectiveConfigStorage,
    action_cost_storage,
    derive_target_storage,
    exact_internal_literal_mask,
    signed_external_to_internal_index,
)
from satact.dataset.compact_records import ActionEvalCompactShardDatasetStorage


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
IndexBuildProgress = IndexBuildProgressStorage


def to_storage_variant(variant: str) -> str:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unsupported SAT-ACT variant: {variant!r}")
    return variant


def from_storage_variant(variant: str) -> str:
    candidate = str(variant)
    if candidate not in MODEL_VARIANTS:
        raise ValueError(f"unsupported storage variant: {variant!r}")
    return candidate


@dataclass(frozen=True)
class ObjectiveConfig:
    tie_margin: float = 0.05
    listwise_temperature: float = 0.25
    censored_penalty: float = 1.0
    native_weight: float = 0.1
    cost_conflicts_weight: float = 1.0
    cost_propagations_weight: float = 0.1
    cost_decisions_weight: float = 0.05

    def __post_init__(self) -> None:
        ObjectiveConfigStorage(**asdict(self))

    def to_storage(self) -> ObjectiveConfigStorage:
        return ObjectiveConfigStorage(**asdict(self))


class ActionEvalLCG(ActionEvalLCGStorage):
    """Portable SAT-ACT graph; its tensor contract is identical to the storage representation."""

    def validate_actioneval(self) -> None:
        self.validate_actioneval_storage()


@dataclass(frozen=True)
class ActionEvalTarget:
    instance_id: str
    variant: str
    eligible_state: int
    record_id: str
    shard_stem: str
    n_literals: int
    eligibility_mask: torch.Tensor
    native_literal_index: int | None
    pair_better: torch.Tensor
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
        _target_to_storage(self).validate()


@dataclass(frozen=True)
class ActionEvalSample:
    graph: ActionEvalLCG
    supervision: ActionEvalTarget


@dataclass(frozen=True)
class TensorDescriptor:
    storage: str
    start: int
    length: int


_LONG_FIELDS: Final[tuple[str, ...]] = (
    "literal_offsets",
    "literal_state",
    "native_literal",
    "pair_better",
    "pair_worse",
    "pair_state",
    "listwise_literals",
    "listwise_state",
    "evaluated_literals",
    "evaluated_state",
)
_FLOAT_FIELDS: Final[tuple[str, ...]] = (
    "state_weights",
    "listwise_target",
    "evaluated_costs",
    "baseline_costs",
)
_BOOL_FIELDS: Final[tuple[str, ...]] = (
    "eligibility",
    "evaluated_complete",
    "baseline_complete",
)


@dataclass(frozen=True)
class FlatSupervision:
    long_storage: torch.Tensor
    float_storage: torch.Tensor
    bool_storage: torch.Tensor
    descriptors: Mapping[str, TensorDescriptor]
    batch_size: int

    def tensor(self, name: str) -> torch.Tensor:
        descriptor = self.descriptors[name]
        storage = getattr(self, descriptor.storage)
        return storage.narrow(0, descriptor.start, descriptor.length)

    @property
    def state_weights(self) -> torch.Tensor:
        return self.tensor("state_weights")

    @property
    def literal_offsets(self) -> torch.Tensor:
        return self.tensor("literal_offsets")

    def pin_memory(self) -> "FlatSupervision":
        return FlatSupervision(
            long_storage=self.long_storage.pin_memory(),
            float_storage=self.float_storage.pin_memory(),
            bool_storage=self.bool_storage.pin_memory(),
            descriptors=self.descriptors,
            batch_size=self.batch_size,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "FlatSupervision":
        return FlatSupervision(
            long_storage=self.long_storage.to(device, non_blocking=non_blocking),
            float_storage=self.float_storage.to(device, non_blocking=non_blocking),
            bool_storage=self.bool_storage.to(device, non_blocking=non_blocking),
            descriptors=self.descriptors,
            batch_size=self.batch_size,
        )


@dataclass(frozen=True)
class ActionEvalBatch:
    graph: Batch
    supervision: FlatSupervision
    variant: str

    @property
    def batch_size(self) -> int:
        return self.supervision.batch_size

    def pin_memory(self) -> "ActionEvalBatch":
        return ActionEvalBatch(
            graph=self.graph.pin_memory(),
            supervision=self.supervision.pin_memory(),
            variant=self.variant,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ActionEvalBatch":
        return ActionEvalBatch(
            graph=self.graph.to(device, non_blocking=non_blocking),
            supervision=self.supervision.to(device, non_blocking=non_blocking),
            variant=self.variant,
        )


def _target_from_storage(target: ActionEvalTargetStorage, variant: str) -> ActionEvalTarget:
    return ActionEvalTarget(
        instance_id=target.instance_id,
        variant=variant,
        eligible_state=target.eligible_state,
        record_id=target.record_id,
        shard_stem=target.shard_stem,
        n_literals=target.n_literals,
        eligibility_mask=target.eligibility_mask,
        native_literal_index=target.native_literal_index,
        pair_better=target.pair_better,
        pair_worse=target.pair_worse,
        listwise_literals=target.listwise_literals,
        listwise_target=target.listwise_target,
        evaluated_literals=target.evaluated_literals,
        evaluated_costs=target.evaluated_costs,
        evaluated_complete=target.evaluated_complete,
        baseline_cost=target.baseline_cost,
        baseline_complete=target.baseline_complete,
        instance_state_count=target.instance_state_count,
        instance_weight=target.instance_weight,
    )


def _target_to_storage(target: ActionEvalTarget) -> ActionEvalTargetStorage:
    return ActionEvalTargetStorage(
        instance_id=target.instance_id,
        variant=to_storage_variant(target.variant),
        eligible_state=target.eligible_state,
        record_id=target.record_id,
        shard_stem=target.shard_stem,
        n_literals=target.n_literals,
        eligibility_mask=target.eligibility_mask,
        native_literal_index=target.native_literal_index,
        pair_better=target.pair_better,
        pair_worse=target.pair_worse,
        listwise_literals=target.listwise_literals,
        listwise_target=target.listwise_target,
        evaluated_literals=target.evaluated_literals,
        evaluated_costs=target.evaluated_costs,
        evaluated_complete=target.evaluated_complete,
        baseline_cost=target.baseline_cost,
        baseline_complete=target.baseline_complete,
        instance_state_count=target.instance_state_count,
        instance_weight=target.instance_weight,
    )


def _graph_from_storage(sample: ActionEvalSampleStorage) -> ActionEvalLCG:
    return ActionEvalLCG(**sample.graph.to_dict())


def _pack_storage(
    fields: Mapping[str, torch.Tensor],
    names: Sequence[str],
    *,
    dtype: torch.dtype,
    storage_name: str,
) -> tuple[torch.Tensor, dict[str, TensorDescriptor]]:
    parts: list[torch.Tensor] = []
    descriptors: dict[str, TensorDescriptor] = {}
    offset = 0
    for name in names:
        values = fields[name].reshape(-1)
        if values.device.type != "cpu" or values.dtype != dtype:
            raise ActionEvalDatasetError(f"flat field {name} must be a CPU {dtype} tensor")
        parts.append(values)
        descriptors[name] = TensorDescriptor(storage_name, offset, values.numel())
        offset += values.numel()
    storage = torch.cat(parts) if parts else torch.empty(0, dtype=dtype)
    return storage, descriptors


def _global_actions(actions: torch.Tensor, literal_offset: int) -> torch.Tensor:
    return torch.where(actions < 0, actions, actions + literal_offset)


def _flatten_targets(targets: Sequence[ActionEvalTarget]) -> FlatSupervision:
    batch_size = len(targets)
    literal_counts = torch.tensor([target.n_literals for target in targets], dtype=torch.long)
    literal_offsets = torch.cat((torch.zeros(1, dtype=torch.long), literal_counts.cumsum(0)))

    literal_state: list[torch.Tensor] = []
    eligibility: list[torch.Tensor] = []
    native_literal: list[int] = []
    pair_better: list[torch.Tensor] = []
    pair_worse: list[torch.Tensor] = []
    pair_state: list[torch.Tensor] = []
    listwise_literals: list[torch.Tensor] = []
    listwise_target: list[torch.Tensor] = []
    listwise_state: list[torch.Tensor] = []
    evaluated_literals: list[torch.Tensor] = []
    evaluated_costs: list[torch.Tensor] = []
    evaluated_complete: list[torch.Tensor] = []
    evaluated_state: list[torch.Tensor] = []
    baseline_costs: list[float] = []
    baseline_complete: list[bool] = []

    for state, target in enumerate(targets):
        offset = int(literal_offsets[state])
        literal_state.append(torch.full((target.n_literals,), state, dtype=torch.long))
        eligibility.append(target.eligibility_mask)
        native_literal.append(
            -1 if target.native_literal_index is None else offset + target.native_literal_index
        )
        pair_better.append(_global_actions(target.pair_better, offset))
        pair_worse.append(_global_actions(target.pair_worse, offset))
        pair_state.append(torch.full((target.pair_better.numel(),), state, dtype=torch.long))
        listwise_literals.append(target.listwise_literals + offset)
        listwise_target.append(target.listwise_target)
        listwise_state.append(
            torch.full((target.listwise_literals.numel(),), state, dtype=torch.long)
        )
        evaluated_literals.append(target.evaluated_literals + offset)
        evaluated_costs.append(target.evaluated_costs)
        evaluated_complete.append(target.evaluated_complete)
        evaluated_state.append(
            torch.full((target.evaluated_literals.numel(),), state, dtype=torch.long)
        )
        baseline_costs.append(math.nan if target.baseline_cost is None else target.baseline_cost)
        baseline_complete.append(target.baseline_complete)

    def concatenate(parts: Sequence[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        return torch.cat(tuple(parts)) if parts else torch.empty(0, dtype=dtype)

    fields = {
        "literal_offsets": literal_offsets,
        "literal_state": concatenate(literal_state, torch.long),
        "native_literal": torch.tensor(native_literal, dtype=torch.long),
        "pair_better": concatenate(pair_better, torch.long),
        "pair_worse": concatenate(pair_worse, torch.long),
        "pair_state": concatenate(pair_state, torch.long),
        "listwise_literals": concatenate(listwise_literals, torch.long),
        "listwise_state": concatenate(listwise_state, torch.long),
        "evaluated_literals": concatenate(evaluated_literals, torch.long),
        "evaluated_state": concatenate(evaluated_state, torch.long),
        "state_weights": torch.tensor([target.instance_weight for target in targets]),
        "listwise_target": concatenate(listwise_target, torch.float32),
        "evaluated_costs": concatenate(evaluated_costs, torch.float32),
        "baseline_costs": torch.tensor(baseline_costs, dtype=torch.float32),
        "eligibility": concatenate(eligibility, torch.bool),
        "evaluated_complete": concatenate(evaluated_complete, torch.bool),
        "baseline_complete": torch.tensor(baseline_complete, dtype=torch.bool),
    }
    long_storage, long_descriptors = _pack_storage(
        fields, _LONG_FIELDS, dtype=torch.long, storage_name="long_storage"
    )
    float_storage, float_descriptors = _pack_storage(
        fields, _FLOAT_FIELDS, dtype=torch.float32, storage_name="float_storage"
    )
    bool_storage, bool_descriptors = _pack_storage(
        fields, _BOOL_FIELDS, dtype=torch.bool, storage_name="bool_storage"
    )
    return FlatSupervision(
        long_storage=long_storage,
        float_storage=float_storage,
        bool_storage=bool_storage,
        descriptors={**long_descriptors, **float_descriptors, **bool_descriptors},
        batch_size=batch_size,
    )


def collate_actioneval(samples: Sequence[ActionEvalSample]) -> ActionEvalBatch:
    if not samples:
        raise ValueError("cannot collate an empty SAT-ACT batch")
    variants = {sample.supervision.variant for sample in samples}
    if len(variants) != 1:
        raise ValueError("one SAT-ACT batch cannot mix objective variants")
    targets = tuple(sample.supervision for sample in samples)
    return ActionEvalBatch(
        graph=Batch.from_data_list([sample.graph for sample in samples]),
        supervision=_flatten_targets(targets),
        variant=next(iter(variants)),
    )


class ActionEvalShardDataset(torch.utils.data.Dataset[ActionEvalSample]):
    """SAT-ACT view over the established materialized-index/storage contract."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        variant: str,
        objective_config: ObjectiveConfig | None = None,
        progress_callback: Callable[[IndexBuildProgressStorage], None] | None = None,
        **kwargs: Any,
    ) -> None:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unsupported SAT-ACT variant: {variant!r}")
        self.variant = variant
        self.objective_config = objective_config or ObjectiveConfig()
        self._backend = ActionEvalCompactShardDatasetStorage(
            split_dir,
            variant=to_storage_variant(variant),
            objective_config=self.objective_config.to_storage(),
            progress_callback=progress_callback,
            **kwargs,
        )
        for name in (
            "split_dir",
            "plan_path",
            "policy_path",
            "part_count",
            "manifest_split",
            "manifest_fingerprint",
            "policy_fingerprint",
            "split_fingerprint",
            "index_cache_path",
            "index_cache_hit",
            "instance_count",
            "shard_ids",
        ):
            setattr(self, name, getattr(self._backend, name))

    def __len__(self) -> int:
        return len(self._backend)

    def __getitem__(self, index: int) -> ActionEvalSample:
        sample = self._backend[index]
        return ActionEvalSample(
            graph=_graph_from_storage(sample),
            supervision=_target_from_storage(sample.supervision, self.variant),
        )

    def close(self) -> None:
        self._backend.close()


ActionEvalCompactShardDataset = ActionEvalShardDataset
DecisionTraceActionEvalDataset = ActionEvalShardDataset


def derive_target(*, variant: str, config: ObjectiveConfig, **kwargs: Any) -> Mapping[str, Any] | None:
    return derive_target_storage(
        variant=to_storage_variant(variant),
        config=config.to_storage(),
        **kwargs,
    )


def action_cost(conflicts: int, propagations: int, decisions: int, config: ObjectiveConfig) -> float:
    return action_cost_storage(conflicts, propagations, decisions, config.to_storage())


__all__ = [
    "ActionEvalBatch",
    "ActionEvalCompactShardDataset",
    "ActionEvalDatasetError",
    "ActionEvalLCG",
    "ActionEvalSample",
    "ActionEvalShardDataset",
    "IndexBuildProgress",
    "ActionEvalTarget",
    "DecisionTraceActionEvalDataset",
    "FlatSupervision",
    "ObjectiveConfig",
    "DEFER_VARIANTS",
    "MODEL_VARIANTS",
    "action_cost",
    "collate_actioneval",
    "derive_target",
    "exact_internal_literal_mask",
    "signed_external_to_internal_index",
    "from_storage_variant",
    "to_storage_variant",
]
