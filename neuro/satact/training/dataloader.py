"""DataLoader construction for the independent ActionEval SAT-ACT stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from satact.dataset.action_preferences import (
    ActionEvalShardDatasetSATACT,
    SATACT_FILTERS,
    collate_actioneval_satact,
)
from satact.dataset.heuristic_supervision import (
    ActionEvalShardDatasetSATACTMultiNative,
    SATACTIndexContract,
    SATACTNativeOutcome,
    validate_satact_index_selection,
)
from satact.training.base_dataloader import (
    EpochIndexSampler,
    RankPartitionSampler,
    ShardLocalitySampler,
)


def create_actioneval_satact_dataloader(
    dataset: ActionEvalShardDatasetSATACT,
    *,
    batch_size: int,
    training: bool,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    shard_shuffle: bool = True,
    shard_shuffle_window: int = 8192,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    drop_last: bool = False,
    **kwargs: Any,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base: Sampler[int] = EpochIndexSampler(
        len(dataset), training=training, seed=seed
    )
    if distributed_world_size > 1:
        base = RankPartitionSampler(
            base,
            rank=distributed_rank,
            world_size=distributed_world_size,
            pad=training,
        )
    sampler: Sampler[int] = base
    if training and shard_shuffle:
        sampler = ShardLocalitySampler(
            base,
            dataset.shard_ids,
            window_size=shard_shuffle_window,
            seed=seed,
        )
    generator = torch.Generator().manual_seed(seed)
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "collate_fn": collate_actioneval_satact,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": bool(drop_last and training),
        "generator": generator,
    }
    if num_workers > 0:
        options["persistent_workers"] = bool(persistent_workers and training)
        if prefetch_factor is not None:
            options["prefetch_factor"] = prefetch_factor
    options.update(kwargs)
    return DataLoader(**options)


def build_actioneval_satact_loader(
    data_dir: str | Path,
    *,
    index_cache_path: str | Path,
    batch_size: int,
    training: bool,
    expected_split: str | None = None,
    pair_filter: str = "all",
    metric_mode: str = "loss",
    include_top_set_supervision: bool = False,
    satact_index_contract: SATACTIndexContract = "preference-only",
    satact_native_outcome: SATACTNativeOutcome = "replay",
    preprocess_workers: int = 1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    verify: bool = True,
    audit_checksum: bool = False,
    index_cache_read_only: bool = False,
    max_states: int | None = None,
    max_open_shards: int = 2,
    max_header_bytes: int = 1024 * 1024 * 1024,
    max_array_bytes: int = 512 * 1024 * 1024,
    shard_shuffle: bool = True,
    shard_shuffle_window: int = 8192,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    drop_last: bool = False,
    **kwargs: Any,
) -> tuple[ActionEvalShardDatasetSATACT, DataLoader]:
    if pair_filter not in SATACT_FILTERS:
        raise ValueError(f"unsupported SATACT pair filter: {pair_filter!r}")
    if metric_mode not in {"loss", "selection", "diagnostics", "full"}:
        raise ValueError(f"unsupported SATACT metric mode: {metric_mode!r}")
    contract, native_outcome = validate_satact_index_selection(
        satact_index_contract, satact_native_outcome
    )
    dataset_class = (
        ActionEvalShardDatasetSATACT
        if contract == "preference-only"
        else ActionEvalShardDatasetSATACTMultiNative
    )
    dataset_options: dict[str, Any] = {}
    if contract == "heuristic-supervision":
        dataset_options["native_outcome"] = native_outcome
    dataset = dataset_class(
        data_dir,
        index_cache_path=index_cache_path,
        pair_filter=pair_filter,
        include_diagnostics=metric_mode in {"diagnostics", "full"},
        include_rule_metrics=metric_mode == "full",
        include_top_set_supervision=(
            include_top_set_supervision
            or metric_mode in {"selection", "diagnostics", "full"}
        ),
        preprocess_workers=preprocess_workers,
        expected_split=expected_split,
        verify_metadata=verify,
        verify_checksums=audit_checksum,
        index_cache_read_only=index_cache_read_only,
        max_states=max_states,
        max_open_shards=max_open_shards,
        max_header_bytes=max_header_bytes,
        max_array_bytes=max_array_bytes,
        **dataset_options,
    )
    loader = create_actioneval_satact_dataloader(
        dataset,
        batch_size=batch_size,
        training=training,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        shard_shuffle=shard_shuffle,
        shard_shuffle_window=shard_shuffle_window,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        drop_last=drop_last,
        **kwargs,
    )
    return dataset, loader


__all__ = [
    "build_actioneval_satact_loader",
    "create_actioneval_satact_dataloader",
]
