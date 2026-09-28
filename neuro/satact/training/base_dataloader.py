"""Deterministic, shard-local DataLoader helpers for DecisionTrace."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from satact.dataset.action_graph import ObjectiveConfig, collate_actioneval
from satact.dataset.compact_action_graph import (
    ActionEvalCompactShardDataset as ActionEvalShardDataset,
)


class EpochIndexSampler(Sampler[int]):
    """Sequential evaluation or deterministic per-epoch training permutation."""

    def __init__(self, size: int, *, training: bool, seed: int) -> None:
        self.size = int(size)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        indices = list(range(self.size))
        if self.training:
            random.Random(self.seed + self.epoch * 1_000_003).shuffle(indices)
        yield from indices


class RankPartitionSampler(Sampler[int]):
    """Partition a deterministic base stream without importing unrelated samplers."""

    def __init__(
        self,
        base: Sampler[int],
        *,
        rank: int,
        world_size: int,
        pad: bool,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank/world_size")
        self.base = base
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.pad = bool(pad)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return math.ceil(len(self.base) / self.world_size) if self.pad else (
            len(self.base) + self.world_size - 1 - self.rank
        ) // self.world_size

    def __iter__(self) -> Iterator[int]:
        values = list(self.base)
        if self.pad and values:
            total = math.ceil(len(values) / self.world_size) * self.world_size
            values.extend(values[index % len(values)] for index in range(total - len(values)))
        yield from values[self.rank :: self.world_size]


class ShardLocalitySampler(Sampler[int]):
    """Regroup bounded permutation windows by shard to reduce random IO."""

    def __init__(
        self,
        base: Sampler[int],
        shard_ids: Sequence[int],
        *,
        window_size: int,
        seed: int,
    ) -> None:
        if window_size < 1:
            raise ValueError("shard shuffle window must be positive")
        self.base = base
        self.shard_ids = tuple(int(value) for value in shard_ids)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.base)

    def __iter__(self) -> Iterator[int]:
        iterator = iter(self.base)
        for window_index in itertools.count():
            window = list(itertools.islice(iterator, self.window_size))
            if not window:
                return
            grouped: dict[int, list[int]] = defaultdict(list)
            for index in window:
                grouped[self.shard_ids[index]].append(index)
            order = list(grouped)
            random.Random(self.seed + self.epoch * 1_000_003 + window_index).shuffle(order)
            for shard_id in order:
                yield from grouped[shard_id]


def create_actioneval_dataloader(
    dataset: ActionEvalShardDataset,
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
    base: Sampler[int] = EpochIndexSampler(len(dataset), training=training, seed=seed)
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "collate_fn": collate_actioneval,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last and training,
        "generator": generator,
    }
    if num_workers > 0:
        options["persistent_workers"] = bool(persistent_workers and training)
        if prefetch_factor is not None:
            options["prefetch_factor"] = prefetch_factor
    options.update(kwargs)
    return DataLoader(**options)


def build_actioneval_loader(
    data_dir: str | Path,
    *,
    variant: str,
    objective_config: ObjectiveConfig | None = None,
    batch_size: int,
    training: bool,
    expected_split: str | None = None,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    verify: bool = True,
    audit_checksum: bool = False,
    index_cache_dir: str | Path | None = None,
    index_cache_path: str | Path | None = None,
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
) -> tuple[ActionEvalShardDataset, DataLoader]:
    """Build a split directly from its final DecisionTrace version directory.

    ``verify`` controls the eager shard-metadata pass. ``audit_checksum`` adds
    the full payload audit and therefore also enables metadata verification.
    """

    if not isinstance(verify, bool) or not isinstance(audit_checksum, bool):
        raise TypeError("verify and audit_checksum must be booleans")
    dataset = ActionEvalShardDataset(
        data_dir,
        variant=variant,
        objective_config=objective_config,
        expected_split=expected_split,
        verify_metadata=verify,
        verify_checksums=audit_checksum,
        index_cache_dir=index_cache_dir,
        index_cache_path=index_cache_path,
        index_cache_read_only=index_cache_read_only,
        max_states=max_states,
        max_open_shards=max_open_shards,
        max_header_bytes=max_header_bytes,
        max_array_bytes=max_array_bytes,
    )
    loader = create_actioneval_dataloader(
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
    "EpochIndexSampler",
    "RankPartitionSampler",
    "ShardLocalitySampler",
    "build_actioneval_loader",
    "create_actioneval_dataloader",
]
