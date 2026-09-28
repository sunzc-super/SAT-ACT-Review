"""Compact index cache for the DecisionTrace action-record dataset.

The shard reader and target derivation remain owned by the storage dataset.
This module only replaces its JSON index serialization with a columnar torch
payload.  When no compact cache exists, an already generated JSON cache is
accepted without rewriting it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .action_records import (
    ActionEvalShardDatasetStorage,
    _IndexEntry,
    _legacy_index_config_storage,
)


COMPACT_INDEX_SCHEMA_STORAGE = "satact-index-compact"
COMPACT_EXPLICIT_INDEX_SCHEMA_STORAGE = "satact-index-compact-explicit"


def _flatten_int(
    entries: Sequence[_IndexEntry], name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for entry in entries:
        offsets.append(offsets[-1] + len(entry.target[name]))
    values = torch.empty(offsets[-1], dtype=torch.int32)
    for index, entry in enumerate(entries):
        begin, end = offsets[index], offsets[index + 1]
        if begin != end:
            values[begin:end] = torch.tensor(entry.target[name], dtype=torch.int32)
    return torch.tensor(offsets, dtype=torch.int64), values


def _flatten_float(
    entries: Sequence[_IndexEntry], name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for entry in entries:
        offsets.append(offsets[-1] + len(entry.target[name]))
    values = torch.empty(offsets[-1], dtype=torch.float32)
    for index, entry in enumerate(entries):
        begin, end = offsets[index], offsets[index + 1]
        if begin != end:
            values[begin:end] = torch.tensor(
                [
                    math.nan if value is None else float(value)
                    for value in entry.target[name]
                ],
                dtype=torch.float32,
            )
    return torch.tensor(offsets, dtype=torch.int64), values


def _flatten_bool(
    entries: Sequence[_IndexEntry], name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for entry in entries:
        offsets.append(offsets[-1] + len(entry.target[name]))
    values = torch.empty(offsets[-1], dtype=torch.bool)
    for index, entry in enumerate(entries):
        begin, end = offsets[index], offsets[index + 1]
        if begin != end:
            values[begin:end] = torch.tensor(entry.target[name], dtype=torch.bool)
    return torch.tensor(offsets, dtype=torch.int64), values


def _pack_entries(
    metadata: Mapping[str, Any],
    entries: Sequence[_IndexEntry],
    *,
    schema: str = COMPACT_INDEX_SCHEMA_STORAGE,
    metadata_key: str = "fingerprint",
) -> dict[str, Any]:
    instance_ids: list[str] = []
    instance_lookup: dict[str, int] = {}
    instance_codes: list[int] = []
    for entry in entries:
        code = instance_lookup.get(entry.instance_id)
        if code is None:
            code = len(instance_ids)
            instance_lookup[entry.instance_id] = code
            instance_ids.append(entry.instance_id)
        instance_codes.append(code)

    pair_offsets, pair_better = _flatten_int(entries, "pair_better")
    _, pair_worse = _flatten_int(entries, "pair_worse")
    list_offsets, listwise_literals = _flatten_int(entries, "listwise_literals")
    _, listwise_target = _flatten_float(entries, "listwise_target")
    evaluated_offsets, evaluated_literals = _flatten_int(entries, "evaluated_literals")
    _, evaluated_costs = _flatten_float(entries, "evaluated_costs")
    _, evaluated_complete = _flatten_bool(entries, "evaluated_complete")

    return {
        "schema": schema,
        metadata_key: dict(metadata),
        "record_ids": [entry.record_id for entry in entries],
        "instance_ids": instance_ids,
        "columns": {
            "shard_index": torch.tensor(
                [entry.shard_index for entry in entries], dtype=torch.int32
            ),
            "record_index": torch.tensor(
                [entry.record_index for entry in entries], dtype=torch.int32
            ),
            "instance_code": torch.tensor(instance_codes, dtype=torch.int32),
            "eligible_state": torch.tensor(
                [entry.eligible_state for entry in entries], dtype=torch.int64
            ),
            "n_literals": torch.tensor(
                [int(entry.target["n_literals"]) for entry in entries], dtype=torch.int32
            ),
            "native_literal_index": torch.tensor(
                [
                    -1
                    if entry.target["native_literal_index"] is None
                    else int(entry.target["native_literal_index"])
                    for entry in entries
                ],
                dtype=torch.int32,
            ),
            "pair_offsets": pair_offsets,
            "pair_better": pair_better,
            "pair_worse": pair_worse,
            "list_offsets": list_offsets,
            "listwise_literals": listwise_literals,
            "listwise_target": listwise_target,
            "evaluated_offsets": evaluated_offsets,
            "evaluated_literals": evaluated_literals,
            "evaluated_costs": evaluated_costs,
            "evaluated_complete": evaluated_complete,
            "baseline_cost": torch.tensor(
                [
                    math.nan
                    if entry.target["baseline_cost"] is None
                    else float(entry.target["baseline_cost"])
                    for entry in entries
                ],
                dtype=torch.float64,
            ),
            "baseline_complete": torch.tensor(
                [bool(entry.target["baseline_complete"]) for entry in entries],
                dtype=torch.bool,
            ),
        },
    }


def _slice(tensor: torch.Tensor, offsets: torch.Tensor, index: int) -> list[Any]:
    begin = int(offsets[index])
    end = int(offsets[index + 1])
    return tensor[begin:end].tolist()


def _unpack_entries(payload: Mapping[str, Any]) -> list[_IndexEntry]:
    columns = payload["columns"]
    record_ids = payload["record_ids"]
    instance_ids = payload["instance_ids"]
    count = len(record_ids)
    entries: list[_IndexEntry] = []
    for index in range(count):
        native = int(columns["native_literal_index"][index])
        baseline_cost = float(columns["baseline_cost"][index])
        target = {
            "n_literals": int(columns["n_literals"][index]),
            "native_literal_index": None if native < 0 else native,
            "pair_better": _slice(
                columns["pair_better"], columns["pair_offsets"], index
            ),
            "pair_worse": _slice(
                columns["pair_worse"], columns["pair_offsets"], index
            ),
            "listwise_literals": _slice(
                columns["listwise_literals"], columns["list_offsets"], index
            ),
            "listwise_target": _slice(
                columns["listwise_target"], columns["list_offsets"], index
            ),
            "evaluated_literals": _slice(
                columns["evaluated_literals"], columns["evaluated_offsets"], index
            ),
            "evaluated_costs": [
                None if math.isnan(float(value)) else float(value)
                for value in _slice(
                    columns["evaluated_costs"], columns["evaluated_offsets"], index
                )
            ],
            "evaluated_complete": _slice(
                columns["evaluated_complete"], columns["evaluated_offsets"], index
            ),
            "baseline_cost": None if math.isnan(baseline_cost) else baseline_cost,
            "baseline_complete": bool(columns["baseline_complete"][index]),
        }
        instance_code = int(columns["instance_code"][index])
        entries.append(
            _IndexEntry(
                shard_index=int(columns["shard_index"][index]),
                record_index=int(columns["record_index"][index]),
                record_id=str(record_ids[index]),
                instance_id=str(instance_ids[instance_code]),
                eligible_state=int(columns["eligible_state"][index]),
                target=target,
            )
        )
    return entries


class ActionEvalCompactShardDatasetStorage(ActionEvalShardDatasetStorage):
    """storage shard dataset using compact caches with legacy JSON fallback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        path = self.index_cache_path
        if path is not None and self.index_cache_hit and not path.is_file():
            legacy_path = path.with_suffix(".json")
            if legacy_path.is_file():
                self.index_cache_path = legacy_path

    def _cache_path(self, root: str | Path | None) -> Path | None:
        legacy_path = super()._cache_path(root)
        return None if legacy_path is None else legacy_path.with_suffix(".pt")

    @staticmethod
    def _load_explicit_cached_index(
        path: Path, index_config: Mapping[str, Any]
    ) -> list[_IndexEntry] | None:
        if not path.is_file():
            return None
        payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        if (
            payload.get("schema") == COMPACT_EXPLICIT_INDEX_SCHEMA_STORAGE
            and payload.get("index_config") == index_config
        ):
            return _unpack_entries(payload)
        if payload.get("schema") == COMPACT_INDEX_SCHEMA_STORAGE:
            try:
                legacy_config = _legacy_index_config_storage(payload["fingerprint"])
            except (KeyError, TypeError, ValueError):
                return None
            if legacy_config == index_config:
                return _unpack_entries(payload)
        return None

    @staticmethod
    def _write_explicit_cached_index(
        path: Path,
        index_config: Mapping[str, Any],
        entries: Sequence[_IndexEntry],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                _pack_entries(
                    index_config,
                    entries,
                    schema=COMPACT_EXPLICIT_INDEX_SCHEMA_STORAGE,
                    metadata_key="index_config",
                ),
                temporary,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _load_cached_index(
        path: Path | None, fingerprint: Mapping[str, Any]
    ) -> list[_IndexEntry] | None:
        if path is None:
            return None
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("schema") == COMPACT_INDEX_SCHEMA_STORAGE
                and payload.get("fingerprint") == fingerprint
            ):
                return _unpack_entries(payload)
        return ActionEvalShardDatasetStorage._load_cached_index(
            path.with_suffix(".json"), fingerprint
        )

    @staticmethod
    def _write_cached_index(
        path: Path | None,
        fingerprint: Mapping[str, Any],
        entries: Sequence[_IndexEntry],
    ) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(_pack_entries(fingerprint, entries), temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


__all__ = [
    "ActionEvalCompactShardDatasetStorage",
    "COMPACT_EXPLICIT_INDEX_SCHEMA_STORAGE",
    "COMPACT_INDEX_SCHEMA_STORAGE",
]
