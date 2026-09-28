"""Metadata-only conversion of legacy DecisionTrace shard layouts."""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from .schema import SHARD_PLAN_SCHEMA, SHARD_POLICY_SCHEMA, canonical_json_bytes, utc_now_text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"legacy plan row {number} is not an object")
            rows.append(value)
    return rows


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite converted metadata: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_group(row: Mapping[str, Any]) -> str:
    declared = row.get("dataset_group")
    if isinstance(declared, str) and declared:
        return declared
    values = {
        item.get("dataset_group")
        for item in row.get("instances", [])
        if isinstance(item, Mapping)
    }
    if len(values) != 1 or not all(isinstance(value, str) and value for value in values):
        raise ValueError("legacy shard plan row does not have one dataset_group")
    return next(iter(values))


def _split(row: Mapping[str, Any]) -> str:
    declared = row.get("split")
    if isinstance(declared, str) and declared:
        return declared
    values = {
        item.get("split")
        for item in row.get("instances", [])
        if isinstance(item, Mapping)
    }
    if len(values) != 1 or not all(isinstance(value, str) and value for value in values):
        raise ValueError("legacy shard plan row does not have one split")
    return next(iter(values))


def convert_legacy(
    root: str | Path, *, overwrite: bool = False, dry_run: bool = False
) -> dict[str, Any]:
    output_root = Path(root).expanduser().resolve(strict=True)
    manifests = output_root / "manifests"
    policy_path = manifests / "pipeline.policy.json"
    plan_path = manifests / "pipeline.parts.jsonl"
    with policy_path.open("r", encoding="utf-8") as stream:
        raw_policy = json.load(stream)
    if not isinstance(raw_policy, Mapping):
        raise ValueError("legacy pipeline policy is not an object")
    legacy_rows = _read_jsonl(plan_path)
    if not legacy_rows:
        raise ValueError("legacy pipeline plan is empty")

    build_id = uuid.uuid4().hex
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(legacy_rows):
        shard_stem = row.get("part_stem")
        if not isinstance(shard_stem, str) or not shard_stem:
            raise ValueError("legacy plan row has no part_stem")
        complete = output_root / "shards" / f"{shard_stem}.complete.json"
        if not complete.is_file():
            raise FileNotFoundError(f"legacy shard completion marker is missing: {complete}")
        source_parts = row.get("source_parts")
        if not isinstance(source_parts, list) or not all(isinstance(item, str) for item in source_parts):
            source_parts = [shard_stem]
        merge = row.get("merge") if isinstance(row.get("merge"), Mapping) else {}
        parts_per_shard = merge.get("parts_per_shard", len(source_parts))
        instances = row.get("instances")
        if not isinstance(instances, list):
            raise ValueError(f"legacy plan row has no instances: {shard_stem}")
        instance_ids = [item.get("instance_id") for item in instances if isinstance(item, Mapping)]
        if len(instance_ids) != len(instances) or not all(isinstance(item, str) for item in instance_ids):
            raise ValueError(f"legacy plan row has invalid instances: {shard_stem}")
        converted.append(
            {
                "schema": SHARD_PLAN_SCHEMA,
                "shard_build_id": build_id,
                "shard_index": index,
                "shard_stem": shard_stem,
                "dataset_group": _dataset_group(row),
                "split": _split(row),
                "parts_per_shard": parts_per_shard,
                "source_part_count": len(source_parts),
                "source_parts": source_parts,
                "source_generations": [],
                "instance_count": len(instance_ids),
                "instance_ids": instance_ids,
            }
        )

    shard_policy = {
        "schema": SHARD_POLICY_SCHEMA,
        "shard_build_id": build_id,
        "created_at": utc_now_text(),
        "source_root": str(raw_policy.get("build_root", output_root)),
        "source_policy_id": hashlib.sha256(canonical_json_bytes(raw_policy)).hexdigest(),
        "source_tag": raw_policy.get("output_tag"),
        "output_root": str(output_root),
        "output_tag": output_root.name.rsplit("-", 1)[-1],
        "parts_per_shard": None,
        "source_part_count": sum(row["source_part_count"] for row in converted),
        "shard_count": len(converted),
        "payload_binding": "legacy-source-provenance",
    }
    policy_out = manifests / "shards" / "policy.json"
    plan_out = manifests / "shards" / "plan.jsonl"
    result = {
        "root": str(output_root),
        "policy": str(policy_out),
        "plan": str(plan_out),
        "shards": len(converted),
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    _atomic_write(policy_out, canonical_json_bytes(shard_policy) + b"\n", overwrite=overwrite)
    _atomic_write(
        plan_out,
        b"".join(canonical_json_bytes(row) + b"\n" for row in converted),
        overwrite=overwrite,
    )
    return result
