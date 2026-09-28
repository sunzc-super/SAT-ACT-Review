"""Independent reader for the raw generator's on-disk contract.

The raw generator lives in the CaDiCaL project.  This package deliberately
does not import it: policy, plan, schema, completion markers and checksums are
validated from files before any shard is committed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .schema import canonical_json_bytes


PIPELINE_POLICY_SCHEMA = "decisiontrace-actioneval-pipeline-policy-v1"
PIPELINE_PLAN_SCHEMA = "decisiontrace-actioneval-pipeline-plan-v1"
PIPELINE_STATS_SCHEMA = "decisiontrace-actioneval-pipeline-part-stats-v1"
RAW_POLICY_SCHEMA_V2 = "decisiontrace-actioneval-raw-policy-v2"
RAW_PLAN_SCHEMA_V2 = "decisiontrace-actioneval-raw-part-plan-v2"
RAW_STATS_SCHEMA_V2 = "decisiontrace-actioneval-raw-part-complete-v2"
WORKER_RAW_SCHEMA = "decisiontrace-actioneval-worker-raw-schema-v1"

TARGET_POLICY_FIELDS = (
    "target_initial_keep_count",
    "target_early_window_end",
    "target_early_sample_count",
    "target_post_restart_keep_count",
    "target_late_fallback_window_end",
)
_REMOVED_TARGET_FIELDS = ("k1", "k2", "k3", "k4", "late_max")
_PART_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

RAW_RECORD_FIELDS: Mapping[str, tuple[str, ...]] = {
    "tr": (
        "instance", "callback", "raw_decision", "eligible_state", "level", "trail",
        "conflicts", "decisions", "propagations", "restarts", "candidates",
        "state_nodes_preflight", "prefix_hash", "eligibility_hash",
        "native_external", "native_internal",
    ),
    "sl": (
        "instance", "eligible_state", "callback", "raw_decision", "level", "trail",
        "conflicts", "decisions", "propagations", "restarts", "prefix_hash",
        "native_external", "assignment_hash", "trail_hash", "clause_canonical_hash",
        "clause_order_hash", "eligibility_hash", "state_nodes", "literal_occurrences",
        "max_clause_length", "estimated_payload_bytes",
    ),
    "st": (
        "instance", "eligible_state", "callback", "raw_decision", "level", "trail",
        "conflicts", "decisions", "propagations", "restarts", "prefix_hash",
        "native_external", "internal_variables", "external_variables", "n_clauses",
        "n_occurrences", "assignment_hash", "trail_hash", "clause_canonical_hash",
        "clause_order_hash", "eligibility_hash", "i2e", "eligibility_bits_hex",
        "assignment_value", "assignment_level", "assignment_source", "trail_position",
        "activity", "saved_phase", "clause_offsets", "clause_literals_internal",
    ),
    "pr": ("instance", "eligible_state", "actions"),
    "ev": (
        "instance", "eligible_state", "callback", "raw_decision", "requested_external",
        "target_reached", "locator_match", "action_eligible", "action_applied",
        "prefix_hash", "native_external", "applied_external", "status", "censored",
        "horizon_hit", "delta_conflicts", "delta_decisions", "delta_propagations",
        "delta_restarts",
    ),
    "out": (
        "instance", "mode", "status", "conflicts", "decisions", "propagations",
        "restarts", "callbacks", "eligible_states", "materialized",
        "materialization_failures",
    ),
    "sf": (
        "instance", "eligible_state", "reason", "state_nodes", "literal_occurrences",
        "max_clause_length", "estimated_payload_bytes",
    ),
    "ef": (
        "instance", "eligible_state", "requested_external", "reason",
        "timeout_seconds",
    ),
}
RAW_STAGE_TAGS: Mapping[str, tuple[str, ...]] = {
    "traces": ("tr", "out"),
    "states": ("sl", "st", "sf", "out"),
    "proposals": ("pr",),
    "evals": ("ev", "ef", "out"),
}
PROPOSAL_ACTION_FIELDS = (
    "external_literal",
    "source_bits",
    "activity",
    "jw",
    "random_key_hex",
)
PROPOSAL_SOURCE_BITS = {
    "native": 1,
    "native_opposite": 2,
    "activity_phase": 4,
    "activity_opposite": 8,
    "jw": 16,
    "random": 32,
}


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RawPolicy:
    path: Path
    build_root: Path
    raw_codec: str
    digest: str
    version: int
    record: Mapping[str, Any]


def load_raw_policy(path: str | Path) -> RawPolicy:
    """Load only fields needed by shard construction and hash the full policy."""

    policy_path = Path(path).expanduser().resolve(strict=True)
    with policy_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or record.get("schema") not in {
        PIPELINE_POLICY_SCHEMA,
        RAW_POLICY_SCHEMA_V2,
    }:
        raise ValueError("unexpected raw pipeline policy schema")
    obsolete = sorted(set(record).intersection(_REMOVED_TARGET_FIELDS))
    if obsolete:
        raise ValueError(
            "raw policy still uses removed target names: " + ",".join(obsolete)
        )
    missing = [name for name in TARGET_POLICY_FIELDS if name not in record]
    if missing:
        raise ValueError("raw policy is missing target fields: " + ",".join(missing))
    initial = _nonnegative_integer(
        record["target_initial_keep_count"], "target_initial_keep_count"
    )
    early_end = _nonnegative_integer(
        record["target_early_window_end"], "target_early_window_end"
    )
    early_count = _nonnegative_integer(
        record["target_early_sample_count"], "target_early_sample_count"
    )
    _nonnegative_integer(
        record["target_post_restart_keep_count"], "target_post_restart_keep_count"
    )
    late_end = _nonnegative_integer(
        record["target_late_fallback_window_end"],
        "target_late_fallback_window_end",
    )
    if initial > early_end or early_end > late_end:
        raise ValueError(
            "target bounds must satisfy initial_keep_count <= early_window_end "
            "<= late_fallback_window_end"
        )
    if early_count > early_end - initial:
        raise ValueError("target_early_sample_count exceeds its eligible window")
    build_root_text = record.get("build_root")
    if not isinstance(build_root_text, str) or not build_root_text:
        raise ValueError("raw policy build_root must be a non-empty path string")
    build_root = Path(build_root_text).expanduser().resolve(strict=False)
    if build_root.exists() and not build_root.is_dir():
        raise NotADirectoryError(f"raw policy build_root is not a directory: {build_root}")
    raw_codec = record.get("raw_codec")
    if raw_codec not in {"none", "gzip"}:
        raise ValueError("raw policy raw_codec must be 'none' or 'gzip'")
    version = 2 if record.get("schema") == RAW_POLICY_SCHEMA_V2 else 1
    if version == 2:
        digest = record.get("policy_id")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}", digest):
            raise ValueError("v2 raw policy has an invalid policy_id")
    else:
        # Legacy plans refer to this small metadata digest.  No payload is read.
        digest = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    return RawPolicy(
        path=policy_path,
        build_root=build_root,
        raw_codec=raw_codec,
        digest=digest,
        version=version,
        record=dict(record),
    )


def load_worker_raw_schema(build_root: str | Path) -> Mapping[str, Any]:
    schema_path = Path(build_root) / "manifests" / "worker.raw.schema.json"
    with schema_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    expected_fields = {name: list(fields) for name, fields in RAW_RECORD_FIELDS.items()}
    expected_tags = {name: list(tags) for name, tags in RAW_STAGE_TAGS.items()}
    if not isinstance(record, dict) or record.get("schema") != WORKER_RAW_SCHEMA:
        raise ValueError(f"unexpected worker raw schema: {schema_path}")
    if record.get("record_fields") != expected_fields:
        raise ValueError("worker raw record fields do not match the shard reader contract")
    if record.get("stage_tags") != expected_tags:
        raise ValueError("worker raw stage tags do not match the shard reader contract")
    if record.get("proposal_action_fields") != list(PROPOSAL_ACTION_FIELDS):
        raise ValueError("worker proposal action fields do not match the shard reader contract")
    if record.get("proposal_source_bits") != PROPOSAL_SOURCE_BITS:
        raise ValueError("worker proposal source bits do not match the shard reader contract")
    return record


def iter_raw_plan(
    path: str | Path, *, expected_policy_sha256: str
) -> Iterator[dict[str, Any]]:
    plan_path = Path(path).expanduser().resolve(strict=True)
    seen_stems: set[str] = set()
    with plan_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid raw plan JSON at line {line_number}") from exc
            record = validate_raw_plan_part(
                record,
                expected_policy_sha256=expected_policy_sha256,
                context=f"line {line_number}",
            )
            stem = record["part_stem"]
            if stem in seen_stems:
                raise ValueError(f"duplicate raw plan part_stem: {stem}")
            seen_stems.add(stem)
            yield record


def validate_raw_plan_part(
    record: Any, *, expected_policy_sha256: str, context: str = "record"
) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema") not in {
        PIPELINE_PLAN_SCHEMA,
        RAW_PLAN_SCHEMA_V2,
    }:
        raise ValueError(f"unexpected raw plan schema at {context}")
    policy_reference = (
        record.get("policy_id")
        if record.get("schema") == RAW_PLAN_SCHEMA_V2
        else record.get("policy_sha256")
    )
    if policy_reference != expected_policy_sha256:
        raise ValueError(f"raw plan policy digest mismatch at {context}")
    stem = record.get("part_stem")
    if not isinstance(stem, str) or not _PART_STEM_RE.fullmatch(stem):
        raise ValueError(f"unsafe raw plan part_stem at {context}")
    instances = record.get("instances")
    if (
        not isinstance(instances, list)
        or not instances
        or not all(isinstance(item, dict) for item in instances)
    ):
        raise ValueError(f"raw plan instances must be objects at {context}")
    instance_ids = [item.get("instance_id") for item in instances]
    if any(not isinstance(value, str) or not value for value in instance_ids):
        raise ValueError(f"raw plan contains an invalid instance_id at {context}")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"raw plan contains duplicate instance_id at {context}")
    return record


def raw_plan_part_digest(record: Mapping[str, Any]) -> str:
    """Return the legacy digest or the v2 lightweight part identity."""

    if not isinstance(record, Mapping):
        raise TypeError("raw plan part must be an object")
    if record.get("schema") == RAW_PLAN_SCHEMA_V2:
        stem = record.get("part_stem")
        if not isinstance(stem, str):
            raise ValueError("v2 raw plan part has no part_stem")
        return stem
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def raw_plan_part_split(
    record: Mapping[str, Any], *, context: str = "raw plan part"
) -> str:
    """Return the single split carried by every instance in one raw part."""

    instances = record.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"{context} has no instances")
    values = {item.get("split") for item in instances if isinstance(item, Mapping)}
    if len(values) != 1 or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{context} must contain exactly one non-empty split")
    split = next(iter(values))
    declared = record.get("split")
    if declared is not None and declared != split:
        raise ValueError(f"{context} split does not match its instances")
    return split


def raw_plan_part_dataset_group(
    record: Mapping[str, Any], *, context: str = "raw plan part"
) -> str:
    """Return and cross-check the source dataset group for one raw part."""

    instances = record.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"{context} has no instances")
    values = {
        item.get("dataset_group")
        for item in instances
        if isinstance(item, Mapping)
    }
    if len(values) != 1 or not all(isinstance(value, str) and value for value in values):
        raise ValueError(
            f"{context} must contain exactly one non-empty dataset_group"
        )
    dataset_group = next(iter(values))
    declared = record.get("dataset_group")
    if declared is not None and declared != dataset_group:
        raise ValueError(f"{context} dataset_group does not match its instances")
    return dataset_group


def find_raw_plan_part(
    path: str | Path, *, expected_policy_sha256: str, part_stem: str
) -> dict[str, Any]:
    found: dict[str, Any] | None = None
    for record in iter_raw_plan(path, expected_policy_sha256=expected_policy_sha256):
        if record["part_stem"] == part_stem:
            found = record
    if found is None:
        raise ValueError(f"part_stem not found in raw plan: {part_stem}")
    return found


def raw_suffix(codec: str) -> str:
    if codec == "gzip":
        return ".jsonl.gz"
    if codec == "none":
        return ".jsonl"
    raise ValueError(f"unsupported raw codec: {codec}")
