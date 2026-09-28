"""Bounded-memory offline runner for DecisionTrace-ActionEval-v1.3.

This module deliberately has no directory-discovery code.  It consumes the
already-built instance manifest, invokes one explicit C++ worker path per
instance, and assembles flat part files.  A capture ``st`` record is copied
from the worker pipe to the part writer without calling ``json.loads`` on it;
the preceding compact ``sl`` record supplies every replay locator needed by
the EVAL subprocesses.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

from .manifest import read_manifest_stream
from .schema import MANIFEST_SCHEMA, canonical_json_bytes, utc_now_text, validate_literal


PIPELINE_POLICY_SCHEMA = "decisiontrace-actioneval-raw-policy-v2"
PIPELINE_PLAN_SCHEMA = "decisiontrace-actioneval-raw-part-plan-v2"
PIPELINE_STATS_SCHEMA = "decisiontrace-actioneval-raw-part-complete-v2"
WORKER_RAW_SCHEMA = "decisiontrace-actioneval-worker-raw-schema-v1"

# Stored once per build by ``write_worker_raw_schema``.  In particular, ``sl``
# is a first-class states-stage record rather than an implementation-only log:
# it is the compact replay locator that lets the runner avoid parsing ``st``.
WORKER_RAW_RECORD_FIELDS: Mapping[str, tuple[str, ...]] = {
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
WORKER_RAW_STAGE_TAGS: Mapping[str, tuple[str, ...]] = {
    "traces": ("tr", "out"),
    "states": ("sl", "st", "sf", "out"),
    "proposals": ("pr",),
    "evals": ("ev", "ef", "out"),
}

_SOLVER_OPTION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{1,16}$")
_PART_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RAW_CODECS = {"none", "gzip"}
_SMALL_RECORD_LIMIT = 16 * 1024 * 1024
_STDERR_LIMIT = 1024 * 1024
_READ_CHUNK = 64 * 1024


class PipelineError(RuntimeError):
    """Base class for pipeline failures."""


class WorkerError(PipelineError):
    """One C++ worker exited abnormally or produced malformed output."""


class WorkerTimeout(WorkerError):
    """One C++ worker exceeded the configured wall-clock limit."""


class PipelinePartError(PipelineError):
    """A part was not committed; its attempt log contains the cause."""

    def __init__(
        self,
        message: str,
        *,
        log_path: Path | None = None,
        statistics_path: Path | None = None,
    ) -> None:
        # Optional defaults are required for BaseException's normal pickle
        # reconstruction in ProcessPoolExecutor.  The exception __dict__ then
        # restores the two diagnostic paths in the parent process.
        super().__init__(message)
        self.log_path = log_path
        self.statistics_path = statistics_path


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class PipelinePolicy:
    """All knobs that affect worker execution or raw part representation.

    A zero C++ budget means "disabled", matching the worker CLI.  The process
    timeout is always enabled, so even a policy with disabled solver budgets
    remains externally bounded.
    """

    policy_id: str
    output_tag: str
    binary: Path
    build_root: Path
    manifest: Path
    cnf_cache_root: Path | None = None

    target_initial_keep_count: int = 2
    target_early_window_end: int = 8
    target_early_sample_count: int = 2
    target_post_restart_keep_count: int = 2
    target_late_fallback_window_end: int = 20

    max_state_nodes: int = 100_000
    max_literal_occurrences: int = 4_000_000
    max_snapshot_bytes: int = 256 * 1024 * 1024
    max_clause_length: int = 0

    max_prefix_conflicts: int = 10_000
    max_prefix_decisions: int = 0
    max_prefix_propagations: int = 10_000_000
    max_prefix_callbacks: int = 1_000

    activity_top_k: int = 2
    jw_top_k: int = 1
    random_top_k: int = 1
    max_actions: int = 8
    max_actions_mode: str = "default"

    eval_max_conflicts: int = 0
    eval_max_decisions: int = 0
    eval_max_propagations: int = 0
    eval_max_timeout: float = 0.0

    seed: int = 1
    plain: bool = False
    solver_settings: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    timeout_seconds: float = 300.0
    raw_codec: str = "gzip"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not re.fullmatch(r"[0-9a-f]{32}", self.policy_id):
            raise ValueError("policy_id must be a 32-character lowercase UUID hex string")
        if not isinstance(self.output_tag, str) or not self.output_tag:
            raise ValueError("output_tag must be a non-empty string")
        binary = Path(self.binary).expanduser().resolve(strict=False)
        build_root = Path(self.build_root).expanduser().resolve(strict=False)
        manifest = Path(self.manifest).expanduser().resolve(strict=False)
        cnf_cache_root = (
            Path(self.cnf_cache_root).expanduser().resolve(strict=False)
            if self.cnf_cache_root is not None
            else None
        )
        object.__setattr__(self, "binary", binary)
        object.__setattr__(self, "build_root", build_root)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "cnf_cache_root", cnf_cache_root)

        for name in (
            "target_initial_keep_count",
            "target_early_window_end",
            "target_early_sample_count",
            "target_post_restart_keep_count",
            "target_late_fallback_window_end",
            "max_state_nodes",
            "max_literal_occurrences",
            "max_snapshot_bytes",
            "max_clause_length",
            "max_prefix_conflicts",
            "max_prefix_decisions",
            "max_prefix_propagations",
            "max_prefix_callbacks",
            "activity_top_k",
            "jw_top_k",
            "random_top_k",
            "max_actions",
            "eval_max_conflicts",
            "eval_max_decisions",
            "eval_max_propagations",
            "seed",
        ):
            _integer(getattr(self, name), name)
        if not (
            self.target_initial_keep_count
            <= self.target_early_window_end
            <= self.target_late_fallback_window_end
        ):
            raise ValueError(
                "target bounds must satisfy target_initial_keep_count <= "
                "target_early_window_end <= target_late_fallback_window_end"
            )
        early_window_size = (
            self.target_early_window_end - self.target_initial_keep_count
        )
        if self.target_early_sample_count > early_window_size:
            raise ValueError(
                "target_early_sample_count cannot exceed the number of states "
                "between target_initial_keep_count and target_early_window_end"
            )
        if self.seed >= 1 << 64:
            raise ValueError("seed must fit uint64")
        if self.max_actions_mode not in {"default", "variable"}:
            raise ValueError("max_actions_mode must be 'default' or 'variable'")
        if not isinstance(self.plain, bool):
            raise ValueError("plain must be a boolean")
        if (
            isinstance(self.eval_max_timeout, bool)
            or not isinstance(self.eval_max_timeout, (int, float))
            or not math.isfinite(float(self.eval_max_timeout))
            or self.eval_max_timeout < 0
        ):
            raise ValueError("eval_max_timeout must be a finite non-negative number")
        object.__setattr__(self, "eval_max_timeout", float(self.eval_max_timeout))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if self.raw_codec not in _RAW_CODECS:
            raise ValueError(f"raw_codec must be one of {sorted(_RAW_CODECS)}")

        normalized_settings: list[tuple[str, int]] = []
        seen: set[str] = set()
        for item in self.solver_settings:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("solver_settings entries must be (name, integer) pairs")
            name, value = item
            if not isinstance(name, str) or not _SOLVER_OPTION_RE.fullmatch(name):
                raise ValueError(f"invalid solver option name: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate solver option: {name}")
            seen.add(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"solver option {name!r} must have an integer value")
            if not -(1 << 31) <= value < (1 << 31):
                raise ValueError(f"solver option {name!r} does not fit int32")
            if name == "quiet" and value != 1:
                raise ValueError("DecisionTrace raw JSON requires solver option quiet=1")
            normalized_settings.append((name, value))
        object.__setattr__(self, "solver_settings", tuple(normalized_settings))

    def check_paths(self) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"worker binary is missing or not executable: {self.binary}")
        if not self.manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {self.manifest}")
        if self.build_root.exists() and not self.build_root.is_dir():
            raise NotADirectoryError(f"build_root is not a directory: {self.build_root}")

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema"] = PIPELINE_POLICY_SCHEMA
        record["binary"] = str(self.binary)
        record["build_root"] = str(self.build_root)
        record["manifest"] = str(self.manifest)
        record["cnf_cache_root"] = (
            str(self.cnf_cache_root) if self.cnf_cache_root is not None else None
        )
        record["solver_settings"] = [list(item) for item in self.solver_settings]
        return record

def pipeline_policy_from_record(record: Mapping[str, Any]) -> PipelinePolicy:
    """Validate a JSON-compatible policy object and construct the policy."""

    if not isinstance(record, Mapping) or record.get("schema") != PIPELINE_POLICY_SCHEMA:
        raise ValueError("unexpected pipeline policy schema")
    expected = set(PipelinePolicy.__dataclass_fields__)
    supplied = set(record) - {"schema"}
    missing = sorted(
        expected
        - supplied
        - {"cnf_cache_root", "eval_max_timeout", "plain", "max_actions_mode"}
    )
    unknown = sorted(supplied - expected)
    if missing or unknown:
        raise ValueError(
            "pipeline policy fields mismatch "
            f"(missing={','.join(missing) or '-'}; unknown={','.join(unknown) or '-'})"
        )
    values = {name: record[name] for name in expected if name in record}
    values.setdefault("eval_max_timeout", 0.0)
    values.setdefault("plain", False)
    values.setdefault("max_actions_mode", "default")
    values.setdefault("cnf_cache_root", None)
    values["solver_settings"] = tuple(tuple(item) for item in values["solver_settings"])
    return PipelinePolicy(**values)


def write_pipeline_policy(
    path: str | Path, policy: PipelinePolicy, *, overwrite: bool = False
) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    _atomic_json(output, policy.as_record(), overwrite=overwrite)
    return output


def load_pipeline_policy(path: str | Path, *, check_paths: bool = True) -> PipelinePolicy:
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    policy = pipeline_policy_from_record(record)
    if check_paths:
        policy.check_paths()
    return policy


def worker_raw_schema_record() -> dict[str, Any]:
    return {
        "schema": WORKER_RAW_SCHEMA,
        "record_fields": {
            tag: list(fields) for tag, fields in WORKER_RAW_RECORD_FIELDS.items()
        },
        "stage_tags": {
            stage: list(tags) for stage, tags in WORKER_RAW_STAGE_TAGS.items()
        },
        "proposal_action_fields": [
            "external_literal",
            "source_bits",
            "activity",
            "jw",
            "random_key_hex",
        ],
        "proposal_source_bits": {
            "native": 1,
            "native_opposite": 2,
            "activity_phase": 4,
            "activity_opposite": 8,
            "jw": 16,
            "random": 32,
        },
    }


def write_worker_raw_schema(
    path: str | Path, *, overwrite: bool = False
) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    _atomic_json(output, worker_raw_schema_record(), overwrite=overwrite)
    return output


@dataclass(frozen=True)
class PipelinePlanResult:
    path: Path
    part_count: int
    instance_count: int
    skipped_count: int


@dataclass(frozen=True)
class PipelinePartResult:
    part_stem: str
    output_files: Mapping[str, Mapping[str, Any]]
    statistics_path: Path
    log_path: Path
    instances_completed: int
    states_materialized: int
    evaluations: int


def _group_slug(dataset_group: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", dataset_group).strip("-").lower()[:48]
    digest = hashlib.sha256(dataset_group.encode("utf-8")).hexdigest()[:8]
    return f"{readable or 'group'}-{digest}"


def _part_stem(dataset_group: str, part_id: int) -> str:
    return f"{_group_slug(dataset_group)}.part-{part_id:05d}"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_pipeline_plan(
    policy: PipelinePolicy,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> PipelinePlanResult:
    """Stream a manifest exactly once and write one JSON record per part.

    The manifest must keep each ``(dataset_group, part_id)`` contiguous.  This
    is the natural order emitted by the manifest builder and permits memory to
    stay bounded by one manifest part rather than the whole dataset.
    """

    policy.check_paths()
    output = (
        Path(path).expanduser().resolve(strict=False)
        if path is not None
        else policy.build_root / "manifests" / "pipeline.parts.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite pipeline plan: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    part_count = instance_count = skipped_count = 0
    current_key: tuple[str, int] | None = None
    current_instances: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, int]] = set()

    def flush(stream: BinaryIO) -> None:
        nonlocal part_count, instance_count, current_instances
        if current_key is None or not current_instances:
            return
        dataset_group, part_id = current_key
        record = {
            "schema": PIPELINE_PLAN_SCHEMA,
            "policy_id": policy.policy_id,
            "dataset_group": dataset_group,
            "part_id": part_id,
            "part_stem": _part_stem(dataset_group, part_id),
            "instances": current_instances,
        }
        encoded = canonical_json_bytes(record) + b"\n"
        stream.write(encoded)
        part_count += 1
        instance_count += len(current_instances)
        current_instances = []

    try:
        with policy.manifest.open("rb") as manifest_stream, temporary.open("xb") as output_stream:
            for record in read_manifest_stream(
                manifest_stream, source=str(policy.manifest), accepted_only=False
            ):
                if not record.get("accepted", False):
                    skipped_count += 1
                    continue
                dataset_group = record.get("dataset_group")
                part_id = record.get("part_id")
                if not isinstance(dataset_group, str) or not dataset_group:
                    raise ValueError("accepted manifest record has no dataset_group")
                _integer(part_id, "manifest part_id")
                key = (dataset_group, part_id)
                if current_key != key:
                    if current_key is not None:
                        flush(output_stream)
                        completed_keys.add(current_key)
                    if key in completed_keys:
                        raise ValueError(
                            "manifest part records are not contiguous; refusing an unbounded regroup"
                        )
                    current_key = key
                if record.get("schema") != MANIFEST_SCHEMA:
                    raise ValueError("unexpected manifest record schema")
                if not isinstance(record.get("instance_id"), str):
                    raise ValueError("accepted manifest record has no instance_id")
                source_path = record.get("source_path")
                if not isinstance(source_path, str) or not Path(source_path).is_absolute():
                    raise ValueError("accepted manifest source_path must be absolute")
                current_instances.append(dict(record))
            flush(output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return PipelinePlanResult(
        path=output,
        part_count=part_count,
        instance_count=instance_count,
        skipped_count=skipped_count,
    )


def read_pipeline_plan(path: str | Path) -> Iterator[dict[str, Any]]:
    seen_stems: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid pipeline plan JSON at line {line_number}") from exc
            if not isinstance(record, dict) or record.get("schema") != PIPELINE_PLAN_SCHEMA:
                raise ValueError(f"unexpected pipeline plan schema at line {line_number}")
            stem = record.get("part_stem")
            if not isinstance(stem, str) or not _PART_STEM_RE.fullmatch(stem):
                raise ValueError(f"unsafe pipeline plan part_stem at line {line_number}")
            if stem in seen_stems:
                raise ValueError(f"duplicate pipeline plan part_stem: {stem}")
            seen_stems.add(stem)
            instances = record.get("instances")
            if not isinstance(instances, list) or not instances or not all(
                isinstance(item, dict) for item in instances
            ):
                raise ValueError(
                    f"pipeline plan instances must be a non-empty object array at line {line_number}"
                )
            instance_ids = [item.get("instance_id") for item in instances]
            if any(not isinstance(value, str) or not value for value in instance_ids):
                raise ValueError(
                    f"pipeline plan contains an invalid instance_id at line {line_number}"
                )
            if len(instance_ids) != len(set(instance_ids)):
                raise ValueError(
                    f"pipeline plan contains duplicate instance_id at line {line_number}"
                )
            yield record


def _trace_fields(record: Sequence[Any]) -> dict[str, Any]:
    if not isinstance(record, list) or len(record) != 17 or record[0] != "tr":
        raise ValueError("invalid tr record")
    return {
        "instance": record[1],
        "callback": record[2],
        "raw_decision": record[3],
        "eligible_state": record[4],
        "level": record[5],
        "trail": record[6],
        "conflicts": record[7],
        "decisions": record[8],
        "propagations": record[9],
        "restarts": record[10],
        "candidates": record[11],
        "state_nodes": record[12],
        "prefix_hash": record[13],
        "eligibility_hash": record[14],
        "native_external": record[15],
    }


def select_target_traces(
    trace_records: Iterable[Sequence[Any]], policy: PipelinePolicy
) -> list[list[Any]]:
    """Select initial, sampled-early, and post-restart target states."""

    normalized: list[tuple[list[Any], dict[str, Any]]] = []
    seen_indices: set[int] = set()
    instance: str | None = None
    for raw in trace_records:
        record = list(raw)
        fields = _trace_fields(record)
        if not isinstance(fields["eligible_state"], int) or fields["eligible_state"] <= 0:
            raise ValueError("tr eligible_state must be positive")
        if fields["eligible_state"] in seen_indices:
            raise ValueError("duplicate eligible_state in discover output")
        seen_indices.add(fields["eligible_state"])
        if instance is None:
            instance = str(fields["instance"])
        elif fields["instance"] != instance:
            raise ValueError("selector received records from multiple instances")
        normalized.append((record, fields))
    if not normalized:
        return []
    normalized.sort(key=lambda item: (item[1]["eligible_state"], item[1]["callback"]))

    selected: dict[int, list[Any]] = {}
    for record, fields in normalized:
        index = fields["eligible_state"]
        if index <= policy.target_initial_keep_count:
            selected[index] = record

    def hash_choose(candidates: list[tuple[list[Any], dict[str, Any]]], count: int, salt: str) -> None:
        ranked: list[tuple[bytes, int, list[Any]]] = []
        for record, fields in candidates:
            payload = (
                f"DecisionTrace-ActionEval-v1\0{policy.seed}\0{salt}\0"
                f"{fields['instance']}\0{fields['eligible_state']}"
            ).encode("utf-8")
            ranked.append((hashlib.sha256(payload).digest(), fields["eligible_state"], record))
        for _, index, record in sorted(ranked)[:count]:
            selected.setdefault(index, record)

    early = [
        item
        for item in normalized
        if policy.target_initial_keep_count
        < item[1]["eligible_state"]
        <= policy.target_early_window_end
    ]
    hash_choose(early, policy.target_early_sample_count, "early")

    # The post-restart quota counts newly selected states, not callbacks that
    # were already retained by the first/early blocks.  If the solve ends
    # before the quota is filled (including no restart), deterministically
    # Backfill from the late window without exceeding the sum of the three
    # target-count quotas.
    post_restart_added = 0
    if policy.target_post_restart_keep_count:
        for record, fields in normalized:
            index = fields["eligible_state"]
            if fields["restarts"] <= 0 or index in selected:
                continue
            selected[index] = record
            post_restart_added += 1
            if post_restart_added == policy.target_post_restart_keep_count:
                break
    missing_post_restart = policy.target_post_restart_keep_count - post_restart_added
    if missing_post_restart > 0:
        late = [
            item
            for item in normalized
            if policy.target_early_window_end < item[1]["eligible_state"] <= policy.target_late_fallback_window_end
            and item[1]["eligible_state"] not in selected
        ]
        hash_choose(late, missing_post_restart, "late-fallback")
    return [selected[index] for index in sorted(selected)]


def _validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 1..16 digit hexadecimal string")
    return value.lower()


def _locator_fields(locator: Sequence[Any]) -> dict[str, Any]:
    if not isinstance(locator, list) or len(locator) != 22 or locator[0] != "sl":
        raise ValueError("invalid sl locator record")
    names = (
        "instance",
        "eligible_state",
        "callback",
        "raw_decision",
        "level",
        "trail",
        "conflicts",
        "decisions",
        "propagations",
        "restarts",
        "prefix_hash",
        "native_external",
        "assignment_hash",
        "trail_hash",
        "clause_canonical_hash",
        "clause_order_hash",
        "eligibility_hash",
        "state_nodes",
        "literal_occurrences",
        "max_clause_length",
        "estimated_payload_bytes",
    )
    fields = dict(zip(names, locator[1:]))
    for name in (
        "prefix_hash",
        "assignment_hash",
        "trail_hash",
        "clause_canonical_hash",
        "clause_order_hash",
        "eligibility_hash",
    ):
        fields[name] = _validate_hash(fields[name], name)
    native = fields["native_external"]
    if isinstance(native, bool) or not isinstance(native, int):
        raise ValueError("sl native_external must be an integer")
    if native:
        validate_literal(native)
    return fields


def forced_evaluation_literals(
    locator: Sequence[Any], proposal: Sequence[Any]
) -> list[int]:
    """Return unique nonzero forced actions, including native when available."""

    fields = _locator_fields(locator)
    if (
        not isinstance(proposal, list)
        or len(proposal) != 4
        or proposal[0] != "pr"
        or proposal[1] != fields["instance"]
        or proposal[2] != fields["eligible_state"]
        or not isinstance(proposal[3], list)
    ):
        raise ValueError("proposal does not match locator")
    candidates: list[int] = []
    if fields["native_external"]:
        candidates.append(validate_literal(fields["native_external"]))
    for action in proposal[3]:
        if not isinstance(action, list) or len(action) != 5:
            raise ValueError("proposal action must contain five fields")
        candidates.append(validate_literal(action[0]))
    return list(dict.fromkeys(candidates))


def build_worker_argv(
    policy: PipelinePolicy,
    *,
    mode: str,
    instance_id: str,
    source_path: str | Path,
    targets: Sequence[int] | None = None,
    locator: Sequence[Any] | None = None,
    force_literal: int | None = None,
) -> list[str]:
    """Build an argv vector without a shell; EVAL uses every ``sl`` guard."""

    if mode not in {"discover", "capture", "eval"}:
        raise ValueError(f"unsupported worker mode: {mode}")
    if not isinstance(instance_id, str) or not instance_id or "\x00" in instance_id:
        raise ValueError("instance_id must be a non-empty NUL-free string")
    source = str(Path(source_path).expanduser().resolve(strict=False))
    argv = [
        str(policy.binary),
        f"--mode={mode}",
        "--output=-",
        f"--instance-id={instance_id}",
        "--no-schema",
        f"--max-state-nodes={policy.max_state_nodes}",
        f"--max-literal-occurrences={policy.max_literal_occurrences}",
        f"--max-snapshot-bytes={policy.max_snapshot_bytes}",
        f"--max-clause-length={policy.max_clause_length}",
        f"--max-prefix-conflicts={policy.max_prefix_conflicts}",
        f"--max-prefix-decisions={policy.max_prefix_decisions}",
        f"--max-prefix-propagations={policy.max_prefix_propagations}",
        f"--max-prefix-callbacks={policy.max_prefix_callbacks}",
        f"--seed={policy.seed}",
    ]
    if policy.plain:
        argv.append("--plain")
    if mode == "discover":
        argv.extend(
            (
                "--discover-eligible-window-end="
                f"{policy.target_late_fallback_window_end}",
                "--discover-post-restart-keep-count="
                f"{policy.target_post_restart_keep_count}",
            )
        )
    elif mode == "capture":
        normalized = sorted(set(targets or ()))
        if not normalized or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in normalized):
            raise ValueError("capture requires positive target indices")
        argv.extend(
            (
                "--targets=" + ",".join(str(value) for value in normalized),
                f"--activity-top-k={policy.activity_top_k}",
                f"--jw-top-k={policy.jw_top_k}",
                f"--random-top-k={policy.random_top_k}",
                f"--max-actions={policy.max_actions}",
                f"--max-actions-mode={policy.max_actions_mode}",
            )
        )
    else:
        fields = _locator_fields(locator if locator is not None else [])
        if force_literal is None or isinstance(force_literal, bool) or not isinstance(force_literal, int):
            raise ValueError("eval requires an integer force_literal (zero means native)")
        if force_literal:
            validate_literal(force_literal)
        argv.extend(
            (
                f"--target={fields['eligible_state']}",
                f"--force-literal={force_literal}",
                f"--expected-callback={fields['callback']}",
                f"--expected-raw-decision={fields['raw_decision']}",
                f"--expected-prefix-hash={fields['prefix_hash']}",
                f"--expected-assignment-hash={fields['assignment_hash']}",
                f"--expected-trail-hash={fields['trail_hash']}",
                f"--expected-clause-canonical-hash={fields['clause_canonical_hash']}",
                f"--expected-clause-order-hash={fields['clause_order_hash']}",
                f"--expected-eligibility-hash={fields['eligibility_hash']}",
                f"--expected-conflicts={fields['conflicts']}",
                f"--expected-decisions={fields['decisions']}",
                f"--expected-propagations={fields['propagations']}",
                f"--expected-restarts={fields['restarts']}",
                f"--expected-trail={fields['trail']}",
                f"--expected-native-literal={fields['native_external']}",
                f"--eval-max-conflicts={policy.eval_max_conflicts}",
                f"--eval-max-decisions={policy.eval_max_decisions}",
                f"--eval-max-propagations={policy.eval_max_propagations}",
                f"--eval-max-timeout={policy.eval_max_timeout:g}",
            )
        )
    argv.extend(f"--set={name}={value}" for name, value in policy.solver_settings)
    argv.append(source)
    return argv


class _CountingFile:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.bytes = 0

    def write(self, payload: bytes) -> int:
        written = self.stream.write(payload)
        if written:
            self.bytes += written
        return written

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()


class AtomicRawWriter:
    """Write one raw part and publish it only after a successful close."""

    def __init__(self, path: Path, codec: str, *, overwrite: bool = False) -> None:
        if codec not in _RAW_CODECS:
            raise ValueError(f"unsupported raw codec: {codec}")
        self.path = Path(path)
        self.codec = codec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite raw part: {self.path}")
        self.temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        self._raw = self.temporary.open("xb")
        self._counted = _CountingFile(self._raw)
        self._encoder: Any = (
            gzip.GzipFile(filename="", mode="wb", fileobj=self._counted, compresslevel=6, mtime=0)
            if codec == "gzip"
            else self._counted
        )
        self.raw_bytes = 0
        self.record_count = 0
        self.finished = False

    def write(self, payload: bytes) -> int:
        if self.finished:
            raise RuntimeError("raw writer is already finished")
        written = self._encoder.write(payload)
        self.raw_bytes += len(payload)
        return written

    def finish_record(self) -> None:
        self.record_count += 1

    def write_json(self, value: Any) -> None:
        self.write(canonical_json_bytes(value) + b"\n")
        self.finish_record()

    def close(self) -> dict[str, Any]:
        if self.finished:
            raise RuntimeError("raw writer is already finished")
        try:
            if self.codec == "gzip":
                self._encoder.close()
            self._counted.flush()
            os.fsync(self._counted.fileno())
            self._raw.close()
            os.replace(self.temporary, self.path)
            _fsync_directory(self.path.parent)
        except BaseException:
            if not self._raw.closed:
                self._raw.close()
            if self.temporary.exists():
                self.temporary.unlink()
            self.finished = True
            raise
        self.finished = True
        return {
            "path": str(self.path),
            "codec": self.codec,
            "bytes": self._counted.bytes,
            "raw_bytes": self.raw_bytes,
            "records": self.record_count,
        }

    def abort(self) -> None:
        if self.finished:
            return
        try:
            if self.codec == "gzip":
                self._encoder.close()
        finally:
            if not self._raw.closed:
                self._raw.close()
            if self.temporary.exists():
                self.temporary.unlink()
            self.finished = True


class _DiscardRecordSink:
    """A sink for already-parsed small records.

    EVAL output is first validated in memory and only then appended to the raw
    part.  Thus a killed EVAL process cannot leave half a JSON line in an
    otherwise useful part.
    """

    def write(self, payload: bytes) -> int:
        return len(payload)

    def finish_record(self) -> None:
        pass


class JsonlDemultiplexer:
    """Route worker JSONL by tag while never materializing unparsed records."""

    def __init__(
        self,
        sinks: Mapping[str, Any],
        *,
        parse_tags: Iterable[str],
        on_record: Callable[[str, Any | None], None] | None = None,
        json_loader: Callable[[bytes], Any] = json.loads,
        small_record_limit: int = _SMALL_RECORD_LIMIT,
    ) -> None:
        if not sinks:
            raise ValueError("demultiplexer requires at least one tag sink")
        self.sinks = dict(sinks)
        self.prefixes = {f'["{tag}",'.encode("ascii"): tag for tag in self.sinks}
        self.max_prefix = max(map(len, self.prefixes))
        self.parse_tags = set(parse_tags)
        unknown = self.parse_tags - set(self.sinks)
        if unknown:
            raise ValueError(f"parse_tags have no sink: {sorted(unknown)}")
        self.on_record = on_record
        self.json_loader = json_loader
        self.small_record_limit = small_record_limit
        self._probe = bytearray()
        self._small = bytearray()
        self._tag: str | None = None
        self.peak_small_buffer_bytes = 0
        self.record_counts: dict[str, int] = {tag: 0 for tag in self.sinks}

    def _classify(self) -> bool:
        probe = bytes(self._probe)
        for prefix, tag in self.prefixes.items():
            if probe.startswith(prefix):
                self._tag = tag
                if tag in self.parse_tags:
                    self._small.extend(self._probe)
                    self.peak_small_buffer_bytes = max(
                        self.peak_small_buffer_bytes, len(self._small)
                    )
                else:
                    self.sinks[tag].write(self._probe)
                self._probe.clear()
                return True
        if any(prefix.startswith(probe) for prefix in self.prefixes):
            return False
        raise WorkerError(f"unknown or malformed worker record prefix: {probe[:64]!r}")

    def _consume(self, payload: bytes) -> None:
        position = 0
        while self._tag is None and position < len(payload):
            take = min(self.max_prefix - len(self._probe), len(payload) - position)
            self._probe.extend(payload[position : position + take])
            position += take
            if self._classify():
                break
            if len(self._probe) >= self.max_prefix:
                raise WorkerError("worker record tag exceeds prefix bound")
        if self._tag is None:
            return
        remainder = payload[position:]
        if self._tag in self.parse_tags:
            self._small.extend(remainder)
            if len(self._small) > self.small_record_limit:
                raise WorkerError(
                    f"small worker record {self._tag!r} exceeds byte ceiling"
                )
            self.peak_small_buffer_bytes = max(self.peak_small_buffer_bytes, len(self._small))
        elif remainder:
            self.sinks[self._tag].write(remainder)

    def feed(self, payload: bytes) -> None:
        position = 0
        while position < len(payload):
            newline = payload.find(b"\n", position)
            if newline < 0:
                self._consume(payload[position:])
                return
            self._consume(payload[position:newline])
            self._end_line()
            position = newline + 1

    def _end_line(self) -> None:
        if self._tag is None:
            if not self._probe:
                return
            raise WorkerError("worker emitted a line without a complete record tag")
        tag = self._tag
        sink = self.sinks[tag]
        parsed: Any | None = None
        if tag in self.parse_tags:
            try:
                parsed = self.json_loader(bytes(self._small))
            except (ValueError, UnicodeDecodeError) as exc:
                raise WorkerError(f"invalid JSON in worker {tag} record") from exc
            if not isinstance(parsed, list) or not parsed or parsed[0] != tag:
                raise WorkerError(f"worker {tag} record has inconsistent tag")
            sink.write(self._small)
        sink.write(b"\n")
        if hasattr(sink, "finish_record"):
            sink.finish_record()
        self.record_counts[tag] += 1
        if self.on_record is not None:
            self.on_record(tag, parsed)
        self._probe.clear()
        self._small.clear()
        self._tag = None

    def finish(self) -> None:
        if self._tag is not None or self._probe or self._small:
            raise WorkerError("worker stdout ended in the middle of a JSONL record")


@dataclass(frozen=True)
class _WorkerResult:
    elapsed_seconds: float
    stderr: str
    record_counts: Mapping[str, int]


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()


def _run_worker(
    argv: Sequence[str],
    demultiplexer: JsonlDemultiplexer,
    *,
    timeout_seconds: float,
) -> _WorkerResult:
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stderr = bytearray()
    deadline = started + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout(f"worker exceeded {timeout_seconds:g}s")
            events = selector.select(min(remaining, 0.5))
            if not events:
                continue
            for key, _ in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), _READ_CHUNK)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if key.data == "stdout":
                    demultiplexer.feed(chunk)
                else:
                    if len(chunk) >= _STDERR_LIMIT:
                        stderr[:] = chunk[-_STDERR_LIMIT:]
                    else:
                        overflow = max(0, len(stderr) + len(chunk) - _STDERR_LIMIT)
                        if overflow:
                            del stderr[:overflow]
                        stderr.extend(chunk)
        return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        demultiplexer.finish()
        stderr_text = stderr.decode("utf-8", errors="replace")
        if return_code != 0:
            raise WorkerError(
                f"worker exited with code {return_code}: {stderr_text[-2000:]}"
            )
        return _WorkerResult(
            elapsed_seconds=time.monotonic() - started,
            stderr=stderr_text,
            record_counts=dict(demultiplexer.record_counts),
        )
    except BaseException:
        if process.poll() is None:
            _kill_process_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


def _validate_out(record: Sequence[Any], instance_id: str, mode: str) -> None:
    if (
        not isinstance(record, list)
        or len(record) != 12
        or record[0] != "out"
        or record[1] != instance_id
        or record[2] != mode
    ):
        raise WorkerError(f"invalid {mode} out record for {instance_id}")


def _cross_check_trace_locator(trace: Sequence[Any], locator: Sequence[Any]) -> None:
    tr = _trace_fields(trace)
    sl = _locator_fields(locator)
    pairs = (
        ("instance", "instance"),
        ("eligible_state", "eligible_state"),
        ("callback", "callback"),
        ("raw_decision", "raw_decision"),
        ("level", "level"),
        ("trail", "trail"),
        ("conflicts", "conflicts"),
        ("decisions", "decisions"),
        ("propagations", "propagations"),
        ("restarts", "restarts"),
        ("prefix_hash", "prefix_hash"),
        ("eligibility_hash", "eligibility_hash"),
        ("native_external", "native_external"),
        ("state_nodes", "state_nodes"),
    )
    mismatches = [left for left, right in pairs if tr[left] != sl[right]]
    if mismatches:
        raise WorkerError(
            f"DISCOVER/CAPTURE locator mismatch at eligible state {sl['eligible_state']}: "
            + ",".join(mismatches)
        )


def _validate_capture_events(
    events: Sequence[tuple[str, Any | None]],
    selected: Sequence[Sequence[Any]],
    instance_id: str,
) -> tuple[list[list[Any]], dict[int, list[Any]], list[list[Any]]]:
    locators: dict[int, list[Any]] = {}
    proposals: dict[int, list[Any]] = {}
    failures: dict[int, list[Any]] = {}
    out_records: list[list[Any]] = []
    position = 0
    while position < len(events):
        tag, record = events[position]
        if tag == "sl":
            if record is None:
                raise WorkerError("parsed sl record is missing")
            fields = _locator_fields(record)
            target = fields["eligible_state"]
            if target in locators:
                raise WorkerError("duplicate sl record")
            if position + 2 >= len(events) or events[position + 1][0] != "st" or events[position + 2][0] != "pr":
                raise WorkerError("capture output must emit sl, st, pr consecutively")
            proposal = events[position + 2][1]
            if (
                not isinstance(proposal, list)
                or len(proposal) != 4
                or proposal[0] != "pr"
                or proposal[1] != instance_id
                or proposal[2] != target
                or not isinstance(proposal[3], list)
            ):
                raise WorkerError("proposal does not match its sl/st state")
            locators[target] = list(record)
            proposals[target] = list(proposal)
            position += 3
            continue
        if tag == "sf":
            if not isinstance(record, list) or len(record) != 8 or record[1] != instance_id:
                raise WorkerError("invalid sf record")
            target = record[2]
            if target in failures:
                raise WorkerError("duplicate sf record")
            failures[target] = list(record)
        elif tag == "out":
            assert record is not None
            _validate_out(record, instance_id, "capture")
            out_records.append(list(record))
        else:
            raise WorkerError(f"unexpected capture event order at {tag}")
        position += 1
    if len(out_records) != 1:
        raise WorkerError("capture worker must emit exactly one out record")

    trace_by_target = {_trace_fields(item)["eligible_state"]: item for item in selected}
    expected = set(trace_by_target)
    observed = set(locators) | set(failures)
    if observed != expected or set(locators) & set(failures):
        raise WorkerError(
            f"capture target coverage mismatch expected={sorted(expected)} observed={sorted(observed)}"
        )
    for target, locator in locators.items():
        _cross_check_trace_locator(trace_by_target[target], locator)
    return [locators[key] for key in sorted(locators)], proposals, [failures[key] for key in sorted(failures)]


def _validate_eval_record(
    record: Sequence[Any], instance_id: str, target: int, requested: int
) -> tuple[bool, bool]:
    if (
        not isinstance(record, list)
        or len(record) != 20
        or record[0] != "ev"
        or record[1] != instance_id
        or record[2] != target
        or record[5] != requested
    ):
        raise WorkerError("invalid ev record identity or requested action")
    replay_valid = bool(record[6] and record[7] and record[8] and record[9])
    outcome_complete = bool(
        replay_valid and not record[14] and not record[15] and record[13] in (10, 20)
    )
    return replay_valid, outcome_complete


def _raw_suffix(codec: str) -> str:
    return ".jsonl.gz" if codec == "gzip" else ".jsonl"


def _write_attempt_event(writer: AtomicRawWriter, kind: str, **values: Any) -> None:
    writer.write_json({"time": utc_now_text(), "event": kind, **values})


def _cnf_cache_suffix(path: Path) -> str:
    lowered = path.name.lower()
    for suffix in (".cnf.gz", ".cnf.xz", ".cnf.bz2", ".cnf"):
        if lowered.endswith(suffix):
            return suffix
    return ".cnf"


def _stage_cnf_in_cache(source: Path, cache_root: Path | None) -> Path | None:
    if cache_root is None:
        return None
    cache_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="decisiontrace-cnf-",
        suffix=_cnf_cache_suffix(source),
        dir=cache_root,
    )
    temporary = Path(temporary_name)
    try:
        with (
            os.fdopen(descriptor, "wb") as destination,
            source.open("rb") as input_stream,
        ):
            shutil.copyfileobj(input_stream, destination, length=1024 * 1024)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _remove_staged_cnf(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def execute_pipeline_part(
    policy: PipelinePolicy,
    part: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> PipelinePartResult:
    """Run DISCOVER -> CAPTURE -> EVAL for one plan record.

    Four raw outputs are committed together in the sense that the completion
    statistics marker is published only after all four files have been fsynced
    and renamed.  On any worker/validation failure all still-temporary raw
    files are removed and a uniquely named failed-attempt report is retained.
    """

    policy.check_paths()
    if part.get("schema") != PIPELINE_PLAN_SCHEMA:
        raise ValueError("unexpected pipeline part schema")
    if part.get("policy_id") != policy.policy_id:
        raise ValueError("pipeline plan was generated with a different policy")
    part_stem = part.get("part_stem")
    instances = part.get("instances")
    if not isinstance(part_stem, str) or not _PART_STEM_RE.fullmatch(part_stem):
        raise ValueError("unsafe pipeline part stem")
    if not isinstance(instances, list) or not instances:
        raise ValueError("pipeline part must contain instances")

    suffix = _raw_suffix(policy.raw_codec)
    raw_paths = {
        "traces": policy.build_root / "traces" / f"{part_stem}{suffix}",
        "states": policy.build_root / "states" / f"{part_stem}{suffix}",
        "proposals": policy.build_root / "proposals" / f"{part_stem}{suffix}",
        "evals": policy.build_root / "evals" / f"{part_stem}{suffix}",
    }
    completion_path = policy.build_root / "statistics" / f"{part_stem}.complete.json"
    had_completion = completion_path.exists()
    if had_completion and not overwrite:
        raise FileExistsError(f"part is already complete: {completion_path}")
    if had_completion:
        # The completion marker is the commit point.  Explicit overwrite first
        # withdraws that marker so readers can never mistake a mixture of old
        # and newly published raw files for one committed part.
        completion_path.unlink()
    attempt_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    log_path = policy.build_root / "logs" / f"{part_stem}.attempt-{attempt_id}.jsonl"
    failed_stats_path = (
        policy.build_root / "statistics" / f"{part_stem}.attempt-{attempt_id}.failed.json"
    )

    raw_writers: dict[str, AtomicRawWriter] = {}
    staged_cnf_path: Path | None = None
    log_writer = AtomicRawWriter(log_path, "none", overwrite=False)
    counters = {
        "instances_completed": 0,
        "targets_selected": 0,
        "states_materialized": 0,
        "capture_failures": 0,
        "evaluations": 0,
        "evaluation_timeouts": 0,
        "replay_valid_evaluations": 0,
        "valid_evaluations": 0,
        "censored_or_rejected_evaluations": 0,
        "worker_processes": 0,
    }
    started = time.monotonic()
    try:
        for stage, path in raw_paths.items():
            # A killed attempt can leave final-named raw files but no commit
            # marker.  Such files are uncommitted by definition and may be
            # replaced on retry without requiring a broad cleanup command.
            raw_writers[stage] = AtomicRawWriter(path, policy.raw_codec, overwrite=True)
        _write_attempt_event(
            log_writer,
            "part_start",
            part_stem=part_stem,
            policy_id=policy.policy_id,
            instances=len(instances),
        )

        for instance_number, manifest_record in enumerate(instances):
            _remove_staged_cnf(staged_cnf_path)
            staged_cnf_path = None
            if not isinstance(manifest_record, Mapping):
                raise ValueError("pipeline instance must be an object")
            instance_id = manifest_record.get("instance_id")
            source_path = manifest_record.get("source_path")
            if not isinstance(instance_id, str) or not isinstance(source_path, str):
                raise ValueError("pipeline instance lacks instance_id/source_path")
            staged_cnf_path = _stage_cnf_in_cache(
                Path(source_path), policy.cnf_cache_root
            )
            worker_source_path = str(staged_cnf_path or source_path)
            _write_attempt_event(
                log_writer,
                "instance_start",
                instance_id=instance_id,
                ordinal=instance_number,
            )

            discover_records: list[tuple[str, Any | None]] = []
            discover_demux = JsonlDemultiplexer(
                {"tr": raw_writers["traces"], "out": raw_writers["traces"]},
                parse_tags={"tr", "out"},
                on_record=lambda tag, record: discover_records.append((tag, record)),
            )
            discover_argv = build_worker_argv(
                policy,
                mode="discover",
                instance_id=instance_id,
                source_path=worker_source_path,
            )
            counters["worker_processes"] += 1
            discover_result = _run_worker(
                discover_argv, discover_demux, timeout_seconds=policy.timeout_seconds
            )
            traces = [record for tag, record in discover_records if tag == "tr" and record is not None]
            outs = [record for tag, record in discover_records if tag == "out" and record is not None]
            if len(outs) != 1:
                raise WorkerError("discover worker must emit exactly one out record")
            _validate_out(outs[0], instance_id, "discover")
            selected = select_target_traces(traces, policy)
            counters["targets_selected"] += len(selected)
            _write_attempt_event(
                log_writer,
                "discover_done",
                instance_id=instance_id,
                elapsed_seconds=discover_result.elapsed_seconds,
                trace_records=len(traces),
                targets=[_trace_fields(record)["eligible_state"] for record in selected],
            )
            if not selected:
                counters["instances_completed"] += 1
                _write_attempt_event(log_writer, "instance_no_targets", instance_id=instance_id)
                continue

            capture_events: list[tuple[str, Any | None]] = []
            capture_demux = JsonlDemultiplexer(
                {
                    "sl": raw_writers["states"],
                    "st": raw_writers["states"],
                    "sf": raw_writers["states"],
                    "pr": raw_writers["proposals"],
                    "out": raw_writers["states"],
                },
                parse_tags={"sl", "sf", "pr", "out"},
                on_record=lambda tag, record: capture_events.append((tag, record)),
            )
            targets = [_trace_fields(record)["eligible_state"] for record in selected]
            capture_argv = build_worker_argv(
                policy,
                mode="capture",
                instance_id=instance_id,
                source_path=worker_source_path,
                targets=targets,
            )
            counters["worker_processes"] += 1
            capture_result = _run_worker(
                capture_argv, capture_demux, timeout_seconds=policy.timeout_seconds
            )
            locators, proposals, failures = _validate_capture_events(
                capture_events, selected, instance_id
            )
            counters["states_materialized"] += len(locators)
            counters["capture_failures"] += len(failures)
            _write_attempt_event(
                log_writer,
                "capture_done",
                instance_id=instance_id,
                elapsed_seconds=capture_result.elapsed_seconds,
                states=len(locators),
                failures=[record[2] for record in failures],
                peak_parsed_record_bytes=capture_demux.peak_small_buffer_bytes,
            )

            for locator in locators:
                fields = _locator_fields(locator)
                proposal = proposals[fields["eligible_state"]]
                # The fallback baseline and a forced replay of the same native
                # literal are intentionally distinct records.
                forced_literals = forced_evaluation_literals(locator, proposal)
                for requested in [0, *forced_literals]:
                    eval_records: list[tuple[str, Any | None]] = []
                    eval_sink = _DiscardRecordSink()
                    eval_demux = JsonlDemultiplexer(
                        {"ev": eval_sink, "out": eval_sink},
                        parse_tags={"ev", "out"},
                        on_record=lambda tag, record: eval_records.append((tag, record)),
                    )
                    eval_argv = build_worker_argv(
                        policy,
                        mode="eval",
                        instance_id=instance_id,
                        source_path=worker_source_path,
                        locator=locator,
                        force_literal=requested,
                    )
                    counters["worker_processes"] += 1
                    try:
                        eval_result = _run_worker(
                            eval_argv, eval_demux, timeout_seconds=policy.timeout_seconds
                        )
                    except WorkerTimeout:
                        raw_writers["evals"].write_json(
                            [
                                "ef",
                                instance_id,
                                fields["eligible_state"],
                                requested,
                                "process_timeout",
                                policy.timeout_seconds,
                            ]
                        )
                        counters["evaluation_timeouts"] += 1
                        counters["censored_or_rejected_evaluations"] += 1
                        _write_attempt_event(
                            log_writer,
                            "eval_timeout",
                            instance_id=instance_id,
                            eligible_state=fields["eligible_state"],
                            requested_literal=requested,
                            timeout_seconds=policy.timeout_seconds,
                        )
                        continue
                    evs = [record for tag, record in eval_records if tag == "ev" and record is not None]
                    eval_outs = [record for tag, record in eval_records if tag == "out" and record is not None]
                    if len(evs) != 1 or len(eval_outs) != 1:
                        raise WorkerError("eval worker must emit exactly one ev and out record")
                    _validate_out(eval_outs[0], instance_id, "eval")
                    raw_writers["evals"].write_json(evs[0])
                    raw_writers["evals"].write_json(eval_outs[0])
                    replay_valid, outcome_complete = _validate_eval_record(
                        evs[0], instance_id, fields["eligible_state"], requested
                    )
                    counters["evaluations"] += 1
                    if replay_valid:
                        counters["replay_valid_evaluations"] += 1
                    if outcome_complete:
                        counters["valid_evaluations"] += 1
                    else:
                        counters["censored_or_rejected_evaluations"] += 1
                    _write_attempt_event(
                        log_writer,
                        "eval_done",
                        instance_id=instance_id,
                        eligible_state=fields["eligible_state"],
                        requested_literal=requested,
                        replay_valid=replay_valid,
                        outcome_complete=outcome_complete,
                        elapsed_seconds=eval_result.elapsed_seconds,
                    )
            counters["instances_completed"] += 1
            _write_attempt_event(log_writer, "instance_done", instance_id=instance_id)

        _remove_staged_cnf(staged_cnf_path)
        staged_cnf_path = None
        _write_attempt_event(log_writer, "part_raw_complete", counters=dict(counters))
        output_descriptors: dict[str, Mapping[str, Any]] = {}
        for stage, writer in raw_writers.items():
            output_descriptors[stage] = writer.close()
        log_descriptor = log_writer.close()
        statistics = {
            "schema": PIPELINE_STATS_SCHEMA,
            "status": "complete",
            "part_stem": part_stem,
            "policy_id": policy.policy_id,
            "generation_id": uuid.uuid4().hex,
            "created_at": utc_now_text(),
            "elapsed_seconds": time.monotonic() - started,
            "counts": {"instances_planned": len(instances), **counters},
            "outputs": output_descriptors,
            "log": log_descriptor,
        }
        _atomic_json(completion_path, statistics, overwrite=overwrite)
        return PipelinePartResult(
            part_stem=part_stem,
            output_files=output_descriptors,
            statistics_path=completion_path,
            log_path=log_path,
            instances_completed=counters["instances_completed"],
            states_materialized=counters["states_materialized"],
            evaluations=counters["evaluations"],
        )
    except BaseException as exc:
        _remove_staged_cnf(staged_cnf_path)
        staged_cnf_path = None
        for writer in raw_writers.values():
            try:
                writer.abort()
            except BaseException:
                pass
        # close() publishes individual files before the one part-level commit
        # marker is written.  If a later close/fsync fails, retract every file
        # published by this attempt so the default retry path remains usable.
        if not completion_path.exists():
            for writer in raw_writers.values():
                if writer.finished and writer.path.exists():
                    try:
                        writer.path.unlink()
                    except OSError:
                        pass
        try:
            _write_attempt_event(
                log_writer,
                "part_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                counters=dict(counters),
            )
            log_descriptor = log_writer.close()
        except BaseException:
            log_writer.abort()
            log_descriptor = {"path": str(log_path), "log_write_failed": True}
        failure = {
            "schema": PIPELINE_STATS_SCHEMA,
            "status": "failed",
            "part_stem": part_stem,
            "policy_id": policy.policy_id,
            "created_at": utc_now_text(),
            "elapsed_seconds": time.monotonic() - started,
            "counts": {"instances_planned": len(instances), **counters},
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "log": log_descriptor,
        }
        try:
            _atomic_json(failed_stats_path, failure, overwrite=False)
        except BaseException:
            pass
        raise PipelinePartError(
            f"part {part_stem} was not committed: {exc}",
            log_path=log_path,
            statistics_path=failed_stats_path,
        ) from exc
