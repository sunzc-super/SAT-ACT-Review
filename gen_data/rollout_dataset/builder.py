"""Stream committed DecisionTrace raw parts into compact typed shards.

No value, regret or ranking label is derived here.  The builder preserves the
captured state, proposed signed literals and replay outcomes and verifies all
cross-file identities before publishing the compact part commit marker.
"""

from __future__ import annotations

import gzip
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .compact import ArrayData, CompactArrayReader, CompactArrayWriter, CompactPartPaths
from .contract import (
    PIPELINE_STATS_SCHEMA,
    RAW_STATS_SCHEMA_V2,
    RAW_RECORD_FIELDS,
    RAW_STAGE_TAGS,
    RawPolicy,
    load_worker_raw_schema,
    raw_plan_part_dataset_group,
    raw_plan_part_digest,
    raw_plan_part_split,
    raw_suffix,
    validate_raw_plan_part,
)
from .schema import PART_DIAGNOSTICS_SCHEMA, SHARD_BUILD_SCHEMA, validate_literal


_HEX64_RE = re.compile(r"^[0-9a-fA-F]{1,16}$")


@dataclass(frozen=True)
class ShardBuildResult:
    paths: CompactPartPaths
    state_count: int
    capture_failure_count: int
    evaluation_count: int


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _open_raw(path: Path, codec: str):
    return gzip.open(path, "rb") if codec == "gzip" else path.open("rb")


def _iter_raw_records(path: Path, codec: str) -> Iterator[list[Any]]:
    with _open_raw(path, codec) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"invalid raw JSON at {path}:{line_number}") from exc
            if not isinstance(value, list) or not value or not isinstance(value[0], str):
                raise ValueError(f"invalid raw record at {path}:{line_number}")
            yield value


def _state_key(record: Sequence[Any]) -> tuple[str, int]:
    if (
        len(record) < 3
        or not isinstance(record[1], str)
        or not record[1]
        or isinstance(record[2], bool)
        or not isinstance(record[2], int)
        or record[2] <= 0
    ):
        raise ValueError("state-keyed record has invalid identity")
    return record[1], record[2]


def _verified_raw_commit(
    policy: RawPolicy, part_stem: str, expected_plan_part_sha256: str
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = policy.build_root
    completion = root / "statistics" / f"{part_stem}.complete.json"
    if not completion.is_file():
        raise FileNotFoundError(f"raw part completion marker is missing: {completion}")
    with completion.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    is_v2 = isinstance(record, Mapping) and record.get("schema") == RAW_STATS_SCHEMA_V2
    policy_reference = (
        record.get("policy_id") if is_v2 else record.get("policy_sha256")
    ) if isinstance(record, Mapping) else None
    plan_matches = bool(
        isinstance(record, Mapping)
        and (is_v2 or record.get("plan_part_sha256") == expected_plan_part_sha256)
    )
    if (
        not isinstance(record, dict)
        or record.get("schema") not in {PIPELINE_STATS_SCHEMA, RAW_STATS_SCHEMA_V2}
        or record.get("status") != "complete"
        or record.get("part_stem") != part_stem
        or policy_reference != policy.digest
        or not plan_matches
    ):
        raise ValueError(
            f"raw part completion marker/policy/plan mismatch: {completion}"
        )
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(RAW_STAGE_TAGS):
        raise ValueError("raw part completion marker has incomplete stage outputs")
    suffix = raw_suffix(policy.raw_codec)
    paths: dict[str, Path] = {}
    for stage in RAW_STAGE_TAGS:
        descriptor = outputs.get(stage)
        expected = root / stage / f"{part_stem}{suffix}"
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"raw output descriptor is missing for {stage}")
        if (
            descriptor.get("codec") != policy.raw_codec
            or Path(str(descriptor.get("path", ""))).name != expected.name
            or isinstance(descriptor.get("bytes"), bool)
            or not isinstance(descriptor.get("bytes"), int)
            or descriptor["bytes"] < 0
            or isinstance(descriptor.get("raw_bytes"), bool)
            or not isinstance(descriptor.get("raw_bytes"), int)
            or descriptor["raw_bytes"] < 0
            or isinstance(descriptor.get("records"), bool)
            or not isinstance(descriptor.get("records"), int)
            or descriptor["records"] < 0
        ):
            raise ValueError(f"invalid committed raw descriptor for {stage}")
        if not expected.is_file() or expected.stat().st_size != descriptor["bytes"]:
            raise ValueError(f"committed raw size mismatch for {stage}: {expected}")
        paths[stage] = expected
    return record, paths


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
    _state_key(locator)
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


def _trace_fields(record: Sequence[Any]) -> dict[str, Any]:
    if not isinstance(record, list) or len(record) != 17 or record[0] != "tr":
        raise ValueError("invalid tr record")
    names = RAW_RECORD_FIELDS["tr"]
    fields = dict(zip(names, record[1:]))
    if not isinstance(fields["instance"], str) or not fields["instance"]:
        raise ValueError("tr instance must be a non-empty string")
    for name in (
        "callback",
        "raw_decision",
        "eligible_state",
        "level",
        "trail",
        "conflicts",
        "decisions",
        "propagations",
        "restarts",
        "candidates",
        "state_nodes_preflight",
    ):
        _integer(fields[name], f"tr {name}")
    if fields["eligible_state"] <= 0:
        raise ValueError("tr eligible_state must be positive")
    fields["prefix_hash"] = _validate_hash(fields["prefix_hash"], "prefix_hash")
    fields["eligibility_hash"] = _validate_hash(
        fields["eligibility_hash"], "eligibility_hash"
    )
    for name in ("native_external", "native_internal"):
        value = fields[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"tr {name} must be an integer")
        if value:
            validate_literal(value)
    return fields


def _validate_out_record(
    record: Sequence[Any], *, expected_mode: str, instance_ids: set[str]
) -> str:
    if (
        not isinstance(record, list)
        or len(record) != 12
        or record[0] != "out"
        or record[2] != expected_mode
        or not isinstance(record[1], str)
        or record[1] not in instance_ids
    ):
        raise ValueError(f"invalid {expected_mode} out record")
    return record[1]


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
        ("state_nodes_preflight", "state_nodes"),
    )
    mismatches = [left for left, right in pairs if tr[left] != sl[right]]
    if mismatches:
        raise ValueError(
            f"tr/sl mismatch at {sl['instance']}:{sl['eligible_state']}: "
            + ",".join(mismatches)
        )


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
        raise ValueError("invalid ev record identity or requested action")
    replay_valid = bool(record[6] and record[7] and record[8] and record[9])
    outcome_complete = bool(
        replay_valid and not record[14] and not record[15] and record[13] in (10, 20)
    )
    return replay_valid, outcome_complete


def _cross_check_locator_state(locator: Sequence[Any], state: Sequence[Any]) -> None:
    fields = _locator_fields(locator)
    if not isinstance(state, list) or len(state) != 32 or state[0] != "st":
        raise ValueError("invalid st record")
    scalar_names = (
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
    )
    mismatches = [
        name
        for offset, name in enumerate(scalar_names, start=1)
        if state[offset] != fields[name]
    ]
    fingerprint_positions = (
        (17, "assignment_hash"),
        (18, "trail_hash"),
        (19, "clause_canonical_hash"),
        (20, "clause_order_hash"),
        (21, "eligibility_hash"),
    )
    mismatches.extend(
        name for position, name in fingerprint_positions if state[position] != fields[name]
    )
    if all(isinstance(state[position], int) for position in (13, 14, 15, 16)):
        nodes = 2 * state[13] + state[15]
        if nodes != fields["state_nodes"]:
            mismatches.append("state_nodes")
        if state[16] != fields["literal_occurrences"]:
            mismatches.append("literal_occurrences")
    if mismatches:
        raise ValueError(
            f"sl/st mismatch at {fields['instance']}:{fields['eligible_state']}: "
            + ",".join(mismatches)
        )


_ST_SCALAR_NAMES = (
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
    "internal_variables",
    "external_variables",
    "n_clauses",
    "n_occurrences",
    "assignment_hash",
    "trail_hash",
    "clause_canonical_hash",
    "clause_order_hash",
    "eligibility_hash",
)

_SHARD_INSTANCE_FIELDS = (
    "instance_id",
    "dataset",
    "dataset_release_id",
    "dataset_group",
    "split",
    "label",
    "difficulty",
    "family",
    "competition_year",
    "track",
    "collection_id",
    "source_relpath",
    "source_instance_group_id",
    "cnf_sha256",
    "size",
)

_DIAGNOSTIC_FIELDS: Mapping[str, tuple[str, ...]] = {
    "tr": RAW_RECORD_FIELDS["tr"],
    "discover_out": RAW_RECORD_FIELDS["out"],
    "capture_out": RAW_RECORD_FIELDS["out"],
    "eval_out": (
        "eligible_state",
        "requested_external",
        *RAW_RECORD_FIELDS["out"],
    ),
}


def _diagnostics_metadata(
    records: Mapping[str, Sequence[Sequence[Any]]]
) -> dict[str, Any]:
    return {
        "schema": PART_DIAGNOSTICS_SCHEMA,
        "fields": {
            name: list(fields) for name, fields in _DIAGNOSTIC_FIELDS.items()
        },
        "records": {
            name: [list(row) for row in records[name]] for name in _DIAGNOSTIC_FIELDS
        },
    }


def _validate_diagnostics_metadata(
    value: Any,
    *,
    instance_ids: set[str],
    expected_capture_instances: set[str] | None = None,
) -> None:
    expected_fields = {
        name: list(fields) for name, fields in _DIAGNOSTIC_FIELDS.items()
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "fields", "records"}
        or value.get("schema") != PART_DIAGNOSTICS_SCHEMA
        or value.get("fields") != expected_fields
        or not isinstance(value.get("records"), Mapping)
        or set(value["records"]) != set(_DIAGNOSTIC_FIELDS)
    ):
        raise ValueError("invalid shard raw_diagnostics schema")
    rows = value["records"]
    if any(not isinstance(rows[name], list) for name in _DIAGNOSTIC_FIELDS):
        raise ValueError("raw_diagnostics records must be positional arrays")

    trace_keys: set[tuple[str, int]] = set()
    for row in rows["tr"]:
        if not isinstance(row, list) or len(row) != len(_DIAGNOSTIC_FIELDS["tr"]):
            raise ValueError("invalid compact tr diagnostic row")
        record = ["tr", *row]
        fields = _trace_fields(record)
        if fields["instance"] not in instance_ids:
            raise ValueError("tr diagnostic instance is absent from the raw plan")
        key = (fields["instance"], fields["eligible_state"])
        if key in trace_keys:
            raise ValueError(f"duplicate tr diagnostic state: {key}")
        trace_keys.add(key)

    out_instances: dict[str, list[str]] = {"discover_out": [], "capture_out": []}
    for name, mode in (("discover_out", "discover"), ("capture_out", "capture")):
        for row in rows[name]:
            if not isinstance(row, list) or len(row) != len(_DIAGNOSTIC_FIELDS[name]):
                raise ValueError(f"invalid compact {name} diagnostic row")
            instance = _validate_out_record(
                ["out", *row], expected_mode=mode, instance_ids=instance_ids
            )
            out_instances[name].append(instance)
        if len(out_instances[name]) != len(set(out_instances[name])):
            raise ValueError(f"duplicate compact {name} diagnostic instance")
    if set(out_instances["discover_out"]) != instance_ids:
        raise ValueError("discover out diagnostics do not cover every raw plan instance")
    if (
        expected_capture_instances is not None
        and set(out_instances["capture_out"]) != expected_capture_instances
    ):
        raise ValueError("capture out diagnostics do not match captured instances")

    eval_keys: set[tuple[str, int, int]] = set()
    for row in rows["eval_out"]:
        if not isinstance(row, list) or len(row) != len(_DIAGNOSTIC_FIELDS["eval_out"]):
            raise ValueError("invalid compact eval_out diagnostic row")
        target, requested = row[0], row[1]
        _integer(target, "eval_out eligible_state", minimum=1)
        if isinstance(requested, bool) or not isinstance(requested, int):
            raise ValueError("eval_out requested_external must be an integer")
        if requested:
            validate_literal(requested)
        out_record = ["out", *row[2:]]
        instance = _validate_out_record(
            out_record, expected_mode="eval", instance_ids=instance_ids
        )
        key = (instance, target, requested)
        if key in eval_keys:
            raise ValueError(f"duplicate compact eval_out diagnostic link: {key}")
        eval_keys.add(key)


def _arrays_for_state(
    state: Sequence[Any],
    proposal: Sequence[Any],
    evaluations: Sequence[Sequence[Any]],
    evaluation_failures: Sequence[Sequence[Any]] = (),
) -> dict[str, ArrayData]:
    if len(state) != 32 or state[0] != "st":
        raise ValueError("invalid st record")
    if (
        len(proposal) != 4
        or proposal[0] != "pr"
        or _state_key(proposal) != _state_key(state)
    ):
        raise ValueError("st/pr association mismatch")
    actions = proposal[3]
    if not isinstance(actions, list):
        raise ValueError("proposal actions must be a list")
    if any(not isinstance(action, list) or len(action) != 5 for action in actions):
        raise ValueError("proposal action must contain five fields")
    for evaluation in evaluations:
        if (
            len(evaluation) != 20
            or evaluation[0] != "ev"
            or _state_key(evaluation) != _state_key(state)
        ):
            raise ValueError("st/ev association mismatch")
    for failure in evaluation_failures:
        if (
            len(failure) != 6
            or failure[0] != "ef"
            or _state_key(failure) != _state_key(state)
            or failure[4] != "process_timeout"
            or isinstance(failure[3], bool)
            or not isinstance(failure[3], int)
            or isinstance(failure[5], bool)
            or not isinstance(failure[5], (int, float))
        ):
            raise ValueError("st/ef association mismatch")
    try:
        eligibility = bytes.fromhex(state[23])
        action_random_key = [int(action[4], 16) for action in actions]
        eval_prefix_hash = [int(evaluation[10], 16) for evaluation in evaluations]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid hexadecimal field in raw state/action/eval") from exc
    internal_variables, external_variables = state[13], state[14]
    clauses, occurrences = state[15], state[16]
    for name, value in (
        ("internal_variables", internal_variables),
        ("external_variables", external_variables),
        ("clauses", clauses),
        ("occurrences", occurrences),
    ):
        _integer(value, f"st {name}")
    internal_arrays = {
        "i2e": state[22],
        "assignment_value": state[24],
        "assignment_level": state[25],
        "assignment_source": state[26],
        "trail_position": state[27],
        "activity": state[28],
        "saved_phase": state[29],
    }
    if any(
        not isinstance(values, list) or len(values) != internal_variables
        for values in internal_arrays.values()
    ):
        raise ValueError("per-variable st array length does not match internal_variables")
    expected_eligibility_bytes = (external_variables + 7) // 8
    if len(eligibility) != expected_eligibility_bytes:
        raise ValueError("eligibility bitset length does not match external_variables")
    offsets, literals = state[30], state[31]
    if (
        not isinstance(offsets, list)
        or len(offsets) != clauses + 1
        or not offsets
        or offsets[0] != 0
        or any(
            not isinstance(value, int)
            or value < 0
            or (index and value < offsets[index - 1])
            for index, value in enumerate(offsets)
        )
        or offsets[-1] != occurrences
    ):
        raise ValueError("clause_offsets are inconsistent with clause/occurrence counts")
    if not isinstance(literals, list) or len(literals) != occurrences:
        raise ValueError("clause literal length does not match n_occurrences")
    for action in actions:
        if isinstance(action[1], bool) or not isinstance(action[1], int) or action[1] < 0:
            raise ValueError("proposal literal/source fields are invalid")
        validate_literal(action[0])
    return {
        "i2e": ArrayData("i4", state[22]),
        "eligibility_bits": ArrayData("u1", eligibility),
        "assignment_value": ArrayData("i1", state[24]),
        "assignment_level": ArrayData("i4", state[25]),
        "assignment_source": ArrayData("u1", state[26]),
        "trail_position": ArrayData("i4", state[27]),
        "activity": ArrayData("f4", state[28]),
        "saved_phase": ArrayData("i1", state[29]),
        "clause_offsets": ArrayData("i8", state[30]),
        "clause_literals_internal": ArrayData("i4", state[31]),
        "action_literal": ArrayData("i4", [action[0] for action in actions]),
        "action_source_bits": ArrayData("u4", [action[1] for action in actions]),
        "action_activity": ArrayData("f4", [action[2] for action in actions]),
        "action_jw": ArrayData("f4", [action[3] for action in actions]),
        "action_random_key": ArrayData("u8", action_random_key),
        "eval_requested_literal": ArrayData("i4", [item[5] for item in evaluations]),
        "eval_target_reached": ArrayData("u1", [item[6] for item in evaluations]),
        "eval_locator_match": ArrayData("u1", [item[7] for item in evaluations]),
        "eval_action_eligible": ArrayData("u1", [item[8] for item in evaluations]),
        "eval_action_applied": ArrayData("u1", [item[9] for item in evaluations]),
        "eval_prefix_hash": ArrayData("u8", eval_prefix_hash),
        "eval_native_literal": ArrayData("i4", [item[11] for item in evaluations]),
        "eval_applied_literal": ArrayData("i4", [item[12] for item in evaluations]),
        "eval_status": ArrayData("i4", [item[13] for item in evaluations]),
        "eval_censored": ArrayData("u1", [item[14] for item in evaluations]),
        "eval_horizon_hit": ArrayData("u1", [item[15] for item in evaluations]),
        "eval_delta_conflicts": ArrayData("i8", [item[16] for item in evaluations]),
        "eval_delta_decisions": ArrayData("i8", [item[17] for item in evaluations]),
        "eval_delta_propagations": ArrayData("i8", [item[18] for item in evaluations]),
        "eval_delta_restarts": ArrayData("i8", [item[19] for item in evaluations]),
        "eval_failure_requested_literal": ArrayData(
            "i4", [item[3] for item in evaluation_failures]
        ),
        "eval_failure_reason_code": ArrayData("u1", [1 for _ in evaluation_failures]),
        "eval_failure_timeout_seconds": ArrayData(
            "f4", [item[5] for item in evaluation_failures]
        ),
    }


def _source_output_summary(raw_commit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            key: descriptor[key]
            for key in ("codec", "bytes", "raw_bytes", "records")
            if key in descriptor
        }
        for stage, descriptor in raw_commit["outputs"].items()
    }


def _record_array_length(record: Mapping[str, Any], name: str) -> int:
    descriptors = record.get("arrays")
    if not isinstance(descriptors, list) or not all(
        isinstance(item, Mapping) for item in descriptors
    ):
        raise ValueError("shard record has invalid array descriptors")
    matches = [item for item in descriptors if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"shard array {name!r} occurs {len(matches)} times")
    length = matches[0].get("length")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError(f"shard array {name!r} has invalid length")
    return length


def _validate_committed_shard_records(
    reader: CompactArrayReader,
    *,
    instance_ids: set[str],
    source_record_counts: Mapping[str, Any],
) -> None:
    """Validate shard records/diagnostics against a complete raw-plan group."""

    metadata = reader.header.get("part_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("shard is missing part metadata")
    captured_instances: set[str] = set()
    seen_record_ids: set[str] = set()
    seen_state_keys: set[tuple[str, int]] = set()
    state_records = capture_failure_records = evaluation_failures = 0
    for shard_record in reader.records():
        record_id = shard_record.get("record_id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in seen_record_ids
        ):
            raise ValueError("shard has missing/duplicate record_id")
        seen_record_ids.add(record_id)
        record_metadata = shard_record.get("metadata")
        if not isinstance(record_metadata, Mapping):
            raise ValueError("shard record is missing metadata")
        kind = record_metadata.get("kind")
        if kind == "state":
            state = record_metadata.get("state")
            if not isinstance(state, Mapping):
                raise ValueError("shard state record is missing scalar metadata")
            instance = state.get("instance")
            eligible_state = state.get("eligible_state")
            if not isinstance(instance, str) or instance not in instance_ids:
                raise ValueError("shard state metadata is absent from the raw plan")
            _integer(eligible_state, "shard eligible_state", minimum=1)
            key = (instance, eligible_state)
            if key in seen_state_keys or record_id != f"{instance}:{eligible_state}":
                raise ValueError(f"duplicate/mismatched shard state identity: {key}")
            seen_state_keys.add(key)
            captured_instances.add(instance)
            state_records += 1
            evaluation_count = _integer(
                record_metadata.get("evaluation_count"),
                "shard evaluation_count",
            )
            failure_count = _integer(
                record_metadata.get("evaluation_failure_count"),
                "shard evaluation_failure_count",
            )
            if _record_array_length(
                shard_record, "eval_requested_literal"
            ) != evaluation_count:
                raise ValueError("shard evaluation array/count mismatch")
            if _record_array_length(
                shard_record, "eval_failure_requested_literal"
            ) != failure_count:
                raise ValueError("shard evaluation-failure array/count mismatch")
            evaluation_failures += failure_count
        elif kind == "capture_failure":
            instance = record_metadata.get("instance")
            eligible_state = record_metadata.get("eligible_state")
            if not isinstance(instance, str) or instance not in instance_ids:
                raise ValueError("shard capture failure is absent from the raw plan")
            _integer(eligible_state, "capture-failure eligible_state", minimum=1)
            key = (instance, eligible_state)
            if key in seen_state_keys or record_id != f"{instance}:{eligible_state}":
                raise ValueError(f"duplicate/mismatched capture-failure identity: {key}")
            seen_state_keys.add(key)
            captured_instances.add(instance)
            capture_failure_records += 1
            if shard_record.get("arrays") != []:
                raise ValueError("capture-failure shard record unexpectedly has arrays")
        else:
            raise ValueError("shard record has an unknown kind")

    _validate_diagnostics_metadata(
        metadata.get("raw_diagnostics"),
        instance_ids=instance_ids,
        expected_capture_instances=captured_instances,
    )
    diagnostics_records = metadata["raw_diagnostics"]["records"]
    expected_raw_records = {
        "traces": len(diagnostics_records["tr"])
        + len(diagnostics_records["discover_out"]),
        "states": 2 * state_records
        + capture_failure_records
        + len(diagnostics_records["capture_out"]),
        "proposals": state_records,
        "evals": 2 * len(diagnostics_records["eval_out"])
        + evaluation_failures,
    }
    actual_raw_records: dict[str, int] = {}
    for stage in RAW_STAGE_TAGS:
        value = source_record_counts.get(stage)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid committed raw record count for {stage}")
        actual_raw_records[stage] = value
    if actual_raw_records != expected_raw_records:
        raise ValueError(
            "shard diagnostics/state counts do not cover the committed raw records: "
            f"expected={expected_raw_records} actual={actual_raw_records}"
        )


def completed_shard_matches(
    policy: RawPolicy, plan_part: Mapping[str, Any]
) -> bool:
    """Validate resume metadata without hashing the large payload again."""

    validated_part = validate_raw_plan_part(
        plan_part,
        expected_policy_sha256=policy.digest,
        context="completed shard plan part",
    )
    part_stem = validated_part["part_stem"]
    plan_part_sha256 = raw_plan_part_digest(validated_part)
    instance_ids = {item["instance_id"] for item in validated_part["instances"]}
    complete = policy.build_root / "shards" / f"{part_stem}.complete.json"
    if not complete.exists():
        return False
    raw_complete = policy.build_root / "statistics" / f"{part_stem}.complete.json"
    if not raw_complete.is_file():
        raise ValueError(f"completed shard has no committed raw source: {complete}")
    with raw_complete.open("r", encoding="utf-8") as stream:
        raw_record = json.load(stream)
    outputs = raw_record.get("outputs") if isinstance(raw_record, Mapping) else None
    is_v2 = isinstance(raw_record, Mapping) and raw_record.get("schema") == RAW_STATS_SCHEMA_V2
    policy_reference = (
        raw_record.get("policy_id") if is_v2 else raw_record.get("policy_sha256")
    ) if isinstance(raw_record, Mapping) else None
    plan_matches = bool(
        isinstance(raw_record, Mapping)
        and (is_v2 or raw_record.get("plan_part_sha256") == plan_part_sha256)
    )
    if (
        not isinstance(raw_record, Mapping)
        or raw_record.get("schema") not in {PIPELINE_STATS_SCHEMA, RAW_STATS_SCHEMA_V2}
        or raw_record.get("status") != "complete"
        or raw_record.get("part_stem") != part_stem
        or policy_reference != policy.digest
        or not plan_matches
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(RAW_STAGE_TAGS)
    ):
        raise ValueError(f"invalid raw source marker for completed shard: {raw_complete}")
    source_outputs = _source_output_summary(raw_record)
    with CompactArrayReader(complete, verify=False) as reader:
        metadata = reader.header.get("part_metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("source_part") != part_stem
            or metadata.get("source_policy_id") != policy.digest
            or metadata.get("source_plan_part_id") != plan_part_sha256
            or metadata.get("source_generation_id")
            != raw_record.get("generation_id", raw_record.get("created_at"))
            or metadata.get("source_raw_outputs") != source_outputs
        ):
            raise ValueError(
                f"existing shard does not match the current raw commit: {complete}"
            )
        _validate_committed_shard_records(
            reader,
            instance_ids=instance_ids,
            source_record_counts={
                stage: source_outputs[stage].get("records")
                for stage in RAW_STAGE_TAGS
            },
        )
    return True


def build_shard_part(
    *,
    policy: RawPolicy,
    plan_part: Mapping[str, Any],
    overwrite: bool = False,
    writer: CompactArrayWriter | None = None,
    output_dir: str | Path | None = None,
    part_metadata_sink: list[dict[str, Any]] | None = None,
    diagnostics_sink: list[dict[str, Any]] | None = None,
) -> ShardBuildResult:
    """Pack one committed raw part without deriving utility or rank labels.

    ``writer`` is intentionally optional.  The normal path owns and commits a
    writer for one part; grouped builds pass one shared writer so several raw
    parts can be streamed into a single shard without creating intermediate
    shard files.
    """

    validated_part = validate_raw_plan_part(
        plan_part,
        expected_policy_sha256=policy.digest,
        context="shard build plan part",
    )
    part_stem = validated_part["part_stem"]
    instance_records = validated_part["instances"]
    plan_part_sha256 = raw_plan_part_digest(validated_part)
    load_worker_raw_schema(policy.build_root)
    raw_commit, all_paths = _verified_raw_commit(
        policy, part_stem, plan_part_sha256
    )
    paths = {
        stage: all_paths[stage]
        for stage in ("traces", "states", "proposals", "evals")
    }
    instance_rows: list[list[Any]] = []
    instance_ids: set[str] = set()
    for manifest_record in instance_records:
        if not isinstance(manifest_record, Mapping):
            raise ValueError("shard instance metadata must contain objects")
        instance_id = manifest_record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in instance_ids:
            raise ValueError("shard instance metadata has missing/duplicate instance_id")
        instance_ids.add(instance_id)
        instance_rows.append([manifest_record.get(name) for name in _SHARD_INSTANCE_FIELDS])

    diagnostic_records: dict[str, list[list[Any]]] = {
        name: [] for name in _DIAGNOSTIC_FIELDS
    }
    traces_by_key: dict[tuple[str, int], list[Any]] = {}
    discover_out_instances: set[str] = set()
    discover_closed_instances: set[str] = set()
    for record in _iter_raw_records(paths["traces"], policy.raw_codec):
        if record[0] == "tr":
            fields = _trace_fields(record)
            instance = fields["instance"]
            if instance not in instance_ids:
                raise ValueError("tr instance is absent from the raw plan part")
            if instance in discover_closed_instances:
                raise ValueError("tr record occurs after its discover out record")
            key = (instance, fields["eligible_state"])
            if key in traces_by_key:
                raise ValueError(f"duplicate tr record for state {key}")
            traces_by_key[key] = record
            diagnostic_records["tr"].append(list(record[1:]))
            continue
        if record[0] == "out":
            instance = _validate_out_record(
                record, expected_mode="discover", instance_ids=instance_ids
            )
            if instance in discover_out_instances:
                raise ValueError(f"duplicate discover out record for {instance}")
            discover_out_instances.add(instance)
            discover_closed_instances.add(instance)
            diagnostic_records["discover_out"].append(list(record[1:]))
            continue
        raise ValueError(f"unexpected record in traces part: {record[0]}")
    if discover_out_instances != instance_ids:
        missing = sorted(instance_ids - discover_out_instances)
        raise ValueError(
            "discover out records do not cover every raw plan instance: "
            + ",".join(missing)
        )

    proposals: dict[tuple[str, int], list[Any]] = {}
    for record in _iter_raw_records(paths["proposals"], policy.raw_codec):
        if record[0] != "pr":
            raise ValueError(f"unexpected record in proposals part: {record[0]}")
        key = _state_key(record)
        if key[0] not in instance_ids:
            raise ValueError("proposal instance is absent from the raw plan part")
        if key in proposals:
            raise ValueError(f"duplicate proposal for state {key}")
        proposals[key] = record
    evaluations: dict[tuple[str, int], list[list[Any]]] = {}
    evaluation_failures: dict[tuple[str, int], list[list[Any]]] = {}
    pending_evaluation: list[Any] | None = None
    for record in _iter_raw_records(paths["evals"], policy.raw_codec):
        if record[0] == "out":
            if pending_evaluation is None:
                raise ValueError("eval out record has no immediately preceding ev record")
            instance = _validate_out_record(
                record, expected_mode="eval", instance_ids=instance_ids
            )
            if instance != pending_evaluation[1]:
                raise ValueError("eval out instance does not match its ev record")
            diagnostic_records["eval_out"].append(
                [pending_evaluation[2], pending_evaluation[5], *record[1:]]
            )
            pending_evaluation = None
            continue
        if record[0] == "ef":
            if pending_evaluation is not None:
                raise ValueError("ef record interrupts an ev/out pair")
            if len(record) != 6 or record[4] != "process_timeout":
                raise ValueError("invalid eval failure record")
            key = _state_key(record)
            if key[0] not in instance_ids:
                raise ValueError("eval failure instance is absent from the raw plan part")
            evaluation_failures.setdefault(key, []).append(record)
            continue
        if record[0] != "ev":
            raise ValueError(f"unexpected record in evals part: {record[0]}")
        if pending_evaluation is not None:
            raise ValueError("ev record is missing its immediately following out record")
        key = _state_key(record)
        if key[0] not in instance_ids:
            raise ValueError("ev instance is absent from the raw plan part")
        _validate_eval_record(record, key[0], key[1], record[5])
        evaluations.setdefault(key, []).append(record)
        pending_evaluation = record
    if pending_evaluation is not None:
        raise ValueError("final ev record is missing its out record")

    locators: dict[tuple[str, int], list[Any]] = {}
    seen_states: set[tuple[str, int]] = set()
    seen_failures: set[tuple[str, int]] = set()
    capture_out_instances: set[str] = set()
    capture_activity_instances: set[str] = set()
    state_count = failure_count = evaluation_count = 0
    shard_dir = policy.build_root / "shards"
    source_outputs = _source_output_summary(raw_commit)
    source_split = raw_plan_part_split(
        validated_part, context=f"source part {part_stem}"
    )
    source_dataset_group = raw_plan_part_dataset_group(
        validated_part, context=f"source part {part_stem}"
    )
    part_metadata = {
        "schema": SHARD_BUILD_SCHEMA,
        "source_part": part_stem,
        "source_split": source_split,
        "source_dataset_group": source_dataset_group,
        "source_policy_id": policy.digest,
        "source_plan_part_id": plan_part_sha256,
        "source_generation_id": raw_commit.get("generation_id", raw_commit.get("created_at")),
        "source_raw_outputs": source_outputs,
        "instance_fields": list(_SHARD_INSTANCE_FIELDS),
        "instances": instance_rows,
        "derived_value_labels": False,
    }
    if part_metadata_sink is not None:
        # The grouped header already has the concatenated plan instances.  Do
        # not duplicate every instance row once per source part; retain only
        # the provenance needed for resume/audit checks.
        part_metadata_sink.append(
            {
                key: part_metadata[key]
                for key in (
                    "schema",
                    "source_part",
                    "source_split",
                    "source_dataset_group",
                    "source_policy_id",
                    "source_plan_part_id",
                    "source_generation_id",
                    "source_raw_outputs",
                    "instance_fields",
                    "derived_value_labels",
                )
            }
        )
    writer_context = (
        CompactArrayWriter(
            output_dir if output_dir is not None else shard_dir,
            part_stem,
            overwrite=overwrite,
            part_metadata=part_metadata,
        )
        if writer is None
        else nullcontext(writer)
    )
    with writer_context as active_writer:
        for record in _iter_raw_records(paths["states"], policy.raw_codec):
            tag = record[0]
            if tag == "out":
                instance = _validate_out_record(
                    record, expected_mode="capture", instance_ids=instance_ids
                )
                if instance in capture_out_instances:
                    raise ValueError(f"duplicate capture out record for {instance}")
                capture_out_instances.add(instance)
                diagnostic_records["capture_out"].append(list(record[1:]))
                continue
            if tag == "sl":
                key = _state_key(record)
                if key[0] not in instance_ids:
                    raise ValueError("sl instance is absent from the raw plan part")
                if key[0] in capture_out_instances:
                    raise ValueError("sl record occurs after its capture out record")
                if key in locators:
                    raise ValueError(f"duplicate locator for state {key}")
                _locator_fields(record)
                trace = traces_by_key.get(key)
                if trace is None:
                    raise ValueError(f"sl state has no discover trace: {key}")
                _cross_check_trace_locator(trace, record)
                locators[key] = record
                capture_activity_instances.add(key[0])
                continue
            if tag == "sf":
                if len(record) != 8:
                    raise ValueError("invalid sf record")
                key = _state_key(record)
                if key[0] not in instance_ids:
                    raise ValueError("sf instance is absent from the raw plan part")
                if key[0] in capture_out_instances:
                    raise ValueError("sf record occurs after its capture out record")
                if key not in traces_by_key:
                    raise ValueError(f"sf state has no discover trace: {key}")
                if key in seen_failures:
                    raise ValueError(f"duplicate capture failure record: {key}")
                seen_failures.add(key)
                capture_activity_instances.add(key[0])
                if key in locators or key in proposals or key in evaluations or key in evaluation_failures:
                    raise ValueError(f"capture failure has state/action records: {key}")
                active_writer.append(
                    f"{key[0]}:{key[1]}",
                    {},
                    metadata={
                        "schema": SHARD_BUILD_SCHEMA,
                        "kind": "capture_failure",
                        "instance": key[0],
                        "eligible_state": key[1],
                        "raw_sf": record[2:],
                    },
                )
                failure_count += 1
                continue
            if tag != "st":
                raise ValueError(f"unexpected record in states part: {tag}")
            key = _state_key(record)
            if key[0] not in instance_ids:
                raise ValueError(f"state is absent from supplied raw plan part: {key[0]}")
            if key in seen_states:
                raise ValueError(f"duplicate state record: {key}")
            seen_states.add(key)
            capture_activity_instances.add(key[0])
            locator = locators.get(key)
            if locator is None:
                raise ValueError(f"state has no preceding locator: {key}")
            locator_fields = _locator_fields(locator)
            proposal = proposals.get(key)
            if proposal is None:
                raise ValueError(f"state has no proposal: {key}")
            state_evals = evaluations.get(key) or []
            state_eval_failures = evaluation_failures.get(key, [])
            expected_requests = {0, *forced_evaluation_literals(locator, proposal)}
            actual_requests: list[int] = []
            for evaluation in state_evals:
                _validate_eval_record(evaluation, key[0], key[1], evaluation[5])
                if evaluation[3] != locator[3] or evaluation[4] != locator[4]:
                    raise ValueError(f"eval callback/raw locator mismatch: {key}")
                if evaluation[10] != locator[11] or evaluation[11] != locator[12]:
                    raise ValueError(f"eval prefix/native locator mismatch: {key}")
                actual_requests.append(evaluation[5])
            actual_requests.extend(failure[3] for failure in state_eval_failures)
            if len(actual_requests) != len(set(actual_requests)):
                raise ValueError(f"duplicate eval request for state: {key}")
            if set(actual_requests) != expected_requests:
                raise ValueError(
                    f"eval request coverage mismatch for {key}: "
                    f"expected={sorted(expected_requests)} actual={sorted(actual_requests)}"
                )
            _cross_check_locator_state(locator, record)
            state_scalars = dict(zip(_ST_SCALAR_NAMES, record[1:22]))
            active_writer.append(
                f"{key[0]}:{key[1]}",
                _arrays_for_state(record, proposal, state_evals, state_eval_failures),
                metadata={
                    "schema": SHARD_BUILD_SCHEMA,
                    "kind": "state",
                    "state": state_scalars,
                    "capture_limits": {
                        name: locator_fields[name]
                        for name in (
                            "state_nodes",
                            "literal_occurrences",
                            "max_clause_length",
                            "estimated_payload_bytes",
                        )
                    },
                    "proposal_count": len(proposal[3]),
                    "evaluation_count": len(state_evals),
                    "evaluation_failure_count": len(state_eval_failures),
                },
            )
            state_count += 1
            evaluation_count += len(state_evals)
        unused_proposals = set(proposals) - seen_states
        unused_evaluations = set(evaluations) - seen_states
        unused_eval_failures = set(evaluation_failures) - seen_states
        unused_locators = set(locators) - seen_states
        if unused_proposals or unused_evaluations or unused_eval_failures or unused_locators:
            raise ValueError(
                f"raw association has orphan records: proposals={len(unused_proposals)} "
                f"evals={len(unused_evaluations)} eval_failures={len(unused_eval_failures)} "
                f"locators={len(unused_locators)}"
            )
        if capture_out_instances != capture_activity_instances:
            raise ValueError(
                "capture out records do not match instances with capture activity: "
                f"out={sorted(capture_out_instances)} "
                f"activity={sorted(capture_activity_instances)}"
            )
        diagnostics = _diagnostics_metadata(diagnostic_records)
        _validate_diagnostics_metadata(
            diagnostics,
            instance_ids=instance_ids,
            expected_capture_instances=capture_activity_instances,
        )
        if writer is None:
            active_writer.update_part_metadata({"raw_diagnostics": diagnostics})
        elif diagnostics_sink is not None:
            # Keep the per-part field declaration together with its rows until
            # the group writer can combine them.  The final merged metadata
            # uses the same diagnostics schema as a normal one-part shard.
            diagnostics_sink.append(
                {"source_part": part_stem, "diagnostics": diagnostics}
            )
    return ShardBuildResult(
        paths=CompactPartPaths(
            header=(Path(output_dir) if output_dir is not None else shard_dir)
            / f"{part_stem}.header.json",
            arrays=(Path(output_dir) if output_dir is not None else shard_dir)
            / f"{part_stem}.arrays.bin",
            complete=(Path(output_dir) if output_dir is not None else shard_dir)
            / f"{part_stem}.complete.json",
        ),
        state_count=state_count,
        capture_failure_count=failure_count,
        evaluation_count=evaluation_count,
    )


def build_shard_group(
    *,
    policy: RawPolicy,
    plan_parts: Sequence[Mapping[str, Any]],
    output_stem: str,
    output_root: str | Path | None = None,
    parts_per_shard: int | None = None,
    overwrite: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ShardBuildResult:
    """Stream several committed raw parts into one compact shard.

    Each input part keeps its normal raw validation and association checks.  A
    shared writer is opened only once, so no intermediate one-part shard is
    created.  Instance IDs must be unique across the group because compact
    record IDs are formed from ``instance_id:eligible_state``.
    """

    if not plan_parts:
        raise ValueError("a shard group must contain at least one raw part")
    validated_parts: list[dict[str, Any]] = []
    group_instances: set[str] = set()
    group_split: str | None = None
    dataset_groups: list[str] = []
    for index, plan_part in enumerate(plan_parts, start=1):
        validated = validate_raw_plan_part(
            plan_part,
            expected_policy_sha256=policy.digest,
            context=f"shard group part {index}",
        )
        part_split = raw_plan_part_split(
            validated, context=f"shard group part {index}"
        )
        if group_split is None:
            group_split = part_split
        elif part_split != group_split:
            raise ValueError(
                "one merged shard cannot contain multiple splits: "
                f"{group_split},{part_split}"
            )
        dataset_group = raw_plan_part_dataset_group(
            validated, context=f"shard group part {index}"
        )
        if dataset_group not in dataset_groups:
            dataset_groups.append(dataset_group)
        for instance in validated["instances"]:
            instance_id = instance["instance_id"]
            if instance_id in group_instances:
                raise ValueError(
                    f"duplicate instance_id across merged parts: {instance_id}"
                )
            group_instances.add(instance_id)
        validated_parts.append(validated)

    destination_root = (
        Path(output_root).expanduser().resolve(strict=False)
        if output_root is not None
        else policy.build_root
    )
    shard_dir = destination_root / "shards"
    source_parts: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    state_count = failure_count = evaluation_count = 0
    group_metadata = {
        "schema": SHARD_BUILD_SCHEMA,
        "group_stem": output_stem,
        "merge": {
            "parts_per_shard": parts_per_shard,
            "source_part_count": len(validated_parts),
        },
        "source_build_root": str(policy.build_root),
        "source_policy_id": policy.digest,
        "split": group_split,
        "dataset_groups": dataset_groups,
        "instance_fields": list(_SHARD_INSTANCE_FIELDS),
        "derived_value_labels": False,
    }
    if extra_metadata:
        group_metadata.update(dict(extra_metadata))
    with CompactArrayWriter(
        shard_dir,
        output_stem,
        overwrite=overwrite,
        part_metadata=group_metadata,
    ) as writer:
        for validated_part in validated_parts:
            result = build_shard_part(
                policy=policy,
                plan_part=validated_part,
                writer=writer,
                output_dir=shard_dir,
                part_metadata_sink=source_parts,
                diagnostics_sink=diagnostics,
            )
            state_count += result.state_count
            failure_count += result.capture_failure_count
            evaluation_count += result.evaluation_count
        merged_diagnostics = _merge_group_diagnostics(
            diagnostics, instance_ids=group_instances
        )
        writer.update_part_metadata(
            {
                "source_parts": source_parts,
                "raw_diagnostics": merged_diagnostics,
            }
        )
    return ShardBuildResult(
        paths=CompactPartPaths(
            header=shard_dir / f"{output_stem}.header.json",
            arrays=shard_dir / f"{output_stem}.arrays.bin",
            complete=shard_dir / f"{output_stem}.complete.json",
        ),
        state_count=state_count,
        capture_failure_count=failure_count,
        evaluation_count=evaluation_count,
    )


def _raw_marker_source_outputs(
    policy: RawPolicy, plan_part: Mapping[str, Any]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Read a raw completion marker for grouped-shard resume checks."""

    validated = validate_raw_plan_part(
        plan_part,
        expected_policy_sha256=policy.digest,
        context="merged shard resume plan part",
    )
    part_stem = validated["part_stem"]
    plan_digest = raw_plan_part_digest(validated)
    complete = policy.build_root / "statistics" / f"{part_stem}.complete.json"
    if not complete.is_file():
        raise FileNotFoundError(f"raw part completion marker is missing: {complete}")
    with complete.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    outputs = record.get("outputs") if isinstance(record, Mapping) else None
    is_v2 = isinstance(record, Mapping) and record.get("schema") == RAW_STATS_SCHEMA_V2
    policy_reference = (
        record.get("policy_id") if is_v2 else record.get("policy_sha256")
    ) if isinstance(record, Mapping) else None
    plan_matches = bool(
        isinstance(record, Mapping)
        and (is_v2 or record.get("plan_part_sha256") == plan_digest)
    )
    if (
        not isinstance(record, Mapping)
        or record.get("schema") not in {PIPELINE_STATS_SCHEMA, RAW_STATS_SCHEMA_V2}
        or record.get("status") != "complete"
        or record.get("part_stem") != part_stem
        or policy_reference != policy.digest
        or not plan_matches
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(RAW_STAGE_TAGS)
    ):
        raise ValueError(f"invalid raw source marker for merged shard: {complete}")
    return part_stem, _source_output_summary(record)


def _merge_group_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], *, instance_ids: set[str]
) -> dict[str, Any]:
    """Combine validated per-part diagnostics for a merged shard."""

    if not diagnostics:
        raise ValueError("merged shard has no per-part diagnostics")
    fields = {name: list(values) for name, values in _DIAGNOSTIC_FIELDS.items()}
    records = {name: [] for name in _DIAGNOSTIC_FIELDS}
    for item in diagnostics:
        source_part = item.get("source_part")
        value = item.get("diagnostics")
        if not isinstance(source_part, str) or not isinstance(value, Mapping):
            raise ValueError("invalid per-part diagnostics in merged shard")
        if value.get("schema") != PART_DIAGNOSTICS_SCHEMA or value.get("fields") != fields:
            raise ValueError(f"diagnostics schema mismatch for merged source part {source_part}")
        part_records = value.get("records")
        if not isinstance(part_records, Mapping):
            raise ValueError(f"diagnostics records are missing for {source_part}")
        for name in _DIAGNOSTIC_FIELDS:
            rows = part_records.get(name)
            if not isinstance(rows, list):
                raise ValueError(f"diagnostics rows are invalid for {source_part}: {name}")
            records[name].extend(rows)
    merged = {
        "schema": PART_DIAGNOSTICS_SCHEMA,
        "fields": fields,
        "records": records,
    }
    _validate_diagnostics_metadata(merged, instance_ids=instance_ids)
    return merged


def completed_shard_group_matches(
    policy: RawPolicy,
    plan_parts: Sequence[Mapping[str, Any]],
    complete_path: str | Path,
    *,
    parts_per_shard: int | None = None,
) -> bool:
    """Fully validate a grouped shard before resume is allowed to skip it."""

    if not plan_parts:
        raise ValueError("merged shard resume group is empty")
    expected: list[
        tuple[str, str, str, str, dict[str, dict[str, Any]]]
    ] = []
    instance_ids: set[str] = set()
    group_split: str | None = None
    dataset_groups: list[str] = []
    source_record_counts = {stage: 0 for stage in RAW_STAGE_TAGS}
    for index, plan_part in enumerate(plan_parts, start=1):
        validated = validate_raw_plan_part(
            plan_part,
            expected_policy_sha256=policy.digest,
            context=f"merged shard resume plan part {index}",
        )
        part_split = raw_plan_part_split(
            validated, context=f"merged shard resume plan part {index}"
        )
        if group_split is None:
            group_split = part_split
        elif part_split != group_split:
            raise ValueError("merged shard resume group crosses splits")
        dataset_group = raw_plan_part_dataset_group(
            validated, context=f"merged shard resume plan part {index}"
        )
        if dataset_group not in dataset_groups:
            dataset_groups.append(dataset_group)
        for item in validated["instances"]:
            instance_id = item["instance_id"]
            if instance_id in instance_ids:
                raise ValueError(
                    f"duplicate instance_id across merged resume parts: {instance_id}"
                )
            instance_ids.add(instance_id)
        part_stem, source_outputs = _raw_marker_source_outputs(policy, validated)
        for stage in RAW_STAGE_TAGS:
            count = source_outputs[stage].get("records")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"invalid committed raw record count for {stage}")
            source_record_counts[stage] += count
        expected.append(
            (
                part_stem,
                part_split,
                dataset_group,
                raw_plan_part_digest(validated),
                source_outputs,
            )
        )
    complete = Path(complete_path).expanduser().resolve(strict=True)
    suffix = ".complete.json"
    if not complete.name.endswith(suffix):
        raise ValueError("merged shard completion marker has an unexpected name")
    group_stem = complete.name[: -len(suffix)]
    with CompactArrayReader(complete, verify=False) as reader:
        metadata = reader.header.get("part_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("merged shard is missing part metadata")
        merge = metadata.get("merge")
        if (
            metadata.get("schema") != SHARD_BUILD_SCHEMA
            or metadata.get("group_stem") != group_stem
            or metadata.get("source_policy_id") != policy.digest
            or metadata.get("split") != group_split
            or metadata.get("dataset_groups") != dataset_groups
            or not isinstance(merge, Mapping)
            or merge.get("source_part_count") != len(expected)
            or merge.get("parts_per_shard") != parts_per_shard
        ):
            return False
        source_parts = metadata.get("source_parts")
        if not isinstance(source_parts, list) or len(source_parts) != len(expected):
            return False
        for actual, (
            stem,
            source_split,
            source_dataset_group,
            plan_digest,
            source_outputs,
        ) in zip(source_parts, expected):
            if (
                not isinstance(actual, Mapping)
                or actual.get("source_part") != stem
                or actual.get("source_split") != source_split
                or actual.get("source_dataset_group") != source_dataset_group
                or actual.get("source_policy_id") != policy.digest
                or actual.get("source_plan_part_id") != plan_digest
                or actual.get("source_raw_outputs") != source_outputs
            ):
                return False
        _validate_committed_shard_records(
            reader,
            instance_ids=instance_ids,
            source_record_counts=source_record_counts,
        )
    return True
