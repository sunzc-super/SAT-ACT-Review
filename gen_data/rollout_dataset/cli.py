"""Command line interface for DecisionTrace ActionEval shard metadata v2."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .builder import build_shard_group
from .compact import CompactArrayReader
from .contract import (
    PIPELINE_STATS_SCHEMA,
    RAW_STAGE_TAGS,
    RAW_STATS_SCHEMA_V2,
    RawPolicy,
    iter_raw_plan,
    load_raw_policy,
    raw_plan_part_dataset_group,
    raw_plan_part_split,
    raw_suffix,
)
from .legacy import convert_legacy
from .schema import SHARD_PLAN_SCHEMA, SHARD_POLICY_SCHEMA, canonical_json_bytes, utc_now_text


_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite metadata: {path}")
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


def _output_root(source_root: Path, output_root: str | None, output_tag: str | None) -> Path:
    if output_root and output_tag:
        raise ValueError("--output-root and --output-tag are mutually exclusive")
    if output_root:
        return Path(output_root).expanduser().resolve(strict=False)
    if not output_tag:
        return source_root
    if not _TAG_RE.fullmatch(output_tag):
        raise ValueError("--output-tag contains unsafe characters")
    if "-" not in source_root.name:
        raise ValueError("cannot replace tag in source directory name; use --output-root")
    prefix, _old_tag = source_root.name.rsplit("-", 1)
    return source_root.parent / f"{prefix}-{output_tag}"


def _completion_generation(policy: RawPolicy, part: Mapping[str, Any]) -> str:
    stem = part["part_stem"]
    path = policy.build_root / "statistics" / f"{stem}.complete.json"
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, Mapping):
        raise ValueError(f"invalid raw completion marker: {path}")
    is_v2 = record.get("schema") == RAW_STATS_SCHEMA_V2
    policy_reference = record.get("policy_id") if is_v2 else record.get("policy_sha256")
    if (
        record.get("schema") not in {PIPELINE_STATS_SCHEMA, RAW_STATS_SCHEMA_V2}
        or record.get("status") != "complete"
        or record.get("part_stem") != stem
        or policy_reference != policy.digest
    ):
        raise ValueError(f"raw completion does not match policy/part: {path}")
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(RAW_STAGE_TAGS):
        raise ValueError(f"raw completion has incomplete outputs: {path}")
    suffix = raw_suffix(policy.raw_codec)
    for stage in RAW_STAGE_TAGS:
        descriptor = outputs[stage]
        expected = policy.build_root / stage / f"{stem}{suffix}"
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("codec") != policy.raw_codec
            or not isinstance(descriptor.get("bytes"), int)
            or not isinstance(descriptor.get("records"), int)
            or not expected.is_file()
            or expected.stat().st_size != descriptor["bytes"]
        ):
            raise ValueError(f"raw output is missing or has wrong size: {expected}")
    generation = record.get("generation_id", record.get("created_at"))
    if not isinstance(generation, str) or not generation:
        raise ValueError(f"raw completion has no generation identity: {path}")
    return generation


def _group_parts(parts: Sequence[Mapping[str, Any]], parts_per_shard: int) -> list[list[Mapping[str, Any]]]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    group_order: list[str] = []
    for part in parts:
        dataset_group = raw_plan_part_dataset_group(part)
        if dataset_group not in by_group:
            by_group[dataset_group] = []
            group_order.append(dataset_group)
        by_group[dataset_group].append(part)
    groups: list[list[Mapping[str, Any]]] = []
    for dataset_group in group_order:
        values = by_group[dataset_group]
        groups.extend(
            [values[start : start + parts_per_shard] for start in range(0, len(values), parts_per_shard)]
        )
    return groups


def _plan_rows(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    generations: Mapping[str, str],
    *,
    shard_build_id: str,
    parts_per_shard: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        dataset_groups = {raw_plan_part_dataset_group(part) for part in group}
        splits = {raw_plan_part_split(part) for part in group}
        if len(dataset_groups) != 1 or len(splits) != 1:
            raise ValueError("one shard group cannot cross dataset_group or split")
        source_parts = [part["part_stem"] for part in group]
        instance_ids = [item["instance_id"] for part in group for item in part["instances"]]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("duplicate instance_id across source parts")
        rows.append(
            {
                "schema": SHARD_PLAN_SCHEMA,
                "shard_build_id": shard_build_id,
                "shard_index": index,
                "shard_stem": f"shard-{index:05d}",
                "dataset_group": next(iter(dataset_groups)),
                "split": next(iter(splits)),
                "parts_per_shard": parts_per_shard,
                "source_part_count": len(group),
                "source_parts": source_parts,
                "source_generations": [generations[stem] for stem in source_parts],
                "instance_count": len(instance_ids),
                "instance_ids": instance_ids,
            }
        )
    return rows


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_plan(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != SHARD_PLAN_SCHEMA:
                raise ValueError(f"invalid shard plan row at line {number}")
            rows.append(row)
    return rows


def _same_plan_shape(existing: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> bool:
    ignored = {"shard_build_id"}
    strip = lambda row: {key: value for key, value in row.items() if key not in ignored}
    return [strip(row) for row in existing] == [strip(row) for row in expected]


def _completed_matches(output_root: Path, row: Mapping[str, Any]) -> bool:
    complete = output_root / "shards" / f"{row['shard_stem']}.complete.json"
    if not complete.exists():
        return False
    with CompactArrayReader(complete, verify=False) as reader:
        metadata = reader.header.get("part_metadata")
        if not isinstance(metadata, Mapping):
            return False
        merge = metadata.get("merge")
        source_parts = metadata.get("source_parts")
        if (
            metadata.get("group_stem") != row["shard_stem"]
            or metadata.get("shard_build_id") != row["shard_build_id"]
            or metadata.get("dataset_groups") != [row["dataset_group"]]
            or not isinstance(merge, Mapping)
            or merge.get("parts_per_shard") != row["parts_per_shard"]
            or merge.get("source_part_count") != row["source_part_count"]
            or not isinstance(source_parts, list)
            or [item.get("source_part") for item in source_parts] != row["source_parts"]
            or [item.get("source_generation_id") for item in source_parts]
            != row["source_generations"]
        ):
            return False
    return True


def _build_one(
    policy_record: Mapping[str, Any],
    plan_parts: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    output_root: str,
    overwrite: bool,
) -> dict[str, Any]:
    from .contract import load_raw_policy

    policy = load_raw_policy(policy_record["path"])
    metadata = {
        "shard_build_id": row["shard_build_id"],
    }
    # build_shard_group owns the compact writer; this field is injected through
    # a copied policy record in its group metadata immediately before writing.
    result = build_shard_group(
        policy=policy,
        plan_parts=plan_parts,
        output_stem=row["shard_stem"],
        output_root=output_root,
        parts_per_shard=row["parts_per_shard"],
        overwrite=overwrite,
        extra_metadata=metadata,
    )
    return {
        "shard_stem": row["shard_stem"],
        "states": result.state_count,
        "capture_failures": result.capture_failure_count,
        "evaluations": result.evaluation_count,
    }


def _run_build(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).expanduser().resolve(strict=True)
    policy_path = source_root / "manifests" / "pipeline.policy.json"
    raw_plan_path = source_root / "manifests" / "pipeline.parts.jsonl"
    policy = load_raw_policy(policy_path)
    if policy.build_root != source_root:
        raise ValueError("raw policy build_root does not match --source-root")
    parts = list(iter_raw_plan(raw_plan_path, expected_policy_sha256=policy.digest))
    if not parts:
        raise ValueError("raw part plan is empty")
    generations = {part["part_stem"]: _completion_generation(policy, part) for part in parts}
    groups = _group_parts(parts, args.parts_per_shard)
    destination = _output_root(source_root, args.output_root, args.output_tag)
    metadata_dir = destination / "manifests" / "shards"
    policy_out = metadata_dir / "policy.json"
    plan_out = metadata_dir / "plan.jsonl"

    build_id = uuid.uuid4().hex
    provisional = _plan_rows(
        groups,
        generations,
        shard_build_id=build_id,
        parts_per_shard=args.parts_per_shard,
    )
    if policy_out.exists() or plan_out.exists():
        if not policy_out.exists() or not plan_out.exists():
            raise ValueError("shard metadata is incomplete; both policy.json and plan.jsonl are required")
        existing_policy = _read_json(policy_out)
        existing_plan = _read_plan(plan_out)
        old_id = existing_policy.get("shard_build_id") if isinstance(existing_policy, Mapping) else None
        reusable = (
            isinstance(old_id, str)
            and existing_policy.get("source_root") == str(source_root)
            and existing_policy.get("output_root") == str(destination)
            and existing_policy.get("parts_per_shard") == args.parts_per_shard
        )
        if reusable:
            provisional = _plan_rows(
                groups,
                generations,
                shard_build_id=old_id,
                parts_per_shard=args.parts_per_shard,
            )
            reusable = _same_plan_shape(existing_plan, provisional)
        if reusable and not args.overwrite:
            build_id = old_id
        elif not args.overwrite:
            raise ValueError("existing shard plan uses different grouping/source data; use --overwrite")

    rows = _plan_rows(
        groups,
        generations,
        shard_build_id=build_id,
        parts_per_shard=args.parts_per_shard,
    )
    shard_policy = {
        "schema": SHARD_POLICY_SCHEMA,
        "shard_build_id": build_id,
        "created_at": utc_now_text(),
        "source_root": str(source_root),
        "source_tag": policy.record.get("output_tag", source_root.name.rsplit("-", 1)[-1]),
        "source_policy_id": policy.digest,
        "source_policy_version": policy.version,
        "output_root": str(destination),
        "output_tag": args.output_tag or policy.record.get("output_tag", destination.name.rsplit("-", 1)[-1]),
        "parts_per_shard": args.parts_per_shard,
        "source_part_count": len(parts),
        "shard_count": len(rows),
        "payload_binding": "shard-plan-v2",
    }
    plan_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    replacing = policy_out.exists() or plan_out.exists()
    _atomic_write(policy_out, canonical_json_bytes(shard_policy) + b"\n", overwrite=replacing)
    _atomic_write(plan_out, plan_payload, overwrite=replacing)

    by_stem = {part["part_stem"]: part for part in parts}
    completed = skipped = failed = 0
    pending: list[tuple[list[Mapping[str, Any]], Mapping[str, Any]]] = []
    for row in rows:
        if not args.overwrite and _completed_matches(destination, row):
            skipped += 1
            _print_json({"status": "skipped_complete", "shard_stem": row["shard_stem"]})
        else:
            pending.append(([by_stem[stem] for stem in row["source_parts"]], row))
    policy_record = {"path": str(policy.path)}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_rows = {
            executor.submit(
                _build_one,
                policy_record,
                group,
                row,
                str(destination),
                args.overwrite or (destination / "shards" / f"{row['shard_stem']}.complete.json").exists(),
            ): row
            for group, row in pending
        }
        for future in concurrent.futures.as_completed(future_rows):
            row = future_rows[future]
            try:
                result = future.result()
            except BaseException as exc:
                failed += 1
                _print_json({"status": "failed", "shard_stem": row["shard_stem"], "error": str(exc)})
                if args.fail_fast:
                    for other in future_rows:
                        other.cancel()
                    break
            else:
                completed += 1
                _print_json({"status": "complete", **result})
    _print_json({"summary": "shards", "completed": completed, "skipped": skipped, "failed": failed})
    return 1 if failed else 0


def _run_convert(args: argparse.Namespace) -> int:
    result = convert_legacy(args.root, overwrite=args.overwrite, dry_run=args.dry_run)
    _print_json(result)
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve(strict=True)
    policy = _read_json(root / "manifests" / "shards" / "policy.json")
    rows = _read_plan(root / "manifests" / "shards" / "plan.jsonl")
    complete = sum((root / "shards" / f"{row['shard_stem']}.complete.json").is_file() for row in rows)
    _print_json({"root": str(root), "policy": policy, "planned_shards": len(rows), "complete_shards": complete})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gen_data.rollout_dataset")
    commands = parser.add_subparsers(dest="subcommand", required=True)

    build = commands.add_parser("build-shards", help="build one complete shard grouping version")
    build.add_argument("--source-root", required=True)
    build.add_argument("--parts-per-shard", type=_positive_integer, default=1)
    build.add_argument("--output-root")
    build.add_argument("--output-tag")
    build.add_argument("--workers", type=_positive_integer, default=1)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--fail-fast", action="store_true")
    build.set_defaults(handler=_run_build)

    convert = commands.add_parser("convert-legacy", help="add v2 metadata to legacy shards")
    convert.add_argument("--root", required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--dry-run", action="store_true")
    convert.set_defaults(handler=_run_convert)

    inspect = commands.add_parser("inspect-shards", help="summarize one v2 shard version")
    inspect.add_argument("--root", required=True)
    inspect.set_defaults(handler=_run_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2
