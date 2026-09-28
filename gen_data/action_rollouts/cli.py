from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .manifest import (
    ManifestLimits,
    build_manifest,
    enumerate_satbench,
    enumerate_satcomp,
)
from .pipeline import (
    PIPELINE_STATS_SCHEMA,
    WORKER_RAW_STAGE_TAGS,
    PipelineError,
    PipelinePolicy,
    execute_pipeline_part,
    load_pipeline_policy,
    pipeline_policy_from_record,
    read_pipeline_plan,
    write_pipeline_plan,
    write_pipeline_policy,
    write_worker_raw_schema,
)
from .schema import (
    DEFAULT_MAX_INPUT_NODES,
    DEFAULT_MAX_LITERAL_OCCURRENCES,
    DEFAULT_OUTPUT_TAG,
    DEFAULT_PART_SIZE,
    build_dir_name,
    canonical_json_bytes,
    parse_named_path,
    validate_output_tag,
)
from .satcomp_split import (
    SATCOMP_SPLITS,
    SATCOMP_STRATIFY_MODES,
    build_satcomp_split_assignment,
    load_satcomp_split_assignment,
    satcomp_source_expectations,
    select_satcomp_sources,
)


def _optional_positive(text: str) -> int | None:
    if text.lower() in {"none", "null", "off"}:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive integer or 'none'"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return value


def _nonnegative(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _positive_int(text: str) -> int:
    value = _nonnegative(text)
    if value == 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return value

def _nonnegative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return value


def _solver_setting(text: str) -> tuple[str, int]:
    name, separator, raw_value = text.partition("=")
    if not separator or not name or not raw_value:
        raise argparse.ArgumentTypeError("expected NAME=INTEGER")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "solver setting value must be an integer"
        ) from exc
    return name, value


def _named_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        try:
            name, path = parse_named_path(value)
        except ValueError as exc:
            raise ValueError(f"{option}: {exc}") from exc
        if name in result:
            raise ValueError(f"{option}: duplicate logical name {name!r}")
        result[name] = path
    if not result:
        raise ValueError(f"{option}: at least one NAME=PATH value is required")
    return result


def _manifest_limits(args: argparse.Namespace) -> ManifestLimits:
    return ManifestLimits(
        max_input_nodes=args.max_input_nodes,
        max_literal_occurrences=args.max_literal_occurrences,
        max_input_cells=args.max_input_cells,
        max_cnf_bytes=args.max_cnf_bytes,
        max_clause_length=args.max_clause_length,
    )


def _task_build_root(
    roots: Mapping[str, Path], output_path: str | None, output_tag: str
) -> Path:
    validate_output_tag(output_tag)
    if output_path is not None:
        return Path(output_path).expanduser().resolve(strict=False)
    if len(roots) != 1:
        raise ValueError(
            "--output-path is required when one invocation has multiple roots"
        )
    task_root = next(iter(roots.values())).resolve(strict=True)
    return task_root / build_dir_name(output_tag)


def _print_json(value: Any) -> None:
    print(canonical_json_bytes(value).decode("utf-8"), flush=True)


def _run_manifest_satbench(args: argparse.Namespace) -> None:
    split_roots = _named_paths(args.split, option="--split")
    sources = enumerate_satbench(
        split_roots,
        difficulty=args.difficulty,
        family=args.family,
    )
    build_root = _task_build_root(split_roots, args.output_path, args.output_tag)
    result = build_manifest(
        sources,
        output_root=build_root / "manifests",
        output_tag=args.output_tag,
        dataset_release_id=args.dataset_release_id,
        limits=_manifest_limits(args),
        part_size=args.part_size,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "manifest": str(result.manifest_path),
            "metadata": str(result.metadata_path),
            "build_root": str(build_root),
            "discovered": result.discovered_count,
            "accepted": result.accepted_count,
            "rejected": result.rejected_count,
        }
    )


def _run_manifest_satcomp(args: argparse.Namespace) -> None:
    collection_roots = _named_paths(args.collection, option="--collection")
    sources = enumerate_satcomp(
        collection_roots,
        cnf_subdir=args.cnf_subdir,
        track=args.track,
    )
    provenance = None
    expected_content = None
    if bool(args.split_assignment) != bool(args.selected_split):
        raise ValueError(
            "--split-assignment and --selected-split must be supplied together"
        )
    if args.split_assignment:
        assignment = load_satcomp_split_assignment(args.split_assignment)
        sources = select_satcomp_sources(
            sources, assignment, args.selected_split, validate_content=False
        )
        expected_content = satcomp_source_expectations(sources, assignment)
        provenance = {
            "satcomp_split_assignment": str(assignment.path),
            "satcomp_split_assignment_sha256": assignment.sha256,
            "satcomp_selected_split": args.selected_split,
        }
    build_root = _task_build_root(
        collection_roots, args.output_path, args.output_tag
    )
    result = build_manifest(
        sources,
        output_root=build_root / "manifests",
        output_tag=args.output_tag,
        dataset_release_id=args.dataset_release_id,
        limits=_manifest_limits(args),
        part_size=args.part_size,
        provenance=provenance,
        workers=args.workers,
        expected_content=expected_content,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "manifest": str(result.manifest_path),
            "metadata": str(result.metadata_path),
            "build_root": str(build_root),
            "discovered": result.discovered_count,
            "accepted": result.accepted_count,
            "rejected": result.rejected_count,
        }
    )


def _run_split_satcomp(args: argparse.Namespace) -> None:
    collection_roots = _named_paths(args.collection, option="--collection")
    sources = enumerate_satcomp(
        collection_roots,
        cnf_subdir=args.cnf_subdir,
        track=args.track,
    )
    result = build_satcomp_split_assignment(
        sources,
        output_path=args.output,
        seed=args.seed,
        split_ratio=args.split_ratio,
        stratify=args.stratify,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "assignment": str(result.path),
            "assignment_sha256": result.sha256,
            "sources": result.source_count,
            "valid_sources": result.valid_source_count,
            "invalid_sources": result.invalid_source_count,
            "generation_sources": result.generation_source_count,
            "source_instance_groups": result.group_count,
            "split_group_counts": result.split_group_counts,
            "split_source_counts": result.split_source_counts,
            "split_generation_source_counts": result.split_generation_source_counts,
        }
    )


def _pipeline_policy_from_args(args: argparse.Namespace) -> PipelinePolicy:
    if args.preset == "satbench":
        target_defaults = (2, 8, 2, 2)
        eval_defaults = (0, 0, 0)
        timeout_default = 300.0
    else:
        target_defaults = (1, 5, 1, 1)
        # SATCompetition has a much heavier tail.  The default fixed horizon
        # makes candidate rollouts comparable without waiting for all solves.
        eval_defaults = (20_000, 0, 50_000_000)
        timeout_default = 900.0

    def chosen(value: int | None, default: int) -> int:
        return default if value is None else value

    return PipelinePolicy(
        policy_id=uuid.uuid4().hex,
        output_tag=args.output_tag,
        binary=Path(args.binary),
        build_root=Path(args.build_root),
        manifest=Path(args.manifest),
        cnf_cache_root=(Path(args.cnf_cache_root) if args.cnf_cache_root else None),
        target_initial_keep_count=chosen(
            args.target_initial_keep_count, target_defaults[0]
        ),
        target_early_window_end=chosen(
            args.target_early_window_end, target_defaults[1]
        ),
        target_early_sample_count=chosen(
            args.target_early_sample_count, target_defaults[2]
        ),
        target_post_restart_keep_count=chosen(
            args.target_post_restart_keep_count, target_defaults[3]
        ),
        target_late_fallback_window_end=args.target_late_fallback_window_end,
        max_state_nodes=args.max_state_nodes,
        max_literal_occurrences=args.max_state_literal_occurrences,
        max_snapshot_bytes=args.max_snapshot_bytes,
        max_clause_length=args.max_state_clause_length,
        max_prefix_conflicts=args.max_prefix_conflicts,
        max_prefix_decisions=args.max_prefix_decisions,
        max_prefix_propagations=args.max_prefix_propagations,
        max_prefix_callbacks=args.max_prefix_callbacks,
        activity_top_k=args.activity_top_k,
        jw_top_k=args.jw_top_k,
        random_top_k=args.random_top_k,
        max_actions=args.max_actions,
        max_actions_mode=args.max_actions_mode,
        eval_max_conflicts=chosen(args.eval_max_conflicts, eval_defaults[0]),
        eval_max_decisions=chosen(args.eval_max_decisions, eval_defaults[1]),
        eval_max_propagations=chosen(
            args.eval_max_propagations, eval_defaults[2]
        ),
        eval_max_timeout=args.eval_max_timeout,
        seed=args.seed,
        plain=args.plain,
        solver_settings=tuple(args.solver_set or ()),
        timeout_seconds=chosen(args.timeout_seconds, timeout_default),
        raw_codec=args.raw_codec,
    )


def _run_pipeline_prepare(args: argparse.Namespace) -> None:
    policy = _pipeline_policy_from_args(args)
    policy.check_paths()
    manifest_parent = policy.build_root / "manifests"
    if policy.manifest.parent != manifest_parent:
        raise ValueError(
            "pipeline manifest must be inside the exact build root: "
            f"expected parent {manifest_parent}, got {policy.manifest.parent}"
        )
    policy_path = (
        Path(args.policy_path).expanduser().resolve(strict=False)
        if args.policy_path
        else manifest_parent / "pipeline.policy.json"
    )
    plan_path = (
        Path(args.plan_path).expanduser().resolve(strict=False)
        if args.plan_path
        else manifest_parent / "pipeline.parts.jsonl"
    )
    schema_path = manifest_parent / "worker.raw.schema.json"
    if not args.overwrite:
        for path in (policy_path, plan_path, schema_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite pipeline output: {path}"
                )
    write_worker_raw_schema(schema_path, overwrite=args.overwrite)
    write_pipeline_policy(policy_path, policy, overwrite=args.overwrite)
    result = write_pipeline_plan(policy, plan_path, overwrite=args.overwrite)
    _print_json(
        {
            "policy": str(policy_path),
            "policy_id": policy.policy_id,
            "plan": str(result.path),
            "raw_schema": str(schema_path),
            "parts": result.part_count,
            "instances": result.instance_count,
            "manifest_skips": result.skipped_count,
        }
    )


def _find_pipeline_part(plan_path: str | Path, part_stem: str) -> dict[str, Any]:
    found: dict[str, Any] | None = None
    for part in read_pipeline_plan(plan_path):
        if part.get("part_stem") == part_stem:
            if found is not None:
                raise ValueError(
                    f"duplicate part_stem in pipeline plan: {part_stem}"
                )
            found = part
    if found is None:
        raise ValueError(f"part_stem not found in pipeline plan: {part_stem}")
    return found


def _part_completion_path(policy: PipelinePolicy, part_stem: str) -> Path:
    return policy.build_root / "statistics" / f"{part_stem}.complete.json"


def _completed_part_matches(
    policy: PipelinePolicy, part: Mapping[str, Any]
) -> bool:
    part_stem = part.get("part_stem")
    if not isinstance(part_stem, str):
        raise ValueError("pipeline plan part has invalid part_stem")
    path = _part_completion_path(policy, part_stem)
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != PIPELINE_STATS_SCHEMA
        or record.get("status") != "complete"
        or record.get("part_stem") != part_stem
        or record.get("policy_id") != policy.policy_id
        or record.get("part_stem") != part.get("part_stem")
        or not isinstance(record.get("generation_id"), str)
    ):
        raise ValueError(
            f"existing completion marker does not match policy/plan: {path}"
        )
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(WORKER_RAW_STAGE_TAGS):
        raise ValueError(f"existing completion marker has incomplete outputs: {path}")
    suffix = ".jsonl.gz" if policy.raw_codec == "gzip" else ".jsonl"
    for stage in WORKER_RAW_STAGE_TAGS:
        descriptor = outputs.get(stage)
        expected = policy.build_root / stage / f"{part_stem}{suffix}"
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"invalid {stage} output descriptor in {path}")
        byte_count = descriptor.get("bytes")
        descriptor_path = descriptor.get("path")
        if (
            descriptor.get("codec") != policy.raw_codec
            or not isinstance(descriptor_path, str)
            or Path(descriptor_path).expanduser().resolve(strict=False) != expected
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError(f"invalid {stage} output descriptor in {path}")
        if not expected.is_file() or expected.stat().st_size != byte_count:
            raise ValueError(f"completed raw output is missing or has wrong size: {expected}")
    return True


def _pipeline_part_worker(
    policy_record: Mapping[str, Any], part: Mapping[str, Any], overwrite: bool
) -> dict[str, Any]:
    policy = pipeline_policy_from_record(policy_record)
    result = execute_pipeline_part(policy, part, overwrite=overwrite)
    return {
        "part_stem": result.part_stem,
        "instances": result.instances_completed,
        "states": result.states_materialized,
        "evaluations": result.evaluations,
        "statistics": str(result.statistics_path),
    }


def _select_pipeline_task(
    parts: Iterable[dict[str, Any]], *, task_count: int, task_index: int
) -> Iterator[dict[str, Any]]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if task_index < 0 or task_index >= task_count:
        raise ValueError(
            f"task_index must be in [0, {task_count}), got {task_index}"
        )
    for ordinal, part in enumerate(parts):
        if ordinal % task_count == task_index:
            yield part


def _run_pipeline_part(args: argparse.Namespace) -> None:
    policy = load_pipeline_policy(args.policy)
    part = _find_pipeline_part(args.plan, args.part_stem)
    if not args.overwrite and _completed_part_matches(policy, part):
        _print_json({"part_stem": args.part_stem, "status": "skipped_complete"})
        return
    _print_json(_pipeline_part_worker(policy.as_record(), part, args.overwrite))


def _run_pipeline_all(args: argparse.Namespace) -> int:
    if args.task_index >= args.task_count:
        raise ValueError(
            f"--task-index must be smaller than --task-count "
            f"({args.task_index} >= {args.task_count})"
        )
    policy = load_pipeline_policy(args.policy)
    policy_record = policy.as_record()
    pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
    failures = completed = skipped = selected = 0
    iterator = iter(
        _select_pipeline_task(
            read_pipeline_plan(args.plan),
            task_count=args.task_count,
            task_index=args.task_index,
        )
    )
    exhausted = False

    def submit_more(executor: concurrent.futures.ProcessPoolExecutor) -> None:
        nonlocal exhausted, selected, skipped
        while not exhausted and len(pending) < max(args.workers, 2 * args.workers):
            try:
                part = next(iterator)
            except StopIteration:
                exhausted = True
                return
            selected += 1
            stem = part.get("part_stem")
            if not isinstance(stem, str):
                raise ValueError("pipeline plan part has invalid part_stem")
            if not args.overwrite and _completed_part_matches(policy, part):
                skipped += 1
                _print_json({"part_stem": stem, "status": "skipped_complete"})
                continue
            future = executor.submit(
                _pipeline_part_worker, policy_record, part, args.overwrite
            )
            pending[future] = stem

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        submit_more(executor)
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                stem = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    failures += 1
                    _print_json(
                        {
                            "part_stem": stem,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    if args.fail_fast:
                        for other in pending:
                            other.cancel()
                        _print_json(
                            {
                                "summary": "raw",
                                "task_count": args.task_count,
                                "task_index": args.task_index,
                                "selected_parts": selected,
                                "completed_parts": completed,
                                "skipped_parts": skipped,
                                "failed_parts": failures,
                            }
                        )
                        return 1
                else:
                    completed += 1
                    _print_json({"status": "complete", **result})
            submit_more(executor)
    _print_json(
        {
            "summary": "raw",
            "task_count": args.task_count,
            "task_index": args.task_index,
            "selected_parts": selected,
            "completed_parts": completed,
            "skipped_parts": skipped,
            "failed_parts": failures,
        }
    )
    return 1 if failures else 0


def _add_manifest_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-release-id", required=True)
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument(
        "--output-path",
        help=(
            "exact task build root; default is "
            "<input-root>/DecisionTrace-ActionEval-v1.3-<tag>"
        ),
    )
    parser.add_argument("--part-size", type=_positive_int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--max-input-nodes",
        type=_optional_positive,
        default=DEFAULT_MAX_INPUT_NODES,
        help="2*V+C ceiling (default: 100000; use 'none' to disable)",
    )
    parser.add_argument(
        "--max-literal-occurrences",
        type=_optional_positive,
        default=DEFAULT_MAX_LITERAL_OCCURRENCES,
        help="input literal-occurrence ceiling (use 'none' to disable)",
    )
    parser.add_argument("--max-input-cells", type=_optional_positive)
    parser.add_argument("--max-cnf-bytes", type=_optional_positive)
    parser.add_argument("--max-clause-length", type=_optional_positive)
    parser.add_argument("--overwrite", action="store_true")


def _add_pipeline_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=("satbench", "satcomp"), required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--cnf-cache-root",
        help=(
            "copy each current CNF once into this local directory and reuse "
            "it for DISCOVER, CAPTURE, and EVAL"
        ),
    )
    parser.add_argument("--policy-path")
    parser.add_argument("--plan-path")
    parser.add_argument("--target-initial-keep-count", type=_nonnegative)
    parser.add_argument("--target-early-window-end", type=_nonnegative)
    parser.add_argument("--target-early-sample-count", type=_nonnegative)
    parser.add_argument("--target-post-restart-keep-count", type=_nonnegative)
    parser.add_argument(
        "--target-late-fallback-window-end", type=_nonnegative, default=20
    )
    parser.add_argument("--max-state-nodes", type=_nonnegative, default=100_000)
    parser.add_argument(
        "--max-state-literal-occurrences", type=_nonnegative, default=4_000_000
    )
    parser.add_argument(
        "--max-snapshot-bytes", type=_nonnegative, default=256 * 1024 * 1024
    )
    parser.add_argument("--max-state-clause-length", type=_nonnegative, default=0)
    parser.add_argument("--max-prefix-conflicts", type=_nonnegative, default=10_000)
    parser.add_argument("--max-prefix-decisions", type=_nonnegative, default=0)
    parser.add_argument(
        "--max-prefix-propagations", type=_nonnegative, default=10_000_000
    )
    parser.add_argument("--max-prefix-callbacks", type=_nonnegative, default=1_000)
    parser.add_argument(
        "--activity-top-k",
        type=_nonnegative,
        default=2,
        help="top activity variables; each may propose both signed literals",
    )
    parser.add_argument(
        "--jw-top-k",
        type=_nonnegative,
        default=1,
        help="top signed Jeroslow-Wang literals",
    )
    parser.add_argument(
        "--random-top-k",
        type=_nonnegative,
        default=1,
        help="deterministically sampled signed literals",
    )
    parser.add_argument(
        "--max-actions",
        type=_nonnegative,
        default=8,
        help=(
            "maximum distinct nonzero signed literals or variables proposed "
            "per state, according to --max-actions-mode; the requested=0 "
            "native baseline is additional"
        ),
    )
    parser.add_argument(
        "--max-actions-mode",
        choices=("default", "variable"),
        default="default",
        help=(
            "default caps distinct signed literals; variable caps distinct "
            "variables and proposes every available polarity"
        ),
    )
    parser.add_argument("--eval-max-conflicts", type=_nonnegative)
    parser.add_argument("--eval-max-decisions", type=_nonnegative)
    parser.add_argument("--eval-max-propagations", type=_nonnegative)
    parser.add_argument("--eval-max-timeout", type=_nonnegative_float, default=0.0)
    parser.add_argument("--seed", type=_nonnegative, default=1)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="disable CaDiCaL preprocessing for every worker process",
    )
    parser.add_argument(
        "--solver-set",
        action="append",
        type=_solver_setting,
        metavar="NAME=INTEGER",
    )
    parser.add_argument("--timeout-seconds", type=_positive_float)
    parser.add_argument("--raw-codec", choices=("gzip", "none"), default="gzip")
    parser.add_argument("--overwrite", action="store_true")


def _add_pipeline_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gen_data.action_rollouts",
        description=(
            "Raw-only DecisionTrace manifest and DISCOVER/CAPTURE/EVAL runner"
        ),
    )
    commands = parser.add_subparsers(dest="subcommand", required=True)

    satbench = commands.add_parser(
        "manifest-satbench",
        help="scan direct CNFs under each explicit SPLIT/{sat,unsat}",
    )
    satbench.add_argument("--split", action="append", required=True, metavar="NAME=PATH")
    satbench.add_argument("--difficulty", required=True)
    satbench.add_argument("--family", required=True)
    _add_manifest_common(satbench)
    satbench.set_defaults(handler=_run_manifest_satbench)

    satcomp = commands.add_parser(
        "manifest-satcomp",
        help="scan direct CNFs under each explicit COLLECTION/cnf",
    )
    satcomp.add_argument(
        "--collection", action="append", required=True, metavar="YEAR=PATH"
    )
    satcomp.add_argument("--cnf-subdir", default="cnf")
    satcomp.add_argument("--track", default="main")
    satcomp.add_argument("--split-assignment")
    satcomp.add_argument("--selected-split", choices=SATCOMP_SPLITS)
    _add_manifest_common(satcomp)
    satcomp.set_defaults(handler=_run_manifest_satcomp)

    split_satcomp = commands.add_parser(
        "split-satcomp",
        help="freeze a leakage-safe train/valid/test assignment by CNF content group",
    )
    split_satcomp.add_argument(
        "--collection", action="append", required=True, metavar="YEAR=PATH"
    )
    split_satcomp.add_argument("--cnf-subdir", default="cnf")
    split_satcomp.add_argument("--track", default="main")
    split_satcomp.add_argument("--output", required=True)
    split_satcomp.add_argument("--seed", type=int, default=42)
    split_satcomp.add_argument(
        "--split-ratio", nargs=3, default=("0.8", "0.1", "0.1"),
        metavar=("TRAIN", "VALID", "TEST"),
    )
    split_satcomp.add_argument(
        "--stratify", choices=SATCOMP_STRATIFY_MODES, default="year-size"
    )
    split_satcomp.add_argument("--overwrite", action="store_true")
    split_satcomp.set_defaults(handler=_run_split_satcomp)

    prepare = commands.add_parser(
        "pipeline-prepare",
        help="write policy, raw record schema, and a single-pass part plan",
    )
    _add_pipeline_prepare_arguments(prepare)
    prepare.set_defaults(handler=_run_pipeline_prepare)

    run_part = commands.add_parser(
        "pipeline-run-part",
        help="run DISCOVER, CAPTURE, and EVAL for one planned part",
    )
    _add_pipeline_execution_arguments(run_part)
    run_part.add_argument("--part-stem", required=True)
    run_part.set_defaults(handler=_run_pipeline_part)

    run_all = commands.add_parser(
        "pipeline-run-all",
        help="run every planned raw part with bounded process parallelism",
    )
    _add_pipeline_execution_arguments(run_all)
    run_all.add_argument(
        "--workers",
        type=_positive_int,
        default=min(4, max(1, os.cpu_count() or 1)),
    )
    run_all.add_argument(
        "--task-count",
        type=_positive_int,
        default=1,
        help="number of stable round-robin tasks covering the full plan",
    )
    run_all.add_argument(
        "--task-index",
        type=_nonnegative,
        default=0,
        help="zero-based task to run; must be smaller than --task-count",
    )
    run_all.add_argument("--fail-fast", action="store_true")
    run_all.set_defaults(handler=_run_pipeline_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, ValueError, TypeError, PipelineError) as exc:
        parser.error(str(exc))
    return result if isinstance(result, int) else 0
