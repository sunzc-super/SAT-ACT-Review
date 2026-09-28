from __future__ import annotations

import concurrent.futures
import hashlib
import json
import lzma
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Iterator, Mapping, Sequence

from .dimacs import DimacsError, DimacsInfo, identity_filename, inspect_dimacs, is_cnf_path
from .schema import (
    DEFAULT_MAX_LITERAL_OCCURRENCES,
    DEFAULT_MAX_INPUT_NODES,
    DEFAULT_PART_SIZE,
    MANIFEST_META_SCHEMA,
    MANIFEST_SCHEMA,
    canonical_json_bytes,
    positive_limit,
    reject_output_source_overlap,
    stable_digest,
    utc_now_text,
    validate_output_tag,
)


@dataclass(frozen=True)
class ManifestLimits:
    max_input_nodes: int | None = DEFAULT_MAX_INPUT_NODES
    max_literal_occurrences: int | None = DEFAULT_MAX_LITERAL_OCCURRENCES
    max_input_cells: int | None = None
    max_cnf_bytes: int | None = None
    max_clause_length: int | None = None

    def validate(self) -> None:
        positive_limit(self.max_input_nodes, "max_input_nodes")
        positive_limit(self.max_literal_occurrences, "max_literal_occurrences")
        positive_limit(self.max_input_cells, "max_input_cells")
        positive_limit(self.max_cnf_bytes, "max_cnf_bytes")
        positive_limit(self.max_clause_length, "max_clause_length")


@dataclass(frozen=True)
class SourceFile:
    dataset: str
    dataset_group: str
    split: str
    class_label: str | None
    source_root: Path
    input_root: Path
    path: Path
    source_relpath: str
    identity_relpath: str
    difficulty: str | None = None
    family: str | None = None
    competition_year: str | None = None
    track: str | None = None
    collection_id: str | None = None
    cnf_subdir: str | None = None


@dataclass(frozen=True)
class ManifestBuildResult:
    output_root: Path
    manifest_path: Path
    metadata_path: Path
    discovered_count: int
    accepted_count: int
    rejected_count: int


def _inspect_source(path: Path) -> tuple[DimacsInfo | None, str | None]:
    try:
        return inspect_dimacs(path), None
    except (DimacsError, EOFError, lzma.LZMAError, OSError, ValueError) as exc:
        return None, str(exc)


def _inspect_sources(
    sources: Sequence[SourceFile], workers: int
) -> Iterator[tuple[DimacsInfo | None, str | None]]:
    paths = (source.path for source in sources)
    if workers == 1:
        for path in paths:
            yield _inspect_source(path)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_inspect_source, paths)


def _direct_cnf_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required input directory does not exist: {directory}")
    entries = sorted(directory.iterdir(), key=lambda path: path.name.encode("utf-8"))
    for entry in entries:
        if entry.is_symlink():
            continue
        if entry.is_file() and is_cnf_path(entry):
            yield entry


def enumerate_satbench(
    split_roots: Mapping[str, Path],
    *,
    difficulty: str,
    family: str,
) -> list[SourceFile]:
    """Enumerate only ROOT/{sat,unsat} immediate CNF children."""
    _validate_logical_name(difficulty, "difficulty")
    _validate_logical_name(family, "family")
    sources: list[SourceFile] = []
    for split, split_root in sorted(split_roots.items()):
        _validate_logical_name(split, "split")
        root = split_root.expanduser().resolve(strict=True)
        for class_label in ("sat", "unsat"):
            class_root = (root / class_label).resolve(strict=True)
            for path in _direct_cnf_files(class_root):
                sources.append(
                    SourceFile(
                        dataset="satbench",
                        dataset_group=f"satbench/{difficulty}/{family}/{split}",
                        split=split,
                        class_label=class_label,
                        source_root=root,
                        input_root=class_root,
                        path=path,
                        source_relpath=f"{class_label}/{path.name}",
                        identity_relpath=(
                            f"{difficulty}/{family}/{split}/{class_label}/{identity_filename(path)}"
                        ),
                        difficulty=difficulty,
                        family=family,
                    )
                )
    return sources


def enumerate_satcomp(
    collection_roots: Mapping[str, Path],
    *,
    cnf_subdir: str = "cnf",
    track: str = "main",
) -> list[SourceFile]:
    """Enumerate only SC_COLLECTION/cnf immediate CNF children by default."""
    if not cnf_subdir or "/" in cnf_subdir or "\\" in cnf_subdir or cnf_subdir == "..":
        raise ValueError("cnf_subdir must be '.' or one plain directory name")
    _validate_logical_name(track, "track")
    sources: list[SourceFile] = []
    for year, collection_root in sorted(collection_roots.items()):
        if not re.fullmatch(r"[0-9]{4}", year):
            raise ValueError(f"competition year must be four digits: {year!r}")
        root = collection_root.expanduser().resolve(strict=True)
        cnf_root = (root if cnf_subdir == "." else root / cnf_subdir).resolve(strict=True)
        for path in _direct_cnf_files(cnf_root):
            physical_relpath = path.name if cnf_subdir == "." else f"{cnf_subdir}/{path.name}"
            sources.append(
                SourceFile(
                    dataset="satcomp",
                    dataset_group=f"satcomp/{year}/{track}",
                    split="all",
                    class_label=None,
                    source_root=root,
                    input_root=cnf_root,
                    path=path,
                    source_relpath=physical_relpath,
                    identity_relpath=identity_filename(path),
                    competition_year=year,
                    track=track,
                    collection_id=f"sc-{year}",
                    cnf_subdir=cnf_subdir,
                )
            )
    return sources


def _limit_reasons(info: DimacsInfo, limits: ManifestLimits) -> list[str]:
    reasons: list[str] = []
    if limits.max_input_nodes is not None and info.input_nodes > limits.max_input_nodes:
        reasons.append("input_nodes_exceeded")
    if (
        limits.max_literal_occurrences is not None
        and info.literal_occurrences > limits.max_literal_occurrences
    ):
        reasons.append("input_literal_occurrences_exceeded")
    if (
        limits.max_input_cells is not None
        and info.input_nodes + info.literal_occurrences > limits.max_input_cells
    ):
        reasons.append("input_cells_exceeded")
    if limits.max_cnf_bytes is not None and info.cnf_bytes > limits.max_cnf_bytes:
        reasons.append("cnf_bytes_exceeded")
    if limits.max_clause_length is not None and info.max_clause_length > limits.max_clause_length:
        reasons.append("max_clause_length_exceeded")
    return reasons


_SAFE_ID_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_-]+")
_LOGICAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_logical_name(value: str, field: str) -> str:
    if not _LOGICAL_NAME_RE.fullmatch(value):
        raise ValueError(f"{field} must be a short path-free identifier: {value!r}")
    return value


def _validate_relative_path(value: str, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe {field}: {value!r}")
    return path


def _safe_id_component(value: str, *, maximum: int) -> str:
    cleaned = _SAFE_ID_COMPONENT_RE.sub("-", value).strip("-_").lower()
    return (cleaned or "unknown")[:maximum]


def _instance_identity(
    source: SourceFile,
    *,
    dataset_release_id: str,
    cnf_sha256: str,
) -> tuple[str, str]:
    payload = b"\0".join(
        part.encode("utf-8")
        for part in (
            "decisiontrace-instance-v1",
            dataset_release_id,
            source.identity_relpath,
            cnf_sha256,
        )
    )
    full_digest = hashlib.sha256(payload).hexdigest()
    release = _safe_id_component(dataset_release_id, maximum=32)
    short_stem = _safe_id_component(Path(source.identity_relpath).stem, maximum=48)
    if source.dataset == "satbench":
        components = (
            "sb",
            release,
            _safe_id_component(source.difficulty or "unknown", maximum=24),
            _safe_id_component(source.family or "unknown", maximum=24),
            _safe_id_component(source.split, maximum=16),
            _safe_id_component(source.class_label or "unknown", maximum=16),
            short_stem,
            full_digest[:32],
        )
    else:
        components = (
            "sc",
            release,
            _safe_id_component(source.competition_year or "unknown", maximum=16),
            _safe_id_component(source.track or "main", maximum=24),
            short_stem,
            full_digest[:32],
        )
    return ".".join(components), full_digest


def _valid_record(
    source: SourceFile,
    info: DimacsInfo,
    *,
    dataset_release_id: str,
    group_index: int,
    part_size: int,
    limits: ManifestLimits,
) -> dict:
    instance_id, instance_digest = _instance_identity(
        source,
        dataset_release_id=dataset_release_id,
        cnf_sha256=info.cnf_sha256,
    )
    reasons = _limit_reasons(info, limits)
    record = {
        "schema": MANIFEST_SCHEMA,
        "instance_id": instance_id,
        "instance_digest_sha256": instance_digest,
        "source_instance_group_id": info.cnf_sha256,
        "dataset": source.dataset,
        "dataset_release_id": dataset_release_id,
        "dataset_group": source.dataset_group,
        "split": source.split,
        "source_path": str(source.path),
        "source_relpath": source.source_relpath,
        "identity_relpath": source.identity_relpath,
        "cnf_sha256": info.cnf_sha256,
        "size": {
            "variables": info.variables,
            "clauses": info.clauses,
            "literal_occurrences": info.literal_occurrences,
            "max_clause_length": info.max_clause_length,
            "nodes_2v_plus_c": info.input_nodes,
            "cells_2v_plus_c_plus_e": info.input_nodes + info.literal_occurrences,
            "cnf_bytes": info.cnf_bytes,
            "source_bytes": info.source_bytes,
        },
        "accepted": not reasons,
        "group_index": group_index,
        "part_id": group_index // part_size,
    }
    if source.class_label is not None:
        record["label"] = source.class_label
    if source.difficulty is not None:
        record["difficulty"] = source.difficulty
    if source.family is not None:
        record["family"] = source.family
    if source.competition_year is not None:
        record["competition_year"] = source.competition_year
    if source.track is not None:
        record["track"] = source.track
    if source.collection_id is not None:
        record["collection_id"] = source.collection_id
    if source.cnf_subdir is not None:
        record["cnf_subdir"] = source.cnf_subdir
    if reasons:
        record["skip_reasons"] = reasons
    return record


def _invalid_record(
    source: SourceFile,
    error: Exception,
    *,
    dataset_release_id: str,
    group_index: int,
    part_size: int,
) -> dict:
    record = {
        "schema": MANIFEST_SCHEMA,
        "record_id": stable_digest(
            (
                "invalid-instance-v1",
                source.dataset,
                source.dataset_group,
                source.split,
                source.source_relpath,
            )
        ),
        "dataset": source.dataset,
        "dataset_release_id": dataset_release_id,
        "dataset_group": source.dataset_group,
        "split": source.split,
        "source_path": str(source.path),
        "source_relpath": source.source_relpath,
        "identity_relpath": source.identity_relpath,
        "accepted": False,
        "skip_reasons": ["invalid_dimacs"],
        "error": str(error),
        "group_index": group_index,
        "part_id": group_index // part_size,
    }
    if source.class_label is not None:
        record["label"] = source.class_label
    if source.difficulty is not None:
        record["difficulty"] = source.difficulty
    if source.family is not None:
        record["family"] = source.family
    if source.competition_year is not None:
        record["competition_year"] = source.competition_year
    if source.track is not None:
        record["track"] = source.track
    if source.collection_id is not None:
        record["collection_id"] = source.collection_id
    if source.cnf_subdir is not None:
        record["cnf_subdir"] = source.cnf_subdir
    return record


def _write_atomic(path: Path, content: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest(
    sources: Sequence[SourceFile],
    *,
    output_root: str | Path,
    output_tag: str,
    dataset_release_id: str,
    limits: ManifestLimits = ManifestLimits(),
    part_size: int = DEFAULT_PART_SIZE,
    provenance: Mapping[str, str] | None = None,
    workers: int = 1,
    expected_content: Mapping[Path, str | None] | None = None,
    overwrite: bool = False,
) -> ManifestBuildResult:
    output_root = Path(output_root).expanduser().resolve(strict=False)
    validate_output_tag(output_tag)
    limits.validate()
    if not isinstance(dataset_release_id, str) or not dataset_release_id.strip():
        raise ValueError("dataset_release_id must not be empty")
    if isinstance(part_size, bool) or not isinstance(part_size, int) or part_size <= 0:
        raise ValueError("part_size must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be positive")
    if provenance is not None and any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in provenance.items()
    ):
        raise ValueError("manifest provenance keys and values must be strings")
    input_roots = sorted({source.input_root.resolve(strict=True) for source in sources})
    reject_output_source_overlap(output_root, input_roots)
    input_root_groups: dict[Path, str] = {}
    source_roots: dict[str, Path] = {}
    for source in sources:
        input_root = source.input_root.resolve(strict=True)
        source_root = source.source_root.resolve(strict=True)
        if not source.path.is_absolute():
            raise ValueError(f"source path must be absolute: {source.path}")
        if source.path.is_symlink() or not source.path.is_file() or not is_cnf_path(source.path):
            raise ValueError(f"source must be a non-symlink regular CNF: {source.path}")
        resolved_source = source.path.resolve(strict=True)
        if resolved_source.parent != input_root:
            raise ValueError(f"source is not an immediate child of input_root: {source.path}")
        relative_path = _validate_relative_path(source.source_relpath, "source_relpath")
        identity_path = _validate_relative_path(source.identity_relpath, "identity_relpath")
        if (source_root / Path(*relative_path.parts)).resolve(strict=True) != resolved_source:
            raise ValueError(f"source_root/source_relpath does not resolve to source: {source.path}")
        if identity_path.name != identity_filename(source.path):
            raise ValueError(f"identity_relpath filename does not match source: {source.path}")
        prior_group = input_root_groups.get(input_root)
        if prior_group is not None and prior_group != source.dataset_group:
            raise ValueError(
                f"the same physical input root was assigned to multiple groups: {input_root}"
            )
        input_root_groups[input_root] = source.dataset_group
        prior_source_root = source_roots.get(source.dataset_group)
        if prior_source_root is not None and prior_source_root != source_root:
            raise ValueError(
                f"dataset group has multiple physical source roots: {source.dataset_group}"
            )
        source_roots[source.dataset_group] = source_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "instances.jsonl"
    metadata_path = output_root / "manifest.meta.json"
    if not overwrite:
        existing = [path for path in (manifest_path, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")

    ordered_sources = sorted(sources, key=lambda item: (item.dataset_group, item.identity_relpath))
    previous_identity_key: tuple[str, str] | None = None
    for source in ordered_sources:
        identity_key = (source.dataset_group, source.identity_relpath)
        if identity_key == previous_identity_key:
            raise ValueError(
                f"duplicate identity_relpath in {source.dataset_group}: {source.identity_relpath}"
            )
        previous_identity_key = identity_key

    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    instance_id_keys: set[bytes] = set()
    group_indices: dict[str, int] = {}
    accepted_count = 0
    try:
        with temporary.open("xb") as stream:
            for source, (info, error) in zip(
                ordered_sources, _inspect_sources(ordered_sources, workers), strict=True
            ):
                group_index = group_indices.get(source.dataset_group, 0)
                group_indices[source.dataset_group] = group_index + 1
                if expected_content is not None:
                    expected_hash = expected_content[source.path]
                    if info is None and expected_hash is not None:
                        raise ValueError(
                            f"SATCOMP source validity changed since split: {source.path}"
                        )
                    if info is not None and info.cnf_sha256 != expected_hash:
                        raise ValueError(
                            f"SATCOMP source content changed since split: {source.path}"
                        )
                if info is not None:
                    record = _valid_record(
                        source,
                        info,
                        dataset_release_id=dataset_release_id,
                        group_index=group_index,
                        part_size=part_size,
                        limits=limits,
                    )
                    instance_id = record["instance_id"]
                    instance_id_key = hashlib.sha256(instance_id.encode("utf-8")).digest()
                    if instance_id_key in instance_id_keys:
                        raise RuntimeError(f"instance_id collision: {instance_id}")
                    instance_id_keys.add(instance_id_key)
                else:
                    record = _invalid_record(
                        source,
                        ValueError(error or "invalid DIMACS"),
                        dataset_release_id=dataset_release_id,
                        group_index=group_index,
                        part_size=part_size,
                    )
                encoded = canonical_json_bytes(record) + b"\n"
                stream.write(encoded)
                accepted_count += bool(record["accepted"])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = {
        "schema": MANIFEST_META_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "output_tag": output_tag,
        "dataset_release_id": dataset_release_id,
        "created_at": utc_now_text(),
        "part_size": part_size,
        "limits": asdict(limits),
        "input_roots": [str(path) for path in input_roots],
        "source_roots": {
            group: str(path) for group, path in sorted(source_roots.items())
        },
        "counts": {
            "discovered": len(ordered_sources),
            "accepted": accepted_count,
            "rejected": len(ordered_sources) - accepted_count,
        },
    }
    if provenance is not None:
        metadata["provenance"] = dict(sorted(provenance.items()))
    _write_atomic(metadata_path, canonical_json_bytes(metadata) + b"\n", overwrite=overwrite)
    return ManifestBuildResult(
        output_root=output_root,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        discovered_count=len(ordered_sources),
        accepted_count=accepted_count,
        rejected_count=len(ordered_sources) - accepted_count,
    )


def read_manifest_stream(
    stream: IO[str] | IO[bytes],
    *,
    source: str = "<manifest stream>",
    accepted_only: bool = False,
) -> Iterator[dict]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid manifest JSON at line {line_number}: {source}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must contain a JSON object")
        if record.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(
                f"unexpected manifest schema at line {line_number}: {record.get('schema')!r}"
            )
        if accepted_only and not record.get("accepted", False):
            continue
        yield record


def read_manifest(path: str | Path, *, accepted_only: bool = False) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        yield from read_manifest_stream(
            stream, source=str(path), accepted_only=accepted_only
        )
