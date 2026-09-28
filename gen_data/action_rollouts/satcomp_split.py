from __future__ import annotations

import hashlib
import json
import lzma
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

from .dimacs import DimacsError, inspect_dimacs
from .manifest import SourceFile
from .schema import canonical_json_bytes, utc_now_text


SATCOMP_SPLIT_SCHEMA = "decisiontrace-satcomp-split-v1"
SATCOMP_SPLITS = ("train", "valid", "test")
SATCOMP_STRATIFY_MODES = ("none", "year", "year-size")


@dataclass(frozen=True)
class SATCOMPSplitAssignment:
    path: Path
    sha256: str
    header: Mapping[str, object]
    sources: Mapping[tuple[str, str, str], Mapping[str, object]]


@dataclass(frozen=True)
class SATCOMPSplitResult:
    path: Path
    sha256: str
    source_count: int
    valid_source_count: int
    invalid_source_count: int
    generation_source_count: int
    group_count: int
    split_group_counts: Mapping[str, int]
    split_source_counts: Mapping[str, int]
    split_generation_source_counts: Mapping[str, int]


def _source_key(source: SourceFile) -> tuple[str, str, str]:
    if source.dataset != "satcomp":
        raise ValueError("SATCOMP split assignment only accepts SATCOMP sources")
    year = source.competition_year
    track = source.track
    if not year or not track:
        raise ValueError("SATCOMP source is missing competition_year or track")
    return year, track, source.source_relpath


def _ratio_fractions(values: Sequence[str | float | int]) -> tuple[Fraction, Fraction, Fraction]:
    if len(values) != 3:
        raise ValueError("split_ratio must contain exactly three values")
    ratios = tuple(Fraction(str(value)) for value in values)
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("split_ratio values must be non-negative with a positive sum")
    return ratios  # type: ignore[return-value]


def _size_bucket(input_nodes: int) -> str:
    if input_nodes <= 10_000:
        return "small"
    if input_nodes <= 50_000:
        return "medium"
    return "large"


def _stable_random(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _stratum(group: Mapping[str, object], mode: str) -> str:
    if mode == "none":
        return "all"
    years = group["competition_years"]
    if not isinstance(years, list) or not years:
        raise ValueError("SATCOMP split group has no competition years")
    primary_year = str(years[0])
    if mode == "year":
        return primary_year
    if mode == "year-size":
        return f"{primary_year}/{group['size_bucket']}"
    raise ValueError(f"unsupported SATCOMP stratification mode: {mode!r}")


def _global_targets(total: int, ratios: tuple[Fraction, Fraction, Fraction]) -> dict[str, int]:
    ratio_sum = sum(ratios)
    train = int(Fraction(total) * ratios[0] / ratio_sum)
    valid = int(Fraction(total) * ratios[1] / ratio_sum)
    return {"train": train, "valid": valid, "test": total - train - valid}


def _stratified_counts(
    stratum_sizes: Mapping[str, int],
    ratios: tuple[Fraction, Fraction, Fraction],
    *,
    seed: int,
) -> dict[str, dict[str, int]]:
    """Hamilton-style allocation with exact global counts and local balance."""
    ratio_sum = sum(ratios)
    targets = _global_targets(sum(stratum_sizes.values()), ratios)
    counts: dict[str, dict[str, int]] = {}
    remaining_slots: dict[str, int] = {}
    candidates: list[tuple[Fraction, str, str, str]] = []
    assigned = {split: 0 for split in SATCOMP_SPLITS}

    for stratum, size in sorted(stratum_sizes.items()):
        local: dict[str, int] = {}
        base_total = 0
        for split, ratio in zip(SATCOMP_SPLITS, ratios):
            exact = Fraction(size) * ratio / ratio_sum
            base = int(exact)
            local[split] = base
            assigned[split] += base
            base_total += base
            tie = hashlib.sha256(
                f"{seed}\0{stratum}\0{split}".encode("utf-8")
            ).hexdigest()
            candidates.append((exact - base, tie, stratum, split))
        counts[stratum] = local
        remaining_slots[stratum] = size - base_total

    deficits = {split: targets[split] - assigned[split] for split in SATCOMP_SPLITS}
    for _, _, stratum, split in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True):
        if remaining_slots[stratum] and deficits[split] > 0:
            counts[stratum][split] += 1
            remaining_slots[stratum] -= 1
            deficits[split] -= 1

    # The fractional pass is sufficient for normal three-way ratios.  Keep a
    # deterministic fallback so unusual ratios cannot leave an unassigned row.
    for stratum in sorted(remaining_slots):
        while remaining_slots[stratum]:
            choices = [split for split in SATCOMP_SPLITS if deficits[split] > 0]
            if not choices:
                raise RuntimeError("SATCOMP split apportionment exhausted its targets")
            split = max(choices, key=lambda name: (deficits[name], name))
            counts[stratum][split] += 1
            remaining_slots[stratum] -= 1
            deficits[split] -= 1
    if any(deficits.values()):
        raise RuntimeError(f"SATCOMP split apportionment mismatch: {deficits}")
    return counts


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing SATCOMP split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_satcomp_split_assignment(
    sources: Sequence[SourceFile],
    *,
    output_path: str | Path,
    seed: int = 42,
    split_ratio: Sequence[str | float | int] = (0.8, 0.1, 0.1),
    stratify: str = "year-size",
    overwrite: bool = False,
) -> SATCOMPSplitResult:
    if stratify not in SATCOMP_STRATIFY_MODES:
        raise ValueError(f"unsupported SATCOMP stratification mode: {stratify!r}")
    ratios = _ratio_fractions(split_ratio)
    ordered_sources = sorted(sources, key=_source_key)
    source_rows: list[dict[str, object]] = []
    groups: dict[str, dict[str, object]] = {}

    for source in ordered_sources:
        year, track, source_relpath = _source_key(source)
        row: dict[str, object] = {
            "schema": SATCOMP_SPLIT_SCHEMA,
            "kind": "source",
            "competition_year": year,
            "track": track,
            "source_relpath": source_relpath,
        }
        try:
            info = inspect_dimacs(source.path)
        except (DimacsError, EOFError, lzma.LZMAError, OSError, ValueError) as exc:
            row.update({
                "valid_dimacs": False,
                "split": "train",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        else:
            group_id = info.cnf_sha256
            row.update({
                "valid_dimacs": True,
                "source_instance_group_id": group_id,
                "cnf_sha256": group_id,
            })
            group = groups.setdefault(
                group_id,
                {
                    "source_instance_group_id": group_id,
                    "competition_years": [],
                    "source_count": 0,
                    "input_nodes": info.input_nodes,
                    "size_bucket": _size_bucket(info.input_nodes),
                },
            )
            if group["input_nodes"] != info.input_nodes:
                raise RuntimeError(f"content hash collision with inconsistent size: {group_id}")
            years = group["competition_years"]
            if not isinstance(years, list):
                raise RuntimeError("internal SATCOMP split group year representation is invalid")
            if year not in years:
                years.append(year)
                years.sort()
            group["source_count"] = int(group["source_count"]) + 1
        source_rows.append(row)

    if not groups:
        raise ValueError("SATCOMP split input contains no valid DIMACS groups")

    by_stratum: dict[str, list[str]] = {}
    for group_id, group in groups.items():
        by_stratum.setdefault(_stratum(group, stratify), []).append(group_id)
    local_counts = _stratified_counts(
        {name: len(group_ids) for name, group_ids in by_stratum.items()},
        ratios,
        seed=seed,
    )
    group_splits: dict[str, str] = {}
    for stratum, group_ids in sorted(by_stratum.items()):
        ordered = sorted(group_ids)
        _stable_random(seed, stratum).shuffle(ordered)
        offset = 0
        for split in SATCOMP_SPLITS:
            count = local_counts[stratum][split]
            for group_id in ordered[offset : offset + count]:
                group_splits[group_id] = split
            offset += count
        if offset != len(ordered):
            raise RuntimeError(f"SATCOMP split did not consume stratum {stratum!r}")

    group_counts = {split: 0 for split in SATCOMP_SPLITS}
    source_counts = {split: 0 for split in SATCOMP_SPLITS}
    generation_source_counts = {split: 0 for split in SATCOMP_SPLITS}
    for group_id, split in group_splits.items():
        groups[group_id]["split"] = split
        groups[group_id]["stratum"] = _stratum(groups[group_id], stratify)
        group_counts[split] += 1
    representatives: dict[str, tuple[str, str, str]] = {}
    for row in source_rows:
        group_id = row.get("source_instance_group_id")
        if isinstance(group_id, str):
            row["split"] = group_splits[group_id]
            key = (
                str(row["competition_year"]),
                str(row["track"]),
                str(row["source_relpath"]),
            )
            representatives.setdefault(group_id, key)
            row["selected_for_generation"] = representatives[group_id] == key
        else:
            # Invalid DIMACS rows remain in train for manifest diagnostics.
            row["selected_for_generation"] = True
        source_counts[str(row["split"])] += 1
        if row["selected_for_generation"]:
            generation_source_counts[str(row["split"])] += 1
    for group_id, key in representatives.items():
        groups[group_id]["representative_source"] = {
            "competition_year": key[0],
            "track": key[1],
            "source_relpath": key[2],
        }

    header = {
        "schema": SATCOMP_SPLIT_SCHEMA,
        "kind": "header",
        "created_at": utc_now_text(),
        "seed": int(seed),
        "split_ratio": [str(value) for value in ratios],
        "stratify": stratify,
        "source_count": len(source_rows),
        "valid_source_count": sum(bool(row.get("valid_dimacs")) for row in source_rows),
        "invalid_source_count": sum(not bool(row.get("valid_dimacs")) for row in source_rows),
        "generation_source_count": sum(generation_source_counts.values()),
        "source_instance_group_count": len(groups),
        "split_group_counts": group_counts,
        "split_source_counts": source_counts,
        "split_generation_source_counts": generation_source_counts,
    }
    records = [header]
    records.extend(
        {
            "schema": SATCOMP_SPLIT_SCHEMA,
            "kind": "group",
            **groups[group_id],
        }
        for group_id in sorted(groups)
    )
    records.extend(source_rows)
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    path = Path(output_path).expanduser().resolve(strict=False)
    _write_atomic(path, payload, overwrite=overwrite)
    return SATCOMPSplitResult(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_count=len(source_rows),
        valid_source_count=int(header["valid_source_count"]),
        invalid_source_count=int(header["invalid_source_count"]),
        generation_source_count=int(header["generation_source_count"]),
        group_count=len(groups),
        split_group_counts=group_counts,
        split_source_counts=source_counts,
        split_generation_source_counts=generation_source_counts,
    )


def load_satcomp_split_assignment(path: str | Path) -> SATCOMPSplitAssignment:
    assignment_path = Path(path).expanduser().resolve(strict=True)
    payload = assignment_path.read_bytes()
    header: Mapping[str, object] | None = None
    sources: dict[tuple[str, str, str], Mapping[str, object]] = {}
    group_splits: dict[str, str] = {}
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SATCOMP split JSON at line {line_number}") from exc
        if not isinstance(row, dict) or row.get("schema") != SATCOMP_SPLIT_SCHEMA:
            raise ValueError(f"unexpected SATCOMP split schema at line {line_number}")
        kind = row.get("kind")
        if kind == "header":
            if header is not None or sources or group_splits:
                raise ValueError("SATCOMP split header must be the first record")
            header = row
        elif kind == "group":
            group_id = row.get("source_instance_group_id")
            split = row.get("split")
            if not isinstance(group_id, str) or split not in SATCOMP_SPLITS:
                raise ValueError(f"invalid SATCOMP split group at line {line_number}")
            if group_id in group_splits:
                raise ValueError(f"duplicate SATCOMP split group at line {line_number}")
            group_splits[group_id] = str(split)
        elif kind == "source":
            key_values = (row.get("competition_year"), row.get("track"), row.get("source_relpath"))
            if not all(isinstance(value, str) and value for value in key_values):
                raise ValueError(f"invalid SATCOMP split source key at line {line_number}")
            key = (str(key_values[0]), str(key_values[1]), str(key_values[2]))
            if key in sources or row.get("split") not in SATCOMP_SPLITS:
                raise ValueError(f"invalid/duplicate SATCOMP split source at line {line_number}")
            group_id = row.get("source_instance_group_id")
            if group_id is not None and group_splits.get(group_id) != row.get("split"):
                raise ValueError(f"SATCOMP source/group split mismatch at line {line_number}")
            sources[key] = row
        else:
            raise ValueError(f"invalid SATCOMP split record kind at line {line_number}")
    if header is None:
        raise ValueError("SATCOMP split assignment has no header")
    if int(header.get("source_count", -1)) != len(sources):
        raise ValueError("SATCOMP split source count does not match its header")
    if int(header.get("source_instance_group_count", -1)) != len(group_splits):
        raise ValueError("SATCOMP split group count does not match its header")
    source_group_counts = Counter(
        str(row["source_instance_group_id"])
        for row in sources.values()
        if isinstance(row.get("source_instance_group_id"), str)
    )
    if set(source_group_counts) != set(group_splits):
        raise ValueError("SATCOMP split source/group inventory does not match")
    if any(
        not isinstance(row.get("selected_for_generation"), bool)
        for row in sources.values()
    ):
        raise ValueError("SATCOMP split source has invalid generation-selection flag")
    selected_group_counts = Counter(
        str(row["source_instance_group_id"])
        for row in sources.values()
        if row.get("selected_for_generation") is True
        and isinstance(row.get("source_instance_group_id"), str)
    )
    if set(selected_group_counts) != set(group_splits) or any(
        count != 1 for count in selected_group_counts.values()
    ):
        raise ValueError("SATCOMP split must select exactly one source per CNF group")
    actual_group_counts = Counter(group_splits.values())
    actual_source_counts = Counter(str(row["split"]) for row in sources.values())
    if header.get("split_group_counts") != {
        split: actual_group_counts[split] for split in SATCOMP_SPLITS
    }:
        raise ValueError("SATCOMP split group counts do not match its header")
    if header.get("split_source_counts") != {
        split: actual_source_counts[split] for split in SATCOMP_SPLITS
    }:
        raise ValueError("SATCOMP split source counts do not match its header")
    actual_generation_counts = Counter(
        str(row["split"])
        for row in sources.values()
        if row.get("selected_for_generation") is True
    )
    if header.get("split_generation_source_counts") != {
        split: actual_generation_counts[split] for split in SATCOMP_SPLITS
    }:
        raise ValueError("SATCOMP split generation counts do not match its header")
    if int(header.get("generation_source_count", -1)) != sum(
        actual_generation_counts.values()
    ):
        raise ValueError("SATCOMP split generation count does not match its header")
    return SATCOMPSplitAssignment(
        path=assignment_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        header=header,
        sources=sources,
    )


def select_satcomp_sources(
    sources: Sequence[SourceFile],
    assignment: SATCOMPSplitAssignment,
    selected_split: str,
    *,
    validate_content: bool = True,
) -> list[SourceFile]:
    if selected_split not in SATCOMP_SPLITS:
        raise ValueError(f"selected_split must be one of {SATCOMP_SPLITS}")
    current_keys = {_source_key(source) for source in sources}
    assignment_keys = set(assignment.sources)
    if current_keys != assignment_keys:
        missing = sorted(current_keys - assignment_keys)
        extra = sorted(assignment_keys - current_keys)
        raise ValueError(
            "SATCOMP split assignment/source inventory mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    selected: list[SourceFile] = []
    for source in sorted(sources, key=_source_key):
        row = assignment.sources[_source_key(source)]
        should_generate = (
            row.get("split") == selected_split
            and row.get("selected_for_generation") is True
        )
        if not should_generate:
            continue
        if validate_content:
            expected_group = row.get("source_instance_group_id")
            try:
                info = inspect_dimacs(source.path)
            except (DimacsError, EOFError, lzma.LZMAError, OSError, ValueError):
                if row.get("valid_dimacs") is not False or expected_group is not None:
                    raise ValueError(f"SATCOMP source validity changed since split: {source.path}")
            else:
                if row.get("valid_dimacs") is not True or expected_group != info.cnf_sha256:
                    raise ValueError(f"SATCOMP source content changed since split: {source.path}")
        selected.append(replace(source, split=selected_split))
    if not selected:
        raise ValueError(f"SATCOMP split {selected_split!r} contains no sources")
    return selected


def satcomp_source_expectations(
    sources: Sequence[SourceFile], assignment: SATCOMPSplitAssignment
) -> dict[Path, str | None]:
    return {
        source.path: assignment.sources[_source_key(source)].get(
            "source_instance_group_id"
        )
        for source in sources
    }
