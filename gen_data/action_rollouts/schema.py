from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA = "decisiontrace-actioneval-instance-manifest-v2"
MANIFEST_META_SCHEMA = "decisiontrace-actioneval-instance-manifest-meta-v2"

DATASET_VERSION = "DecisionTrace-ActionEval-v1.3"
DEFAULT_OUTPUT_TAG = "260819b"
DEFAULT_MAX_INPUT_NODES = 100_000
DEFAULT_MAX_LITERAL_OCCURRENCES = 4_000_000
DEFAULT_PART_SIZE = 512

_OUTPUT_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESERVED_OUTPUT_TAGS = {
    ".tmp",
    "cnf",
    "eval",
    "logs",
    "manifest",
    "proposal",
    "raw",
    "raw-builds",
    "sat",
    "shard-builds",
    "state",
    "statistics",
    "trace",
    "unsat",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_digest(parts: Iterable[str], *, digest_bytes: int = 16) -> str:
    if not 1 <= digest_bytes <= 32:
        raise ValueError("digest_bytes must be in 1..32")
    hasher = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()[: 2 * digest_bytes]


def action_id(state_id: str, signed_external_literal: int) -> str:
    validate_literal(signed_external_literal)
    return stable_digest(("action-v1", state_id, str(signed_external_literal)))


def validate_literal(literal: int) -> int:
    if isinstance(literal, bool) or not isinstance(literal, int):
        raise TypeError("literal must be an integer")
    if literal == 0:
        raise ValueError("literal 0 is callback abstain, not an action")
    if not -(1 << 31) < literal < (1 << 31):
        raise ValueError("literal must fit signed int32 and be safely negatable")
    return literal


def validate_output_tag(tag: str) -> str:
    if not isinstance(tag, str):
        raise TypeError("output tag must be a string")
    if not _OUTPUT_TAG_RE.fullmatch(tag):
        raise ValueError(
            "output tag must start with an alphanumeric character and contain "
            "only alphanumerics, '.', '_' or '-' (maximum 64 characters)"
        )
    if tag.lower() in _RESERVED_OUTPUT_TAGS:
        raise ValueError(f"output tag is reserved: {tag!r}")
    return tag


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_dir_name(output_tag: str) -> str:
    validate_output_tag(output_tag)
    prefix = f"{DATASET_VERSION}-"
    if output_tag == DATASET_VERSION or output_tag.lower().startswith(prefix.lower()):
        return output_tag
    return f"{prefix}{output_tag}"


def resolve_output_root(output_path: str | Path | None, output_tag: str) -> Path:
    validate_output_tag(output_tag)
    if output_path is not None:
        return Path(output_path).expanduser().resolve(strict=False)
    return (Path.cwd() / "decisiontrace_actioneval_outputs" / build_dir_name(output_tag)).resolve(
        strict=False
    )


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def reject_output_source_overlap(output_root: Path, source_roots: Iterable[Path]) -> None:
    output = output_root.resolve(strict=False)
    for source_root in source_roots:
        source = source_root.resolve(strict=False)
        if output == source or is_relative_to(output, source) or is_relative_to(source, output):
            raise ValueError(
                f"output and source roots must be separate and non-nested: "
                f"output={output} source={source}"
            )


def parse_named_path(text: str) -> tuple[str, Path]:
    name, separator, raw_path = text.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"expected NAME=PATH, got {text!r}")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"invalid logical name: {name!r}")
    return name, Path(raw_path).expanduser().resolve(strict=False)


def positive_limit(value: int | None, option_name: str) -> int | None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{option_name} must be an integer or null")
        if value <= 0:
            raise ValueError(f"{option_name} must be positive when enabled")
    return value
