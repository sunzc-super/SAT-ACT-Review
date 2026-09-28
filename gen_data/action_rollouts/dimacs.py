from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class DimacsError(ValueError):
    pass


@dataclass(frozen=True)
class DimacsInfo:
    variables: int
    clauses: int
    literal_occurrences: int
    max_clause_length: int
    input_nodes: int
    cnf_bytes: int
    source_bytes: int
    cnf_sha256: str
    compression: str | None


_CNF_SUFFIXES = (".cnf", ".cnf.gz", ".cnf.xz", ".cnf.bz2")


def is_cnf_path(path: Path) -> bool:
    return path.name.lower().endswith(_CNF_SUFFIXES)


def identity_filename(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    for suffix in (".gz", ".xz", ".bz2"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _compression(path: Path) -> str | None:
    lowered = path.name.lower()
    if lowered.endswith(".gz"):
        return "gzip"
    if lowered.endswith(".xz"):
        return "xz"
    if lowered.endswith(".bz2"):
        return "bzip2"
    return None


def open_cnf_binary(path: Path) -> BinaryIO:
    compression = _compression(path)
    if compression == "gzip":
        return gzip.open(path, "rb")
    if compression == "xz":
        return lzma.open(path, "rb")
    if compression == "bzip2":
        return bz2.open(path, "rb")
    return path.open("rb")


_TOKEN_OR_NEWLINE = re.compile(rb"[^\s]+|\n")
_TRAILING_TOKEN = re.compile(rb"[^\s]+$")
_INTEGER_TOKEN = re.compile(rb"[+-]?[0-9]+$")
_READ_BYTES = 1 << 20
_MAX_TOKEN_BYTES = 256


def _stream_tokens(
    raw: BinaryIO,
    hasher: object,
    byte_count: list[int],
):
    """Yield ``(line_number, is_first_token, token)`` with bounded buffering."""
    pending = b""
    line_number = 1
    first_token = True
    while True:
        chunk = raw.read(_READ_BYTES)
        if not chunk:
            break
        if not chunk.isascii():
            raise DimacsError("DIMACS must be ASCII")
        hasher.update(chunk)
        byte_count[0] += len(chunk)
        data = pending + chunk
        trailing = _TRAILING_TOKEN.search(data)
        if trailing is None:
            complete = data
            pending = b""
        else:
            complete = data[: trailing.start()]
            pending = trailing.group(0)
            if len(pending) > _MAX_TOKEN_BYTES:
                raise DimacsError("DIMACS token exceeds 256 bytes")
        for match in _TOKEN_OR_NEWLINE.finditer(complete):
            token = match.group(0)
            if token == b"\n":
                line_number += 1
                first_token = True
            else:
                if len(token) > _MAX_TOKEN_BYTES:
                    raise DimacsError("DIMACS token exceeds 256 bytes")
                yield line_number, first_token, token
                first_token = False
    if pending:
        yield line_number, first_token, pending


def inspect_dimacs(path: str | Path) -> DimacsInfo:
    source_path = Path(path)
    source_bytes = source_path.stat().st_size
    content_hasher = hashlib.sha256()
    content_bytes = [0]
    variables: int | None = None
    declared_clauses: int | None = None
    parsed_clauses = 0
    literal_occurrences = 0
    current_clause_length = 0
    max_clause_length = 0
    terminated = False
    ignored_line: int | None = None
    header_line: int | None = None
    header_fields: list[bytes] = []

    def finish_header() -> None:
        nonlocal variables, declared_clauses, header_line, header_fields
        if header_line is None:
            return
        line_number = header_line
        if len(header_fields) != 4 or header_fields[0] != b"p" or header_fields[1].lower() != b"cnf":
            raise DimacsError(f"line {line_number}: expected 'p cnf V C'")
        try:
            if not _INTEGER_TOKEN.fullmatch(header_fields[2]) or not _INTEGER_TOKEN.fullmatch(
                header_fields[3]
            ):
                raise ValueError
            variables = int(header_fields[2])
            declared_clauses = int(header_fields[3])
        except ValueError as exc:
            raise DimacsError(f"line {line_number}: invalid header integer") from exc
        if variables < 0 or declared_clauses < 0:
            raise DimacsError(f"line {line_number}: negative DIMACS count")
        header_line = None
        header_fields = []

    with open_cnf_binary(source_path) as raw:
        for line_number, first_token, token in _stream_tokens(
            raw, content_hasher, content_bytes
        ):
            if header_line is not None and line_number != header_line:
                finish_header()
            if terminated:
                continue
            if ignored_line == line_number:
                continue
            if first_token and token.startswith(b"c"):
                ignored_line = line_number
                continue
            if first_token and token.startswith(b"%"):
                terminated = True
                continue
            if header_line is not None:
                header_fields.append(token)
                if len(header_fields) > 4:
                    raise DimacsError(f"line {line_number}: expected 'p cnf V C'")
                continue
            if first_token and token == b"p":
                if variables is not None:
                    raise DimacsError(f"line {line_number}: duplicate p cnf header")
                header_line = line_number
                header_fields = [token]
                continue
            if variables is None or declared_clauses is None:
                raise DimacsError(f"line {line_number}: clause data before p cnf header")
            try:
                if not _INTEGER_TOKEN.fullmatch(token):
                    raise ValueError
                literal = int(token)
            except ValueError as exc:
                raise DimacsError(f"line {line_number}: invalid literal {token!r}") from exc
            if literal == 0:
                parsed_clauses += 1
                max_clause_length = max(max_clause_length, current_clause_length)
                current_clause_length = 0
            else:
                if abs(literal) > variables:
                    raise DimacsError(
                        f"line {line_number}: literal {literal} exceeds declared V={variables}"
                    )
                literal_occurrences += 1
                current_clause_length += 1
    finish_header()

    if variables is None or declared_clauses is None:
        raise DimacsError("missing p cnf header")
    if current_clause_length:
        raise DimacsError("final clause is missing terminating 0")
    if parsed_clauses != declared_clauses:
        raise DimacsError(
            f"declared C={declared_clauses}, parsed {parsed_clauses} clause terminators"
        )
    return DimacsInfo(
        variables=variables,
        clauses=declared_clauses,
        literal_occurrences=literal_occurrences,
        max_clause_length=max_clause_length,
        input_nodes=2 * variables + declared_clauses,
        cnf_bytes=content_bytes[0],
        source_bytes=source_bytes,
        cnf_sha256=content_hasher.hexdigest(),
        compression=_compression(source_path),
    )
