from __future__ import annotations

import array
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .schema import RAW_ARRAY_SCHEMA, RAW_COMPLETE_SCHEMA, canonical_json_bytes, utc_now_text


_DTYPES: dict[str, tuple[str, int]] = {
    "i1": ("b", 1),
    "u1": ("B", 1),
    "i2": ("h", 2),
    "u2": ("H", 2),
    "i4": ("i", 4),
    "u4": ("I", 4),
    "i8": ("q", 8),
    "u8": ("Q", 8),
    "f4": ("f", 4),
    "f8": ("d", 8),
}
_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IO_CHUNK_BYTES = 1 << 20
_RAW_MAGIC = "DTAEARR1"


@dataclass(frozen=True)
class ArrayData:
    """One canonical little-endian typed array."""

    dtype: str
    values: Iterable[int | float] | array.array
    shape: Sequence[int] | None = None


@dataclass(frozen=True)
class CompactPartPaths:
    header: Path
    arrays: Path
    complete: Path


def part_paths(output_dir: str | Path, stem: str) -> CompactPartPaths:
    if not _STEM_RE.fullmatch(stem):
        raise ValueError("part stem contains unsafe characters")
    root = Path(output_dir)
    return CompactPartPaths(
        header=root / f"{stem}.header.json",
        arrays=root / f"{stem}.arrays.bin",
        complete=root / f"{stem}.complete.json",
    )


def _shape_for(shape: Sequence[int] | None, count: int) -> list[int]:
    normalized = [count] if shape is None else list(shape)
    product = 1
    for dimension in normalized:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError("array shape dimensions must be non-negative integers")
        product *= dimension
    if product != count:
        raise ValueError(f"array shape has {product} elements, but values contain {count}")
    return normalized


def _coerce_array(
    data: ArrayData, *, max_array_bytes: int
) -> tuple[array.array, int, list[int]]:
    try:
        typecode, itemsize = _DTYPES[data.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {data.dtype!r}") from exc
    try:
        value_count = len(data.values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("array values must be sized so RAM can be preflighted") from exc
    if value_count * itemsize > max_array_bytes:
        raise ValueError("array exceeds writer byte ceiling")
    if isinstance(data.values, array.array) and data.values.typecode == typecode:
        values = data.values
    else:
        values = array.array(typecode, data.values)
    if values.itemsize != itemsize:
        raise RuntimeError(
            f"platform array typecode {typecode!r} has itemsize {values.itemsize}, expected {itemsize}"
        )
    shape = _shape_for(data.shape, len(values))
    if sys.byteorder == "big" and itemsize > 1:
        values = array.array(typecode, values)
        values.byteswap()
    return values, itemsize, shape


def _write_chunks(stream: Any, payload: memoryview) -> None:
    for start in range(0, len(payload), _IO_CHUNK_BYTES):
        chunk = payload[start : start + _IO_CHUNK_BYTES]
        stream.write(chunk)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CompactArrayWriter:
    """Stream a part, then publish arrays/header and an atomic commit marker.

    The writer retains no prior array payloads. A reader only accepts the part
    after the final ``.complete.json`` marker exists and file sizes match.
    """

    def __init__(
        self,
        output_dir: str | Path,
        stem: str,
        *,
        overwrite: bool = False,
        max_array_bytes: int = 256 * 1024 * 1024,
        part_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(max_array_bytes, bool)
            or not isinstance(max_array_bytes, int)
            or max_array_bytes <= 0
        ):
            raise ValueError("max_array_bytes must be a positive integer")
        self.max_array_bytes = max_array_bytes
        self.part_metadata = dict(part_metadata or {})
        canonical_json_bytes(self.part_metadata)
        self.output_dir = Path(output_dir).expanduser().resolve(strict=False)
        self.paths = part_paths(self.output_dir, stem)
        self.overwrite = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        self._lock_path = self.output_dir / f".{self.paths.complete.stem}.writer.lock"
        self._header_temp = self.output_dir / f".{self.paths.header.name}{suffix}"
        self._arrays_temp = self.output_dir / f".{self.paths.arrays.name}{suffix}"
        self._complete_temp = self.output_dir / f".{self.paths.complete.name}{suffix}"
        try:
            with self._lock_path.open("xb") as lock:
                lock.write(f"pid={os.getpid()}\n".encode("ascii"))
                lock.flush()
                os.fsync(lock.fileno())
        except FileExistsError as exc:
            raise FileExistsError(f"another writer owns compact part lock: {self._lock_path}") from exc
        if not overwrite and self.paths.complete.exists():
            self._lock_path.unlink()
            raise FileExistsError(
                f"refusing to overwrite committed compact part: {self.paths.complete}"
            )
        if overwrite and self.paths.complete.exists():
            # The completion marker is the commit point.  Withdraw it before
            # replacing header/arrays so readers never accept a mixed version.
            self.paths.complete.unlink()
        # Header/arrays without a completion marker are uncommitted remnants
        # of an interrupted close.  The final os.replace safely supersedes
        # those exact part files, so normal retry needs no directory scan.
        try:
            self._header = self._header_temp.open("xb")
        except BaseException:
            self._lock_path.unlink()
            raise
        try:
            self._arrays = self._arrays_temp.open("xb")
        except BaseException:
            self._header.close()
            self._header_temp.unlink()
            self._lock_path.unlink()
            raise
        self._record_count = 0
        self._array_bytes = 0
        self._finished = False
        self._committed = False
        self._failed = False
        prefix = b'{"records":['
        self._header.write(prefix)

    def append(
        self,
        record_id: str,
        arrays: Mapping[str, ArrayData],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError("compact part is already finished")
        if self._failed:
            raise RuntimeError("compact part was poisoned by an earlier append failure")
        try:
            self._append_record(record_id, arrays, metadata=metadata)
        except BaseException:
            self._failed = True
            raise

    def update_part_metadata(self, values: Mapping[str, Any]) -> None:
        """Add part-level metadata after streaming records but before commit.

        This is useful for diagnostics whose final contents are only known
        after the input stream has been validated.  Array payloads stay
        streaming, while the metadata object itself remains in memory and is
        written uncompressed into the JSON header.
        """

        if self._finished:
            raise RuntimeError("compact part is already finished")
        if self._failed:
            raise RuntimeError("compact part was poisoned by an earlier append failure")
        if not isinstance(values, Mapping):
            raise TypeError("part metadata update must be a mapping")
        duplicate = set(values).intersection(self.part_metadata)
        if duplicate:
            raise ValueError(
                "part metadata keys already exist: " + ",".join(sorted(duplicate))
            )
        updated = {**self.part_metadata, **dict(values)}
        canonical_json_bytes(updated)
        self.part_metadata = updated

    def _append_record(
        self,
        record_id: str,
        arrays: Mapping[str, ArrayData],
        *,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        array_directory: list[dict[str, Any]] = []
        for name in sorted(arrays):
            if not isinstance(name, str) or not name:
                raise ValueError("array names must be non-empty strings")
            data = arrays[name]
            if not isinstance(data, ArrayData):
                raise TypeError(f"array {name!r} must be ArrayData")
            values, itemsize, shape = _coerce_array(
                data, max_array_bytes=self.max_array_bytes
            )
            byte_length = len(values) * itemsize
            descriptor = {
                "name": name,
                "offset": self._array_bytes,
                "length": len(values),
                "dtype": data.dtype,
                "shape": shape,
                "codec": "none",
            }
            array_directory.append(descriptor)
            payload = memoryview(values).cast("B")
            _write_chunks(self._arrays, payload)
            self._array_bytes += byte_length

        record: dict[str, Any] = {"record_id": record_id, "arrays": array_directory}
        if metadata:
            record["metadata"] = dict(metadata)
        encoded = canonical_json_bytes(record)
        separator = b"," if self._record_count else b""
        self._header.write(separator)
        self._header.write(encoded)
        self._record_count += 1

    def close(self) -> CompactPartPaths:
        if self._finished:
            if self._committed:
                return self.paths
            raise RuntimeError("compact part was aborted and cannot be committed")
        if self._failed:
            self.abort()
            raise RuntimeError("refusing to commit a compact part after an append failure")
        try:
            trailer = canonical_json_bytes(
                {
                    "array_schema": RAW_ARRAY_SCHEMA,
                    "endianness": "little",
                    "magic": _RAW_MAGIC,
                    "part_metadata": self.part_metadata,
                    "record_count": self._record_count,
                }
            )[1:]
            # Replace the opening brace of the canonical suffix with the comma
            # that closes the already-streamed records member.
            trailer = b"]," + trailer
            self._header.write(trailer)
            for stream in (self._arrays, self._header):
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()

            header_bytes = self._header_temp.stat().st_size
            arrays_bytes = self._arrays_temp.stat().st_size
            complete = {
                "schema": RAW_COMPLETE_SCHEMA,
                "array_schema": RAW_ARRAY_SCHEMA,
                "endianness": "little",
                "magic": _RAW_MAGIC,
                "created_at": utc_now_text(),
                "record_count": self._record_count,
                "files": {
                    "header": {
                        "name": self.paths.header.name,
                        "bytes": header_bytes,
                    },
                    "arrays": {
                        "name": self.paths.arrays.name,
                        "bytes": arrays_bytes,
                    },
                },
            }
            with self._complete_temp.open("xb") as stream:
                stream.write(canonical_json_bytes(complete) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(self._arrays_temp, self.paths.arrays)
            os.replace(self._header_temp, self.paths.header)
            os.replace(self._complete_temp, self.paths.complete)
            _fsync_directory(self.output_dir)
        except BaseException:
            for stream in (self._arrays, self._header):
                if not stream.closed:
                    stream.close()
            self._finished = True
            self._cleanup_temps()
            raise
        self._finished = True
        self._committed = True
        self._cleanup_temps()
        return self.paths

    def abort(self) -> None:
        if self._finished:
            return
        for stream in (self._arrays, self._header):
            if not stream.closed:
                stream.close()
        self._finished = True
        self._cleanup_temps()

    def _cleanup_temps(self) -> None:
        for path in (self._arrays_temp, self._header_temp, self._complete_temp):
            if path.exists():
                path.unlink()
        if self._lock_path.exists():
            self._lock_path.unlink()

    def __enter__(self) -> "CompactArrayWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class CompactArrayReader:
    def __init__(
        self,
        complete_path: str | Path,
        *,
        verify: bool = True,
        max_header_bytes: int = 64 * 1024 * 1024,
        max_array_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(max_header_bytes, bool)
            or not isinstance(max_header_bytes, int)
            or max_header_bytes <= 0
        ):
            raise ValueError("max_header_bytes must be a positive integer")
        if (
            isinstance(max_array_bytes, bool)
            or not isinstance(max_array_bytes, int)
            or max_array_bytes <= 0
        ):
            raise ValueError("max_array_bytes must be a positive integer")
        self.max_array_bytes = max_array_bytes
        self.complete_path = Path(complete_path).expanduser().resolve(strict=True)
        with self.complete_path.open("r", encoding="utf-8") as stream:
            self.complete = json.load(stream)
        if not isinstance(self.complete, dict):
            raise ValueError("compact-part commit marker must be a JSON object")
        if self.complete.get("schema") != RAW_COMPLETE_SCHEMA:
            raise ValueError("unexpected compact-part commit schema")
        if self.complete.get("array_schema") != RAW_ARRAY_SCHEMA:
            raise ValueError("unexpected compact-part array schema")
        if self.complete.get("magic") != _RAW_MAGIC or self.complete.get("endianness") != "little":
            raise ValueError("unexpected compact-part magic or endianness")
        if not isinstance(self.complete.get("files"), Mapping):
            raise ValueError("compact-part commit files must be an object")
        self.header_path = self._resolve_file("header")
        self.arrays_path = self._resolve_file("arrays")
        header_stream = self._open_part_file("header", self.header_path, verify=verify)
        try:
            if os.fstat(header_stream.fileno()).st_size > max_header_bytes:
                raise ValueError("compact-array header exceeds reader byte ceiling")
            self._arrays = self._open_part_file("arrays", self.arrays_path, verify=verify)
            self._arrays_size = os.fstat(self._arrays.fileno()).st_size
            self.header = json.load(header_stream)
        except BaseException:
            if hasattr(self, "_arrays"):
                self._arrays.close()
            raise
        finally:
            header_stream.close()
        try:
            if not isinstance(self.header, dict):
                raise ValueError("compact-array header must be a JSON object")
            if self.header.get("array_schema") != RAW_ARRAY_SCHEMA:
                raise ValueError("unexpected compact-array header schema")
            if self.header.get("magic") != _RAW_MAGIC or self.header.get("endianness") != "little":
                raise ValueError("unexpected compact-array header magic or endianness")
            records = self.header.get("records")
            if not isinstance(records, list) or not all(
                isinstance(record, dict) for record in records
            ):
                raise ValueError("compact-array header records must be an array of objects")
            if self.header.get("record_count") != len(records):
                raise ValueError("compact-array header record count mismatch")
            if self.complete.get("record_count") != self.header.get("record_count"):
                raise ValueError("commit/header record count mismatch")
        except BaseException:
            self._arrays.close()
            raise

    def _resolve_file(self, kind: str) -> Path:
        descriptor = self.complete.get("files", {}).get(kind, {})
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"invalid {kind} descriptor in commit marker")
        name = descriptor.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe {kind} filename in commit marker")
        return self.complete_path.parent / name

    def _open_part_file(self, kind: str, path: Path, *, verify: bool):
        descriptor = self.complete["files"][kind]
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(path, flags)
        stream = os.fdopen(file_descriptor, "rb")
        try:
            if os.fstat(file_descriptor).st_size != descriptor.get("bytes"):
                raise ValueError(f"{kind} size mismatch")
            if verify and isinstance(descriptor.get("sha256"), str):
                hasher = hashlib.sha256()
                while True:
                    chunk = stream.read(_IO_CHUNK_BYTES)
                    if not chunk:
                        break
                    hasher.update(chunk)
                if hasher.hexdigest() != descriptor.get("sha256"):
                    raise ValueError(f"{kind} checksum mismatch")
                stream.seek(0)
            return stream
        except BaseException:
            stream.close()
            raise

    def records(self) -> Iterator[dict[str, Any]]:
        yield from self.header["records"]

    def _read_array_descriptor(self, descriptor: Mapping[str, Any]) -> array.array:
        if descriptor.get("codec") != "none":
            raise ValueError(f"unsupported array codec: {descriptor.get('codec')!r}")
        dtype = descriptor.get("dtype")
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype in header: {dtype!r}")
        typecode, itemsize = _DTYPES[dtype]
        length = descriptor.get("length")
        offset = descriptor.get("offset")
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("invalid array offset or length")
        byte_length = length * itemsize
        _shape_for(descriptor.get("shape"), length)
        if byte_length > self.max_array_bytes:
            raise ValueError("array exceeds reader byte ceiling")
        if offset + byte_length > self._arrays_size:
            raise ValueError("array descriptor exceeds binary payload")
        self._arrays.seek(offset)
        values = array.array(typecode)
        try:
            values.fromfile(self._arrays, length)
        except EOFError as exc:
            raise ValueError("truncated binary array payload") from exc
        if sys.byteorder == "big" and itemsize > 1:
            values.byteswap()
        return values

    def read_arrays(
        self, record: Mapping[str, Any], names: Sequence[str]
    ) -> dict[str, array.array]:
        descriptors = {
            descriptor["name"]: descriptor for descriptor in record["arrays"]
        }
        return {
            name: self._read_array_descriptor(descriptors[name]) for name in names
        }

    def read_array(self, record: Mapping[str, Any], name: str) -> array.array:
        return self.read_arrays(record, (name,))[name]

    def close(self) -> None:
        self._arrays.close()

    def __enter__(self) -> "CompactArrayReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def inspect_compact_part(complete_path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    with CompactArrayReader(complete_path, verify=verify) as reader:
        return {
            "schema": reader.complete["schema"],
            "record_count": reader.complete["record_count"],
            "header_bytes": reader.complete["files"]["header"]["bytes"],
            "array_bytes": reader.complete["files"]["arrays"]["bytes"],
            "verified": verify,
        }
