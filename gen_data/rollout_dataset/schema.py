"""Stable schemas shared only by DecisionTrace shard files.

This module intentionally does not import the raw-data generator.  The shard
package validates the raw JSON contract at its input boundary and owns only
the compact array representation that it writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


RAW_ARRAY_SCHEMA = "decisiontrace-actioneval-compact-arrays-v2"
RAW_COMPLETE_SCHEMA = "decisiontrace-actioneval-compact-arrays-complete-v2"
SHARD_BUILD_SCHEMA = "decisiontrace-actioneval-raw-shard-build-v2"
PART_DIAGNOSTICS_SCHEMA = "decisiontrace-actioneval-part-raw-diagnostics-v2"
SHARD_POLICY_SCHEMA = "decisiontrace-actioneval-shard-policy-v2"
SHARD_PLAN_SCHEMA = "decisiontrace-actioneval-shard-plan-v2"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_literal(literal: int) -> int:
    if isinstance(literal, bool) or not isinstance(literal, int):
        raise TypeError("literal must be an integer")
    if literal == 0:
        raise ValueError("literal 0 means native baseline, not a forced action")
    if not -(1 << 31) < literal < (1 << 31):
        raise ValueError("literal must fit signed int32 and be safely negatable")
    return literal
