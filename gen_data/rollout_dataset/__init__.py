"""Raw-to-shard v2 tools for DecisionTrace-ActionEval-v1."""

__version__ = "2.0.0"

from .builder import (
    ShardBuildResult,
    build_shard_group,
    build_shard_part,
    completed_shard_group_matches,
    completed_shard_matches,
)
from .compact import (
    ArrayData,
    CompactArrayReader,
    CompactArrayWriter,
    CompactPartPaths,
    inspect_compact_part,
)
from .contract import RawPolicy, load_raw_policy

__all__ = [
    "ArrayData",
    "CompactArrayReader",
    "CompactArrayWriter",
    "CompactPartPaths",
    "RawPolicy",
    "ShardBuildResult",
    "build_shard_group",
    "build_shard_part",
    "completed_shard_group_matches",
    "completed_shard_matches",
    "inspect_compact_part",
    "load_raw_policy",
]
