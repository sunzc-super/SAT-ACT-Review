"""Raw/part v2 data generation for DecisionTrace-ActionEval-v1.3.

This package owns manifest construction and the offline
DISCOVER -> CAPTURE -> EVAL pipeline.  Raw-to-shard conversion deliberately
lives outside this C++ source project.
"""

from .pipeline import PipelinePolicy, select_target_traces
from .schema import MANIFEST_SCHEMA, action_id

__all__ = [
    "MANIFEST_SCHEMA",
    "PipelinePolicy",
    "action_id",
    "select_target_traces",
]

__version__ = "2.1.0"
