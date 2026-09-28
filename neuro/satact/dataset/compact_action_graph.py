"""Compact-index facade for ActionEval.

The implementation lives in the SAT-ACT data module because its storage backend
deliberately opens the already materialized compatible indexes.
"""

from satact.dataset.action_graph import (
    ActionEvalCompactShardDataset,
    ActionEvalShardDataset,
)


__all__ = ["ActionEvalCompactShardDataset", "ActionEvalShardDataset"]
