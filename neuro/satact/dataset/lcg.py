from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data


def _scalar_to_int(value: Any) -> int:
    """将标量或单元素张量转换为 Python int.
    Convert a scalar or one-element tensor to a Python int.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got shape {tuple(value.shape)}")
        return int(value.item())
    return int(value)


class LCG(Data):
    """保存单个 Literal-Clause Graph 样本.
    Store one Literal-Clause Graph sample.
    """

    def __init__(
        self,
        n_vars: Any = None,
        n_clauses: Any = None,
        l_edge_index: torch.Tensor | None = None,
        c_edge_index: torch.Tensor | None = None,
        label_var_mask: torch.Tensor | None = None,
        label_clause_mask: torch.Tensor | None = None,
        l_batch: torch.Tensor | None = None,
        c_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 LCG 字段.
        Initialize LCG fields.
        """
        super().__init__(**kwargs)
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index
        self.label_var_mask = label_var_mask
        if label_clause_mask is None:
            object.__setattr__(self, "label_clause_mask", None)
        else:
            self.label_clause_mask = label_clause_mask
        self.l_batch = l_batch
        self.c_batch = c_batch

    @property
    def num_edges(self) -> int:
        """返回 literal-clause occurrence 边数量.
        Return the number of literal-clause occurrence edges.
        """
        if self.c_edge_index is None:
            return 0
        return int(self.c_edge_index.numel())

    @property
    def num_variables(self) -> int:
        """返回变量数量的 Python int 形式.
        Return the variable count as a Python int.
        """
        return _scalar_to_int(self.n_vars)

    @property
    def num_clauses_lcg(self) -> int:
        """返回子句数量的 Python int 形式.
        Return the clause count as a Python int.
        """
        return _scalar_to_int(self.n_clauses)

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> int:
        """定义 PyG 合批时各字段的索引偏移.
        Define index increments for PyG batching.
        """
        if key == "l_edge_index":
            return self.num_variables * 2
        if key == "c_edge_index":
            return self.num_clauses_lcg
        if key in {"l_batch", "c_batch", "positive_index"}:
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    def validate_lcg(self) -> None:
        """检查 LCG 字段之间的一致性.
        Validate consistency among LCG fields.
        """
        n_vars = self.num_variables
        n_clauses = self.num_clauses_lcg
        if n_vars < 0:
            raise ValueError("n_vars must be non-negative")
        if n_clauses < 0:
            raise ValueError("n_clauses must be non-negative")
        if self.l_edge_index is None or self.c_edge_index is None:
            raise ValueError("l_edge_index and c_edge_index are required")
        if self.l_edge_index.dim() != 1 or self.c_edge_index.dim() != 1:
            raise ValueError("l_edge_index and c_edge_index must be 1-D tensors")
        if self.l_edge_index.numel() != self.c_edge_index.numel():
            raise ValueError("l_edge_index and c_edge_index must have the same length")
        if self.l_edge_index.numel() > 0:
            if int(self.l_edge_index.min().item()) < 0:
                raise ValueError("l_edge_index contains negative literal indices")
            if int(self.l_edge_index.max().item()) >= 2 * n_vars:
                raise ValueError("l_edge_index contains out-of-range literal indices")
            if int(self.c_edge_index.min().item()) < 0:
                raise ValueError("c_edge_index contains negative clause indices")
            if int(self.c_edge_index.max().item()) >= n_clauses:
                raise ValueError("c_edge_index contains out-of-range clause indices")
        label_var_mask = getattr(self, "label_var_mask", None)
        label_clause_mask = getattr(self, "label_clause_mask", None)
        if label_var_mask is not None and label_var_mask.numel() != n_vars:
            raise ValueError("label_var_mask length must equal n_vars")
        if label_clause_mask is not None and label_clause_mask.numel() != n_clauses:
            raise ValueError("label_clause_mask length must equal n_clauses")
        if self.l_batch is not None and self.l_batch.numel() != 2 * n_vars:
            raise ValueError("l_batch length must equal 2 * n_vars")
        if self.c_batch is not None and self.c_batch.numel() != n_clauses:
            raise ValueError("c_batch length must equal n_clauses")
