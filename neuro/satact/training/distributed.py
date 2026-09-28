from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


T = TypeVar("T")


@dataclass(frozen=True)
class PendingWeightReduction:
    global_weight_sum: torch.Tensor
    work: Any | None
    world_size: int

    def scale(self, local_mean: torch.Tensor, local_weight_sum: torch.Tensor) -> torch.Tensor:
        if self.work is not None:
            self.work.wait()
        return local_mean * local_weight_sum * self.world_size / self.global_weight_sum


def _environment_int(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"distributed environment is missing {name}")
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"distributed environment variable {name} must be an integer") from exc


@dataclass(frozen=True)
class DistributedContext:
    """Describe one torchrun worker and provide rank-safe collective helpers."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def call_on_main(self, function: Callable[[], T]) -> T:
        """Run a filesystem-mutating operation on rank zero and broadcast its result."""
        if not self.enabled:
            return function()
        payload: list[tuple[str, Any] | None] = [None]
        if self.is_main:
            try:
                payload[0] = ("ok", function())
            except Exception as exc:  # noqa: BLE001 - propagate rank-zero failure to every worker
                payload[0] = ("error", f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(payload, src=0)
        status, value = payload[0]  # type: ignore[misc]
        if status == "error":
            raise RuntimeError(f"rank-zero operation failed: {value}")
        return value

    def all_gather_object(self, value: T) -> list[T]:
        if not self.enabled:
            return [value]
        gathered: list[T | None] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return [item for item in gathered if item is not None]

    def scale_weighted_objective(
        self,
        local_mean: torch.Tensor,
        local_weight_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Scale a local weighted mean so DDP averaging yields the global objective gradient."""
        if not self.enabled:
            return local_mean
        pending = self.begin_weighted_objective(local_weight_sum)
        return pending.scale(local_mean, local_weight_sum)

    def begin_weighted_objective(
        self,
        local_weight_sum: torch.Tensor,
    ) -> PendingWeightReduction:
        """Start the scalar reduction early so it overlaps model computation."""

        global_weight_sum = local_weight_sum.detach().clone()
        work = None
        if self.enabled:
            work = dist.all_reduce(
                global_weight_sum,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
        return PendingWeightReduction(
            global_weight_sum=global_weight_sum,
            work=work,
            world_size=self.world_size,
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if not self.enabled:
            return model
        if self.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        return DistributedDataParallel(
            model,
            broadcast_buffers=False,
        )

    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DistributedDataParallel) else model

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(
    requested_device: str,
    requested_backend: str = "auto",
    timeout_minutes: float = 30.0,
) -> DistributedContext:
    """Initialize torchrun DDP when WORLD_SIZE is greater than one."""
    if requested_backend not in {"auto", "nccl", "gloo"}:
        raise ValueError("distributed backend must be auto, nccl, or gloo")
    if timeout_minutes <= 0:
        raise ValueError("distributed timeout must be positive")

    world_size = _environment_int("WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    enabled = world_size > 1
    rank = _environment_int("RANK") if enabled else 0
    local_rank = _environment_int("LOCAL_RANK") if enabled else 0
    if enabled and not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")

    requested = torch.device(requested_device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            if enabled:
                raise RuntimeError("distributed CUDA training requested but CUDA is unavailable")
            device = torch.device("cpu")
        elif enabled:
            device_count = torch.cuda.device_count()
            if not 0 <= local_rank < device_count:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} cannot be mapped to {device_count} visible CUDA devices"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = requested
    else:
        device = requested

    backend: str | None = None
    if enabled:
        backend = "nccl" if requested_backend == "auto" and device.type == "cuda" else requested_backend
        if backend == "auto":
            backend = "gloo"
        if backend == "nccl" and device.type != "cuda":
            raise ValueError("NCCL requires a CUDA device")
        if backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("this PyTorch build does not provide NCCL")
        if backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("this PyTorch build does not provide Gloo")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=float(timeout_minutes)),
        )

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
    )
