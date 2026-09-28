from __future__ import annotations

import csv
import fcntl
import json
import os
import shlex
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

RUN_REGISTRY_FIELDS = ("run_id", "start_time", "command", "checkpoint_dir")


@dataclass(frozen=True)
class TrainingRun:
    """保存一次训练运行的目录和标识.
    Store the directory and identifier for one training run.
    """

    run_id: str
    checkpoint_dir: Path
    log_path: Path
    config_path: Path


class TeeLogger:
    """同时写入 stdout 和日志文件.
    Write messages to both stdout and a log file.
    """

    def __init__(self, log_path: str | Path) -> None:
        """打开日志文件并创建父目录.
        Open the log file and create its parent directory.
        """
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def close(self) -> None:
        """关闭日志文件.
        Close the log file.
        """
        self.log_file.close()

    def write(self, message: str) -> None:
        """向 stdout 和日志文件写入一行消息.
        Write one message line to stdout and the log file.
        """
        print(message, flush=True)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def __enter__(self) -> TeeLogger:
        """返回当前 logger 以支持 with 语法.
        Return this logger for with-statement usage.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """退出 with 语法时关闭日志文件.
        Close the log file when leaving a with statement.
        """
        self.close()


def utc_now() -> str:
    """返回 UTC ISO 时间字符串.
    Return a UTC ISO timestamp string.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_timestamp_run_id() -> str:
    """生成基于 UTC 时间的训练 run id.
    Generate a UTC timestamp-based training run id.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def default_command() -> str:
    """返回当前 Python 启动命令.
    Return the current Python launch command.
    """
    return shlex.join([sys.executable, *sys.argv])


@contextmanager
def locked_run_registry(registry_path: str | Path) -> Iterator[None]:
    """加锁访问 run registry 文件.
    Lock access to the run registry file.
    """
    path = Path(registry_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_run_registry(registry_path: str | Path) -> list[dict[str, str]]:
    """读取 run registry CSV.
    Read the run registry CSV.
    """
    path = Path(registry_path).expanduser()
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_run_registry(registry_path: str | Path, rows: list[dict[str, str]]) -> None:
    """原子写入 run registry CSV.
    Atomically write the run registry CSV.
    """
    path = Path(registry_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RUN_REGISTRY_FIELDS})
    os.replace(tmp_path, path)


def append_run_registry(registry_path: str | Path, row: Mapping[str, str]) -> None:
    """向 run registry 追加一条记录.
    Append one record to the run registry.
    """
    path = Path(registry_path).expanduser()
    with locked_run_registry(path):
        rows = read_run_registry(path)
        rows.append({field: str(row.get(field, "")) for field in RUN_REGISTRY_FIELDS})
        write_run_registry(path, rows)


def make_numeric_train_id(registry_path: str | Path, command: str | None = None, checkpoint_dir: str | Path | None = None) -> int:
    """生成递增数字 train id 并写入 registry.
    Generate an incremental numeric train id and write it to the registry.
    """
    path = Path(registry_path).expanduser()
    with locked_run_registry(path):
        rows = read_run_registry(path)
        numeric_ids: list[int] = []
        for row in rows:
            try:
                numeric_ids.append(int(row.get("run_id", "0")))
            except ValueError:
                continue
        run_id = max(numeric_ids, default=0) + 1
        rows.append(
            {
                "run_id": str(run_id),
                "start_time": utc_now(),
                "command": command or default_command(),
                "checkpoint_dir": "" if checkpoint_dir is None else str(Path(checkpoint_dir).expanduser()),
            }
        )
        write_run_registry(path, rows)
    return run_id


def create_training_run(
    save_dir: str | Path,
    args: Mapping[str, Any] | object | None = None,
    command: str | None = None,
    run_id: str | None = None,
    run_registry: str | Path | None = None,
) -> TrainingRun:
    """创建训练 run 目录并写入配置和 registry.
    Create a training run directory and write config and registry records.
    """
    resolved_run_id = run_id or make_timestamp_run_id()
    checkpoint_dir = Path(save_dir).expanduser().resolve() / f"run_{resolved_run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    run = TrainingRun(
        run_id=str(resolved_run_id),
        checkpoint_dir=checkpoint_dir,
        log_path=checkpoint_dir / "train.log",
        config_path=checkpoint_dir / "run_config.json",
    )
    resolved_command = command or default_command()
    write_run_config(run, args=args, command=resolved_command)
    if run_registry is not None:
        append_run_registry(
            run_registry,
            {
                "run_id": run.run_id,
                "start_time": utc_now(),
                "command": resolved_command,
                "checkpoint_dir": str(run.checkpoint_dir),
            },
        )
    return run


def _args_to_dict(args: Mapping[str, Any] | object | None) -> dict[str, Any]:
    """将配置对象转换为可 JSON 序列化的字典.
    Convert a config object to a JSON-serializable dictionary.
    """
    if args is None:
        return {}
    if isinstance(args, Mapping):
        return dict(args)
    if hasattr(args, "__dict__"):
        return vars(args)
    raise TypeError("args must be a mapping, an object with __dict__, or None")


def write_run_config(
    run: TrainingRun,
    args: Mapping[str, Any] | object | None = None,
    command: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """写入 run_config.json.
    Write run_config.json.
    """
    config = {
        "run_id": run.run_id,
        "checkpoint_dir": str(run.checkpoint_dir),
        "command": command or default_command(),
        "argv": [sys.executable, *sys.argv],
        "args": _args_to_dict(args),
    }
    if extra:
        config.update(dict(extra))
    with open(run.config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True, default=str)
        config_file.write("\n")


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    run_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    opts: Mapping[str, Any] | object | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 checkpoint 保存载荷.
    Build a checkpoint payload.
    """
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "run_id": run_id,
        "metrics": dict(metrics or {}),
        "opts": _args_to_dict(opts),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(dict(extra))
    return payload


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    run_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    opts: Mapping[str, Any] | object | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """保存 checkpoint 到指定路径.
    Save a checkpoint to the given path.
    """
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            run_id=run_id,
            metrics=metrics,
            opts=opts,
            extra=extra,
        ),
        checkpoint_path,
    )
    return checkpoint_path


def save_run_checkpoint(
    run: TrainingRun,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    metrics: Mapping[str, Any] | None = None,
    opts: Mapping[str, Any] | object | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """按名称保存 run 目录下的 checkpoint.
    Save a named checkpoint under a run directory.
    """
    filename = name if name.endswith(".pt") else f"{name}.pt"
    return save_checkpoint(
        run.checkpoint_dir / filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        run_id=run.run_id,
        metrics=metrics,
        opts=opts,
        extra=extra,
    )


def _torch_load_checkpoint(path: Path, map_location: str | torch.device | None) -> dict[str, Any]:
    """兼容不同 PyTorch 版本读取 checkpoint.
    Load a checkpoint across different PyTorch versions.
    """
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint must contain a dict, got {type(loaded).__name__}")
    return loaded


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """读取 checkpoint 并可选恢复训练状态.
    Load a checkpoint and optionally restore training state.
    """
    checkpoint_path = Path(path).expanduser()
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=map_location)
    if model is not None:
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def resume_training_state(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> tuple[int, dict[str, Any]]:
    """恢复训练状态并返回下一轮 epoch.
    Restore training state and return the next epoch.
    """
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=map_location,
        strict=strict,
    )
    return int(checkpoint.get("epoch", 0)) + 1, checkpoint
