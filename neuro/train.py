#!/usr/bin/env python3
"""Train the independent DecisionTrace ActionEval SAT-ACT pair policy."""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

NEURO_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = NEURO_ROOT.parent
for search_path in (REPOSITORY_ROOT, NEURO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from satact.dataset.action_preferences import (
    SATACT_FILTERS,
    HEURISTIC_SUPERVISION_MODES,
    SATACT_VARIANT,
    ObjectiveConfigSATACT,
    canonicalize_satact_variant,
)
from satact.dataset.heuristic_supervision import (
    SATACT_INDEX_CONTRACTS,
    SATACT_NATIVE_OUTCOMES,
    validate_satact_index_selection,
)
from satact.models.model_factory import READOUTS
from satact.models.satact_factory import build_actioneval_model_satact
from satact.training.checkpoint import (
    RunSATACT,
    append_metrics_satact,
    capture_rng_satact,
    create_run_satact,
    load_checkpoint_satact,
    restore_rng_satact,
    save_checkpoint_satact,
    validate_init_checkpoint_satact,
    validate_resume_satact,
)
from satact.training.distributed import DistributedContext, initialize_distributed
from satact.training.run_management import TeeLogger


class _NullLogger:
    def write(self, message: str) -> None:
        del message

    def __enter__(self) -> "_NullLogger":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    for split in ("train", "valid", "test"):
        data.add_argument(f"--{split}_dir", required=split == "train")
        data.add_argument(f"--{split}_index_cache_path", required=split == "train")
        data.add_argument(f"--max_{split}_states", type=int, default=0)
    data.add_argument("--batch_size", type=int, default=32)
    data.add_argument("--valid_batch_size", type=int, default=None)
    data.add_argument("--test_batch_size", type=int, default=None)
    data.add_argument("--pair_filter", choices=SATACT_FILTERS, default="all")
    data.add_argument(
        "--index_contract",
        dest="satact_index_contract",
        choices=SATACT_INDEX_CONTRACTS,
        default="preference-only",
    )
    data.add_argument(
        "--heuristic_action_source",
        dest="satact_native_outcome",
        choices=SATACT_NATIVE_OUTCOMES,
        default="replay",
    )
    data.add_argument("--verify", choices=("none", "metadata"), default="metadata")
    data.add_argument("--max_open_shards", type=int, default=1)
    data.add_argument("--max_header_mb", type=int, default=1024)
    data.add_argument("--max_array_mb", type=int, default=512)
    data.add_argument("--shard_shuffle", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--shard_shuffle_window", type=int, default=8192)
    data.add_argument("--num_workers", type=int, default=0)
    data.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--prefetch_factor", type=int, default=2)
    data.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=(SATACT_VARIANT,), default=SATACT_VARIANT)
    model.add_argument("--model", choices=("neurocore", "neurosat"), default="neurocore")
    model.add_argument("--embedding_dim", type=int, default=64)
    model.add_argument("--num_rounds", type=int, default=4)
    model.add_argument("--mlp_layers", type=int, default=2)
    model.add_argument("--readout", choices=READOUTS, default="contextual")
    model.add_argument("--shared_updates", action=argparse.BooleanOptionalAction, default=False)
    model.add_argument("--use_layer_norm", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--use_residual", action=argparse.BooleanOptionalAction, default=True)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument(
        "--heuristic_supervision_weight", dest="native_weight", type=float, default=0.0
    )
    optimization.add_argument(
        "--heuristic_supervision_mode",
        dest="native_supervision",
        choices=HEURISTIC_SUPERVISION_MODES,
        default="always",
    )
    optimization.add_argument("--top_set_weight", type=float, default=0.0)
    optimization.add_argument("--epochs", type=int, default=10)
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--weight_decay", type=float, default=0.0)
    optimization.add_argument("--adam_beta1", type=float, default=0.9)
    optimization.add_argument("--adam_beta2", type=float, default=0.999)
    optimization.add_argument("--adam_eps", type=float, default=1e-8)
    optimization.add_argument("--gradient_clip_norm", type=float, default=0.0)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    optimization.add_argument(
        "--train_metrics", choices=("loss", "selection", "diagnostics", "full"), default="loss"
    )
    optimization.add_argument(
        "--validation_metrics",
        dest="satact_valid_metrics",
        choices=("loss", "selection", "diagnostics", "full"),
        default="diagnostics",
    )
    optimization.add_argument(
        "--validation_every_epochs", dest="satact_valid_every_epochs", type=int, default=1
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--distributed_backend", choices=("auto", "nccl", "gloo"), default="auto")
    runtime.add_argument("--distributed_timeout_minutes", type=float, default=30.0)
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)

    run = parser.add_argument_group("run")
    run.add_argument("--run_root", required=True)
    run.add_argument("--experiment_key", default="default_neurocore_seed42")
    run.add_argument("--run_id", default=None)
    initialization = run.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None)
    initialization.add_argument(
        "--init_checkpoint",
        default=None,
        help=(
            "Strictly load model weights only; optimizer, scaler, epoch, best "
            "scores, RNG, and source split fingerprints are not restored"
        ),
    )
    run.add_argument("--checkpoint_every_epochs", type=int, default=0,
                     help="Save an additional epoch checkpoint every N epochs; 0 disables it")
    run.add_argument("--allow_data_change_on_resume", action=argparse.BooleanOptionalAction, default=False)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "resume", None) and getattr(args, "init_checkpoint", None):
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")
    args.variant = canonicalize_satact_variant(args.variant)
    args.satact_index_contract, args.satact_native_outcome = validate_satact_index_selection(
        args.satact_index_contract, args.satact_native_outcome
    )
    for name in (
        "batch_size",
        "epochs",
        "embedding_dim",
        "num_rounds",
        "mlp_layers",
        "max_open_shards",
        "max_header_mb",
        "max_array_mb",
        "shard_shuffle_window",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    for split in ("valid", "test"):
        directory = getattr(args, f"{split}_dir")
        path = getattr(args, f"{split}_index_cache_path")
        if (directory is None) != (path is None):
            raise ValueError(f"--{split}_dir and --{split}_index_cache_path must be supplied together")
    for name in ("max_train_states", "max_valid_states", "max_test_states", "num_workers", "satact_valid_every_epochs"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.num_workers == 0 and args.persistent_workers:
        raise ValueError("--persistent_workers requires --num_workers > 0")
    if getattr(args, "checkpoint_every_epochs", 0) < 0:
        raise ValueError("--checkpoint_every_epochs must be non-negative")
    if not math.isfinite(args.native_weight) or args.native_weight < 0:
        raise ValueError("--heuristic_supervision_weight must be finite and non-negative")
    if not math.isfinite(args.top_set_weight) or args.top_set_weight < 0:
        raise ValueError("--top_set_weight must be finite and non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight_decay must be finite and non-negative")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if not math.isfinite(args.adam_eps) or args.adam_eps <= 0:
        raise ValueError("--adam_eps must be finite and positive")
    if args.amp and args.device == "cpu":
        raise ValueError("--amp requires CUDA")


def _batch_size(args: argparse.Namespace, split: str) -> int:
    value = getattr(args, f"{split}_batch_size", None)
    return args.batch_size if value is None else value


def _metric_mode(args: argparse.Namespace, split: str) -> str:
    if split == "train":
        return args.train_metrics
    if split == "valid":
        return args.satact_valid_metrics
    return "full"


def build_split_loader_satact(
    args: argparse.Namespace, split: str, distributed: DistributedContext
):
    from satact.training.dataloader import build_actioneval_satact_loader

    directory = getattr(args, f"{split}_dir")
    if directory is None:
        return None, None
    return build_actioneval_satact_loader(
        directory,
        index_cache_path=getattr(args, f"{split}_index_cache_path"),
        batch_size=_batch_size(args, split),
        training=split == "train",
        expected_split=split,
        pair_filter=args.pair_filter,
        metric_mode=_metric_mode(args, split),
        include_top_set_supervision=(
            args.top_set_weight > 0
            or (
                args.native_weight > 0
                and args.native_supervision == "top-set"
            )
        ),
        satact_index_contract=args.satact_index_contract,
        satact_native_outcome=args.satact_native_outcome,
        preprocess_workers=1,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and distributed.device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        verify=args.verify != "none",
        audit_checksum=args.verify == "checksum",
        index_cache_read_only=True,
        max_states=getattr(args, f"max_{split}_states") or None,
        max_open_shards=args.max_open_shards,
        max_header_bytes=args.max_header_mb * (1 << 20),
        max_array_bytes=args.max_array_mb * (1 << 20),
        shard_shuffle=args.shard_shuffle,
        shard_shuffle_window=args.shard_shuffle_window,
        distributed_rank=distributed.rank,
        distributed_world_size=distributed.world_size,
    )


def _set_epoch(loader: Any, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def run_epoch_satact(
    model: torch.nn.Module,
    loader: Any,
    *,
    config: ObjectiveConfigSATACT,
    metric_mode: str,
    device: torch.device,
    distributed: DistributedContext,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 0.0,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float | int]:
    from satact.training.objective import SATACTMetricsAccumulator, compute_satact_loss

    training = optimizer is not None
    model.train(training)
    accumulator = SATACTMetricsAccumulator(metric_mode)  # type: ignore[arg-type]
    batches = 0
    edges = 0
    started = time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            edges += batch.graph.l_edge_index.numel()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pending = (
                distributed.begin_weighted_objective(batch.supervision.state_weights.sum())
                if training
                else None
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp and device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast:
                output = model(batch.graph)
            with torch.autocast(device_type=device.type, enabled=False):
                result = compute_satact_loss(output, batch, config, metric_mode=metric_mode)
            if optimizer is not None:
                loss = result.loss if pending is None else pending.scale(result.loss, result.weight_sum)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            accumulator.update(result)
            batches += 1
    reduced = accumulator.reduce(distributed=distributed.enabled)
    metrics: dict[str, float | int] = reduced.summary()
    counters = torch.tensor([batches, edges], dtype=torch.float64, device=device)
    elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    if distributed.enabled:
        torch.distributed.all_reduce(counters)
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
    total_batches, total_edges = counters.cpu().tolist()
    seconds = float(elapsed.cpu())
    metrics.update(
        batch_count=int(total_batches),
        optimizer_steps=batches if training else 0,
        elapsed_seconds=seconds,
        states_per_second=float(metrics["state_count"]) / seconds,
        edges_per_second=float(total_edges) / seconds,
    )
    return metrics


def _format(metrics: Mapping[str, Any]) -> str:
    return " ".join(
        f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}"
        for name, value in sorted(metrics.items())
        if isinstance(value, (float, int, str, bool))
    )


def _should_validate(epoch: int, epochs: int, interval: int) -> bool:
    return epoch == epochs or (interval > 0 and epoch % interval == 0)


def _primary_best_selector_satact(valid_metric_mode: str) -> str:
    return "best_objective" if valid_metric_mode == "loss" else "best_pair_accuracy"


def _load_selected_model_state_satact(
    path: Path, *, expected_selector: str
) -> tuple[int, dict[str, torch.Tensor]] | None:
    if not path.is_file():
        return None
    checkpoint = load_checkpoint_satact(path, map_location="cpu")
    selector = str(checkpoint.get("selector", ""))
    if selector != expected_selector:
        raise ValueError(
            f"SATACT test checkpoint selector mismatch: expected {expected_selector!r}, "
            f"got {selector!r}"
        )
    return int(checkpoint["epoch"]), dict(checkpoint["model"])


def _clone_model_state_cpu_satact(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _apply_optimizer_options_satact(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["weight_decay"] = args.weight_decay
        group["betas"] = (args.adam_beta1, args.adam_beta2)
        group["eps"] = args.adam_eps


def _save(
    path: Path,
    *,
    run: RunSATACT,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    best_scores: Mapping[str, float],
    rng_states: list[Mapping[str, Any]],
    selector: str,
    scaler: torch.amp.GradScaler,
) -> None:
    del run
    save_checkpoint_satact(
        path,
        model=model,
        optimizer=optimizer,
        args=args,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        split_fingerprints=fingerprints,
        best_scores=best_scores,
        rng_states=rng_states,
        selector=selector,
        scaler=scaler,
    )


def train_and_evaluate(args: argparse.Namespace) -> Path:
    validate_args(args)
    distributed = initialize_distributed(
        args.device, args.distributed_backend, args.distributed_timeout_minutes
    )
    datasets: dict[str, Any] = {}
    try:
        random.seed(args.seed + distributed.rank)
        torch.manual_seed(args.seed + distributed.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + distributed.rank)
        torch.use_deterministic_algorithms(args.deterministic)
        init_checkpoint: dict[str, Any] | None = None
        if args.init_checkpoint:
            init_checkpoint = distributed.call_on_main(
                lambda: load_checkpoint_satact(args.init_checkpoint, map_location="cpu")
            )
            validate_init_checkpoint_satact(init_checkpoint, args)
            args.initialization = {
                "scope": "full",
                "source_path": str(
                    Path(args.init_checkpoint).expanduser().resolve()
                ),
                "source_variant": str(init_checkpoint.get("variant", "")),
                "source_epoch": int(init_checkpoint.get("epoch", 0)),
                "source_selector": str(init_checkpoint.get("selector", "")),
            }
        run = distributed.call_on_main(lambda: create_run_satact(args, run_id=args.run_id))
        logger_context: Any = TeeLogger(run.log_path) if distributed.is_main else _NullLogger()
        with logger_context as logger:
            logger.write(f"run_id={run.run_id}")
            logger.write(f"run_dir={run.run_dir}")
            logger.write(
                f"device={distributed.device} distributed={distributed.enabled} "
                f"rank={distributed.rank}/{distributed.world_size} backend={distributed.backend}"
            )
            logger.write(f"checkpoints={run.checkpoint_dir}")
            loaders: dict[str, Any] = {}
            for split in ("train", "valid", "test"):
                dataset, loader = build_split_loader_satact(args, split, distributed)
                datasets[split], loaders[split] = dataset, loader
                if dataset is not None:
                    logger.write(
                        f"split={split} states={len(dataset)} instances={dataset.instance_count} "
                        f"metric_mode={_metric_mode(args, split)} filter={args.pair_filter}"
                    )
            if not len(datasets["train"]):
                raise ValueError("train split has no SATACT pair states after filtering")
            fingerprints = {
                split: dataset.split_fingerprint
                for split, dataset in datasets.items()
                if dataset is not None
            }
            base_model = build_actioneval_model_satact(args).to(distributed.device)
            if init_checkpoint is not None:
                base_model.load_state_dict(init_checkpoint["model"], strict=True)
                logger.write(
                    "init_checkpoint "
                    f"scope=full source={args.initialization['source_path']} "
                    f"epoch={args.initialization['source_epoch']} "
                    f"selector={args.initialization['source_selector']} "
                    "optimizer=fresh scaler=fresh start_epoch=1 global_step=0"
                )
                del init_checkpoint
            model = distributed.wrap_model(base_model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
            )
            scaler = torch.amp.GradScaler(
                "cuda", enabled=args.amp and distributed.device.type == "cuda"
            )
            start_epoch, global_step = 1, 0
            best_scores: dict[str, float] = {}
            if args.resume:
                checkpoint = distributed.call_on_main(
                    lambda: load_checkpoint_satact(args.resume, map_location="cpu")
                )
                validate_resume_satact(checkpoint, args)
                previous = dict(checkpoint.get("split_fingerprints") or {})
                if previous != fingerprints and not args.allow_data_change_on_resume:
                    raise ValueError("resume split fingerprints differ from current SATACT indexes")
                distributed.unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                _apply_optimizer_options_satact(optimizer, args)
                logger.write(
                    "resume_optimizer "
                    f"lr={args.lr:.12g} weight_decay={args.weight_decay:.12g} "
                    f"betas=({args.adam_beta1:.12g},{args.adam_beta2:.12g}) "
                    f"eps={args.adam_eps:.12g}"
                )
                if checkpoint.get("scaler") is not None:
                    scaler.load_state_dict(checkpoint["scaler"])
                start_epoch = int(checkpoint["epoch"]) + 1
                global_step = int(checkpoint.get("global_step", 0))
                best_scores = {str(k): float(v) for k, v in dict(checkpoint.get("best_scores") or {}).items()}
                states = list(checkpoint.get("rng_states") or [])
                if len(states) == distributed.world_size:
                    restore_rng_satact(states[distributed.rank])
            if start_epoch > args.epochs:
                raise ValueError(
                    f"--epochs={args.epochs} is before the next resume epoch {start_epoch}"
                )
            config = ObjectiveConfigSATACT(
                native_weight=args.native_weight,
                native_supervision=args.native_supervision,
                top_set_weight=args.top_set_weight,
            )
            final_metrics: dict[str, Any] = {}
            final_epoch = start_epoch - 1
            for epoch in range(start_epoch, args.epochs + 1):
                final_epoch = epoch
                _set_epoch(loaders["train"], epoch)
                train_metrics = run_epoch_satact(
                    model,
                    loaders["train"],
                    config=config,
                    metric_mode=args.train_metrics,
                    device=distributed.device,
                    distributed=distributed,
                    optimizer=optimizer,
                    gradient_clip_norm=args.gradient_clip_norm,
                    amp=args.amp,
                    scaler=scaler,
                )
                global_step += int(train_metrics["optimizer_steps"])
                final_metrics = {"train": train_metrics}
                logger.write(f"epoch={epoch} stage=train {_format(train_metrics)}")
                if distributed.is_main:
                    append_metrics_satact(run, epoch=epoch, stage="train", metrics=train_metrics)

                improved_objective = False
                improved_pair = False
                improved_top_set = False
                if loaders["valid"] is not None and _should_validate(
                    epoch, args.epochs, args.satact_valid_every_epochs
                ):
                    valid_metrics = run_epoch_satact(
                        distributed.unwrap_model(model),
                        loaders["valid"],
                        config=config,
                        metric_mode=args.satact_valid_metrics,
                        device=distributed.device,
                        distributed=distributed,
                        amp=args.amp,
                    )
                    final_metrics["valid"] = valid_metrics
                    logger.write(f"epoch={epoch} stage=valid {_format(valid_metrics)}")
                    if distributed.is_main:
                        append_metrics_satact(run, epoch=epoch, stage="valid", metrics=valid_metrics)
                    if args.satact_valid_metrics == "loss":
                        objective_value = float(valid_metrics["objective_loss"])
                        improved_objective = objective_value < best_scores.get(
                            "best_objective", math.inf
                        )
                        if improved_objective:
                            best_scores["best_objective"] = objective_value
                    else:
                        pair_value = float(valid_metrics["pair_accuracy"])
                        improved_pair = pair_value > best_scores.get("best_pair_accuracy", -math.inf)
                        if improved_pair:
                            best_scores["best_pair_accuracy"] = pair_value
                        if (
                            args.top_set_weight > 0
                            or (
                                args.native_weight > 0
                                and args.native_supervision == "top-set"
                            )
                        ):
                            top_set_value = float(valid_metrics["top_set_accuracy"])
                            improved_top_set = top_set_value > best_scores.get(
                                "best_top_set_accuracy", -math.inf
                            )
                            if improved_top_set:
                                best_scores["best_top_set_accuracy"] = top_set_value

                rng_states = distributed.all_gather_object(capture_rng_satact())

                def save_epoch() -> None:
                    common = dict(
                        run=run,
                        model=distributed.unwrap_model(model),
                        optimizer=optimizer,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        metrics=final_metrics,
                        fingerprints=fingerprints,
                        best_scores=best_scores,
                        rng_states=rng_states,
                        scaler=scaler,
                    )
                    _save(run.checkpoint_dir / "latest.pt", selector="latest", **common)
                    checkpoint_interval = getattr(args, "checkpoint_every_epochs", 0)
                    if checkpoint_interval and epoch % checkpoint_interval == 0:
                        _save(run.checkpoint_dir / f"epoch_{epoch:04d}.pt", selector="epoch", **common)
                    if improved_objective:
                        _save(run.checkpoint_dir / "best_objective.pt", selector="best_objective", **common)
                    if improved_pair:
                        _save(run.checkpoint_dir / "best_pair_accuracy.pt", selector="best_pair_accuracy", **common)
                    if improved_top_set:
                        _save(
                            run.checkpoint_dir / "best_top_set_accuracy.pt",
                            selector="best_top_set_accuracy",
                            **common,
                        )

                distributed.call_on_main(save_epoch)

            unwrapped_model = distributed.unwrap_model(model)
            if loaders["test"] is not None:
                test_metrics = run_epoch_satact(
                    unwrapped_model,
                    loaders["test"],
                    config=config,
                    metric_mode="full",
                    device=distributed.device,
                    distributed=distributed,
                    amp=args.amp,
                )
                final_metrics["test"] = test_metrics
                logger.write(f"epoch={final_epoch} stage=test {_format(test_metrics)}")
                if distributed.is_main:
                    append_metrics_satact(run, epoch=final_epoch, stage="test", metrics=test_metrics)
            rng_states = distributed.all_gather_object(capture_rng_satact())

            if loaders["test"] is not None:
                primary_selector = _primary_best_selector_satact(args.satact_valid_metrics)
                best_test_stage = f"test_{primary_selector}"
                if loaders["valid"] is None:
                    logger.write(
                        f"epoch={final_epoch} stage={best_test_stage} "
                        "skipped=no_validation_split"
                    )
                else:
                    best_path = run.checkpoint_dir / f"{primary_selector}.pt"
                    selected = distributed.call_on_main(
                        lambda: _load_selected_model_state_satact(
                            best_path, expected_selector=primary_selector
                        )
                    )
                    if selected is None:
                        logger.write(
                            f"epoch={final_epoch} stage={best_test_stage} "
                            f"skipped=checkpoint_unavailable checkpoint={best_path}"
                        )
                    else:
                        best_epoch, best_state = selected
                        del selected
                        if best_epoch == final_epoch:
                            best_test_metrics = dict(final_metrics["test"])
                            del best_state
                            logger.write(
                                f"epoch={best_epoch} stage={best_test_stage} "
                                f"reused_final_test=true checkpoint={best_path} "
                                f"{_format(best_test_metrics)}"
                            )
                        else:
                            final_state = _clone_model_state_cpu_satact(unwrapped_model)
                            try:
                                unwrapped_model.load_state_dict(best_state, strict=True)
                                del best_state
                                best_test_metrics = run_epoch_satact(
                                    unwrapped_model,
                                    loaders["test"],
                                    config=config,
                                    metric_mode="full",
                                    device=distributed.device,
                                    distributed=distributed,
                                    amp=args.amp,
                                )
                            finally:
                                unwrapped_model.load_state_dict(final_state, strict=True)
                                del final_state
                            logger.write(
                                f"epoch={best_epoch} stage={best_test_stage} "
                                f"checkpoint={best_path} {_format(best_test_metrics)}"
                            )
                        final_metrics[best_test_stage] = best_test_metrics
                        if distributed.is_main:
                            append_metrics_satact(
                                run,
                                epoch=best_epoch,
                                stage=best_test_stage,
                                metrics=best_test_metrics,
                            )

            distributed.call_on_main(
                lambda: _save(
                    run.checkpoint_dir / "last.pt",
                    run=run,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    args=args,
                    epoch=final_epoch,
                    global_step=global_step,
                    metrics=final_metrics,
                    fingerprints=fingerprints,
                    best_scores=best_scores,
                    rng_states=rng_states,
                    selector="last",
                    scaler=scaler,
                )
            )
            return run.checkpoint_dir
    finally:
        for dataset in datasets.values():
            if dataset is not None:
                dataset.close()
        distributed.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"checkpoints={train_and_evaluate(args)}", flush=True)


if __name__ == "__main__":
    main()
