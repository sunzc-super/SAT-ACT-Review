"""Pair objective and four-level metrics for ActionEval SAT-ACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, Mapping

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch_geometric.utils import scatter

from satact.dataset.action_preferences import (
    ActionEvalBatchSATACT,
    ObjectiveConfigSATACT,
)


MetricModeSATACT = Literal["loss", "selection", "diagnostics", "full"]
_MODE_LEVEL: Final[dict[str, int]] = {
    "loss": 0,
    "selection": 1,
    "diagnostics": 2,
    "full": 3,
}
METRIC_FIELDS_SATACT: Final[tuple[str, ...]] = (
    "state_count",
    "instance_weight_sum",
    "objective_weighted_sum",
    "value_weighted_sum",
    "top_set_weighted_sum",
    "top_set_weight_sum",
    "native_weighted_sum",
    "native_present_sum",
    "native_ce_active_sum",
    "pair_correct_sum",
    "pair_total",
    "top_set_correct_sum",
    "top_set_total",
    "top_set_probability_mass_sum",
    "top_set_probability_mass_count",
    "native_top_set_sum",
    "native_top_set_total",
    "native_override_correct_sum",
    "native_override_total",
    "top1_total",
    "top1_evaluated_sum",
    "top1_regret_conflicts_sum",
    "top1_regret_conflicts_count",
    "top1_regret_propagations_sum",
    "top1_regret_propagations_count",
    "top1_regret_decisions_sum",
    "top1_regret_decisions_count",
    "top1_regret_restarts_sum",
    "top1_regret_restarts_count",
    "native_count",
    "native_top1_sum",
    "native_top3_sum",
    "native_top5_sum",
    "native_top10_sum",
    "native_reciprocal_rank_sum",
    "r1_correct_sum",
    "r1_total",
    "r2_correct_sum",
    "r2_total",
    "r3_correct_sum",
    "r3_total",
    "r1_only_correct_sum",
    "r1_only_total",
    "r2_only_correct_sum",
    "r2_only_total",
    "r3_only_correct_sum",
    "r3_only_total",
    "r2_r3_overlap_correct_sum",
    "r2_r3_overlap_total",
)
_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(METRIC_FIELDS_SATACT)
}


def _segment_sum(values: torch.Tensor, state: torch.Tensor, size: int) -> torch.Tensor:
    return scatter(values, state, dim=0, dim_size=size, reduce="sum")


def _segment_max(values: torch.Tensor, state: torch.Tensor, size: int) -> torch.Tensor:
    return scatter(values, state, dim=0, dim_size=size, reduce="max")


def _segment_min(values: torch.Tensor, state: torch.Tensor, size: int) -> torch.Tensor:
    return scatter(values, state, dim=0, dim_size=size, reduce="min")


def _segment_logsumexp(
    values: torch.Tensor, state: torch.Tensor, size: int
) -> torch.Tensor:
    maximum = _segment_max(values, state, size)
    shifted = torch.exp(values - maximum[state])
    return maximum + torch.log(_segment_sum(shifted, state, size))


def _native_losses(
    logits: torch.Tensor, batch: ActionEvalBatchSATACT
) -> tuple[torch.Tensor, torch.Tensor]:
    supervision = batch.supervision
    eligibility = supervision.tensor("eligibility")
    literal_state = supervision.tensor("literal_state")
    native = supervision.tensor("native_literal")
    denominator = _segment_logsumexp(
        logits[eligibility], literal_state[eligibility], batch.batch_size
    )
    present = native >= 0
    safe_native = native.clamp_min(0)
    losses = torch.where(
        present,
        denominator - logits[safe_native],
        torch.zeros_like(denominator),
    )
    return losses, present


def _pair_values(
    logits: torch.Tensor, batch: ActionEvalBatchSATACT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supervision = batch.supervision
    better = logits[supervision.tensor("pair_better")]
    worse = logits[supervision.tensor("pair_worse")]
    state = supervision.tensor("pair_state")
    losses = F.softplus(-(better - worse))
    totals = _segment_sum(losses, state, batch.batch_size)
    counts = _segment_sum(torch.ones_like(losses), state, batch.batch_size)
    return totals / counts, better.detach(), worse.detach()


def _top_set_values(
    logits: torch.Tensor, batch: ActionEvalBatchSATACT
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    supervision = batch.supervision
    size = batch.batch_size
    evaluated = supervision.tensor("evaluated_literals")
    evaluated_state = supervision.tensor("evaluated_state")
    top = supervision.tensor("top_literals")
    top_state = supervision.tensor("top_state")
    available = supervision.tensor("top_set_available")
    if not bool(torch.all(available)):
        raise ValueError("Top-Set supervision is unavailable for part of the SATACT batch")

    denominator = _segment_logsumexp(logits[evaluated], evaluated_state, size)
    numerator = _segment_logsumexp(logits[top], top_state, size)
    losses = denominator - numerator

    evaluated_score = logits[evaluated]
    maximum = _segment_max(evaluated_score, evaluated_state, size)
    candidate = evaluated_score == maximum[evaluated_state]
    sentinel = torch.full_like(evaluated, logits.numel())
    selected = _segment_min(
        torch.where(candidate, evaluated, sentinel), evaluated_state, size
    )
    top_match = top == selected[top_state]
    top_correct = _segment_sum(top_match.to(torch.long), top_state, size) > 0

    native = supervision.tensor("native_literal")
    native_evaluated = supervision.tensor("native_evaluated")
    native_in_top = supervision.tensor("native_in_top")
    pair_state = supervision.tensor("pair_state")
    into_native = (
        supervision.tensor("pair_worse") == native[pair_state]
    ) & native_evaluated[pair_state]
    override_edge_correct = (
        logits[supervision.tensor("pair_better")]
        > logits[supervision.tensor("pair_worse")]
    ) & into_native
    override_edges = _segment_sum(into_native.to(torch.long), pair_state, size)
    override_correct = _segment_sum(
        override_edge_correct.to(torch.long), pair_state, size
    ) > 0
    override_state = override_edges > 0

    dtype = torch.float64
    values = {
        "top_set_correct_sum": top_correct.to(dtype).sum(),
        "top_set_total": torch.tensor(float(size), dtype=dtype, device=logits.device),
        "top_set_probability_mass_sum": torch.exp(-losses.detach()).to(dtype).sum(),
        "top_set_probability_mass_count": torch.tensor(
            float(size), dtype=dtype, device=logits.device
        ),
        "native_top_set_sum": (native_in_top & native_evaluated).to(dtype).sum(),
        "native_top_set_total": native_evaluated.to(dtype).sum(),
        "native_override_correct_sum": (
            override_correct & override_state
        ).to(dtype).sum(),
        "native_override_total": override_state.to(dtype).sum(),
    }
    return losses, values


def _selected_literals(logits: torch.Tensor, batch: ActionEvalBatchSATACT) -> torch.Tensor:
    supervision = batch.supervision
    eligibility = supervision.tensor("eligibility")
    literal_state = supervision.tensor("literal_state")
    eligible_literal = torch.nonzero(eligibility, as_tuple=False).flatten()
    eligible_state = literal_state[eligible_literal]
    eligible_score = logits[eligible_literal]
    maximum = _segment_max(eligible_score, eligible_state, batch.batch_size)
    candidate = eligible_score == maximum[eligible_state]
    sentinel = torch.full_like(eligible_literal, logits.numel())
    return _segment_min(
        torch.where(candidate, eligible_literal, sentinel),
        eligible_state,
        batch.batch_size,
    )


def _diagnostic_values(
    logits: torch.Tensor, batch: ActionEvalBatchSATACT
) -> dict[str, torch.Tensor]:
    supervision = batch.supervision
    size = batch.batch_size
    dtype = torch.float64
    device = logits.device
    selected = _selected_literals(logits, batch)
    evaluated = supervision.tensor("evaluated_literals")
    evaluated_state = supervision.tensor("evaluated_state")
    complete = supervision.tensor("evaluated_complete")
    match = evaluated == selected[evaluated_state]
    match_count = _segment_sum(match.to(torch.long), evaluated_state, size)
    result: dict[str, torch.Tensor] = {
        "top1_total": torch.tensor(float(size), dtype=dtype, device=device),
        "top1_evaluated_sum": (match_count == 1).to(dtype).sum(),
    }
    complete_state = evaluated_state[complete]
    chosen_complete = match & complete
    chosen_count = _segment_sum(chosen_complete.to(torch.long), evaluated_state, size)
    for resource in ("conflicts", "propagations", "decisions", "restarts"):
        values = supervision.tensor(f"evaluated_{resource}").to(dtype)
        best = _segment_min(values[complete], complete_state, size)
        chosen = _segment_sum(
            torch.where(chosen_complete, values, torch.zeros_like(values)),
            evaluated_state,
            size,
        )
        valid = chosen_count == 1
        regret = chosen - best
        result[f"top1_regret_{resource}_sum"] = torch.where(
            valid, regret, torch.zeros_like(regret)
        ).sum()
        result[f"top1_regret_{resource}_count"] = valid.to(dtype).sum()
    return result


def _full_values(
    logits: torch.Tensor,
    batch: ActionEvalBatchSATACT,
    better: torch.Tensor,
    worse: torch.Tensor,
) -> dict[str, torch.Tensor]:
    supervision = batch.supervision
    size = batch.batch_size
    dtype = torch.float64
    eligibility = supervision.tensor("eligibility")
    literal_state = supervision.tensor("literal_state")
    native = supervision.tensor("native_literal")
    eligible_literal = torch.nonzero(eligibility, as_tuple=False).flatten()
    eligible_state = literal_state[eligible_literal]
    eligible_score = logits[eligible_literal]
    present = native >= 0
    safe_native = native.clamp_min(0)
    native_score = logits[safe_native]
    precedes = (eligible_score > native_score[eligible_state]) | (
        (eligible_score == native_score[eligible_state])
        & (eligible_literal < safe_native[eligible_state])
    )
    rank = _segment_sum(precedes.to(dtype), eligible_state, size) + 1.0
    result: dict[str, torch.Tensor] = {
        "native_count": present.to(dtype).sum(),
        "native_top1_sum": ((rank <= 1) & present).to(dtype).sum(),
        "native_top3_sum": ((rank <= 3) & present).to(dtype).sum(),
        "native_top5_sum": ((rank <= 5) & present).to(dtype).sum(),
        "native_top10_sum": ((rank <= 10) & present).to(dtype).sum(),
        "native_reciprocal_rank_sum": torch.where(
            present, rank.reciprocal(), torch.zeros_like(rank)
        ).sum(),
    }
    correct = better > worse
    masks = supervision.tensor("pair_rule_mask")
    for name, rule in (("r1", 1), ("r2", 2), ("r3", 4)):
        selected = (masks & rule) != 0
        result[f"{name}_correct_sum"] = (correct & selected).to(dtype).sum()
        result[f"{name}_total"] = selected.to(dtype).sum()
    for name, exact in (
        ("r1_only", 1),
        ("r2_only", 2),
        ("r3_only", 4),
        ("r2_r3_overlap", 6),
    ):
        selected = masks == exact
        result[f"{name}_correct_sum"] = (correct & selected).to(dtype).sum()
        result[f"{name}_total"] = selected.to(dtype).sum()
    return result


@dataclass(frozen=True)
class SATACTMetricSums:
    values: torch.Tensor
    mode: MetricModeSATACT

    def to_dict(self) -> dict[str, float]:
        numbers = self.values.detach().cpu().tolist()
        return {name: float(numbers[index]) for index, name in enumerate(METRIC_FIELDS_SATACT)}

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any], *, mode: MetricModeSATACT = "full"
    ) -> "SATACTMetricSums":
        return cls(
            torch.tensor(
                [float(values.get(name, 0.0)) for name in METRIC_FIELDS_SATACT],
                dtype=torch.float64,
            ),
            mode,
        )

    def summary(self) -> dict[str, float]:
        raw = self.to_dict()

        def ratio(numerator: str, denominator: str) -> float:
            return raw[numerator] / raw[denominator] if raw[denominator] else 0.0

        result = {
            "objective_loss": ratio("objective_weighted_sum", "instance_weight_sum"),
            "value_loss": ratio("value_weighted_sum", "instance_weight_sum"),
            "top_set_loss": ratio("top_set_weighted_sum", "top_set_weight_sum"),
            "native_loss": ratio("native_weighted_sum", "instance_weight_sum"),
            "native_ce_active_rate": ratio(
                "native_ce_active_sum", "native_present_sum"
            ),
            "state_count": raw["state_count"],
        }
        if _MODE_LEVEL[self.mode] >= 1:
            result.update(
                pair_accuracy=ratio("pair_correct_sum", "pair_total"),
                pair_count=raw["pair_total"],
                top_set_accuracy=ratio("top_set_correct_sum", "top_set_total"),
                top_set_count=raw["top_set_total"],
                top_set_probability_mass=ratio(
                    "top_set_probability_mass_sum",
                    "top_set_probability_mass_count",
                ),
                native_top_set_rate=ratio(
                    "native_top_set_sum", "native_top_set_total"
                ),
                native_top_set_count=raw["native_top_set_total"],
                native_override_accuracy=ratio(
                    "native_override_correct_sum", "native_override_total"
                ),
                native_override_count=raw["native_override_total"],
            )
        if _MODE_LEVEL[self.mode] >= 2:
            result["top1_evaluated_coverage"] = ratio(
                "top1_evaluated_sum", "top1_total"
            )
            for resource in ("conflicts", "propagations", "decisions", "restarts"):
                result[f"top1_regret_{resource}"] = ratio(
                    f"top1_regret_{resource}_sum",
                    f"top1_regret_{resource}_count",
                )
        if _MODE_LEVEL[self.mode] >= 3:
            for k in (1, 3, 5, 10):
                result[f"native_top{k}"] = ratio(f"native_top{k}_sum", "native_count")
            result["native_mrr"] = ratio("native_reciprocal_rank_sum", "native_count")
            for name in (
                "r1",
                "r2",
                "r3",
                "r1_only",
                "r2_only",
                "r3_only",
                "r2_r3_overlap",
            ):
                result[f"{name}_accuracy"] = ratio(
                    f"{name}_correct_sum", f"{name}_total"
                )
                result[f"{name}_count"] = raw[f"{name}_total"]
        return result


class SATACTMetricsAccumulator:
    def __init__(self, mode: MetricModeSATACT = "loss") -> None:
        self.values: torch.Tensor | None = None
        self.mode = mode

    def update(self, result: "SATACTLossResult" | SATACTMetricSums | Mapping[str, Any]) -> None:
        if isinstance(result, SATACTLossResult):
            sums = result.metric_sums
        elif isinstance(result, SATACTMetricSums):
            sums = result
        else:
            sums = SATACTMetricSums.from_mapping(result, mode=self.mode)
        values = sums.values.detach()
        self.values = values.clone() if self.values is None else self.values + values
        if _MODE_LEVEL[sums.mode] > _MODE_LEVEL[self.mode]:
            self.mode = sums.mode

    def reduce(self, *, distributed: bool) -> SATACTMetricSums:
        values = (
            torch.zeros(len(METRIC_FIELDS_SATACT), dtype=torch.float64)
            if self.values is None
            else self.values.clone()
        )
        if distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return SATACTMetricSums(values, self.mode)

    def summary(self) -> dict[str, float]:
        return self.reduce(distributed=False).summary()


@dataclass(frozen=True)
class SATACTLossResult:
    loss: torch.Tensor
    value_loss: torch.Tensor
    top_set_loss: torch.Tensor
    native_loss: torch.Tensor
    weight_sum: torch.Tensor
    metric_sums: SATACTMetricSums

    @property
    def total_loss(self) -> torch.Tensor:
        return self.loss

    @property
    def metrics(self) -> dict[str, float]:
        return self.metric_sums.summary()


def compute_satact_loss(
    output: Any,
    batch: ActionEvalBatchSATACT,
    config: ObjectiveConfigSATACT | None = None,
    *,
    metric_mode: MetricModeSATACT = "full",
) -> SATACTLossResult:
    if metric_mode not in _MODE_LEVEL:
        raise ValueError(f"unsupported SATACT metric mode: {metric_mode!r}")
    config = config or ObjectiveConfigSATACT()
    logits = output.literal_logits.float()
    pair_state, better, worse = _pair_values(logits, batch)
    native = batch.supervision.tensor("native_literal")
    native_present = native >= 0
    top_available = batch.supervision.tensor("top_set_available")
    require_top = (
        config.top_set_weight > 0
        or (
            config.native_weight > 0
            and config.native_supervision == "top-set"
        )
        or _MODE_LEVEL[metric_mode] >= 1
    )
    if require_top and not bool(torch.all(top_available)):
        raise ValueError(
            "SATACT Top-Set supervision is required by the objective or metric mode"
        )
    top_metrics: dict[str, torch.Tensor] = {}
    if bool(torch.all(top_available)):
        top_state, top_metrics = _top_set_values(logits, batch)
    else:
        top_state = torch.zeros_like(pair_state)
    if config.native_weight > 0:
        native_raw, native_present = _native_losses(logits, batch)
        native_active = native_present
        if config.native_supervision == "top-set":
            native_active = native_active & batch.supervision.tensor("native_in_top")
        native_state = torch.where(
            native_active, native_raw, torch.zeros_like(native_raw)
        )
    else:
        native_active = torch.zeros_like(native_present)
        native_state = torch.zeros_like(pair_state)
    state_loss = (
        pair_state
        + config.top_set_weight * top_state
        + config.native_weight * native_state
    )
    weights = batch.supervision.state_weights
    weight_sum = weights.sum()
    weighted_loss = weights * state_loss
    weighted_value = weights * pair_state
    weighted_top = weights * top_state
    weighted_native = weights * native_state
    loss = weighted_loss.sum() / weight_sum
    value_loss = weighted_value.sum() / weight_sum
    top_set_loss = weighted_top.sum() / weight_sum
    native_loss = weighted_native.sum() / weight_sum

    metrics = torch.zeros(
        len(METRIC_FIELDS_SATACT), dtype=torch.float64, device=logits.device
    )
    basics = {
        "state_count": torch.tensor(float(batch.batch_size), device=logits.device),
        "instance_weight_sum": weight_sum.detach(),
        "objective_weighted_sum": weighted_loss.detach().sum(),
        "value_weighted_sum": weighted_value.detach().sum(),
        "top_set_weighted_sum": weighted_top.detach().sum(),
        "top_set_weight_sum": (
            weights[top_available].detach().sum()
            if bool(torch.any(top_available))
            else torch.zeros((), device=logits.device)
        ),
        "native_weighted_sum": weighted_native.detach().sum(),
        "native_present_sum": native_present.to(torch.float64).sum(),
        "native_ce_active_sum": native_active.to(torch.float64).sum(),
    }
    for name, value in basics.items():
        metrics[_INDEX[name]] = value.to(torch.float64)
    if _MODE_LEVEL[metric_mode] >= 1:
        correct = better > worse
        metrics[_INDEX["pair_correct_sum"]] = correct.to(torch.float64).sum()
        metrics[_INDEX["pair_total"]] = float(correct.numel())
        for name, value in top_metrics.items():
            metrics[_INDEX[name]] = value
    if _MODE_LEVEL[metric_mode] >= 2:
        with torch.no_grad():
            for name, value in _diagnostic_values(logits.detach(), batch).items():
                metrics[_INDEX[name]] = value
    if _MODE_LEVEL[metric_mode] >= 3:
        with torch.no_grad():
            for name, value in _full_values(
                logits.detach(), batch, better, worse
            ).items():
                metrics[_INDEX[name]] = value
    return SATACTLossResult(
        loss=loss,
        value_loss=value_loss,
        top_set_loss=top_set_loss,
        native_loss=native_loss,
        weight_sum=weight_sum.detach(),
        metric_sums=SATACTMetricSums(metrics, metric_mode),
    )


__all__ = [
    "SATACTLossResult",
    "SATACTMetricSums",
    "SATACTMetricsAccumulator",
    "METRIC_FIELDS_SATACT",
    "MetricModeSATACT",
    "compute_satact_loss",
]
