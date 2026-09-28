"""Serving adapter for SAT-ACT checkpoints on the stable satact wire contract."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from satact.dataset.action_preferences import SATACT_VARIANT, SATACT_WIRE_VARIANT
from satact.models.policy import ActionEvalPolicyOutput
from satact.models.satact_factory import build_actioneval_model_satact
from satact.serving.common import (
    BranchPrediction,
    build_online_lcg,
    resolve_device,
)
from satact.training.checkpoint import load_checkpoint_satact


class SATActInference:
    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.device = resolve_device(device)
        self.checkpoint = load_checkpoint_satact(checkpoint, map_location=self.device)
        self.opts = dict(self.checkpoint["opts"])
        self.variant = SATACT_VARIANT
        self.wire_variant = SATACT_WIRE_VARIANT
        self.model = build_actioneval_model_satact(self.opts).to(self.device)
        self.model.load_state_dict(self.checkpoint["model"], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_branch(
        self, *, model_variant: str, full: bool = False, **request: Any
    ) -> BranchPrediction:
        if model_variant != self.wire_variant:
            raise ValueError(
                f"wire model_variant must be {self.wire_variant!r}, got {model_variant!r}"
            )
        sample, eligibility = build_online_lcg(**request)
        sample = sample.to(self.device)
        eligibility = eligibility.to(self.device)
        started = perf_counter()
        output = self.model(sample)
        elapsed = perf_counter() - started
        if not isinstance(output, ActionEvalPolicyOutput):
            raise TypeError("SATACT model returned an invalid output")
        logits = output.literal_logits
        masked = logits.masked_fill(~eligibility, -torch.inf)
        selected = int(torch.argmax(masked).item())
        return BranchPrediction(
            model_variant=self.wire_variant,
            action="literal",
            selected_literal_index=selected,
            selected_log_probability=float(torch.log_softmax(masked, dim=0)[selected].item()),
            n_secs_inference=elapsed,
            action_logits=logits.detach().cpu().tolist() if full else [],
        )


__all__ = ["SATActInference"]
