from __future__ import annotations

import logging

from solver.inference.generated import satact_trace_pb2, satact_trace_pb2_grpc
from satact.serving.inference import SATActInference


_LOG = logging.getLogger(__name__)


class SATActTraceServicer(satact_trace_pb2_grpc.SATActTraceServerServicer):
    """A deliberately per-request SAT-ACT service (no dynamic micro-batching)."""

    def __init__(self, inference: SATActInference) -> None:
        self.inference = inference

    def query_branch(self, request, context):
        del context
        try:
            if request.response_payload not in {
                satact_trace_pb2.RESPONSE_PAYLOAD_COMPACT,
                satact_trace_pb2.RESPONSE_PAYLOAD_FULL,
            }:
                raise ValueError("response_payload must be COMPACT or FULL")
            prediction = self.inference.predict_branch(
                n_vars=request.n_vars,
                n_clauses=request.n_clauses,
                c_idxs=list(request.c_idxs),
                l_idxs=list(request.l_idxs),
                assignment_value=list(request.assignment_value),
                assignment_level=list(request.assignment_level),
                decision_level=request.decision_level,
                candidate_variable=list(request.candidate_variable),
                model_variant=request.model_variant,
                full=request.response_payload == satact_trace_pb2.RESPONSE_PAYLOAD_FULL,
            )
            action = (
                satact_trace_pb2.ACTION_LITERAL
                if prediction.action == "literal"
                else satact_trace_pb2.ACTION_DEFER
            )
            return satact_trace_pb2.BranchDecision(
                success=True,
                msg="ok",
                n_secs_inference=prediction.n_secs_inference,
                model_variant=prediction.model_variant,
                action=action,
                selected_literal_index=prediction.selected_literal_index,
                selected_log_probability=prediction.selected_log_probability,
                response_payload=request.response_payload,
                action_logits=prediction.action_logits,
            )
        except Exception as exc:
            _LOG.exception("SAT-ACT inference failed")
            return satact_trace_pb2.BranchDecision(
                success=False,
                msg=f"{type(exc).__name__}: {exc}",
                model_variant=self.inference.wire_variant,
                action=satact_trace_pb2.ACTION_UNSPECIFIED,
                selected_literal_index=-1,
                response_payload=(
                    request.response_payload
                    if request.response_payload
                    in {
                        satact_trace_pb2.RESPONSE_PAYLOAD_COMPACT,
                        satact_trace_pb2.RESPONSE_PAYLOAD_FULL,
                    }
                    else satact_trace_pb2.RESPONSE_PAYLOAD_UNSPECIFIED
                ),
            )
