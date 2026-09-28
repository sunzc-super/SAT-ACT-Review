#!/usr/bin/env python3
"""Serve SAT-ACT through the existing satact protobuf service."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent import futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "neuro"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import grpc

from solver.inference.generated import satact_trace_pb2_grpc
from satact.serving.inference import SATActInference
from solver.inference.service import SATActTraceServicer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bind", default="0.0.0.0:41070")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_message_mb", type=int, default=1024)
    return parser


def serve(args: argparse.Namespace) -> None:
    inference = SATActInference(args.checkpoint, device=args.device)
    limit = args.max_message_mb * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=(
            ("grpc.max_send_message_length", limit),
            ("grpc.max_receive_message_length", limit),
        ),
    )
    satact_trace_pb2_grpc.add_SATActTraceServerServicer_to_server(
        SATActTraceServicer(inference), server
    )
    if server.add_insecure_port(args.bind) == 0:
        raise RuntimeError(f"unable to bind {args.bind}")
    server.start()
    logging.info("SAT-ACT serving wire variant satact on %s", args.bind)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0).wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    if args.workers < 1 or args.max_message_mb < 1:
        raise ValueError("workers and max_message_mb must be positive")
    serve(args)


if __name__ == "__main__":
    main()
