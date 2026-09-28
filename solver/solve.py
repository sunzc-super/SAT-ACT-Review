#!/usr/bin/env python3
"""Run SAT-ACT inference and the neural-assisted CaDiCaL solver together."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def wait_for_server(host: str, port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("SAT-ACT inference server exited before becoming ready")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("timed out waiting for the SAT-ACT inference server")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-bin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41070)
    parser.add_argument("--neuro-calls", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    python_path = [str(root), str(root / "neuro")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    server_log = (args.output_dir / "server.log").open("wb")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "solver.inference.server",
            "--checkpoint",
            str(args.checkpoint),
            "--bind",
            f"{args.host}:{args.port}",
            "--device",
            args.device,
        ],
        cwd=root,
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_server(args.host, args.port, server)
        with (args.output_dir / "solver.stdout").open("wb") as stdout, (
            args.output_dir / "solver.stderr"
        ).open("wb") as stderr:
            result = subprocess.run(
                [
                    str(args.solver_bin),
                    "--mode",
                    "SATACT",
                    "--model_variant",
                    "satact",
                    "--branch_server",
                    f"{args.host}:{args.port}",
                    "--neuro_calls",
                    str(args.neuro_calls),
                    "--timeout_s",
                    str(args.timeout),
                    "--neuro_outfile",
                    str(args.output_dir / "neural_trace.txt"),
                    str(args.cnf),
                ],
                cwd=root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout + 30,
                check=False,
            )
        if result.returncode not in (10, 20):
            raise SystemExit(result.returncode)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        server_log.close()


if __name__ == "__main__":
    main()
