#!/usr/bin/env python3
"""Run grouped SAT-ACT and CaDiCaL baseline evaluations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.evaluation.common import (  # noqa: E402
    FAMILIES,
    LABELS,
    SPLITS,
    load_manifest,
    select_group,
    write_json,
)


BASELINE_NAME = "cadical_baseline"
MODEL_MODE = "SATACT"
BASELINE_MODE = "CADICAL-BASELINE"
TAIL_LIMIT = 65536
RESULT_FIELDS = (
    "status",
    "cpu_time",
    "wall_time",
    "request_time",
    "decision_opportunities",
    "requests",
    "model_calls",
    "rpc_errors",
    "invalid_responses",
    "literal_actions",
    "defer_actions",
    "preflight_skips",
    "pause_skips",
    "budget_skips",
    "inference_time",
    "conflicts",
    "decisions",
    "propagations",
    "restarts",
)


def parse_solver_result(path: Path) -> dict[str, Any]:
    values = path.read_text(encoding="utf-8").split()
    if len(values) != len(RESULT_FIELDS):
        raise ValueError(f"unexpected solver result in {path}")
    result: dict[str, Any] = {"status": values[0]}
    for index, name in enumerate(RESULT_FIELDS[1:15], start=1):
        result[name] = float(values[index]) if name.endswith("_time") else int(values[index])
    for index, name in enumerate(RESULT_FIELDS[15:], start=15):
        result[name] = int(values[index])
    return result


def solver_command(
    solver_bin: Path,
    cnf: Path,
    result_path: Path,
    mode: str,
    solver_seed: int,
    timeout: int,
    neural_calls: int,
    server_address: str,
) -> list[str]:
    command = [
        str(solver_bin),
        "--mode",
        mode,
        "--timeout_s",
        str(timeout),
        "--neuro_outfile",
        str(result_path),
        "--stabilize=1",
        "--stabilizeonly=1",
        "--score=1",
        "--preprocesslight=0",
        f"--seed={solver_seed}",
        "-t",
        str(timeout),
    ]
    if mode == MODEL_MODE:
        command[3:3] = [
            "--model_variant",
            "satact",
            "--decide_strategy",
            "FIRST",
            "--branch_server",
            server_address,
            "--n_secs_pause",
            "0",
            "--n_secs_pause_inc",
            "1",
            "--max_lclause_size",
            "500",
            "--max_n_nodes_cells",
            "2000000",
            "--neuro_calls",
            str(neural_calls),
            "--response_payload",
            "compact",
        ]
    command.append(str(cnf))
    return command


def run_instance(
    row: dict[str, str],
    *,
    data_dir: Path,
    solver_bin: Path,
    mode: str,
    neural_calls: int,
    server_address: str,
    timeout: int,
    temporary_dir: Path,
) -> dict[str, Any]:
    relative = row["relative_path"]
    cnf = data_dir / relative
    handle, name = tempfile.mkstemp(suffix=".result", dir=temporary_dir)
    os.close(handle)
    result_path = Path(name)
    command = solver_command(
        solver_bin,
        cnf,
        result_path,
        mode,
        int(row["solver_seed"]),
        timeout,
        neural_calls,
        server_address,
    )
    error: str | None = None
    stdout = ""
    stderr = ""
    returncode = -1
    result: dict[str, Any] = {"status": "ERROR"}
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout + 30,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        if result_path.stat().st_size:
            result = parse_solver_result(result_path)
        else:
            error = "solver did not write a result record"
    except subprocess.TimeoutExpired as exc:
        error = "process timeout"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        result_path.unlink(missing_ok=True)

    expected = "SAT" if row["label"] == "sat" else "UNSAT"
    solved = result.get("status") == expected and returncode == {"SAT": 10, "UNSAT": 20}[expected]
    timed_out = result.get("status") == "UNKNOWN" and returncode == 0
    complete = error is None and (solved or timed_out)
    record: dict[str, Any] = {
        "instance": relative,
        "family": row["family"],
        "split": row["split"],
        "label": row["label"],
        "group": int(row["group"]),
        "solver_seed": int(row["solver_seed"]),
        "mode": BASELINE_NAME if mode == BASELINE_MODE else "satact",
        "neural_calls": neural_calls,
        "returncode": returncode,
        "complete": complete,
        "error": error,
        "result": result,
    }
    if not complete:
        record["diagnostic"] = {
            "stdout_tail": stdout[-TAIL_LIMIT:],
            "stderr_tail": stderr[-TAIL_LIMIT:],
        }
    return record


def read_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def configuration_complete(
    directory: Path, parameters: dict[str, Any], rows: list[dict[str, str]]
) -> bool:
    try:
        stored = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        results = read_results(directory / "results.jsonl")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {row["relative_path"] for row in rows}
    return (
        stored == parameters
        and len(results) == len(rows)
        and {row.get("instance") for row in results} == expected
        and all(row.get("complete") for row in results)
    )


def run_configuration(
    directory: Path,
    parameters: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    data_dir: Path,
    solver_bin: Path,
    mode: str,
    neural_calls: int,
    server_address: str,
    timeout: int,
    workers: int,
    local_temp_dir: Path | None,
) -> None:
    if configuration_complete(directory, parameters, rows):
        print(f"complete {directory}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "parameters.json", parameters)
    with tempfile.TemporaryDirectory(
        prefix="satact-evaluation-", dir=local_temp_dir
    ) as temporary:
        worker = partial(
            run_instance,
            data_dir=data_dir,
            solver_bin=solver_bin,
            mode=mode,
            neural_calls=neural_calls,
            server_address=server_address,
            timeout=timeout,
            temporary_dir=Path(temporary),
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker, rows))
    temporary_results = directory / "results.jsonl.tmp"
    with temporary_results.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary_results.replace(directory / "results.jsonl")
    failures = [result for result in results if not result["complete"]]
    failure_path = directory / "failures.jsonl"
    if failures:
        with failure_path.open("w", encoding="utf-8") as stream:
            for failure in failures:
                stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise RuntimeError(f"{len(failures)} incomplete instances in {directory}")
    failure_path.unlink(missing_ok=True)
    print(f"finished {directory}", flush=True)


def wait_for_server(address: str, process: subprocess.Popen[bytes]) -> None:
    host, port_text = address.rsplit(":", 1)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("inference server exited before becoming ready")
        try:
            with socket.create_connection((host, int(port_text)), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("timed out waiting for the inference server")


def start_server(
    checkpoint: Path,
    device: str,
    address: str,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], Any]:
    environment = os.environ.copy()
    python_path = [str(PROJECT_ROOT), str(PROJECT_ROOT / "neuro")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "solver.inference.server",
            "--checkpoint",
            str(checkpoint),
            "--bind",
            address,
            "--device",
            device,
            "--workers",
            "1",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_server(address, process)
    except Exception:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        log.close()
        raise
    return process, log


def stop_server(process: subprocess.Popen[bytes], log: Any) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    log.close()


def parameters_for(
    rows: list[dict[str, str]],
    family: str,
    split: str,
    group: int,
    model: str,
    neural_calls: int,
    solver_bin: Path,
    timeout: int,
    workers: int,
) -> dict[str, Any]:
    return {
        "schema": "satact-solver-evaluation",
        "family": family,
        "split": split,
        "group": group,
        "solver_seed": group,
        "model": model,
        "neural_calls": neural_calls,
        "solver": str(solver_bin),
        "timeout": timeout,
        "workers": workers,
        "solver_settings": {
            "stabilize": 1,
            "stabilizeonly": 1,
            "score": 1,
            "preprocesslight": 0,
        },
        "instances": [row["relative_path"] for row in rows],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-bin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--neural-calls", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41070)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=100)
    parser.add_argument("--local-temp-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.workers < 1 or args.timeout < 1:
        raise ValueError("workers and timeout must be positive")
    if any(group < 0 for group in args.groups):
        raise ValueError("groups must be non-negative")
    if any(calls < 1 for calls in args.neural_calls):
        raise ValueError("neural-calls must be positive")
    solver_bin = args.solver_bin.resolve()
    checkpoint = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    if not solver_bin.is_file():
        raise FileNotFoundError(solver_bin)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows = load_manifest(data_dir)
    selected: dict[int, list[dict[str, str]]] = {}
    for group in args.groups:
        group_rows = select_group(rows, args.family, args.split, group)
        if not group_rows:
            raise ValueError(f"no instances for {args.family}/{args.split}/group_{group}")
        counts = {label: sum(row["label"] == label for row in group_rows) for label in LABELS}
        if len(set(counts.values())) != 1:
            raise ValueError(f"unbalanced group {group}: {counts}")
        for row in group_rows:
            if not (data_dir / row["relative_path"]).is_file():
                raise FileNotFoundError(data_dir / row["relative_path"])
        selected[group] = group_rows

    model_name = checkpoint.stem
    server_address = f"{args.host}:{args.port}"
    planned: list[tuple[str, int, int, Path, dict[str, Any]]] = []
    for group, group_rows in selected.items():
        baseline_dir = (
            args.output_dir / args.family / args.split / f"group_{group}" / BASELINE_NAME
        )
        planned.append(
            (
                BASELINE_MODE,
                group,
                0,
                baseline_dir,
                parameters_for(
                    group_rows,
                    args.family,
                    args.split,
                    group,
                    BASELINE_NAME,
                    0,
                    solver_bin,
                    args.timeout,
                    args.workers,
                ),
            )
        )
        for calls in args.neural_calls:
            model_dir = (
                args.output_dir
                / args.family
                / args.split
                / f"group_{group}"
                / model_name
                / f"calls_{calls}"
            )
            planned.append(
                (
                    MODEL_MODE,
                    group,
                    calls,
                    model_dir,
                    parameters_for(
                        group_rows,
                        args.family,
                        args.split,
                        group,
                        model_name,
                        calls,
                        solver_bin,
                        args.timeout,
                        args.workers,
                    ),
                )
            )
    if args.dry_run:
        for mode, group, calls, directory, parameters in planned:
            print(
                json.dumps(
                    {
                        "mode": BASELINE_NAME if mode == BASELINE_MODE else "satact",
                        "group": group,
                        "neural_calls": calls,
                        "directory": str(directory),
                        "parameters": parameters,
                    },
                    ensure_ascii=False,
                )
            )
        return

    for mode, group, calls, directory, parameters in planned:
        if mode != BASELINE_MODE:
            continue
        run_configuration(
            directory,
            parameters,
            selected[group],
            data_dir=data_dir,
            solver_bin=solver_bin,
            mode=mode,
            neural_calls=calls,
            server_address=server_address,
            timeout=args.timeout,
            workers=args.workers,
            local_temp_dir=args.local_temp_dir,
        )

    pending_models = [
        item
        for item in planned
        if item[0] == MODEL_MODE
        and not configuration_complete(item[3], item[4], selected[item[1]])
    ]
    if not pending_models:
        print("all model configurations are complete", flush=True)
        return
    server, log = start_server(
        checkpoint,
        args.device,
        server_address,
        args.output_dir / args.family / "inference_server.log",
    )
    try:
        for mode, group, calls, directory, parameters in pending_models:
            run_configuration(
                directory,
                parameters,
                selected[group],
                data_dir=data_dir,
                solver_bin=solver_bin,
                mode=mode,
                neural_calls=calls,
                server_address=server_address,
                timeout=args.timeout,
                workers=args.workers,
                local_temp_dir=args.local_temp_dir,
            )
    finally:
        stop_server(server, log)


if __name__ == "__main__":
    main()
