# Neural-assisted CaDiCaL

This directory contains the CaDiCaL-based SAT-ACT solver, inference service,
and runner. Glucose is not included.

## Build requirements

See [../ENVIRONMENT.md](../ENVIRONMENT.md).

## Build

`GRPC_PREFIX` is the installed gRPC C++/Protobuf C++ prefix. It must contain
the CMake package files under `lib/cmake`; it is not the gRPC source tree. The
example below assumes gRPC was installed in `/opt/grpc-1.76.0`.

```bash
# Build the neural-assisted CaDiCaL solver.
GRPC_PREFIX=/opt/grpc-1.76.0
solver/build.sh solver/build "$GRPC_PREFIX" 8
```

Generated protobuf sources are included. Regenerate them only after changing
the protocol:

```bash
# Regenerate the C++ and Python protocol bindings.
GRPC_PREFIX=/opt/grpc-1.76.0
PYTHON=/path/to/python
solver/generate_proto.sh "$GRPC_PREFIX" "$PYTHON"
```

## CA

```bash
# Solve one CNF with the CA Full checkpoint.
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

python solver/solve.py \
  --solver-bin solver/build/cadical-satact \
  --checkpoint neuro/checkpoints/ca_satact_full.pt \
  --cnf solver/fixtures/sat_branch.cnf \
  --output-dir "$OUTPUT_ROOT/solver-validation/ca" \
  --device "$DEVICE"
```

## SR

```bash
# Solve one CNF with the SR Full checkpoint.
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

python solver/solve.py \
  --solver-bin solver/build/cadical-satact \
  --checkpoint neuro/checkpoints/sr_satact_full.pt \
  --cnf solver/fixtures/sat_branch.cnf \
  --output-dir "$OUTPUT_ROOT/solver-validation/sr" \
  --device "$DEVICE"
```

## PS

```bash
# Solve one CNF with the PS Full checkpoint.
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

python solver/solve.py \
  --solver-bin solver/build/cadical-satact \
  --checkpoint neuro/checkpoints/ps_satact_full.pt \
  --cnf solver/fixtures/sat_branch.cnf \
  --output-dir "$OUTPUT_ROOT/solver-validation/ps" \
  --device "$DEVICE"
```

Replace `full` with `base` to run a Base checkpoint. Use
`solver/fixtures/unsat_branch.cnf` for the UNSAT fixture.

## Benchmark preparation and solver evaluation

The evaluation workflow samples grouped CA, SR, and PS instances from the
G4SATBench Medium validation and test splits. It runs the CaDiCaL baseline and
the selected SAT-ACT checkpoint with three and five neural calls.

```bash
# Prepare five evaluation groups for each family and split.
CA_SOURCE=/path/to/g4satbench/medium/ca
SR_SOURCE=/path/to/g4satbench/medium/sr
PS_SOURCE=/path/to/g4satbench/medium/ps
EVALUATION_DATA=/path/to/satact-evaluation-data

python solver/evaluation/prepare_benchmark.py \
  --ca-source "$CA_SOURCE" \
  --sr-source "$SR_SOURCE" \
  --ps-source "$PS_SOURCE" \
  --output-dir "$EVALUATION_DATA" \
  --seed 1 --groups 5 --instances-per-label 100
```

```bash
# Evaluate the CA Full checkpoint on all validation groups.
EVALUATION_DATA=/path/to/satact-evaluation-data
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

python solver/evaluation/run_evaluation.py \
  --solver-bin solver/build/cadical-satact \
  --checkpoint neuro/checkpoints/ca_satact_full.pt \
  --data-dir "$EVALUATION_DATA" \
  --family ca --split valid --groups 0 1 2 3 4 \
  --neural-calls 3 5 \
  --output-dir "$OUTPUT_ROOT/solver-evaluation" \
  --device "$DEVICE" --workers 8 --timeout 100
```

The root README contains separate validation, test, and summary commands for
CA, SR, and PS.
