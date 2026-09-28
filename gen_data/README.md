# Data preparation

Use the G4SATBench `medium` CA, SR, and PS datasets, organized as
`<g4satbench-root>/medium/<family>/<split>/{sat,unsat}`.

This directory starts from those CNFs and generates solver actions, evaluated
outcomes, and preference pairs. It does not generate the original CNFs.

## Build requirements

See [../ENVIRONMENT.md](../ENVIRONMENT.md).

## Build

```bash
# Build the DecisionTrace data generator.
cmake -S gen_data/cadical/src -B gen_data/cadical/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build gen_data/cadical/build --target decisiontrace -j 8
```

## Example: CA train split

```bash
# Generate action and pair data for the CA training split.
G4SATBENCH_ROOT=/path/to/g4satbench
OUTPUT_ROOT=/path/to/satact-output

python gen_data/prepare_dataset.py all \
  --family ca --split train \
  --input-dir "$G4SATBENCH_ROOT/medium/ca/train" \
  --output-dir "$OUTPUT_ROOT/datasets/ca/train" \
  --binary gen_data/cadical/build/decisiontrace \
  --cache-dir "$OUTPUT_ROOT/cnf-cache/ca/train" \
  --workers 8 --seed 1
```

Use separate commands with `--split valid` and `--split test` for the remaining
splits. The complete CA, SR, and PS commands are listed in the root README.

The family determines the propagation horizon: CA 1000, SR 5000, and PS 15000.
C1/C2/C3 in raw records are outcome criteria, not model names.
