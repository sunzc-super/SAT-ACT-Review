# SAT-ACT neural model

Training first builds explicit indexes and then launches the selected config.
The root README contains complete CA, SR, and PS data-path examples.
During training, `valid` selects the best checkpoint and `test` evaluates that
selected checkpoint at the end.

## Environment

See [../ENVIRONMENT.md](../ENVIRONMENT.md).

## CA

```bash
# Train the CA Base and Full models.
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

# Train SAT-ACT Base on CA.
python neuro/scripts/train_from_config.py \
  --config neuro/configs/ca_satact_base.json \
  --train-dir "$OUTPUT_ROOT/datasets/ca/train" \
  --valid-dir "$OUTPUT_ROOT/datasets/ca/valid" \
  --test-dir "$OUTPUT_ROOT/datasets/ca/test" \
  --cache-dir "$OUTPUT_ROOT/indexes/ca/base" \
  --output-dir "$OUTPUT_ROOT/runs/ca" --device "$DEVICE"

# Train SAT-ACT Full on CA.
python neuro/scripts/train_from_config.py \
  --config neuro/configs/ca_satact_full.json \
  --train-dir "$OUTPUT_ROOT/datasets/ca/train" \
  --valid-dir "$OUTPUT_ROOT/datasets/ca/valid" \
  --test-dir "$OUTPUT_ROOT/datasets/ca/test" \
  --cache-dir "$OUTPUT_ROOT/indexes/ca/full" \
  --output-dir "$OUTPUT_ROOT/runs/ca" --device "$DEVICE"
```

## SR

```bash
# Train the SR Base and Full models.
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

# Train SAT-ACT Base on SR.
python neuro/scripts/train_from_config.py \
  --config neuro/configs/sr_satact_base.json \
  --train-dir "$OUTPUT_ROOT/datasets/sr/train" \
  --valid-dir "$OUTPUT_ROOT/datasets/sr/valid" \
  --test-dir "$OUTPUT_ROOT/datasets/sr/test" \
  --cache-dir "$OUTPUT_ROOT/indexes/sr/base" \
  --output-dir "$OUTPUT_ROOT/runs/sr" --device "$DEVICE"

# Train SAT-ACT Full on SR.
python neuro/scripts/train_from_config.py \
  --config neuro/configs/sr_satact_full.json \
  --train-dir "$OUTPUT_ROOT/datasets/sr/train" \
  --valid-dir "$OUTPUT_ROOT/datasets/sr/valid" \
  --test-dir "$OUTPUT_ROOT/datasets/sr/test" \
  --cache-dir "$OUTPUT_ROOT/indexes/sr/full" \
  --output-dir "$OUTPUT_ROOT/runs/sr" --device "$DEVICE"
```

## PS

```bash
# Train the PS Base and Full models.
OUTPUT_ROOT=/path/to/satact-output
DEVICE=cuda

# Train SAT-ACT Base on PS.
python neuro/scripts/train_from_config.py \
  --config neuro/configs/ps_satact_base.json \
  --train-dir "$OUTPUT_ROOT/datasets/ps/train" \
  --valid-dir "$OUTPUT_ROOT/datasets/ps/valid" \
  --test-dir "$OUTPUT_ROOT/datasets/ps/test" \
  --cache-dir "$OUTPUT_ROOT/indexes/ps/base" \
  --output-dir "$OUTPUT_ROOT/runs/ps" --device "$DEVICE"

# Train SAT-ACT Full on PS.
python neuro/scripts/train_from_config.py \
  --config neuro/configs/ps_satact_full.json \
  --train-dir "$OUTPUT_ROOT/datasets/ps/train" \
  --valid-dir "$OUTPUT_ROOT/datasets/ps/valid" \
  --test-dir "$OUTPUT_ROOT/datasets/ps/test" \
  --cache-dir "$OUTPUT_ROOT/indexes/ps/full" \
  --output-dir "$OUTPUT_ROOT/runs/ps" --device "$DEVICE"
```

Archived-checkpoint inference checks are separated by family in the root
README. They check checkpoint loading and one fixed branch query; they do not
evaluate the validation datasets.
`--valid-dir` and `--test-dir` may be omitted for a training-only smoke test.
