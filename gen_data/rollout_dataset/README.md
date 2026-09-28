# Pair construction

This package converts completed action-outcome parts into compact shards used by
SAT-ACT training. It preserves the C1/C2/C3 outcome criteria stored in the raw
records and constructs the corresponding preference pairs.

Run the low-level interface from the archive root:

```bash
python -m gen_data.rollout_dataset --help
```

The recommended end-to-end entry point is `gen_data/prepare_dataset.py`.
