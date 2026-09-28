# Action rollout

This package constructs manifests and execution plans, invokes the instrumented
`decisiontrace` binary, and records solver states, candidate actions, and their
outcomes. It is normally called through `gen_data/prepare_dataset.py`.

Run the low-level interface from the archive root:

```bash
python -m gen_data.action_rollouts --help
```

Dataset locations, the solver binary, cache locations, and output locations are
always supplied by command-line arguments.
