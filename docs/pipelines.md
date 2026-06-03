# Pipelines

Pipelines combine the package modules into reproducible workflows. They are
kept separate from runtime adapters, model implementations, and scoring code.

The supervised memory reference pipeline runs:

```text
runtime collection
  -> normalized feature rows
  -> supervised module datasets
  -> module checkpoint manifests
  -> stack manifest
  -> manifest-loaded replay
  -> pipeline summary
```

Run it with:

```bash
python scripts/run_supervised_memory_pipeline.py \
  --config configs/pipelines/fixture_supervised_memory.yaml
```

This is a scaffold for reproducibility and packaging. It is not a task
benchmark, and it does not make a runtime competence claim.
