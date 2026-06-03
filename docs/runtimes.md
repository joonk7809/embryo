# Runtimes

Runtime adapters live under `embryo.runtimes`. Each runtime owns environment
construction, reset/step semantics, optional dependencies, and the conversion
from backend observations into deployable observation dictionaries.

This keeps environment suites separate from memory, model, and scoring code.
Adding another runtime should not require changing the Crafter adapter.

## Layout

```text
embryo/runtimes/
  base.py              shared runtime protocol and step records
  registry.py          runtime registry and factory
  fixture.py           deterministic package-test runtime
  crafter/
    adapter.py         optional Crafter adapter
```

## Registered Runtimes

- `fixture_memory`: deterministic fixture runtime for tests and smoke checks.
- `crafter_memory`: optional Crafter RGB runtime.

## CLI

```bash
python scripts/collect_runtime.py \
  --runtime fixture_memory \
  --steps 4 \
  --out runs/runtime_collect/fixture_trace.jsonl \
  --features-out runs/runtime_collect/fixture_features.jsonl
```

Collection can write two artifacts:

- a runtime trace with observations, chosen actions, rewards, and post-action
  diagnostics
- normalized feature rows for replay and dataset-building scaffolds

The collection trace is diagnostic. It is not a task-performance benchmark.
