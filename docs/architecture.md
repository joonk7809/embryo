# Architecture

Embryo is organized around a small deployable perception-memory-action stack:

```text
RGB / RGB delta / previous action
  -> visual fact writer
  -> sparse facts
  -> query memory
  -> freshness-aware router
  -> BC action policy
  -> diagnostic replay
  -> memory-grounded scoring
```

The forward package is intentionally smaller than the historical experiment tree.
Old experiment scripts are evidence, not API.

## Module Map

- `embryo.core`: shared types, spaces, config, registry.
- `embryo.core.contracts`: deployable observation contract.
- `embryo.runtimes`: runtime adapters and suite-specific environment wrappers.
- `embryo.datasets`: runtime-trace to supervised-dataset builders.
- `embryo.memory`: facts, queries, freshness, router logic.
- `embryo.models`: learned module interfaces, reference models, and checkpoint manifests.
- `embryo.eval`: memory-grounded scoring, guardrails, contamination, traces.
- `embryo.pipelines`: config-driven workflows that compose collection, datasets, training, and replay.
- `embryo.run`: collection, replay, training, evaluation entry points.

## Runnable Path

The package includes a tiny reference replay path that exercises the same module
boundaries without depending on historical experiment scripts:

```bash
python scripts/replay_learned_stack.py --config configs/learned_stack_replay.yaml
python scripts/evaluate_trace.py --trace data/fixtures/tiny_memory_score_trace.jsonl
python scripts/collect_runtime.py --runtime fixture_memory --steps 4
python scripts/build_datasets.py --config configs/datasets/fixture_supervised.yaml
python scripts/run_supervised_memory_pipeline.py --config configs/pipelines/fixture_supervised_memory.yaml
```
