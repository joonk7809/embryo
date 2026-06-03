# Contributing

Embryo is organized as a small forward package. Keep changes scoped, tested,
and independent of historical experiment scripts.

## Local Checks

Run these before sharing changes:

```bash
python -m compileall -q embryo scripts tests examples
python -m unittest discover -s tests
python scripts/run_supervised_memory_pipeline.py --config configs/pipelines/fixture_supervised_memory.yaml --out /tmp/embryo_pipeline_check
python scripts/check_release.py
```

## Public Surface

Public docs and configs should describe architecture and claims in ordinary
language. Keep migration notes, scratch plans, and private experiment
bookkeeping out of public docs and configs.

## Boundaries

Do not import from historical experiment directories. Port required behavior
into `embryo/` modules with focused tests.

Do not claim task competence, benchmark performance, or reinforcement learning
success unless the relevant runtime, training, and evaluation gates are present.
