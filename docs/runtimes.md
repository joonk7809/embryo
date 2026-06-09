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
  popgym/
    adapter.py         optional POPGym adapter
```

## Registered Runtimes

- `fixture_memory`: deterministic fixture runtime for tests and smoke checks,
  including a tiny visual-anchor sequence for cached-memory coverage.
- `crafter_memory`: optional Crafter RGB runtime.
- `popgym_repeat_first`: optional POPGym RepeatFirst runtime for diagnostic
  recall checks over deployable discrete observations.

The Crafter adapter exposes a small reference RGB scaffold with visual change,
center-patch salience, and patch-hash fields derived only from current and
previous pixels plus previous action. These fields are deployable diagnostics
for long-run memory anchors, not semantic object labels or backend state.

For evaluation, the Crafter runtime defaults to a deterministic backend patch
that orders Crafter spawn-balancing sets before sampling. Protocol manifests
surface this as `backend_determinism_patch:
crafter_balance_object_order_v1`. This stabilizes fixed-seed replay but means
public long-run results using that setting are from the deterministic Crafter
adapter, not untouched Crafter. It does not add actor/query inputs.

`reference_exploration_sweep` is a deterministic deployable diagnostic arm used
to create visual-anchor opportunities; it is not a learned policy or competence
baseline.

Memory-control arms use the same deterministic exploration fallback until cached
query memory is active, so clean and corrupted arms are matched before memory
content is exercised.

The POPGym RepeatFirst adapter exposes only the current discrete observation,
previous discrete observation, and previous action as deployable inputs. Reward
and task success are emitted only as post-action `eval_only` fields. This
runtime is for causal memory-protocol checks, not RL training or benchmark
leaderboard claims.

Install it with `python -m pip install -e ".[popgym]"`. The task-native
recurrent comparator additionally needs `python -m pip install -e ".[popgym-recurrent]"`.

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
