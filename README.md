# Embryo: Diagnostic Framework for Deployable Memory in Embodied Agents

**One-line:** Experiment testing whether learned perception-memory-action stacks can produce *causally verifiable* memory use in visual embodied environments, using only deployable observations and no privileged state.

## Core Result

In closed-loop Crafter diagnostic replay, the learned stack preserved memory-grounded separation from corrupted and absent-memory controls.

The stack includes:

- RGB-derived visual fact writer
- query/freshness router
- behavior-cloned policy
- memory-grounded scorer

**Key Metrics:**

| Control | Grounded Gap | Resource Divergence |
| --- | ---: | ---: |
| No memory | 18.75 | n/a |
| Shuffled query | 19.01 | 0.856 |
| Stale memory | 19.10 | 0.997 |
| Wrong binding | 18.71 | 0.932 |

Additional checks:

- Route preservation = 1.00
- Event self-trigger = 0.00
- Invalid action rate = 0.00
- Contamination failures = 0

This provides evidence that structured queryable memory with freshness controls can causally influence behavior beyond immediate observations.

**Explicitly not claimed:** Crafter task performance, RL success, online fine-tuning, or generalization.

## Experimental Approach

We enforce hard boundaries:

**Allowed actor/query inputs:**

- raw RGB observations and bounded deltas
- previous action
- inferred visual events
- query results and freshness bits

**Forbidden actor/query inputs:**

- reward
- done flags
- inventory
- semantic maps
- player position
- achievements
- seed/source metadata
- backend simulator state

Validated against no-memory, shuffled, stale, and wrong-binding controls.

## Architecture

```text
Runtime trace
  -> Deployable features (RGB / delta / previous action)
  -> Visual fact writer (learned)
  -> Sparse facts
  -> Queryable memory
  -> Freshness-aware router (learned)
  -> Behavior-cloned policy
  -> Closed-loop replay + memory-grounded scoring
```

## Limitations

- Behavior-cloned policy only; no online RL or fine-tuning yet.
- Diagnostic Crafter replay, not task-solving performance.
- Small models, single environment.
- No transfer demonstrated to Craftax, Minecraft, or robotics.
- Mechanism research, not a production agent system.

## Quickstart

Clone the repo, install the package, and run tests:

```bash
git clone https://github.com/joonk7809/embryo.git
cd embryo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m unittest discover -s tests
```

Run the reproducibility check:

```bash
python scripts/run_supervised_memory_pipeline.py \
  --out runs/quickstart
```

Inspect the generated summary:

```bash
cat runs/quickstart/pipeline_summary.json
```

This command runs a tiny built-in toy runtime to verify the full scaffold:

```text
collect toy runtime observations
  -> build deployable feature rows
  -> build supervised module datasets
  -> train reference modules
  -> assemble a stack manifest
  -> replay the stack
  -> compute memory-grounded diagnostics
```

This is only a reproducibility check. It does not run Crafter, train an agent, or evaluate task performance.

Run the public package checks:

```bash
python scripts/check_release.py
```

## Next Steps

- Reproduce the full diagnostic result in the clean package with real Crafter training runs.
- Add inference optimizations to fact writing and routing.
- Expand runtime support.

## License

Embryo is released under the Apache License 2.0. See [LICENSE](LICENSE).
