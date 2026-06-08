# Embryo: Diagnostic Framework for Deployable Memory in Embodied Agents

**One-line:** Experiment testing whether learned perception-memory-action stacks can produce *causally verifiable* memory use in visual embodied environments, using only deployable observations and no privileged state.

## Current Diagnostic Result

In fixed-seed Crafter long-run replay, `learned_fact_writer_v0` makes the
dev16/h512 seed block memory-evaluable where the reference RGB scaffold does
not.

Current strongest supported statement:

```text
learned_fact_writer_v0 improves Crafter memory-evaluability and produces a
large diagnostic clean-vs-control score under the prior collapsed-control
long-run protocol.
```

This is a protocol/mechanism result. It is not a Crafter competence result and
not final causal proof under separate shuffled, stale, and wrong-binding control
families.

**Explicitly not claimed:** Crafter task performance, memory-driven benchmark
improvement, RL success, online fine-tuning, or generalization.

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
- privileged backend inventory
- semantic maps
- player position
- achievements
- seed/source metadata
- backend simulator state

Deployable inventory belief inferred from observations or agent-owned events is
a future memory surface, not a privileged actor input.

The forward long-run protocol tests clean memory against no-memory plus
separate shuffled, stale, and wrong-binding corrupt-memory controls. The
existing strongest supported result still comes from the collapsed-control path;
the separated controls need fresh validation before supporting a stronger claim.

## Architecture

```text
Runtime trace
  -> Deployable features (RGB / delta / previous action)
  -> Visual fact writer (learned)
  -> Sparse facts
  -> Queryable memory
  -> Freshness-aware router / reference action policy
  -> Closed-loop replay + memory-grounded scoring
```

## Limitations

- The current memory result does not rely on PPO/RL.
- A separate experimental Crafter survival actor training lane exists, but it
  does not currently support a memory, competence, or benchmark claim.
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

- Align and validate corrupt-control families under the learned fact writer.
- Replicate a competent Crafter actor through an official or exact-baseline
  path before making task-performance claims.
- Keep survival training results separate from memory-mechanism claims.

## License

Embryo is released under the Apache License 2.0. See [LICENSE](LICENSE).
