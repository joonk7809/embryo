# AGENTS.md

## North Star

Embryo is a public research codebase for testing whether deployable memory can
causally influence embodied-agent behavior under controlled diagnostics.

The project should stay focused on bounded, reproducible evidence:

- actor and query inputs come from deployable observations, not simulator truth
- memory claims are tested against corrupted and absent-memory controls
- runtime adapters stay separate from memory, model, training, and scoring code
- generated outputs remain disposable
- public docs describe what the repo can reproduce, not aspirational ability

Do not claim Crafter competence, reinforcement-learning success, online
fine-tuning, task mastery, or broad generalization unless the repo can reproduce
that evidence directly.

## Project Shape

This repository is the active development root. Do not read from or modify
files outside the repository unless the user explicitly asks for outside
context.

Important package areas:

- `embryo.core`: shared types, spaces, config, registry, observation contracts.
- `embryo.runtimes`: runtime adapters and suite-specific environment wrappers.
- `embryo.datasets`: runtime-trace to supervised-dataset builders.
- `embryo.memory`: facts, queries, freshness, and router logic.
- `embryo.models`: learned module interfaces, reference models, and manifests.
- `embryo.eval`: memory-grounded scoring, guardrails, contamination, traces, and
  long-run metrics.
- `embryo.pipelines`: config-driven workflows.
- `embryo.run`: reusable CLI entry points.
- `configs/`: public package configs.
- `docs/`: public docs and claim boundaries.
- `tests/`: fixture-backed package tests.
- `runs/`: generated outputs, ignored by git.
- `internal/`: private notes, ignored by git.

Keep `.gitignore` ignoring `runs/`, `internal/`, caches, and checkpoints.

## Research Boundaries

Allowed actor/query/router inputs include:

- RGB observations and bounded RGB deltas
- previous action
- previous action outcome inferred from deployable observation
- learned or reference visual facts derived from deployable observation
- query results
- freshness/cache bits
- future deployable inventory belief only if inferred from observations or
  agent-owned events

Forbidden actor/query/router inputs include:

- reward
- done flags
- privileged backend inventory
- semantic maps
- player position
- achievements
- backend object ids
- seed/source metadata
- future labels
- backend simulator state

Forbidden fields may be used only for post-action evaluation or supervised
labels with clear provenance.

## Current Long-Run Direction

The near-term Crafter question is:

```text
In a persistent Crafter world, does deployable memory help an embodied agent
maintain useful behavior over long horizons, under controls where corrupted
memory should hurt?
```

Use fixed seed blocks:

- `train`: module fitting only
- `dev`: metric and protocol tuning
- `test`: final claim only

Protocol work should separate:

- protocol validity: actions valid, contamination absent, artifacts complete,
  deterministic replay checked
- memory evaluability: enough deployable anchor opportunities exist for fair
  clean-vs-corrupt comparison
- task performance: secondary and not claimed until competence is reproducible

Before Tier 2, keep anchor coverage explicit:

- `anchor_opportunity_count`
- `cached_anchor_use_count`
- `evaluable_anchor_count`
- `evaluable_seed_rate`
- configured coverage thresholds, including `min_evaluable_seed_rate`
- deterministic replay digest/pass status; use fresh-process replay for Crafter
  checks because in-process reconstruction can exercise library process state

Do not treat `GO_long_run_protocol_supported` as a memory claim. Memory
evaluable seed blocks should be reported separately with
`GO_memory_evaluable_seed_block` only when configured coverage thresholds pass;
insufficient anchor coverage should be reported as
`NOT_EVALUABLE_insufficient_anchor_coverage`.

The current Crafter RGB scaffold is a reference visual-anchor scaffold, not a
semantic resource detector or learned perception module. Treat it as a
calibration surface for protocol development.

## Control Arms

Use forward-facing arm names:

- `no_memory`
- `query_memory_clean`
- `query_memory_shuffled`
- `query_memory_stale`
- `query_memory_wrong_binding`
- `random_valid_action`

Oracle or scripted diagnostic arms may be useful for calibration, but they must
not be used for actor-input claims.

## Development Commands

Use the project Python environment. In this local shell, `python` may be absent;
use `python3` when needed.

Required validation before handoff:

```sh
python3 -m compileall -q embryo scripts tests examples
python3 -m unittest discover -s tests
python3 scripts/check_release.py
```

If touching the supervised memory pipeline, also run:

```sh
python3 scripts/run_supervised_memory_pipeline.py --out runs/quickstart
```

For the default long-run fixture sanity run:

```sh
python3 scripts/run_long_run_protocol.py \
  --config configs/long_run_sanity.yaml \
  --out runs/long_run_sanity_fixture
```

For Crafter smoke checks, keep Crafter optional. If Crafter is unavailable, the
protocol should produce `NOT_EVALUABLE_runtime_unavailable` artifacts rather
than fail the package test suite.

## Working Rules

- Prefer small, tested package modules over one-off experiment scripts.
- Do not introduce private migration notes, internal milestone terminology, or
  historical experiment framing into public docs.
- Keep runtime adapters independent from memory, model, scoring, and training
  logic.
- Keep hard-coded behavior clearly named as a reference baseline or scaffold.
- Do not weaken release checks to make a change pass.
- Use structured parsers and existing helpers where available.
- Keep generated artifacts in `runs/`, checkpoints out of git, and private notes
  out of public docs.
- Preserve unrelated user changes in the working tree.

## Before Finishing

Run the validation commands that match the changed surface. Report any command
that could not be run and why.

For review requests, lead with findings and file/line references. For
implementation requests, carry the work through code, tests, and a concise
handoff.
