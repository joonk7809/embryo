# Model Interfaces

The clean package separates model contracts from training jobs. A module that
writes facts, routes queries, or chooses actions should expose a small typed
interface and a checkpoint manifest.

## Interfaces

- `VisualFactWriter` consumes deployable visual features and emits sparse facts.
- `RouterModel` consumes query/freshness state and emits a route decision.
- `BCPolicy` consumes routed features and emits an action decision.

The package includes deterministic reference implementations for tests and
smoke replays:

- `ThresholdFactWriter`
- `RuleRouterModel`
- `RouteBCPolicy`

These are not meant to replace learned models. They keep fixtures reproducible
while future training code can load learned weights behind the same interfaces.

## Supervised Training Scaffold

The package includes tiny reference trainers for the three module boundaries:

```bash
python scripts/train_fact_writer.py --config configs/train_fact_writer.yaml
python scripts/train_router.py --config configs/train_router.yaml
python scripts/train_bc.py --config configs/train_bc.yaml
```

The reference trainers write `checkpoint_manifest.json`, `metrics.json`, and
`manifest_validation.json`. They are API and packaging checks, not final model
training.

## Checkpoint Manifests

Manifests record the public boundary around a model:

- `model_type`
- `model_name`
- `version`
- `feature_schema`
- `artifact_paths`
- `metadata`

Replay code should validate manifests before loading artifacts. Training jobs
can add metadata, but actor inputs must still obey the observation contract.

## Stack Manifests

A stack manifest points to one manifest per deployable module:

- `fact_writer`
- `router`
- `bc_policy`

Replay can load the stack directly:

```bash
python scripts/assemble_stack.py \
  --fact-writer runs/train_fact_writer_reference/checkpoint_manifest.json \
  --router runs/train_router_reference/checkpoint_manifest.json \
  --bc-policy runs/train_bc_reference/checkpoint_manifest.json \
  --out runs/reference_stack/stack_manifest.json

python scripts/replay_learned_stack.py \
  --stack-manifest runs/reference_stack/stack_manifest.json
```

## Reference Replay

`scripts/replay_learned_stack.py` composes the reference writer, router, and
policy over tiny deployable feature fixtures. It is useful for API and artifact
checks. It is not evidence of task competence.
