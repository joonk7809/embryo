# Configs

Config files are the reproducibility surface for the Embryo testbed.

- `base.yaml`: shared defaults.
- `smoke.yaml`: tiny local smoke.
- `crafter_memory.yaml`: frozen Crafter memory-contract defaults.
- `learned_stack_replay.yaml`: future learned-stack replay config.
- `learned_stack_manifest_replay.yaml`: replay config for manifest-loaded stacks.
- `train_fact_writer.yaml`, `train_router.yaml`, `train_bc.yaml`: tiny supervised reference trainers.
- `runtimes/`: runtime-specific defaults kept separate from model and memory configs.
- `datasets/`: dataset-builder configs for turning traces into module datasets.
- `pipelines/`: end-to-end workflow configs.
