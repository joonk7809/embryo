# Datasets

Dataset builders live under `embryo.datasets`. They convert collected runtime
traces into module-specific supervised datasets while keeping actor inputs,
router state, and training/evaluation labels explicit.

The current builder writes:

- `fact_writer_dataset.jsonl`
- `router_dataset.jsonl`
- `bc_dataset.jsonl`
- `dataset_manifest.json`

The manifest records source path, schemas, row counts, label policy, and
contamination results.

## CLI

```bash
python scripts/build_datasets.py --config configs/datasets/fixture_supervised.yaml
```

Collected runtime features can be used directly:

```bash
python scripts/collect_runtime.py --runtime fixture_memory --features-out runs/runtime_collect/fixture_features.jsonl
python scripts/build_datasets.py --input runs/runtime_collect/fixture_features.jsonl
```

The fixture builder is a packaging and reproducibility scaffold. Real runtime
datasets should keep post-action diagnostic labels out of actor/query/router
inputs and document label provenance in the manifest.
