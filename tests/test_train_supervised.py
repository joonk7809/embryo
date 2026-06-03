from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.eval.traces import read_jsonl
from embryo.models import ThresholdFactWriter, build_model_from_manifest, load_checkpoint_manifest
from embryo.run.train_supervised import train_bc_policy, train_fact_writer, train_module, train_router, write_training_artifacts


ROOT = Path(__file__).resolve().parents[1]


class SupervisedTrainingTests(unittest.TestCase):
    def test_fact_writer_training_fits_threshold_manifest(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_fact_writer_dataset.jsonl")
        result = train_fact_writer(rows)

        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        writer = build_model_from_manifest(result["manifest"])
        self.assertIsInstance(writer, ThresholdFactWriter)
        self.assertTrue(writer.predict({"candidate_score": 0.9}).predicted)
        self.assertFalse(writer.predict({"candidate_score": 0.1}).predicted)

    def test_router_training_validates_rule_contract(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_router_dataset.jsonl")
        result = train_router(rows)

        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual(result["manifest"].model_type, "rule_router")

    def test_bc_training_emits_route_action_mapping(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_bc_dataset.jsonl")
        result = train_bc_policy(rows)

        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        mapping = result["manifest"].metadata["parameters"]["action_by_route"]
        self.assertEqual(mapping["resource"], "move_forward")

    def test_training_artifacts_round_trip(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_fact_writer_dataset.jsonl")
        result = train_module("fact_writer", rows)
        with TemporaryDirectory() as tmpdir:
            write_training_artifacts(result, tmpdir)
            manifest = load_checkpoint_manifest(Path(tmpdir) / "checkpoint_manifest.json")
            self.assertEqual(manifest.model_type, "threshold_fact_writer")
            self.assertTrue((Path(tmpdir) / "metrics.json").exists())
            self.assertTrue((Path(tmpdir) / "manifest_validation.json").exists())


if __name__ == "__main__":
    unittest.main()
