from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.datasets import build_supervised_datasets, write_dataset_bundle
from embryo.eval.traces import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class DatasetBuilderTests(unittest.TestCase):
    def test_builds_all_supervised_module_datasets(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_replay_features.jsonl")
        result = build_supervised_datasets(rows, source_name="fixture")

        self.assertEqual(len(result.fact_writer_rows), len(rows))
        self.assertEqual(len(result.router_rows), len(rows))
        self.assertEqual(len(result.bc_rows), len(rows))
        self.assertTrue(result.manifest["contamination"]["passed"])
        self.assertEqual(result.router_rows[0]["target_route"], "resource")
        self.assertEqual(result.bc_rows[0]["target_action"], "move_forward")

    def test_writes_dataset_bundle(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_replay_features.jsonl")
        result = build_supervised_datasets(rows)
        with TemporaryDirectory() as tmpdir:
            write_dataset_bundle(result, tmpdir)
            root = Path(tmpdir)

            self.assertTrue((root / "fact_writer_dataset.jsonl").exists())
            self.assertTrue((root / "router_dataset.jsonl").exists())
            self.assertTrue((root / "bc_dataset.jsonl").exists())
            self.assertTrue((root / "dataset_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
