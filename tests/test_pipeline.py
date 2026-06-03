from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.pipelines.supervised_memory import run_supervised_memory_pipeline


class SupervisedMemoryPipelineTests(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self) -> None:
        with TemporaryDirectory() as tmpdir:
            summary = run_supervised_memory_pipeline(out=tmpdir, runtime_name="fixture_memory", steps=4, seed=0)
            root = Path(tmpdir)

            self.assertEqual(summary["decision"], "GO_supervised_memory_pipeline_supported")
            self.assertEqual(summary["feature_rows"], 4)
            self.assertTrue((root / "collection" / "features.jsonl").exists())
            self.assertTrue((root / "datasets" / "dataset_manifest.json").exists())
            self.assertTrue((root / "models" / "fact_writer" / "checkpoint_manifest.json").exists())
            self.assertTrue((root / "stack" / "stack_manifest.json").exists())
            self.assertTrue((root / "replay" / "replay_summary.json").exists())
            written = json.loads((root / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["decision"], summary["decision"])


if __name__ == "__main__":
    unittest.main()
