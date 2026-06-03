from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.eval.traces import read_jsonl
from embryo.run.evaluate import evaluate_trace
from embryo.models import assemble_stack_manifest, save_stack_manifest
from embryo.run.replay import DEFAULT_ARMS, main as replay_main, run_reference_replay, write_replay_artifacts
from embryo.run.train_supervised import train_bc_policy, train_fact_writer, train_router, write_training_artifacts


ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def test_reference_replay_separates_clean_memory(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_replay_features.jsonl")
        result = run_reference_replay(rows, arms=DEFAULT_ARMS)

        self.assertEqual(result["summary"]["decision"], "GO_reference_replay_supported")
        self.assertEqual(result["summary"]["tick_count"], len(rows) * len(DEFAULT_ARMS))
        gaps = result["summary"]["score"]["memory_grounded_gaps_clean_minus_controls"]
        for gap in gaps.values():
            self.assertIsNotNone(gap)
            self.assertGreater(gap, 0.0)

    def test_replay_artifacts_are_written(self) -> None:
        rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_replay_features.jsonl")
        result = run_reference_replay(rows)
        with TemporaryDirectory() as tmpdir:
            write_replay_artifacts(result, tmpdir)
            self.assertTrue((Path(tmpdir) / "replay_summary.json").exists())
            self.assertTrue((Path(tmpdir) / "replay_ticks.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "contamination_scan.json").exists())

    def test_evaluate_trace_reads_fixture(self) -> None:
        result = evaluate_trace(ROOT / "data" / "fixtures" / "tiny_memory_score_trace.jsonl")

        self.assertEqual(result["tick_count"], 5)
        self.assertEqual(result["score"]["best_arm"], "clean")

    def test_replay_cli_accepts_stack_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_training_artifacts(train_fact_writer(read_jsonl(ROOT / "data" / "fixtures" / "tiny_fact_writer_dataset.jsonl")), root / "fact_writer")
            write_training_artifacts(train_router(read_jsonl(ROOT / "data" / "fixtures" / "tiny_router_dataset.jsonl")), root / "router")
            write_training_artifacts(train_bc_policy(read_jsonl(ROOT / "data" / "fixtures" / "tiny_bc_dataset.jsonl")), root / "bc_policy")
            stack_path = root / "stack_manifest.json"
            save_stack_manifest(
                assemble_stack_manifest(
                    fact_writer="fact_writer/checkpoint_manifest.json",
                    router="router/checkpoint_manifest.json",
                    bc_policy="bc_policy/checkpoint_manifest.json",
                ),
                stack_path,
            )
            out = root / "replay"
            exit_code = replay_main(["--stack-manifest", str(stack_path), "--out", str(out)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((out / "stack_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
