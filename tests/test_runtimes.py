from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.eval.traces import read_jsonl
from embryo.run.collect import collect_feature_rows, collect_runtime_trace, scripted_action
from embryo.runtimes.base import action_to_backend
from embryo.runtimes import make_runtime, runtime_names
from embryo.runtimes.crafter import CRAFTER_SPEC, CrafterUnavailable


class RuntimeTests(unittest.TestCase):
    def test_builtin_runtime_registry_contains_separable_suites(self) -> None:
        names = runtime_names()

        self.assertIn("fixture_memory", names)
        self.assertIn("crafter_memory", names)

    def test_fixture_runtime_reset_and_step(self) -> None:
        runtime = make_runtime("fixture_memory", max_steps=2)
        try:
            first = runtime.reset(seed=1)
            second = runtime.step("move_forward")
        finally:
            runtime.close()

        self.assertEqual(runtime.spec.suite, "fixture")
        self.assertIn("raw_rgb_frame", first.observation)
        self.assertEqual(second.reward, 1.0)
        self.assertEqual(scripted_action(runtime.spec, first.observation), "move_forward")

    def test_collect_runtime_trace_uses_registry(self) -> None:
        rows = collect_runtime_trace("fixture_memory", steps=3, seed=1)

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["actor_contamination"]["passed"] for row in rows))
        self.assertEqual(rows[0]["runtime"], "fixture_memory")
        self.assertIn("runtime_action_names", rows[0])

    def test_collect_feature_rows_normalizes_runtime_trace(self) -> None:
        trace = collect_runtime_trace("fixture_memory", steps=3, seed=1)
        features = collect_feature_rows(trace)

        self.assertEqual(len(features), 3)
        self.assertIn("candidate_score", features[0])
        self.assertEqual(features[0]["target_action"], "move_forward")
        self.assertEqual(features[-1]["target_action"], "turn_left")

    def test_collect_script_artifact_shape(self) -> None:
        from embryo.run.collect import main

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            features_path = Path(tmpdir) / "features.jsonl"
            exit_code = main(["--runtime", "fixture_memory", "--steps", "2", "--out", str(path), "--features-out", str(features_path)])
            rows = read_jsonl(path)
            feature_rows = read_jsonl(features_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(feature_rows), 2)

    def test_crafter_runtime_failure_type_is_public(self) -> None:
        self.assertTrue(issubclass(CrafterUnavailable, RuntimeError))

    def test_crafter_action_map_is_available_without_importing_crafter(self) -> None:
        self.assertEqual(action_to_backend(CRAFTER_SPEC, "move_up"), 3)
        self.assertEqual(CRAFTER_SPEC.resource_action, "move_up")


if __name__ == "__main__":
    unittest.main()
