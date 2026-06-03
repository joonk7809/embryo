from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO
import unittest

from embryo.eval.long_run import (
    GO_LONG_RUN_PROTOCOL_SUPPORTED,
    NOT_EVALUABLE_NO_MEMORY_ANCHORS,
    memory_grounded_protocol_score,
)
from embryo.eval.traces import read_jsonl
from embryo.run.long_run import DEFAULT_ARMS, action_is_valid, main as long_run_main, run_long_run_protocol
from embryo.runtimes.fixture import FIXTURE_SPEC


def fixture_config(*, detail_ticks: bool = False) -> dict:
    return {
        "runtime": {"name": "fixture_memory", "split": "dev", "seed_start": 10000, "seed_count": 2},
        "protocol": {"horizons": [8], "max_episodes_per_seed": 1, "detail_ticks": detail_ticks},
        "arms": list(DEFAULT_ARMS),
        "metrics": {
            "survival": True,
            "valid_actions": True,
            "loop_rate": True,
            "event_self_trigger": True,
            "memory_grounded_score": True,
            "contamination": True,
        },
    }


class LongRunProtocolTests(unittest.TestCase):
    def test_fixture_long_run_protocol_emits_required_namespaces(self) -> None:
        result = run_long_run_protocol(fixture_config(detail_ticks=True))

        summary = result["summary"]
        self.assertEqual(summary["decision"], GO_LONG_RUN_PROTOCOL_SUPPORTED)
        self.assertEqual(summary["episode_count"], 2 * len(DEFAULT_ARMS))
        self.assertGreater(summary["tick_count"], 0)
        self.assertEqual(summary["contamination"]["failure_count"], 0)
        self.assertEqual(summary["memory_grounded_score"]["status"], "computed")
        self.assertGreater(summary["memory_grounded_score"]["anchor_count"], 0)

        first = result["episodes"][0]
        self.assertIn("actor", first)
        self.assertIn("metrics", first)
        self.assertIn("eval_only", first)
        self.assertIn("contamination", first)
        self.assertEqual(first["actor"]["split"], "dev")
        self.assertIn(first["actor"]["arm"], DEFAULT_ARMS)
        self.assertEqual(first["metrics"]["valid_action_rate"], 1.0)
        self.assertEqual(first["metrics"]["contamination_failure_count"], 0)
        self.assertIn("repeated_action_loop_rate", first["metrics"])
        self.assertIn("survival_steps", first["metrics"])
        self.assertIn("reward_sum_eval_only", first["eval_only"])
        no_memory_ticks = [row for row in result["ticks"] if row["actor"]["arm"] == "no_memory"]
        self.assertTrue(no_memory_ticks)
        self.assertTrue(all(row["metrics"]["query_content_hash"] == "" for row in no_memory_ticks))

    def test_memory_score_reports_not_evaluable_without_anchors(self) -> None:
        ticks = []
        for arm in ("query_memory_clean", "no_memory", "query_memory_shuffled"):
            ticks.append(
                {
                    "actor": {
                        "runtime": "fixture_memory",
                        "arm": arm,
                        "seed": 1,
                        "horizon": 4,
                        "split": "dev",
                        "episode_index": 0,
                        "episode_id": "dev:seed-1:horizon-4:episode-0",
                        "tick": 0,
                        "action": "noop",
                    },
                    "metrics": {
                        "invalid_or_unknown_action": False,
                        "resource_memory_critical": False,
                        "resource_route_preserved": arm == "query_memory_clean",
                        "fallback_triggered": False,
                        "event_self_triggered": False,
                        "repeated_action_loop": False,
                    },
                    "eval_only": {"reward_delta_eval_only": 0.0},
                    "contamination": {"failure_count": 0, "failures": []},
                }
            )

        score = memory_grounded_protocol_score(ticks)

        self.assertEqual(score["status"], NOT_EVALUABLE_NO_MEMORY_ANCHORS)
        self.assertEqual(score["anchor_count"], 0)

    def test_cli_writes_required_artifacts_and_optional_ticks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "long_run"
            with redirect_stdout(StringIO()):
                exit_code = long_run_main(
                    [
                        "--runtime",
                        "fixture_memory",
                        "--seed-block",
                        "dev",
                        "--seed-count",
                        "1",
                        "--horizon",
                        "4",
                        "--detail-ticks",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out / "long_run_summary.json").exists())
            self.assertTrue((out / "long_run_summary.md").exists())
            self.assertTrue((out / "long_run_episodes.jsonl").exists())
            self.assertTrue((out / "protocol_manifest.json").exists())
            self.assertTrue((out / "contamination_scan.json").exists())
            self.assertTrue((out / "long_run_ticks.jsonl").exists())

            summary = json.loads((out / "long_run_summary.json").read_text(encoding="utf-8"))
            episodes = read_jsonl(out / "long_run_episodes.jsonl")
            ticks = read_jsonl(out / "long_run_ticks.jsonl")

        self.assertEqual(summary["decision"], GO_LONG_RUN_PROTOCOL_SUPPORTED)
        self.assertEqual(len(episodes), len(DEFAULT_ARMS))
        self.assertGreater(len(ticks), 0)

    def test_numeric_strings_are_not_implicitly_valid_actions(self) -> None:
        self.assertTrue(action_is_valid(FIXTURE_SPEC, "noop"))
        self.assertFalse(action_is_valid(FIXTURE_SPEC, "999"))


if __name__ == "__main__":
    unittest.main()
