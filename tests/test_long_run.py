from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np

from embryo.datasets.fact_writer import DEFAULT_ACTION_VOCAB, feature_schema_for_actions
from embryo.eval.long_run import (
    GO_LONG_RUN_PROTOCOL_SUPPORTED,
    GO_MEMORY_EVALUABLE_SEED_BLOCK,
    NOT_EVALUABLE_INSUFFICIENT_ANCHOR_COVERAGE,
    NOT_EVALUABLE_NO_MEMORY_ANCHORS,
    memory_grounded_protocol_score,
    memory_score_rows,
    replay_digest,
    summarize_long_run_protocol,
)
from embryo.eval.traces import read_jsonl
from embryo.models import LearnedFactWriterV0, save_fact_writer_checkpoint
from embryo.run.long_run import DEFAULT_ARMS, action_is_valid, main as long_run_main, resolve_protocol_manifest, run_long_run_protocol
from embryo.run.long_run_fact_surface import (
    ALLOWED_LEARNED_INPUTS,
    LEARNED_VISUAL_ANCHOR_FAMILY,
    REQUIRED_FACT_FIELDS,
    learned_model_inputs,
    load_fact_surface,
)
from embryo.runtimes.fixture import FIXTURE_SPEC


def fixture_config(*, detail_ticks: bool = False) -> dict:
    return {
        "runtime": {"name": "fixture_memory", "split": "dev", "seed_start": 10000, "seed_count": 2},
        "protocol": {"horizons": [8], "max_episodes_per_seed": 1, "detail_ticks": detail_ticks},
        "arms": list(DEFAULT_ARMS),
    }


def constant_learned_writer(*, probability_bias: float) -> LearnedFactWriterV0:
    schema = feature_schema_for_actions(DEFAULT_ACTION_VOCAB)
    zeros = tuple(0.0 for _ in schema)
    ones = tuple(1.0 for _ in schema)
    return LearnedFactWriterV0(
        feature_schema=schema,
        feature_mean=zeros,
        feature_scale=ones,
        visible_weights=zeros,
        visible_bias=probability_bias,
        salience_weights=zeros,
        salience_bias=0.5,
        change_weights=zeros,
        change_bias=-1.0,
    )


def rgb_observation() -> dict:
    image = np.zeros((24, 24, 3), dtype=np.float32) + 0.25
    image[9:15, 9:15, 0] = 1.0
    previous = np.zeros_like(image) + 0.25
    return {
        "raw_rgb_frame": image,
        "previous_rgb_frame": previous,
        "previous_action": "noop",
        "rgb_delta_score": 0.01,
        "visual_change_event": True,
        "failed_action_event": False,
    }


class LongRunProtocolTests(unittest.TestCase):
    def test_fixture_long_run_protocol_emits_required_namespaces(self) -> None:
        result = run_long_run_protocol(fixture_config(detail_ticks=True))

        summary = result["summary"]
        self.assertEqual(summary["decision"], GO_LONG_RUN_PROTOCOL_SUPPORTED)
        self.assertEqual(summary["memory_evaluability_decision"], GO_MEMORY_EVALUABLE_SEED_BLOCK)
        self.assertEqual(summary["episode_count"], 2 * len(DEFAULT_ARMS))
        self.assertGreater(summary["tick_count"], 0)
        self.assertTrue(summary["deterministic_replay"]["passed"])
        self.assertEqual(summary["deterministic_replay"]["mode"], "fresh_process")
        self.assertIn("all_arm_anchor_opportunity_count", summary["anchor_coverage"])
        self.assertIn("clean_arm_anchor_opportunity_count", summary["anchor_coverage"])
        self.assertNotIn("anchor_opportunity_count", summary["anchor_coverage"])
        self.assertGreaterEqual(
            summary["anchor_coverage"]["all_arm_anchor_opportunity_count"],
            summary["anchor_coverage"]["clean_arm_anchor_opportunity_count"],
        )
        self.assertGreater(summary["anchor_coverage"]["cached_anchor_use_count"], 0)
        self.assertGreaterEqual(summary["anchor_coverage"]["evaluable_seed_rate"], 0.5)
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

    def test_protocol_go_is_separate_from_memory_evaluability(self) -> None:
        summary = summarize_long_run_protocol(
            protocol_manifest=fixture_config(),
            episodes=[],
            ticks=[],
            contamination={"failure_count": 0, "failures": [], "passed": True},
            deterministic_replay={"enabled": True, "passed": True, "primary_digest": "x", "repeat_digest": "x"},
        )

        self.assertEqual(summary["decision"], GO_LONG_RUN_PROTOCOL_SUPPORTED)
        self.assertEqual(summary["memory_evaluability_decision"], NOT_EVALUABLE_INSUFFICIENT_ANCHOR_COVERAGE)
        self.assertEqual(summary["anchor_coverage"]["evaluable_anchor_count"], 0)

    def test_memory_evaluability_requires_configured_seed_rate(self) -> None:
        manifest = fixture_config()
        manifest["runtime"]["seeds"] = [1, 2, 3, 4]
        manifest["runtime"]["seed_count"] = 4
        manifest["protocol"]["min_evaluable_seed_rate"] = 0.5
        ticks = []
        corrupt_arms = ("query_memory_shuffled", "query_memory_stale", "query_memory_wrong_binding")
        for seed in (1, 2, 3, 4):
            arms = ("query_memory_clean", *corrupt_arms) if seed == 1 else ("query_memory_clean",)
            for arm in arms:
                ticks.append(
                    {
                        "actor": {
                            "runtime": "fixture_memory",
                            "arm": arm,
                            "seed": seed,
                            "horizon": 4,
                            "split": "dev",
                            "episode_index": 0,
                            "episode_id": f"dev:seed-{seed}:horizon-4:episode-0",
                            "tick": 1,
                            "action": "move_forward" if arm == "query_memory_clean" else "noop",
                        },
                        "metrics": {
                            "invalid_or_unknown_action": False,
                            "resource_memory_critical": False,
                            "memory_anchor_critical": arm == "query_memory_clean",
                            "anchor_currently_visible": False,
                            "query_used_cached_fact": True,
                            "event_self_triggered": False,
                            "repeated_action_loop": False,
                        },
                        "eval_only": {"reward_delta_eval_only": 0.0},
                        "contamination": {"failure_count": 0, "failures": []},
                    }
                )

        summary = summarize_long_run_protocol(
            protocol_manifest=manifest,
            episodes=[],
            ticks=ticks,
            contamination={"failure_count": 0, "failures": [], "passed": True},
            deterministic_replay={"enabled": True, "passed": True, "primary_digest": "x", "repeat_digest": "x"},
        )

        self.assertEqual(summary["anchor_coverage"]["evaluable_seed_rate"], 0.25)
        self.assertEqual(summary["memory_evaluability_decision"], NOT_EVALUABLE_INSUFFICIENT_ANCHOR_COVERAGE)

    def test_replay_digest_includes_eval_only_fields(self) -> None:
        base = {
            "actor": {"arm": "query_memory_clean", "seed": 1, "tick": 0},
            "metrics": {"valid_action": True},
            "eval_only": {"reward_delta_eval_only": 0.0},
            "contamination": {"failure_count": 0, "failures": []},
        }
        changed = dict(base)
        changed["eval_only"] = {"reward_delta_eval_only": 1.0}

        self.assertNotEqual(replay_digest([base]), replay_digest([changed]))

    def test_crafter_backend_patch_is_manifested_and_configurable(self) -> None:
        enabled = resolve_protocol_manifest({"runtime": {"name": "crafter_memory"}})
        disabled = resolve_protocol_manifest({"runtime": {"name": "crafter_memory", "deterministic_backend_patch": False}})
        fixture = resolve_protocol_manifest({"runtime": {"name": "fixture_memory", "deterministic_backend_patch": True}})

        self.assertTrue(enabled["runtime"]["deterministic_backend_patch"])
        self.assertEqual(enabled["runtime"]["backend_determinism_patch"], "crafter_balance_object_order_v1")
        self.assertFalse(disabled["runtime"]["deterministic_backend_patch"])
        self.assertIsNone(disabled["runtime"]["backend_determinism_patch"])
        self.assertFalse(fixture["runtime"]["deterministic_backend_patch"])
        self.assertIsNone(fixture["runtime"]["backend_determinism_patch"])

    def test_fact_surface_defaults_to_reference_scaffold(self) -> None:
        manifest = resolve_protocol_manifest({})

        self.assertEqual(manifest["fact_surface"]["name"], "reference_rgb_scaffold")

    def test_manifest_records_learned_fact_surface_and_checkpoint(self) -> None:
        manifest = resolve_protocol_manifest(
            {
                "fact_surface": {
                    "name": "learned_fact_writer_v0",
                    "checkpoint": "runs/fact_writer_sanity/checkpoint/fact_writer.json",
                    "threshold": 0.4,
                }
            }
        )

        self.assertEqual(manifest["fact_surface"]["name"], "learned_fact_writer_v0")
        self.assertEqual(manifest["fact_surface"]["checkpoint"], "runs/fact_writer_sanity/checkpoint/fact_writer.json")
        self.assertEqual(manifest["fact_surface"]["threshold"], 0.4)

    def test_learned_fact_surface_loads_checkpoint_and_emits_required_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint" / "fact_writer.json"
            save_fact_writer_checkpoint(constant_learned_writer(probability_bias=1.0), checkpoint)
            surface = load_fact_surface(
                {
                    "name": "learned_fact_writer_v0",
                    "checkpoint": str(checkpoint),
                    "threshold": 0.5,
                }
            )

            first = surface.apply(rgb_observation())
            second = surface.apply(rgb_observation())

        for field in REQUIRED_FACT_FIELDS:
            self.assertIn(field, first)
        self.assertEqual(first["visual_anchor_family"], LEARNED_VISUAL_ANCHOR_FAMILY)
        self.assertTrue(first["visual_anchor_visible"])
        self.assertEqual(first["center_patch_hash"], second["center_patch_hash"])
        self.assertEqual(first["center_salience_score"], second["center_salience_score"])

    def test_learned_fact_surface_protocol_records_actor_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint" / "fact_writer.json"
            save_fact_writer_checkpoint(constant_learned_writer(probability_bias=1.0), checkpoint)
            config = fixture_config(detail_ticks=True)
            config["runtime"]["seed_count"] = 1
            config["protocol"]["horizons"] = [2]
            config["fact_surface"] = {
                "name": "learned_fact_writer_v0",
                "checkpoint": str(checkpoint),
                "threshold": 0.5,
            }

            result = run_long_run_protocol(config)

        first_actor = result["ticks"][0]["actor"]
        self.assertEqual(first_actor["fact_surface"], "learned_fact_writer_v0")
        self.assertEqual(first_actor["fact_surface_checkpoint"], str(checkpoint))
        self.assertEqual(first_actor["fact_surface_threshold"], 0.5)
        self.assertEqual(result["summary"]["protocol"]["fact_surface"], "learned_fact_writer_v0")
        self.assertEqual(result["summary"]["protocol"]["fact_surface_checkpoint"], str(checkpoint))

    def test_learned_fact_surface_model_inputs_ignore_forbidden_fields(self) -> None:
        observation = rgb_observation()
        observation.update(
            {
                "reward": 1.0,
                "done": False,
                "seed": 10000,
                "achievements": {"collect_wood": 1},
                "info": {"inventory": {"wood": 1}, "player_pos": [1, 2]},
            }
        )

        inputs = learned_model_inputs(observation)

        self.assertEqual(tuple(inputs), ALLOWED_LEARNED_INPUTS)
        self.assertNotIn("reward", inputs)
        self.assertNotIn("done", inputs)
        self.assertNotIn("seed", inputs)
        self.assertNotIn("info", inputs)

    def test_reference_exploration_sweep_is_valid_deterministic_and_nonmemory(self) -> None:
        config = {
            "runtime": {"name": "fixture_memory", "split": "dev", "seed_start": 10000, "seed_count": 1},
            "protocol": {"horizons": [8], "max_episodes_per_seed": 1, "detail_ticks": True},
            "arms": ["reference_exploration_sweep"],
        }

        first = run_long_run_protocol(config)
        second = run_long_run_protocol(config)
        episode = first["episodes"][0]
        ticks = first["ticks"]

        self.assertEqual(episode["metrics"]["valid_action_rate"], 1.0)
        self.assertEqual(episode["contamination"]["failure_count"], 0)
        self.assertTrue(all(row["metrics"]["query_content_hash"] == "" for row in ticks))
        self.assertEqual(replay_digest(ticks), replay_digest(second["ticks"]))
        self.assertFalse(memory_score_rows(ticks))

    def test_memory_arms_use_matched_exploration_before_cached_memory(self) -> None:
        arms = ["query_memory_clean", "query_memory_shuffled", "query_memory_stale", "query_memory_wrong_binding"]
        effective_fields = (
            "effective_candidate_score",
            "effective_cache_age",
            "effective_invalidated",
            "effective_center_patch_hash",
            "effective_fact_value",
            "effective_memory_arm",
        )
        result = run_long_run_protocol(
            {
                "runtime": {"name": "fixture_memory", "split": "dev", "seed_start": 10000, "seed_count": 1},
                "protocol": {"horizons": [4], "max_episodes_per_seed": 1, "detail_ticks": True},
                "arms": arms,
            }
        )
        rows_by_tick = {}
        for row in result["ticks"]:
            rows_by_tick.setdefault(row["actor"]["tick"], []).append(row)
            for field in effective_fields:
                self.assertIn(field, row["metrics"])

        for tick in (0, 1):
            rows = rows_by_tick[tick]
            self.assertEqual({row["actor"]["arm"] for row in rows}, set(arms))
            self.assertEqual({row["actor"]["action"] for row in rows}, {"move_forward"})
            self.assertEqual({row["actor"]["route_mode"] for row in rows}, {"matched_exploration_fallback"})
            self.assertTrue(all(not row["metrics"]["query_used_cached_fact"] for row in rows))
            self.assertEqual({row["metrics"]["query_content_hash"] for row in rows}, {""})
            for row in rows:
                for field in effective_fields:
                    self.assertIsNone(row["metrics"][field])

        cached_rows = [row for row in result["ticks"] if row["metrics"]["query_used_cached_fact"]]
        self.assertTrue(cached_rows)
        cached_by_arm = {row["actor"]["arm"]: row for row in cached_rows}
        self.assertEqual(set(cached_by_arm), set(arms))
        for row in cached_rows:
            metrics = row["metrics"]
            self.assertIsNotNone(metrics["effective_candidate_score"])
            self.assertIsNotNone(metrics["effective_cache_age"])
            self.assertIsNotNone(metrics["effective_invalidated"])
            self.assertIsNotNone(metrics["effective_center_patch_hash"])
            self.assertIsNotNone(metrics["effective_fact_value"])
            self.assertEqual(metrics["effective_memory_arm"], row["actor"]["arm"])
            self.assertIn("memory_residual_target_action", metrics)

        clean = cached_by_arm["query_memory_clean"]["metrics"]
        shuffled = cached_by_arm["query_memory_shuffled"]["metrics"]
        stale = cached_by_arm["query_memory_stale"]["metrics"]
        wrong_binding = cached_by_arm["query_memory_wrong_binding"]["metrics"]
        self.assertEqual(clean["memory_residual_target_action"], "move_forward")
        self.assertEqual(wrong_binding["memory_residual_target_action"], "turn_left")
        self.assertIsNone(stale["memory_residual_target_action"])
        self.assertFalse(clean["effective_invalidated"])
        self.assertFalse(shuffled["effective_invalidated"])
        self.assertTrue(stale["effective_invalidated"])
        self.assertFalse(wrong_binding["effective_invalidated"])
        self.assertNotEqual(clean["effective_center_patch_hash"], shuffled["effective_center_patch_hash"])
        self.assertEqual(clean["effective_center_patch_hash"], wrong_binding["effective_center_patch_hash"])
        self.assertEqual(clean["effective_center_patch_hash"], stale["effective_center_patch_hash"])
        self.assertEqual(
            len({row["metrics"]["query_content_hash"] for row in cached_by_arm.values()}),
            len(arms),
        )

    def test_visual_anchor_scaffold_can_compute_memory_score_without_reward(self) -> None:
        ticks = []
        specs = {
            "query_memory_clean": ("move_forward", True, 1.0),
            "no_memory": ("noop", False, 0.0),
            "query_memory_shuffled": ("noop", False, 0.0),
            "query_memory_stale": ("noop", False, 0.0),
            "query_memory_wrong_binding": ("noop", False, 0.0),
        }
        for arm, (action, anchor, followthrough) in specs.items():
            ticks.append(
                {
                    "actor": {
                        "runtime": "crafter_memory",
                        "arm": arm,
                        "seed": 1,
                        "horizon": 4,
                        "split": "dev",
                        "episode_index": 0,
                        "episode_id": "dev:seed-1:horizon-4:episode-0",
                        "tick": 1,
                        "action": action,
                    },
                    "metrics": {
                        "invalid_or_unknown_action": False,
                        "resource_memory_critical": False,
                        "memory_anchor_critical": anchor,
                        "anchor_family": "visual_salience_v1",
                        "anchor_fact_age": 1,
                        "anchor_currently_visible": False,
                        "query_used_cached_fact": arm != "no_memory",
                        "resource_route_preserved": False,
                        "fallback_triggered": False,
                        "event_self_triggered": False,
                        "repeated_action_loop": False,
                        "memory_followthrough_delta": followthrough,
                    },
                    "eval_only": {"reward_delta_eval_only": 0.0},
                    "contamination": {"failure_count": 0, "failures": []},
                }
            )

        score = memory_grounded_protocol_score(ticks)

        self.assertEqual(score["status"], "computed")
        self.assertEqual(score["anchor_count"], 1)
        gaps = score["score"]["memory_grounded_gaps_clean_minus_controls"]
        self.assertGreater(gaps["no_memory"], 0.0)

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
        self.assertIn("memory_evaluability_decision", summary)
        self.assertIn("anchor_coverage", summary)
        self.assertTrue(summary["deterministic_replay"]["passed"])
        self.assertEqual(summary["deterministic_replay"]["mode"], "fresh_process")
        self.assertEqual(len(episodes), len(DEFAULT_ARMS))
        self.assertGreater(len(ticks), 0)

    def test_numeric_strings_are_not_implicitly_valid_actions(self) -> None:
        self.assertTrue(action_is_valid(FIXTURE_SPEC, "noop"))
        self.assertFalse(action_is_valid(FIXTURE_SPEC, "999"))


if __name__ == "__main__":
    unittest.main()
