from __future__ import annotations

import unittest

from embryo.eval.long_run import GO_BENCH_RECALL_EVALUABLE, INCONCLUSIVE_LOW_ERE, bench_recall_protocol_summary
from embryo.models import ActorDecision, MemoryResidualPolicy
from embryo.run.long_run_arms import EpisodeState, select_action_with_actor_sidecar
from embryo.runtimes.base import RuntimeSpec


BENCH_SPEC = RuntimeSpec(
    name="bench_fixture",
    suite="fixture",
    observation_keys=("raw_rgb_frame", "previous_rgb_frame", "previous_action", "failed_action_event"),
    action_names=("noop", "move_left", "move_right", "move_up", "move_down", "place_table", "make_wood_pickaxe"),
    noop_action="noop",
    resource_action="move_up",
    fallback_action="move_left",
)


def bench_decision_sequence(arm: str) -> tuple[dict, dict]:
    state = EpisodeState(seed=11, arm=arm, horizon=4, bench_recall_config={"h_lstm": 1, "ttl": 8})
    residual = MemoryResidualPolicy(strength=1.0, max_abs_bias=8.0)
    base = ActorDecision(action="place_table", logits={name: 0.0 for name in BENCH_SPEC.action_names} | {"place_table": 4.0})
    placed = {"raw_rgb_frame": "a", "previous_rgb_frame": "z", "previous_action": "place_table", "failed_action_event": False}
    moved_away = {"raw_rgb_frame": "b", "previous_rgb_frame": "a", "previous_action": "move_right", "failed_action_event": False}

    first = select_action_with_actor_sidecar(BENCH_SPEC, placed, state, arm, base, residual, memory_residual_bias=8.0)
    state.observe(first["action"], first["route_mode"])
    second = select_action_with_actor_sidecar(BENCH_SPEC, moved_away, state, arm, base, residual, memory_residual_bias=8.0)
    return first, second


class BenchRecallLongRunTests(unittest.TestCase):
    def test_clean_bench_recall_suppresses_repeat_place_at_center(self) -> None:
        first, _ = bench_decision_sequence("bench_recall_clean")

        self.assertTrue(first["bench_fact_present"])
        self.assertEqual(first["bench_fact_bearing"], "center")
        self.assertEqual(first["action"], "noop")
        self.assertTrue(first["bench_repeat_place_suppressed"])

    def test_bench_recall_arms_diverge_after_table_is_offscreen(self) -> None:
        _, off = bench_decision_sequence("bench_recall_off")
        _, clean = bench_decision_sequence("bench_recall_clean")
        _, stale = bench_decision_sequence("bench_recall_stale")
        _, wrong = bench_decision_sequence("bench_recall_wrong_binding")
        _, shuffled = bench_decision_sequence("bench_recall_shuffled")

        self.assertTrue(clean["bench_recall_ere"])
        self.assertEqual(clean["bench_fact_bearing"], "left")
        self.assertEqual(clean["bench_recall_target_action"], "move_left")
        self.assertEqual(clean["action"], "move_left")
        self.assertTrue(clean["bench_recall_consistent_action"])
        self.assertEqual(off["action"], "place_table")
        self.assertFalse(off["bench_recall_active"])
        self.assertEqual(stale["action"], "place_table")
        self.assertFalse(stale["bench_recall_active"])
        self.assertEqual(wrong["bench_recall_effective_target_action"], "move_right")
        self.assertFalse(wrong["bench_recall_consistent_action"])
        self.assertNotEqual(shuffled["bench_recall_effective_target_action"], clean["bench_recall_target_action"])

    def test_bench_recall_summary_reports_low_ere_separately(self) -> None:
        manifest = {
            "protocol": {"bench_recall": {"min_ere_count": 2, "min_ere_episode_rate": 0.5}},
            "arms": ["bench_recall_off", "bench_recall_clean"],
        }
        row = bench_tick_row(arm="bench_recall_clean", seed=1, tick=0, ere=True, action="move_left", target="move_left")

        summary = bench_recall_protocol_summary([row], protocol_manifest=manifest)

        self.assertEqual(summary["status"], INCONCLUSIVE_LOW_ERE)
        self.assertEqual(summary["clean_ere_count"], 1)

    def test_bench_recall_summary_computes_clean_control_contrast(self) -> None:
        manifest = {
            "protocol": {"bench_recall": {"min_ere_count": 1, "min_ere_episode_rate": 0.5}},
            "arms": ["bench_recall_off", "bench_recall_clean"],
        }
        ticks = [
            bench_tick_row(arm="bench_recall_clean", seed=1, tick=0, ere=True, action="move_left", target="move_left"),
            bench_tick_row(arm="bench_recall_off", seed=1, tick=0, ere=False, action="place_table", target="move_left"),
        ]

        summary = bench_recall_protocol_summary(ticks, protocol_manifest=manifest)

        self.assertEqual(summary["status"], GO_BENCH_RECALL_EVALUABLE)
        self.assertEqual(summary["contrasts"]["bench_recall_off"]["comparable_ere_count"], 1)
        self.assertEqual(summary["contrasts"]["bench_recall_off"]["clean_minus_control_recall_consistent_action_rate"], 1.0)


def bench_tick_row(*, arm: str, seed: int, tick: int, ere: bool, action: str, target: str) -> dict:
    return {
        "actor": {
            "arm": arm,
            "seed": seed,
            "episode_id": f"dev:seed-{seed}:horizon-4:episode-0",
            "episode_index": 0,
            "tick": tick,
            "action": action,
        },
        "metrics": {
            "bench_recall_ere": ere,
            "bench_fact_present": True,
            "bench_need_active": True,
            "bench_recall_active": arm == "bench_recall_clean",
            "bench_recall_consistent_action": action == target,
            "bench_fact_age": 2,
        },
        "eval_only": {},
        "contamination": {"failure_count": 0, "failures": []},
    }


if __name__ == "__main__":
    unittest.main()
