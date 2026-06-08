from __future__ import annotations

import unittest

import numpy as np

from embryo.eval.long_run import GO_WATER_RECALL_EVALUABLE, INCONCLUSIVE_LOW_ERE, water_recall_protocol_summary
from embryo.models import ActorDecision, MemoryResidualPolicy
from embryo.run.long_run_arms import EpisodeState, select_action_with_actor_sidecar
from embryo.runtimes.base import RuntimeSpec


WATER_SPEC = RuntimeSpec(
    name="water_fixture",
    suite="fixture",
    observation_keys=("raw_rgb_frame", "previous_rgb_frame", "previous_action"),
    action_names=("noop", "move_left", "move_right", "move_up", "move_down", "do"),
    noop_action="noop",
    resource_action="move_up",
    fallback_action="move_left",
)


def water_frame(*, x0: int | None = None, y0: int = 6) -> np.ndarray:
    frame = np.zeros((16, 16, 3), dtype=np.uint8) + 40
    if x0 is not None:
        frame[y0 : y0 + 4, x0 : x0 + 4] = np.array([20, 80, 230], dtype=np.uint8)
    return frame


def water_arm_second_decision(arm: str) -> dict:
    state = EpisodeState(seed=7, arm=arm, horizon=4, water_recall_config={"need_after_steps": 0, "h_lstm": 1, "ttl": 4})
    base = ActorDecision(action="noop", logits={name: 0.0 for name in WATER_SPEC.action_names})
    residual = MemoryResidualPolicy(strength=1.0, max_abs_bias=8.0)
    visible = {"raw_rgb_frame": water_frame(x0=11), "previous_rgb_frame": None, "previous_action": "noop"}
    hidden = {"raw_rgb_frame": water_frame(x0=None), "previous_rgb_frame": visible["raw_rgb_frame"], "previous_action": "move_up"}

    first = select_action_with_actor_sidecar(WATER_SPEC, visible, state, arm, base, residual, memory_residual_bias=8.0)
    state.observe(first["action"], first["route_mode"])
    return select_action_with_actor_sidecar(WATER_SPEC, hidden, state, arm, base, residual, memory_residual_bias=8.0)


class WaterRecallLongRunTests(unittest.TestCase):
    def test_water_recall_arms_diverge_only_on_occluded_evaluable_recall(self) -> None:
        off = water_arm_second_decision("water_recall_off")
        clean = water_arm_second_decision("water_recall_clean")
        shuffled = water_arm_second_decision("water_recall_shuffled")
        stale = water_arm_second_decision("water_recall_stale")
        wrong = water_arm_second_decision("water_recall_wrong_binding")

        self.assertTrue(clean["water_recall_ere"])
        self.assertEqual(clean["water_recall_target_action"], "move_right")
        self.assertEqual(clean["action"], "move_right")
        self.assertTrue(clean["water_recall_consistent_action"])
        self.assertEqual(off["action"], "noop")
        self.assertFalse(off["water_recall_active"])
        self.assertEqual(stale["action"], "noop")
        self.assertFalse(stale["water_recall_active"])
        self.assertNotEqual(shuffled["water_recall_effective_target_action"], clean["water_recall_target_action"])
        self.assertFalse(shuffled["water_recall_consistent_action"])
        self.assertEqual(wrong["water_recall_effective_target_action"], "move_left")
        self.assertFalse(wrong["water_recall_consistent_action"])

    def test_water_recall_summary_reports_low_ere_separately(self) -> None:
        manifest = {
            "protocol": {"water_recall": {"min_ere_count": 2, "min_ere_episode_rate": 0.5}},
            "arms": ["water_recall_off", "water_recall_clean"],
        }
        row = water_tick_row(arm="water_recall_clean", seed=1, tick=0, ere=True, action="move_right", target="move_right")

        summary = water_recall_protocol_summary([row], protocol_manifest=manifest)

        self.assertEqual(summary["status"], INCONCLUSIVE_LOW_ERE)
        self.assertEqual(summary["clean_ere_count"], 1)

    def test_water_recall_summary_computes_clean_control_contrast(self) -> None:
        manifest = {
            "protocol": {"water_recall": {"min_ere_count": 1, "min_ere_episode_rate": 0.5}},
            "arms": ["water_recall_off", "water_recall_clean"],
        }
        ticks = [
            water_tick_row(arm="water_recall_clean", seed=1, tick=0, ere=True, action="move_right", target="move_right"),
            water_tick_row(arm="water_recall_off", seed=1, tick=0, ere=False, action="noop", target="move_right"),
        ]

        summary = water_recall_protocol_summary(ticks, protocol_manifest=manifest)

        self.assertEqual(summary["status"], GO_WATER_RECALL_EVALUABLE)
        self.assertEqual(summary["contrasts"]["water_recall_off"]["comparable_ere_count"], 1)
        self.assertEqual(summary["contrasts"]["water_recall_off"]["clean_minus_control_recall_consistent_action_rate"], 1.0)


def water_tick_row(*, arm: str, seed: int, tick: int, ere: bool, action: str, target: str) -> dict:
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
            "water_recall_ere": ere,
            "water_visible": False,
            "water_need_active": True,
            "water_recall_active": arm == "water_recall_clean",
            "water_recall_consistent_action": action == target,
            "water_fact_age": 2,
        },
        "eval_only": {"drink_eval_only": 0.0},
        "contamination": {"failure_count": 0, "failures": []},
    }


if __name__ == "__main__":
    unittest.main()
