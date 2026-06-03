from __future__ import annotations

import unittest

from embryo.eval.guardrails import event_repetition_guardrail, resource_binding_guardrail
from embryo.eval.memory_grounded_score import ScoreConfig, compute_memory_grounded_scores, compute_window_sensitivity


def tick(
    arm: str,
    tick_id: int,
    action: str,
    *,
    progress: float = 0.0,
    resource: bool = True,
    fallback: bool = False,
    loop: bool = False,
) -> dict:
    return {
        "arm": arm,
        "seed": 1,
        "episode_id": "e1",
        "episode_index": 0,
        "tick": tick_id,
        "action": action,
        "resource_memory_critical": resource,
        "resource_route_preserved": arm == "clean",
        "diagnostic_progress_delta_teacher_only": progress,
        "fallback_triggered": fallback,
        "action_entropy_proxy": 0.1,
        "repeated_action_loop": loop,
        "event_self_triggered": False,
        "selected_action": {"action_name": action, "invalid_or_unknown": False},
    }


def fixture_ticks() -> list[dict]:
    rows: list[dict] = []
    specs = {
        "clean": ("right", 2.0, False, False),
        "wrong_binding": ("left", 0.2, True, False),
        "shuffled": ("left", 0.1, True, False),
        "stale": ("noop", 0.0, False, False),
        "no_memory": ("noop", 0.0, False, True),
    }
    for arm, (action, progress, fallback, loop) in specs.items():
        rows.append(tick(arm, 0, action, progress=0.0, resource=True, fallback=fallback, loop=loop))
        rows.append(tick(arm, 1, action, progress=progress, resource=True, fallback=fallback, loop=loop))
        rows.append(tick(arm, 2, "noop", progress=0.0, resource=False, loop=loop))
    return rows


class MemoryGroundedScoreTests(unittest.TestCase):
    def test_clean_separates_from_controls(self) -> None:
        result = compute_memory_grounded_scores(fixture_ticks(), config=ScoreConfig(followthrough_window=2))

        self.assertEqual(result["best_arm"], "clean")
        for gap in result["memory_grounded_gaps_clean_minus_controls"].values():
            self.assertIsNotNone(gap)
            self.assertGreater(gap, 0.0)

    def test_window_sensitivity_stable(self) -> None:
        sensitivity = compute_window_sensitivity(fixture_ticks(), windows=(1, 2, 4), config=ScoreConfig())

        self.assertTrue(sensitivity["stable_for_at_least_two_windows"])
        self.assertGreaterEqual(len(sensitivity["stable_windows"]), 2)

    def test_resource_binding_guardrail_passes(self) -> None:
        guardrail = resource_binding_guardrail(fixture_ticks())

        self.assertTrue(guardrail["passed"])
        self.assertGreaterEqual(guardrail["route_preservation"], 0.8)

    def test_event_repetition_guardrail_passes(self) -> None:
        guardrail = event_repetition_guardrail(fixture_ticks())

        self.assertTrue(guardrail["passed"])
        self.assertLessEqual(guardrail["event_self_trigger_rate"], 0.15)
        self.assertLess(guardrail["repeated_action_loop_excess_vs_no_memory"], 0.0)

    def test_score_handles_missing_clean_arm(self) -> None:
        result = compute_memory_grounded_scores([tick("wrong_binding", 0, "left")])

        self.assertEqual(result["error"], "clean_arm_missing")
        self.assertEqual(result["anchor_count"], 0)


if __name__ == "__main__":
    unittest.main()
