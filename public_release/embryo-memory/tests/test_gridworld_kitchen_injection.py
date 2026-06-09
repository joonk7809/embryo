import unittest

from embryo.run.probe_gridworld_injection import (
    GO_GRIDWORLD_INJECTION_SUPPORTED,
    NO_GO_MEMORY_INJECTION_FAILURE,
    decide_gridworld_injection,
    run_arm_episode,
    run_gridworld_injection_probe,
)
from embryo.runtimes.gridworld_kitchen import GoalConditionedNavigator, make_kitchen_episode


class GridworldKitchenInjectionTests(unittest.TestCase):
    def test_goal_conditioned_navigator_reaches_drawer(self):
        episode = make_kitchen_episode(seed=3, drawer_count=8)
        navigator = GoalConditionedNavigator()
        position = episode.start_position
        goal = episode.drawer_position(episode.target_drawer_id)
        for _ in range(20):
            action = navigator.action(position, goal)
            if action == "open":
                break
            x, y = position
            if action == "right":
                x += 1
            elif action == "left":
                x -= 1
            elif action == "down":
                y += 1
            elif action == "up":
                y -= 1
            position = (x, y)
        self.assertEqual(action, "open")
        self.assertEqual(position, goal)

    def test_clean_memory_succeeds_and_wrong_binding_recovers_after_wrong_goal(self):
        episode = make_kitchen_episode(seed=12, drawer_count=10)
        clean = run_arm_episode(episode, arm="memory_clean", max_steps=64)
        wrong = run_arm_episode(episode, arm="memory_wrong_binding", max_steps=128)
        self.assertTrue(clean["retrieval_success"])
        self.assertTrue(clean["goal_injection_accuracy"])
        self.assertTrue(wrong["retrieval_success"])
        self.assertTrue(wrong["wrong_drawer_opened"])
        self.assertTrue(wrong["failed_memory_goal"])
        self.assertGreater(wrong["steps_to_retrieve"], clean["steps_to_retrieve"])

    def test_decision_accepts_separated_controls(self):
        metrics = {
            "oracle_goal_eval_only": {"retrieval_success_rate": 1.0, "mean_steps_to_retrieve": 5.0, "mean_drawers_opened_to_success": 1.0},
            "memory_clean": {"retrieval_success_rate": 1.0, "goal_injection_accuracy": 1.0, "mean_steps_to_retrieve": 5.0, "mean_drawers_opened_to_success": 1.0},
            "no_memory_search": {"retrieval_success_rate": 1.0, "mean_steps_to_retrieve": 30.0, "mean_drawers_opened_to_success": 8.0},
            "memory_shuffled_location": {"retrieval_success_rate": 1.0, "wrong_drawer_rate": 1.0, "mean_steps_to_retrieve": 35.0, "mean_drawers_opened_to_success": 8.0},
            "memory_wrong_binding": {"retrieval_success_rate": 1.0, "wrong_drawer_rate": 1.0, "mean_steps_to_retrieve": 36.0, "mean_drawers_opened_to_success": 8.0},
            "memory_stale": {"retrieval_success_rate": 1.0, "wrong_drawer_rate": 1.0, "mean_steps_to_retrieve": 34.0, "mean_drawers_opened_to_success": 8.0},
        }
        curve = {
            "6": {"pairwise": {"clean_minus_no_memory_drawers_opened": 2.0, "wrong_binding_minus_no_memory_steps": 2.0}},
            "18": {"pairwise": {"clean_minus_no_memory_drawers_opened": 7.0, "wrong_binding_minus_no_memory_steps": 6.0}},
        }
        decision, _ = decide_gridworld_injection(metrics, curve_by_drawer_count=curve, contamination={"failure_count": 0}, protocol_cfg={})
        self.assertEqual(decision, GO_GRIDWORLD_INJECTION_SUPPORTED)

    def test_decision_rejects_clean_failure(self):
        metrics = {
            "oracle_goal_eval_only": {"retrieval_success_rate": 1.0},
            "memory_clean": {"retrieval_success_rate": 0.0, "goal_injection_accuracy": 0.0, "mean_steps_to_retrieve": 0.0, "mean_drawers_opened_to_success": 0.0},
            "no_memory_search": {"retrieval_success_rate": 1.0, "mean_steps_to_retrieve": 20.0, "mean_drawers_opened_to_success": 5.0},
            "memory_shuffled_location": {"retrieval_success_rate": 1.0, "wrong_drawer_rate": 1.0},
            "memory_wrong_binding": {"retrieval_success_rate": 1.0, "wrong_drawer_rate": 1.0},
            "memory_stale": {"retrieval_success_rate": 1.0, "wrong_drawer_rate": 1.0},
        }
        decision, _ = decide_gridworld_injection(metrics, contamination={"failure_count": 0}, protocol_cfg={})
        self.assertEqual(decision, NO_GO_MEMORY_INJECTION_FAILURE)

    def test_probe_reports_drawer_count_curve(self):
        result = run_gridworld_injection_probe(
            {
                "runtime": {"width": 7, "height": 7, "drawer_counts": [6, 10], "object_count": 4, "seed_start": 12000, "seed_count": 4},
                "protocol": {"max_steps": 256, "min_drawer_gap_growth": 0.0, "min_wrong_binding_step_cost": 0.0},
            }
        )
        self.assertIn("6", result["summary"]["curve_by_drawer_count"])
        self.assertEqual(result["metrics_by_arm"]["memory_clean"]["mean_drawers_opened_to_success"], 1.0)
        self.assertGreater(result["metrics_by_arm"]["no_memory_search"]["mean_drawers_opened_to_success"], 1.0)


if __name__ == "__main__":
    unittest.main()
