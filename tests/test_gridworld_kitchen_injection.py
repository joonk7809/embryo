import unittest

from embryo.run.probe_gridworld_injection import (
    GO_GRIDWORLD_INJECTION_SUPPORTED,
    NO_GO_MEMORY_INJECTION_FAILURE,
    decide_gridworld_injection,
    run_arm_episode,
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

    def test_clean_memory_succeeds_and_wrong_binding_fails(self):
        episode = make_kitchen_episode(seed=12, drawer_count=10)
        clean = run_arm_episode(episode, arm="memory_clean", max_steps=64)
        wrong = run_arm_episode(episode, arm="memory_wrong_binding", max_steps=64)
        self.assertTrue(clean["retrieval_success"])
        self.assertTrue(clean["goal_injection_accuracy"])
        self.assertFalse(wrong["retrieval_success"])
        self.assertTrue(wrong["wrong_drawer_opened"])

    def test_decision_accepts_separated_controls(self):
        metrics = {
            "oracle_goal_eval_only": {"retrieval_success_rate": 1.0},
            "memory_clean": {"retrieval_success_rate": 1.0, "goal_injection_accuracy": 1.0, "mean_steps_to_retrieve": 5.0},
            "no_memory_search": {"retrieval_success_rate": 1.0, "mean_steps_to_retrieve": 20.0},
            "memory_shuffled_location": {"retrieval_success_rate": 0.0, "wrong_drawer_rate": 1.0},
            "memory_wrong_binding": {"retrieval_success_rate": 0.0, "wrong_drawer_rate": 1.0},
            "memory_stale": {"retrieval_success_rate": 0.0, "wrong_drawer_rate": 1.0},
        }
        decision, _ = decide_gridworld_injection(metrics, contamination={"failure_count": 0}, protocol_cfg={})
        self.assertEqual(decision, GO_GRIDWORLD_INJECTION_SUPPORTED)

    def test_decision_rejects_clean_failure(self):
        metrics = {
            "oracle_goal_eval_only": {"retrieval_success_rate": 1.0},
            "memory_clean": {"retrieval_success_rate": 0.0, "goal_injection_accuracy": 0.0, "mean_steps_to_retrieve": 0.0},
            "no_memory_search": {"retrieval_success_rate": 1.0, "mean_steps_to_retrieve": 20.0},
            "memory_shuffled_location": {"retrieval_success_rate": 0.0, "wrong_drawer_rate": 1.0},
            "memory_wrong_binding": {"retrieval_success_rate": 0.0, "wrong_drawer_rate": 1.0},
            "memory_stale": {"retrieval_success_rate": 0.0, "wrong_drawer_rate": 1.0},
        }
        decision, _ = decide_gridworld_injection(metrics, contamination={"failure_count": 0}, protocol_cfg={})
        self.assertEqual(decision, NO_GO_MEMORY_INJECTION_FAILURE)


if __name__ == "__main__":
    unittest.main()
