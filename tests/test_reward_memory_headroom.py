import unittest

from embryo.run.probe_reward_memory_headroom import (
    GO_REWARD_MEMORY_HEADROOM_FOUND,
    NO_GO_BASELINE_NOT_COMPETENT,
    ObservationMemoryTeacher,
    decide_headroom,
)


class RewardMemoryHeadroomTests(unittest.TestCase):
    def test_observation_memory_teacher_reuses_seen_pair(self):
        teacher = ObservationMemoryTeacher(facedown_value=2)
        self.assertEqual(teacher.act([2, 2, 2, 2]), 0)
        self.assertEqual(teacher.act([1, 2, 2, 2]), 1)
        self.assertEqual(teacher.act([1, 0, 2, 2]), 2)
        self.assertEqual(teacher.act([2, 0, 1, 2]), 0)
        self.assertEqual(teacher.act([1, 2, 1, 2]), 3)
        self.assertEqual(teacher.act([2, 2, 2, 0]), 1)

    def test_decision_reports_headroom_when_short_competent_and_long_drops(self):
        metrics = [
            {
                "recurrent_baseline": {"completion_rate": 0.90},
                "random_valid_action": {"completion_rate": 0.10},
            },
            {
                "recurrent_baseline": {"completion_rate": 0.40},
                "random_valid_action": {"completion_rate": 0.05},
            },
        ]
        decision, reasons = decide_headroom(metrics, contamination={"failure_count": 0}, evaluation_cfg={}, audit={})
        self.assertEqual(decision, GO_REWARD_MEMORY_HEADROOM_FOUND)
        self.assertEqual(reasons, ["recurrent_baseline_competent_short_and_degrades_with_demand"])

    def test_decision_rejects_incompetent_short_baseline(self):
        metrics = [
            {
                "recurrent_baseline": {"completion_rate": 0.20},
                "random_valid_action": {"completion_rate": 0.10},
            },
            {
                "recurrent_baseline": {"completion_rate": 0.05},
                "random_valid_action": {"completion_rate": 0.05},
            },
        ]
        decision, _ = decide_headroom(metrics, contamination={"failure_count": 0}, evaluation_cfg={}, audit={})
        self.assertEqual(decision, NO_GO_BASELINE_NOT_COMPETENT)


if __name__ == "__main__":
    unittest.main()
