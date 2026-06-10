import tempfile
import unittest

from embryo.eval.contamination import scan_actor_context
from embryo.memory.distractor_stream import cue_rates, make_distractor_stream_episode
from embryo.run.probe_distractor_stream_pressure import (
    GO_DISTRACTOR_STREAM_PRESSURE_CONFIRMED,
    deployable_fact_context,
    decide_distractor_stream_pressure,
    run_distractor_stream_pressure_probe,
    run_pressure_arm,
    write_distractor_stream_pressure_artifacts,
)


class DistractorStreamPressureTests(unittest.TestCase):
    def test_task_has_randomized_relevant_positions_and_deployable_actor_view(self):
        episode = make_distractor_stream_episode(
            seed=13000,
            relevant_count=4,
            budget=4,
            distractor_ratio=5,
            p_cue=0.5,
            key_space=128,
            value_space=8,
        )
        relevant_positions = [fact.stream_index for fact in episode.queries]
        self.assertNotEqual(relevant_positions, list(range(4)))
        context = deployable_fact_context(episode.facts[0])
        self.assertEqual(scan_actor_context(context)["failure_count"], 0)
        self.assertNotIn("relevant_eval_only", str(context))
        self.assertNotIn("seed", context)

    def test_cue_semantics_are_explicit(self):
        true_positive_rate, false_positive_rate = cue_rates(p_cue=0.8, cue_model="symmetric_label_accuracy")
        self.assertAlmostEqual(true_positive_rate, 0.8)
        self.assertAlmostEqual(false_positive_rate, 0.2)
        self.assertEqual(
            cue_rates(
                p_cue=0.0,
                cue_model="explicit_rates",
                cue_true_positive_rate=0.7,
                cue_false_positive_rate=0.1,
            ),
            (0.7, 0.1),
        )

    def test_oracle_beats_fifo_under_pressure(self):
        episode = make_distractor_stream_episode(
            seed=13001,
            relevant_count=8,
            budget=8,
            distractor_ratio=15,
            p_cue=0.5,
            key_space=4096,
            value_space=16,
        )
        oracle, _ = run_pressure_arm(episode, arm="oracle_selection")
        fifo, _ = run_pressure_arm(episode, arm="store_everything_fifo")
        self.assertEqual(oracle["recall_success_rate"], 1.0)
        self.assertLess(fifo["recall_success_rate"], 0.5)
        self.assertGreater(oracle["retention_precision"], fifo["retention_precision"])

    def test_random_admission_is_occupancy_matched_and_not_fifo_path(self):
        episode = make_distractor_stream_episode(
            seed=13001,
            relevant_count=8,
            budget=8,
            distractor_ratio=15,
            p_cue=0.5,
            key_space=4096,
            value_space=16,
        )
        fifo, _ = run_pressure_arm(episode, arm="store_everything_fifo")
        random, _ = run_pressure_arm(episode, arm="random_admission")
        self.assertEqual(random["admitted_count"], 8)
        self.assertEqual(random["occupancy"], fifo["occupancy"])
        self.assertNotEqual(random["memory_content_hash"], fifo["memory_content_hash"])

    def test_decision_accepts_pressure_confirmed_metrics(self):
        rows = []
        for seed in range(10):
            rows.append({"seed": seed, "arm": "oracle_selection", "distractor_ratio": 15, "budget": 8, "p_cue": 0.5, "recall_success_rate": 1.0})
            rows.append({"seed": seed, "arm": "store_everything_fifo", "distractor_ratio": 15, "budget": 8, "p_cue": 0.5, "recall_success_rate": 0.1})
            rows.append({"seed": seed, "arm": "random_admission", "distractor_ratio": 15, "budget": 8, "p_cue": 0.5, "recall_success_rate": 0.1})
        decision, reasons = decide_distractor_stream_pressure(
            rows,
            contamination={"failure_count": 0},
            protocol_cfg={"primary_distractor_ratio": 15, "primary_budget": 8, "primary_p_cue": 0.5},
        )
        self.assertEqual(decision, GO_DISTRACTOR_STREAM_PRESSURE_CONFIRMED)
        self.assertTrue(reasons)

    def test_probe_writes_required_artifacts(self):
        config = {
            "task": {
                "seed_start": 13000,
                "seed_count": 4,
                "tuning_seed_start": 13000,
                "tuning_seed_count": 2,
                "relevant_count": 4,
                "distractor_ratios": [1, 7],
                "budgets": [4],
                "p_cues": [0.5, 0.8],
                "cue_model": "symmetric_label_accuracy",
                "key_space": 512,
                "value_space": 8,
            },
            "protocol": {
                "primary_distractor_ratio": 7,
                "primary_budget": 4,
                "primary_p_cue": 0.5,
                "threshold_candidates": [0.0, 0.5, 1.1],
                "min_oracle_fifo_gap": 0.0,
            },
        }
        result = run_distractor_stream_pressure_probe(config)
        self.assertIn("metrics_by_arm", result["summary"])
        self.assertIn("cue_threshold_gate", result["summary"]["metrics_by_arm"])
        self.assertIn("thresholds_by_p_cue", result["summary"])
        self.assertIn("fifo_random_diagnostic", result["summary"])
        self.assertEqual(result["contamination"]["failure_count"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_distractor_stream_pressure_artifacts(result, tmp)
            self.assertIn("summary_json", artifacts)
            self.assertIn("episodes", artifacts)


if __name__ == "__main__":
    unittest.main()
