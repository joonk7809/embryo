import tempfile
import unittest

from embryo.eval.contamination import scan_actor_context
from embryo.memory.popgym_count_recall import (
    CountRecallMemory,
    count_recall_shuffled_query,
    count_recall_wrong_binding_query,
)
from embryo.run.probe_popgym_count_recall_injection import (
    GO_POPGYM_COUNT_RECALL_INJECTION_SUPPORTED,
    NO_GO_CLEAN_MEMORY_FAILURE,
    deployable_count_recall_context,
    decide_count_recall_action,
    decide_popgym_count_recall_injection,
    run_popgym_count_recall_injection_probe,
    write_popgym_count_recall_injection_artifacts,
)

try:
    import popgym  # noqa: F401

    POPGYM_AVAILABLE = True
except Exception:
    POPGYM_AVAILABLE = False


class PopGymCountRecallInjectionTests(unittest.TestCase):
    def test_count_memory_updates_from_deployable_observation(self):
        memory = CountRecallMemory(symbol_count=4)
        memory.update({"popgym_observation": [2, 1]})
        memory.update({"popgym_observation": [2, 3]})
        memory.update({"popgym_observation": [1, 2]})
        self.assertEqual(memory.answer(2), 2)
        self.assertEqual(memory.answer(1), 1)
        self.assertEqual(memory.stale_counts(lag=2), [0, 0, 1, 0])

    def test_corrupt_queries_change_lookup_binding(self):
        memory = CountRecallMemory(symbol_count=4)
        for obs in ([0, 0], [0, 0], [1, 0], [2, 0]):
            memory.update({"popgym_observation": obs})
        clean = decide_count_recall_action(
            arm="count_recall_memory_clean",
            memory=memory,
            query=0,
            target_count_eval_only=2,
            seed=7,
            tick=3,
            stale_lag=2,
        )
        wrong = decide_count_recall_action(
            arm="count_recall_memory_wrong_binding",
            memory=memory,
            query=0,
            target_count_eval_only=2,
            seed=7,
            tick=3,
            stale_lag=2,
        )
        self.assertEqual(clean["action"], 2)
        self.assertEqual(wrong["effective_query_symbol"], count_recall_wrong_binding_query(0, 4))
        self.assertNotEqual(clean["query_content_hash"], wrong["query_content_hash"])

    def test_shuffled_query_avoids_identity_when_possible(self):
        for query in range(4):
            self.assertNotEqual(count_recall_shuffled_query(query, 4, seed=11), query)

    def test_actor_context_excludes_forbidden_fields(self):
        context = deployable_count_recall_context([1, 2], previous_obs=[0, 1], previous_action=3)
        scan = scan_actor_context(context)
        self.assertEqual(scan["failure_count"], 0)
        self.assertNotIn("reward", context)
        self.assertNotIn("done", context)
        self.assertNotIn("seed", context)

    def test_decision_accepts_clean_count_memory(self):
        metrics = {
            "count_recall_oracle_eval_only": {"query_accuracy": 1.0},
            "count_recall_memory_clean": {"query_accuracy": 1.0},
            "count_recall_no_memory": {"query_accuracy": 0.05},
            "count_recall_memory_shuffled": {"query_accuracy": 0.08},
            "count_recall_memory_wrong_binding": {"query_accuracy": 0.08},
            "count_recall_memory_stale": {"query_accuracy": 0.10},
        }
        decision, _ = decide_popgym_count_recall_injection(metrics, contamination={"failure_count": 0}, protocol_cfg={})
        self.assertEqual(decision, GO_POPGYM_COUNT_RECALL_INJECTION_SUPPORTED)

    def test_decision_rejects_clean_failure(self):
        metrics = {
            "count_recall_oracle_eval_only": {"query_accuracy": 1.0},
            "count_recall_memory_clean": {"query_accuracy": 0.50},
            "count_recall_no_memory": {"query_accuracy": 0.05},
            "count_recall_memory_shuffled": {"query_accuracy": 0.08},
            "count_recall_memory_wrong_binding": {"query_accuracy": 0.08},
            "count_recall_memory_stale": {"query_accuracy": 0.10},
        }
        decision, _ = decide_popgym_count_recall_injection(metrics, contamination={"failure_count": 0}, protocol_cfg={})
        self.assertEqual(decision, NO_GO_CLEAN_MEMORY_FAILURE)

    @unittest.skipUnless(POPGYM_AVAILABLE, "popgym unavailable")
    def test_probe_writes_required_artifacts(self):
        config = {
            "runtime": {"task": "count_recall_hard", "seed_start": 12000, "seed_count": 2},
            "protocol": {"stale_lag": 8, "max_corrupt_accuracy": 0.50},
        }
        result = run_popgym_count_recall_injection_probe(config)
        self.assertIn("decision", result["summary"])
        self.assertGreater(result["metrics_by_arm"]["count_recall_memory_clean"]["query_accuracy"], 0.99)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_popgym_count_recall_injection_artifacts(result, tmp)
            self.assertIn("popgym_count_recall_summary_json", artifacts)
            self.assertIn("ticks", artifacts)


if __name__ == "__main__":
    unittest.main()
