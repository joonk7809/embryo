import tempfile
import unittest
from pathlib import Path

from embryo.run.train_popgym_recurrent import (
    GO_POPGYM_RECURRENT_H_LSTM_MEASURED,
    GO_POPGYM_RECURRENT_H_LSTM_NOT_OBSERVED,
    collect_sequence,
    contamination_summary,
    decision_from_metrics,
    extract_h_lstm,
)


try:
    import popgym  # noqa: F401

    POPGYM_AVAILABLE = True
except ModuleNotFoundError:
    POPGYM_AVAILABLE = False


try:
    from embryo.models.popgym_recurrent import (
        RepeatFirstGRUPolicy,
        RepeatFirstRecurrentActor,
        require_torch,
        save_popgym_recurrent_checkpoint,
    )

    require_torch()
    TORCH_AVAILABLE = True
except RuntimeError:
    TORCH_AVAILABLE = False


class PopGymRecurrentTests(unittest.TestCase):
    def test_h_lstm_extraction_measured(self):
        rows = [{"gap": gap, "success_rate": 1.0 if gap < 5 else 0.25} for gap in range(1, 12)]
        result = extract_h_lstm(rows, chance_success_rate=0.25, tolerance=0.05, tail_bins=4)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["h_lstm"], 5)

    def test_h_lstm_extraction_not_observed(self):
        rows = [{"gap": gap, "success_rate": 0.95} for gap in range(1, 12)]
        result = extract_h_lstm(rows, chance_success_rate=0.25, tolerance=0.05, tail_bins=4)
        self.assertEqual(result["status"], "not_observed")
        self.assertIsNone(result["h_lstm"])

    def test_decision_allows_not_observed_when_competent(self):
        decision, reasons = decision_from_metrics(
            metrics_by_split={
                "dev": {"short_success_rate": 1.0},
                "test": {"short_success_rate": 1.0},
            },
            contamination={"failure_count": 0},
            h_lstm={"status": "not_observed"},
            min_short_success=0.90,
            competence_max_gap=32,
        )
        self.assertEqual(decision, GO_POPGYM_RECURRENT_H_LSTM_NOT_OBSERVED)
        self.assertEqual(reasons, ["falloff_not_observed_within_sweep"])

    def test_decision_reports_measured_when_competent(self):
        decision, reasons = decision_from_metrics(
            metrics_by_split={
                "dev": {"short_success_rate": 1.0},
                "test": {"short_success_rate": 1.0},
            },
            contamination={"failure_count": 0},
            h_lstm={"status": "measured", "h_lstm": 17},
            min_short_success=0.90,
            competence_max_gap=32,
        )
        self.assertEqual(decision, GO_POPGYM_RECURRENT_H_LSTM_MEASURED)
        self.assertEqual(reasons, [])

    @unittest.skipUnless(POPGYM_AVAILABLE, "popgym unavailable")
    def test_sequence_context_has_no_forbidden_fields(self):
        _, _, contexts = collect_sequence(task="repeat_first_easy", seed=3, max_steps=8)
        scan = contamination_summary(contexts)
        self.assertEqual(scan["failure_count"], 0)
        self.assertIn("popgym_observation", contexts[0])
        self.assertNotIn("reward", contexts[0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
    def test_checkpoint_actor_loads_and_emits_valid_action(self):
        torch, _ = require_torch()
        model = RepeatFirstGRUPolicy.build(action_count=4, embedding_dim=4, hidden_size=8)
        with tempfile.TemporaryDirectory() as tmp:
            save_popgym_recurrent_checkpoint(
                Path(tmp),
                model=model,
                manifest={
                    "action_count": 4,
                    "embedding_dim": 4,
                    "hidden_size": 8,
                    "weights": "model.pt",
                },
            )
            actor = RepeatFirstRecurrentActor(checkpoint_path=tmp, action_names=("suit_0", "suit_1", "suit_2", "suit_3"))
            actor.reset(seed=1)
            decision = actor.act({"popgym_observation": torch.tensor(2).item()})
            self.assertIn(decision.action, {"suit_0", "suit_1", "suit_2", "suit_3"})
            self.assertEqual(set(decision.logits), {"suit_0", "suit_1", "suit_2", "suit_3"})


if __name__ == "__main__":
    unittest.main()
