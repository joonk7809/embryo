import unittest

from embryo.run.train_popgym_autoencode import (
    GO_POPGYM_AUTOENCODE_H_LSTM_MEASURED,
    autoencode_decision,
    baseline_reconciliation,
    collect_autoencode_sequence,
    contamination_summary,
)
from embryo.memory.popgym_autoencode import OrderedSequenceMemory
from embryo.models.popgym_associative_memory import ExplicitPositionDeltaMemoryPolicy, content_corrupt_suits
from embryo.models.popgym_recurrent import require_torch


try:
    import popgym  # noqa: F401

    POPGYM_AVAILABLE = True
except ModuleNotFoundError:
    POPGYM_AVAILABLE = False


class PopGymAutoencodeTests(unittest.TestCase):
    @unittest.skipUnless(POPGYM_AVAILABLE, "popgym unavailable")
    def test_autoencode_sequence_has_play_labels_and_gap(self):
        row = collect_autoencode_sequence(task="autoencode_easy", seed=5)
        self.assertGreater(sum(row["masks"]), 0)
        self.assertEqual(max(row["gaps"]), sum(row["masks"]))
        self.assertEqual(len(row["tokens"]), len(row["labels"]))

    @unittest.skipUnless(POPGYM_AVAILABLE, "popgym unavailable")
    def test_autoencode_context_has_no_forbidden_fields(self):
        row = collect_autoencode_sequence(task="autoencode_easy", seed=6)
        scan = contamination_summary(row["contexts"])
        self.assertEqual(scan["failure_count"], 0)
        self.assertIn("popgym_observation", row["contexts"][0])
        self.assertNotIn("reward", row["contexts"][0])

    def test_autoencode_decision_reports_measured_task(self):
        decision, reasons = autoencode_decision(
            metrics_by_task={"autoencode_easy": {"play_success_rate": 0.50, "short_success_rate": 0.95}},
            h_lstm_by_task={"autoencode_easy": {"status": "measured", "h_lstm": 42}},
            contamination={"failure_count": 0},
            min_easy_success=0.90,
        )
        self.assertEqual(decision, GO_POPGYM_AUTOENCODE_H_LSTM_MEASURED)
        self.assertEqual(reasons, ["measured_tasks=autoencode_easy"])

    def test_ordered_sequence_memory_separates_content_and_order(self):
        memory = OrderedSequenceMemory.from_observed_suits([0, 1, 2, 3])
        self.assertEqual(memory.recall_reverse(1), 3)
        self.assertEqual(memory.recall_reverse(4), 0)
        self.assertEqual(memory.recall_content_corrupt(1), 0)
        self.assertEqual(memory.recall_order_corrupt(1), 0)
        self.assertEqual(memory.recall_order_corrupt(4), 3)

    def test_baseline_reconciliation_reports_match(self):
        memory_by_gap = {
            "1": {"autoencode_memory_off": {"success_rate": 0.95, "available_count": 8}},
            "2": {"autoencode_memory_off": {"success_rate": 0.50, "available_count": 8}},
        }
        baseline_rows = [{"gap": 1, "success_rate": 0.95, "count": 8}, {"gap": 2, "success_rate": 0.50, "count": 8}]
        reconciliation = baseline_reconciliation(memory_by_gap, baseline_rows, gaps=(1, 2), comparable_seed_block=True)
        self.assertEqual(reconciliation["status"], "matched")
        self.assertEqual(reconciliation["rows"][0]["absolute_delta"], 0.0)

    def test_explicit_position_delta_memory_fits_tiny_recall(self):
        try:
            torch, _ = require_torch()
        except RuntimeError:
            self.skipTest("torch unavailable")
        torch.manual_seed(7)
        model = ExplicitPositionDeltaMemoryPolicy.build(max_positions=4, suit_count=4, value_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        watch_suits = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
        read_positions = torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]], dtype=torch.long)
        targets = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=torch.long)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            logits = model(watch_suits, read_positions)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
            loss.backward()
            optimizer.step()
        clean = model(watch_suits, read_positions).argmax(dim=-1)
        corrupt = model(watch_suits, read_positions, write_suits=content_corrupt_suits(watch_suits, suit_count=4)).argmax(dim=-1)
        self.assertTrue(bool((clean == targets).all()))
        self.assertEqual(int((corrupt == targets).sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
