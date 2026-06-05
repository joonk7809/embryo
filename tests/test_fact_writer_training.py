from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from embryo.datasets.fact_writer import ALLOWED_SOURCE_FIELDS, LABEL_SOURCE, FactWriterDataset, LabelProfile, example_from_observation
from embryo.models import load_fact_writer_checkpoint, save_fact_writer_checkpoint
from embryo.run.train_fact_writer import (
    GO_FACT_WRITER_OFFLINE_SUPPORTED,
    train_fact_writer_from_dataset,
    write_fact_writer_artifacts,
)


class FactWriterTrainingTests(unittest.TestCase):
    def test_dataset_rows_keep_allowed_source_inputs_only(self) -> None:
        row = example_from_observation(observation(True), split="train", seed=1, tick=0)

        self.assertEqual(tuple(row["source_inputs"]), ALLOWED_SOURCE_FIELDS)
        self.assertEqual(row["label_source"], LABEL_SOURCE)
        self.assertIn("label_visual_anchor_visible", row["labels"])
        self.assertIn("center_vs_global_contrast", row["scaffold_diagnostics"])
        self.assertIn("center_patch_hash", row["scaffold_diagnostics"])
        self.assertNotIn("reward", row["source_inputs"])
        self.assertNotIn("inventory", row["source_inputs"])

    def test_forbidden_fields_are_rejected_by_contamination_scan(self) -> None:
        payload = observation(True)
        payload["info"] = {"inventory": {"wood": 1}}

        row = example_from_observation(payload, split="train", seed=1, tick=0)

        self.assertGreater(row["contamination"]["failure_count"], 0)
        self.assertIn("inventory", row["contamination"]["forbidden_present"])

    def test_label_profile_can_create_negatives_from_permissive_scaffold(self) -> None:
        row = example_from_observation(
            observation(True),
            split="train",
            seed=1,
            tick=0,
            label_profile=LabelProfile(center_vs_global_contrast_threshold=0.5),
        )

        self.assertFalse(row["labels"]["label_visual_anchor_visible"])
        self.assertTrue(row["scaffold_diagnostics"]["visual_anchor_visible"])

    def test_tiny_separable_dataset_trains_supported_model(self) -> None:
        result = train_fact_writer_from_dataset(tiny_dataset(), tiny_config())

        self.assertEqual(result["summary"]["decision"], GO_FACT_WRITER_OFFLINE_SUPPORTED)
        self.assertEqual(result["metrics_by_split"]["test"]["f1"], 1.0)
        self.assertEqual(result["contamination"]["failure_count"], 0)

    def test_checkpoint_save_load_preserves_predictions(self) -> None:
        result = train_fact_writer_from_dataset(tiny_dataset(), tiny_config())
        model = result["model"]
        features = tiny_dataset().rows_by_split["test"][0]["features"]

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_fact_writer_checkpoint(model, path)
            loaded = load_fact_writer_checkpoint(path)

        self.assertEqual(model.predict_scores(features), loaded.predict_scores(features))

    def test_summary_emits_required_artifacts_and_decision_fields(self) -> None:
        result = train_fact_writer_from_dataset(tiny_dataset(), tiny_config())

        with TemporaryDirectory() as tmpdir:
            paths = write_fact_writer_artifacts(result, tmpdir)
            root = Path(tmpdir)
            summary = json.loads((root / "fact_writer_summary.json").read_text(encoding="utf-8"))

            expected = {
                "fact_writer_summary_json",
                "fact_writer_summary_md",
                "dataset_manifest",
                "metrics_by_split",
                "checkpoint_manifest",
                "contamination_scan",
                "scaffold_diagnostics",
                "checkpoint",
            }
            self.assertEqual(set(paths), expected)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            self.assertIn("decision", summary)
            self.assertIn("metrics_by_split", summary)
            self.assertTrue((root / "checkpoint").exists())


def tiny_dataset() -> FactWriterDataset:
    rows_by_split = {
        "train": make_rows(12, "train"),
        "dev": make_rows(6, "dev"),
        "test": make_rows(6, "test"),
    }
    failures = [failure for rows in rows_by_split.values() for row in rows for failure in row["contamination"]["failures"]]
    feature_schema = list(rows_by_split["train"][0]["features"])
    manifest = {
        "builder": "test_fact_writer_dataset",
        "runtime": {"name": "fixture_rgb_arrays"},
        "splits": {split: {"row_count": len(rows)} for split, rows in rows_by_split.items()},
        "source_input_fields": list(ALLOWED_SOURCE_FIELDS),
        "feature_schema": feature_schema,
        "label_schema": [
            "label_visual_anchor_visible",
            "label_center_salience_score",
            "label_visual_change_event",
        ],
        "label_source": LABEL_SOURCE,
        "contamination": {"passed": not failures, "failure_count": len(failures), "failures": failures},
    }
    diagnostics = {split: [dict(row["scaffold_diagnostics"]) for row in rows] for split, rows in rows_by_split.items()}
    return FactWriterDataset(
        rows_by_split=rows_by_split,
        diagnostics_by_split=diagnostics,
        manifest=manifest,
        contamination=manifest["contamination"],
    )


def make_rows(count: int, split: str) -> list[dict]:
    rows = []
    for index in range(count):
        visible = index % 2 == 0
        rows.append(example_from_observation(observation(visible), split=split, seed=100 + index, tick=index))
    return rows


def observation(visible: bool) -> dict:
    image = np.zeros((24, 24, 3), dtype=np.float32) + 0.25
    if visible:
        image[9:15, 9:15, 0] = 1.0
        image[9:15, 9:15, 1] = 0.0
        image[9:15, 9:15, 2] = 0.0
    previous = np.zeros_like(image) + 0.25
    return {
        "raw_rgb_frame": image,
        "previous_rgb_frame": previous,
        "previous_action": "noop",
        "visual_anchor_visible": visible,
        "center_salience_score": 0.9 if visible else 0.0,
        "visual_change_event": False,
    }


def tiny_config() -> dict:
    return {
        "training": {"epochs": 300, "learning_rate": 0.3, "l2": 0.0001, "ridge": 0.0001},
        "decision": {
            "min_test_f1": 0.90,
            "max_train_test_f1_gap": 0.15,
            "max_dev_test_f1_gap": 0.15,
            "min_predicted_positive_rate": 0.02,
            "max_predicted_positive_rate": 0.80,
            "min_label_positive_rate": 0.05,
            "max_label_positive_rate": 0.80,
            "min_test_positive_count": 2,
        },
    }


if __name__ == "__main__":
    unittest.main()
