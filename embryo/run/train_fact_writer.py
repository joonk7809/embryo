"""Offline supervised training for fact-writer v0."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embryo.core.config import load_config
from embryo.eval.traces import write_jsonl
from embryo.datasets.fact_writer import (
    FactWriterDataset,
    build_fact_writer_dataset,
    change_labels,
    feature_matrix,
    salience_labels,
    visible_labels,
)
from embryo.models import CheckpointManifest, save_fact_writer_checkpoint, save_manifest, train_learned_fact_writer_v0


GO_FACT_WRITER_OFFLINE_SUPPORTED = "GO_fact_writer_offline_supported"
NO_GO_FACT_WRITER_OVERFIT = "NO_GO_fact_writer_overfit"
NO_GO_FACT_WRITER_COLLAPSE = "NO_GO_fact_writer_collapse"
NO_GO_FACT_WRITER_LOW_SIGNAL = "NO_GO_fact_writer_low_signal"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"


def run_fact_writer_training(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = build_fact_writer_dataset(config)
    return train_fact_writer_from_dataset(dataset, config)


def train_fact_writer_from_dataset(dataset: FactWriterDataset, config: Mapping[str, Any]) -> dict[str, Any]:
    train_cfg = mapping(config.get("training"))
    decision_cfg = mapping(config.get("decision"))
    feature_schema = tuple(str(item) for item in dataset.manifest["feature_schema"])
    train_rows = dataset.rows_by_split.get("train", [])
    x_train = feature_matrix(train_rows, feature_schema)
    model = train_learned_fact_writer_v0(
        x_train,
        visible=visible_labels(train_rows),
        salience=salience_labels(train_rows),
        change=change_labels(train_rows),
        feature_schema=feature_schema,
        epochs=int(train_cfg.get("epochs", 500)),
        learning_rate=float(train_cfg.get("learning_rate", 0.2)),
        l2=float(train_cfg.get("l2", 1e-4)),
        ridge=float(train_cfg.get("ridge", 1e-4)),
    )
    metrics_by_split = {
        split: evaluate_split(model, rows, feature_schema)
        for split, rows in sorted(dataset.rows_by_split.items())
    }
    overfit = overfit_checks(metrics_by_split)
    decision, reasons = fact_writer_decision(
        metrics_by_split,
        contamination=dataset.contamination,
        thresholds=decision_cfg,
        overfit=overfit,
    )
    checkpoint_manifest = CheckpointManifest(
        model_type="learned_fact_writer_v0",
        model_name="compact_rgb_fact_writer_v0",
        version="0",
        feature_schema=feature_schema,
        artifact_paths={"weights": "checkpoint/fact_writer.json"},
        metadata={
            "label_source": dataset.manifest["label_source"],
            "training_boundary": "offline_supervised_fact_writer_v0",
            "decision": decision,
            "decision_reasons": reasons,
        },
    )
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "offline_fact_writer_v0",
        "label_source": dataset.manifest["label_source"],
        "model_type": checkpoint_manifest.model_type,
        "metrics_by_split": metrics_by_split,
        "overfit_checks": overfit,
        "contamination": dataset.contamination,
        "dataset": {
            "runtime": dataset.manifest["runtime"],
            "splits": dataset.manifest["splits"],
            "feature_count": len(feature_schema),
            "label_profile": dataset.manifest.get("label_profile", {}),
        },
    }
    return {
        "summary": summary,
        "dataset_manifest": dataset.manifest,
        "metrics_by_split": metrics_by_split,
        "checkpoint_manifest": checkpoint_manifest,
        "contamination": dataset.contamination,
        "scaffold_diagnostics": [
            diagnostic
            for split in sorted(dataset.diagnostics_by_split)
            for diagnostic in dataset.diagnostics_by_split[split]
        ],
        "model": model,
    }


def evaluate_split(model: Any, rows: Sequence[Mapping[str, Any]], feature_schema: Sequence[str]) -> dict[str, Any]:
    x = feature_matrix(rows, feature_schema)
    labels = visible_labels(rows)
    salience = salience_labels(rows)
    change = change_labels(rows)
    if len(rows) == 0:
        return empty_metrics()
    predictions = model.predict_matrix(x)
    visible_probability = predictions["visible_probability"]
    predicted = visible_probability >= 0.5
    visible_metrics = binary_metrics(predicted, labels, probabilities=visible_probability)
    salience_mse = float(np.mean((predictions["salience"] - salience) ** 2))
    change_metrics = binary_metrics(predictions["change_probability"] >= 0.5, change, probabilities=predictions["change_probability"])
    visible_metrics.update(
        {
            "brier_or_mse": round(float(np.mean((visible_probability - labels) ** 2)), 6),
            "salience_mse": round(salience_mse, 6),
            "visual_change_f1": change_metrics["f1"],
            "visual_change_accuracy": change_metrics["accuracy"],
        }
    )
    return visible_metrics


def binary_metrics(predicted: np.ndarray, labels: np.ndarray, *, probabilities: np.ndarray) -> dict[str, Any]:
    positives = labels == 1.0
    negatives = labels == 0.0
    tp = int(np.logical_and(predicted, positives).sum())
    tn = int(np.logical_and(~predicted, negatives).sum())
    fp = int(np.logical_and(predicted, negatives).sum())
    fn = int(np.logical_and(~predicted, positives).sum())
    total = int(len(labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "row_count": total,
        "positive_count": int(positives.sum()),
        "negative_count": int(negatives.sum()),
        "positive_rate": round(float(positives.mean()), 6) if total else 0.0,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round((tp + tn) / total, 6) if total else 0.0,
        "predicted_positive_rate": round(float(predicted.mean()), 6) if total else 0.0,
        "false_positive_rate": round(fp / int(negatives.sum()), 6) if int(negatives.sum()) else 0.0,
        "false_negative_rate": round(fn / int(positives.sum()), 6) if int(positives.sum()) else 0.0,
        "mean_probability": round(float(probabilities.mean()), 6) if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "row_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "positive_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "brier_or_mse": 0.0,
        "predicted_positive_rate": 0.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "salience_mse": 0.0,
        "visual_change_f1": 0.0,
        "visual_change_accuracy": 0.0,
    }


def overfit_checks(metrics_by_split: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    train_f1 = float(mapping(metrics_by_split.get("train")).get("f1", 0.0))
    dev_f1 = float(mapping(metrics_by_split.get("dev")).get("f1", 0.0))
    test_f1 = float(mapping(metrics_by_split.get("test")).get("f1", 0.0))
    return {
        "train_f1_minus_dev_f1": round(train_f1 - dev_f1, 6),
        "train_f1_minus_test_f1": round(train_f1 - test_f1, 6),
        "dev_f1_minus_test_f1": round(dev_f1 - test_f1, 6),
    }


def fact_writer_decision(
    metrics_by_split: Mapping[str, Mapping[str, Any]],
    *,
    contamination: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    overfit: Mapping[str, float],
) -> tuple[str, list[str]]:
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count_nonzero"]
    min_test_f1 = float(thresholds.get("min_test_f1", 0.70))
    min_test_positive = int(thresholds.get("min_test_positive_count", 50))
    max_train_test_gap = float(thresholds.get("max_train_test_f1_gap", 0.15))
    max_dev_test_gap = float(thresholds.get("max_dev_test_f1_gap", 0.15))
    min_predicted_rate = float(thresholds.get("min_predicted_positive_rate", 0.02))
    max_predicted_rate = float(thresholds.get("max_predicted_positive_rate", 0.80))
    min_label_rate = float(thresholds.get("min_label_positive_rate", 0.05))
    max_label_rate = float(thresholds.get("max_label_positive_rate", 0.80))
    test = mapping(metrics_by_split.get("test"))
    reasons: list[str] = []
    label_balance_reasons = [
        f"{split}_label_positive_rate={metrics.get('positive_rate')}"
        for split, metrics in metrics_by_split.items()
        if not (min_label_rate <= float(metrics.get("positive_rate", 0.0)) <= max_label_rate)
    ]
    label_balance_reasons.extend(
        f"{split}_negative_count=0"
        for split, metrics in metrics_by_split.items()
        if int(metrics.get("negative_count", 0)) == 0
    )
    label_balance_reasons.extend(
        f"{split}_positive_count=0"
        for split, metrics in metrics_by_split.items()
        if int(metrics.get("positive_count", 0)) == 0
    )
    if label_balance_reasons:
        return NO_GO_FACT_WRITER_LOW_SIGNAL, label_balance_reasons
    collapse_reasons = [
        f"{split}_predicted_positive_rate={metrics.get('predicted_positive_rate')}"
        for split, metrics in metrics_by_split.items()
        if not (min_predicted_rate <= float(metrics.get("predicted_positive_rate", 0.0)) <= max_predicted_rate)
    ]
    if collapse_reasons:
        return NO_GO_FACT_WRITER_COLLAPSE, collapse_reasons
    if int(test.get("positive_count", 0)) < min_test_positive:
        reasons.append(f"test_positive_count<{min_test_positive}")
    if float(test.get("f1", 0.0)) < min_test_f1:
        reasons.append(f"test_f1<{min_test_f1}")
    if reasons:
        return NO_GO_FACT_WRITER_LOW_SIGNAL, reasons
    if abs(float(overfit.get("train_f1_minus_test_f1", 0.0))) > max_train_test_gap:
        reasons.append(f"abs_train_test_f1_gap>{max_train_test_gap}")
    if abs(float(overfit.get("dev_f1_minus_test_f1", 0.0))) > max_dev_test_gap:
        reasons.append(f"abs_dev_test_f1_gap>{max_dev_test_gap}")
    if reasons:
        return NO_GO_FACT_WRITER_OVERFIT, reasons
    return GO_FACT_WRITER_OFFLINE_SUPPORTED, []


def write_fact_writer_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    checkpoint_dir = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fact_writer_summary_json": root / "fact_writer_summary.json",
        "fact_writer_summary_md": root / "fact_writer_summary.md",
        "dataset_manifest": root / "dataset_manifest.json",
        "metrics_by_split": root / "metrics_by_split.json",
        "checkpoint_manifest": root / "checkpoint_manifest.json",
        "contamination_scan": root / "contamination_scan.json",
        "scaffold_diagnostics": root / "scaffold_diagnostics.jsonl",
        "checkpoint": checkpoint_dir / "fact_writer.json",
    }
    paths["fact_writer_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["fact_writer_summary_md"].write_text(format_fact_writer_summary_markdown(result["summary"]), encoding="utf-8")
    paths["dataset_manifest"].write_text(json.dumps(result["dataset_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics_by_split"].write_text(json.dumps(result["metrics_by_split"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_manifest(result["checkpoint_manifest"], paths["checkpoint_manifest"])
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(paths["scaffold_diagnostics"], result.get("scaffold_diagnostics", ()))
    save_fact_writer_checkpoint(result["model"], paths["checkpoint"])
    return {key: str(path) for key, path in paths.items()}


def format_fact_writer_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Fact Writer V0 Summary",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Label source: `{summary.get('label_source')}`",
        f"- Model type: `{summary.get('model_type')}`",
        f"- Contamination failures: `{mapping(summary.get('contamination')).get('failure_count')}`",
        "",
        "## Metrics",
        "",
    ]
    for split, metrics in mapping(summary.get("metrics_by_split")).items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Rows: `{metrics.get('row_count')}`",
                f"- Positive count: `{metrics.get('positive_count')}`",
                f"- F1: `{metrics.get('f1')}`",
                f"- Accuracy: `{metrics.get('accuracy')}`",
                f"- Predicted positive rate: `{metrics.get('predicted_positive_rate')}`",
                f"- Brier/MSE: `{metrics.get('brier_or_mse')}`",
                "",
            ]
        )
    return "\n".join(lines)


def config_with_output(config: Mapping[str, Any], root: Path, out: str) -> tuple[dict[str, Any], Path]:
    selected = dict(config)
    output = out or str(mapping(config.get("output")).get("path", "runs/fact_writer_sanity"))
    output_path = resolve_path(root, output)
    return selected, output_path


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train the offline fact-writer v0 model.")
    parser.add_argument("--config", default=str(root / "configs" / "fact_writer_sanity.yaml"))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config, out = config_with_output(config, root, args.out)
    result = run_fact_writer_training(config)
    paths = write_fact_writer_artifacts(result, out)
    print(json.dumps({"decision": result["summary"]["decision"], "out": str(out), "artifacts": paths}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
