"""Small supervised training scaffolds.

These trainers are intentionally lightweight. They establish the artifact and
manifest contracts for future learned modules without importing historical
experiment code or requiring a deep learning framework.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.core.config import load_config
from embryo.eval.traces import read_jsonl
from embryo.models import CheckpointManifest, RouteBCPolicy, RuleRouterModel, ThresholdFactWriter, save_manifest, validate_manifest


DEFAULT_MODULE_INPUTS = {
    "fact_writer": "data/fixtures/tiny_fact_writer_dataset.jsonl",
    "router": "data/fixtures/tiny_router_dataset.jsonl",
    "bc_policy": "data/fixtures/tiny_bc_dataset.jsonl",
}


def train_fact_writer(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    threshold, metrics = fit_threshold(rows)
    manifest = CheckpointManifest(
        model_type="threshold_fact_writer",
        model_name="threshold_visual_fact_writer",
        version="1",
        feature_schema=("candidate_score",),
        metadata={
            "parameters": {"feature_name": "candidate_score", "threshold": threshold},
            "metrics": metrics,
            "training_boundary": "supervised_reference_scaffold",
        },
    )
    return {"manifest": manifest, "metrics": metrics}


def train_router(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    router = RuleRouterModel()
    predictions = [router.predict(row).route_mode for row in rows]
    labels = [str(row.get("target_route", "hold")) for row in rows]
    metrics = classification_metrics(predictions, labels)
    manifest = CheckpointManifest(
        model_type="rule_router",
        model_name="freshness_rule_router",
        version="1",
        feature_schema=("facing_candidate", "failed_action_event", "cache_age", "invalidated", "repeat_count"),
        metadata={
            "parameters": {"repeat_cap": 2},
            "metrics": metrics,
            "training_boundary": "contract_rule_validation",
        },
    )
    return {"manifest": manifest, "metrics": metrics}


def train_bc_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_by_route: dict[str, str] = {}
    for route, pairs in group_actions_by_route(rows).items():
        action_by_route[route] = Counter(pairs).most_common(1)[0][0]
    policy = RouteBCPolicy(action_by_route=action_by_route)
    predictions = [policy.act({"route_mode": row.get("route_mode", "hold")}).action for row in rows]
    labels = [str(row.get("target_action", "noop")) for row in rows]
    metrics = classification_metrics(predictions, labels)
    manifest = CheckpointManifest(
        model_type="route_bc_policy",
        model_name="route_bc_policy",
        version="1",
        feature_schema=("route_mode",),
        metadata={
            "parameters": {"action_by_route": action_by_route},
            "metrics": metrics,
            "training_boundary": "supervised_reference_scaffold",
        },
    )
    return {"manifest": manifest, "metrics": metrics}


def fit_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    scores = sorted({float(row.get("candidate_score", 0.0)) for row in rows})
    candidates = sorted(set([0.0, 0.5, 1.0, *scores, *[(a + b) / 2.0 for a, b in zip(scores, scores[1:])]]))
    best_threshold = 0.5
    best_accuracy = -1.0
    best_predictions: list[bool] = []
    labels = [bool(row.get("target_facing_candidate", False)) for row in rows]
    for threshold in candidates:
        predictions = [float(row.get("candidate_score", 0.0)) >= threshold for row in rows]
        accuracy = sum(pred == label for pred, label in zip(predictions, labels)) / len(labels) if labels else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
            best_predictions = predictions
    metrics = binary_metrics(best_predictions, labels)
    metrics["threshold"] = round(best_threshold, 6)
    return best_threshold, metrics


def binary_metrics(predictions: Sequence[bool], labels: Sequence[bool]) -> dict[str, Any]:
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
    fp = sum(pred and (not label) for pred, label in zip(predictions, labels))
    fn = sum((not pred) and label for pred, label in zip(predictions, labels))
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": total,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def classification_metrics(predictions: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    total = len(labels)
    correct = sum(pred == label for pred, label in zip(predictions, labels))
    classes = sorted(set(labels) | set(predictions))
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in classes:
        tp = sum(pred == label and actual == label for pred, actual in zip(predictions, labels))
        fp = sum(pred == label and actual != label for pred, actual in zip(predictions, labels))
        fn = sum(pred != label and actual == label for pred, actual in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    return {
        "sample_count": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
    }


def group_actions_by_route(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        route = str(row.get("route_mode", "hold"))
        grouped.setdefault(route, []).append(str(row.get("target_action", "noop")))
    return grouped


def train_module(module: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if module == "fact_writer":
        return train_fact_writer(rows)
    if module == "router":
        return train_router(rows)
    if module == "bc_policy":
        return train_bc_policy(rows)
    raise ValueError(f"Unsupported supervised module: {module}")


def write_training_artifacts(result: Mapping[str, Any], out: str | Path) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = result["manifest"]
    save_manifest(manifest, root / "checkpoint_manifest.json")
    (root / "metrics.json").write_text(json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_manifest(manifest)
    (root / "manifest_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_input_for(module: str, root: Path) -> Path:
    return root / DEFAULT_MODULE_INPUTS[module]


def module_from_config(config: Mapping[str, Any], fallback: str) -> str:
    train = config.get("train", {})
    if isinstance(train, Mapping):
        return str(train.get("module", fallback))
    return fallback


def train_config_value(config: Mapping[str, Any], key: str) -> str:
    train = config.get("train", {})
    if isinstance(train, Mapping) and train.get(key):
        return str(train[key])
    return ""


def resolve_path(root: Path, cli_value: str, config_value: str, fallback: Path) -> Path:
    selected = cli_value or config_value
    if not selected:
        return fallback
    path = Path(selected)
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a tiny supervised reference trainer.")
    parser.add_argument("--module", choices=sorted(DEFAULT_MODULE_INPUTS), default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else {}
    module = args.module or module_from_config(config, "fact_writer")
    config_input = train_config_value(config, "input")
    config_output = train_config_value(config, "output")
    input_path = resolve_path(root, args.input, config_input, default_input_for(module, root))
    out = resolve_path(root, args.out, config_output, root / "runs" / f"train_{module}_reference")
    result = train_module(module, read_jsonl(input_path))
    write_training_artifacts(result, out)
    print(json.dumps({"module": module, "out": str(out), "metrics": result["metrics"]}, sort_keys=True))
    return 0
