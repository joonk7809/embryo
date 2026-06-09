"""Train and evaluate a task-native recurrent POPGym RepeatFirst baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.models.popgym_recurrent import RepeatFirstGRUPolicy, require_torch, save_popgym_recurrent_checkpoint


GO_POPGYM_RECURRENT_H_LSTM_MEASURED = "GO_popgym_recurrent_h_lstm_measured"
GO_POPGYM_RECURRENT_H_LSTM_NOT_OBSERVED = "GO_popgym_recurrent_h_lstm_not_observed"
NO_GO_POPGYM_RECURRENT_LOW_COMPETENCE = "NO_GO_popgym_recurrent_low_competence"
NO_GO_POPGYM_RECURRENT_RUNTIME_UNAVAILABLE = "NO_GO_popgym_recurrent_runtime_unavailable"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"


def run_popgym_recurrent_training(config: Mapping[str, Any]) -> dict[str, Any]:
    torch, _ = require_torch()
    set_determinism(int(mapping(config.get("training")).get("seed", 20260608)))
    runtime_cfg = mapping(config.get("runtime"))
    model_cfg = mapping(config.get("model"))
    training_cfg = mapping(config.get("training"))
    eval_cfg = mapping(config.get("evaluation"))
    splits = mapping(config.get("splits"))

    action_count = int(model_cfg.get("action_count", 4))
    model = RepeatFirstGRUPolicy.build(
        action_count=action_count,
        embedding_dim=int(model_cfg.get("embedding_dim", 16)),
        hidden_size=int(model_cfg.get("hidden_size", 32)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 0.01)))
    train_data = collect_split(runtime_cfg, splits, "train", max_steps=int(training_cfg.get("train_max_steps", 96)))
    train_log = train_model(
        model,
        optimizer,
        train_data,
        epochs=int(training_cfg.get("epochs", 30)),
        batch_size=int(training_cfg.get("batch_size", 64)),
    )
    eval_max_steps = int(eval_cfg.get("max_steps", 256))
    datasets = {
        split: collect_split(runtime_cfg, splits, split, max_steps=eval_max_steps)
        for split in ("train", "dev", "test")
    }
    metrics_by_split = {split: evaluate_model(model, data) for split, data in datasets.items()}
    metrics_by_gap = {split: gap_metrics(evaluate_sequences(model, data)) for split, data in datasets.items()}
    h_lstm = extract_h_lstm(
        metrics_by_gap["test"],
        chance_success_rate=float(eval_cfg.get("chance_success_rate", 0.25)),
        tolerance=float(eval_cfg.get("falloff_tolerance", 0.05)),
        tail_bins=int(eval_cfg.get("falloff_tail_bins", 8)),
    )
    contamination = combine_contamination(datasets)
    decision, reasons = decision_from_metrics(
        metrics_by_split=metrics_by_split,
        contamination=contamination,
        h_lstm=h_lstm,
        min_short_success=float(eval_cfg.get("min_short_success_rate", 0.90)),
        competence_max_gap=int(eval_cfg.get("competence_max_gap", 32)),
    )
    manifest = {
        "model_type": "popgym_repeat_first_gru_v0",
        "task": str(runtime_cfg.get("task", "repeat_first_medium")),
        "action_count": action_count,
        "embedding_dim": int(model_cfg.get("embedding_dim", 16)),
        "hidden_size": int(model_cfg.get("hidden_size", 32)),
        "weights": "model.pt",
        "decision": decision,
        "decision_reasons": reasons,
        "training_boundary": "supervised_task_native_recurrent_baseline",
    }
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "popgym_repeat_first_recurrent_h_lstm",
        "runtime": {"name": "popgym_repeat_first", "task": str(runtime_cfg.get("task", "repeat_first_medium"))},
        "model": manifest,
        "splits": split_manifest(splits),
        "metrics_by_split": metrics_by_split,
        "h_lstm": h_lstm,
        "contamination": contamination,
        "portability_targets": portability_targets(),
    }
    return {
        "summary": summary,
        "metrics_by_split": metrics_by_split,
        "metrics_by_gap": metrics_by_gap,
        "training_log": train_log,
        "config_manifest": json_safe(config),
        "checkpoint_manifest": manifest,
        "contamination": contamination,
        "model": model,
    }


def train_model(model: Any, optimizer: Any, data: Mapping[str, np.ndarray], *, epochs: int, batch_size: int) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    tokens = torch.tensor(data["tokens"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    row_count = int(tokens.shape[0])
    log: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(row_count)
        losses: list[float] = []
        for start in range(0, row_count, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_tokens = tokens[idx]
            batch_labels = labels[idx]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_tokens)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            metrics = evaluate_model(model, data)
            log.append({"epoch": epoch + 1, "loss": round(sum(losses) / len(losses), 6), "success_rate": metrics["success_rate"]})
    return log


def evaluate_model(model: Any, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = evaluate_sequences(model, data)
    total = len(rows)
    success = sum(int(row["success"]) for row in rows)
    short_rows = [row for row in rows if 1 <= int(row["gap"]) <= 32]
    return {
        "sequence_count": int(data["tokens"].shape[0]),
        "tick_count": total,
        "success_count": success,
        "success_rate": round(success / total, 6) if total else 0.0,
        "short_success_rate": round(sum(int(row["success"]) for row in short_rows) / len(short_rows), 6) if short_rows else 0.0,
    }


def evaluate_sequences(model: Any, data: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    model.eval()
    tokens = torch.tensor(data["tokens"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(tokens)
    predicted = logits.argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    for seq_idx in range(tokens.shape[0]):
        for tick in range(tokens.shape[1]):
            gap = int(tick)
            if gap == 0:
                continue
            label = int(labels[seq_idx, tick].item())
            pred = int(predicted[seq_idx, tick].item())
            rows.append(
                {
                    "sequence_index": int(seq_idx),
                    "gap": gap,
                    "target": label,
                    "observation": int(tokens[seq_idx, tick].item()),
                    "predicted": pred,
                    "success": pred == label,
                }
            )
    return rows


def gap_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_gap: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_gap.setdefault(int(row["gap"]), []).append(row)
    return [
        {
            "gap": gap,
            "count": len(items),
            "success_rate": round(sum(int(item["success"]) for item in items) / len(items), 6),
        }
        for gap, items in sorted(by_gap.items())
    ]


def extract_h_lstm(
    rows: Sequence[Mapping[str, Any]],
    *,
    chance_success_rate: float,
    tolerance: float,
    tail_bins: int,
) -> dict[str, Any]:
    threshold = float(chance_success_rate) + float(tolerance)
    rates = [(int(row["gap"]), float(row["success_rate"])) for row in rows]
    for idx, (gap, rate) in enumerate(rates):
        tail = rates[idx : idx + int(tail_bins)]
        if len(tail) < int(tail_bins):
            break
        if rate <= threshold and all(value <= threshold for _, value in tail):
            return {
                "status": "measured",
                "h_lstm": gap,
                "chance_success_rate": float(chance_success_rate),
                "falloff_threshold": round(threshold, 6),
                "tail_bins": int(tail_bins),
            }
    return {
        "status": "not_observed",
        "h_lstm": None,
        "chance_success_rate": float(chance_success_rate),
        "falloff_threshold": round(threshold, 6),
        "tail_bins": int(tail_bins),
        "max_gap": max((gap for gap, _ in rates), default=0),
        "min_success_rate": round(min((rate for _, rate in rates), default=0.0), 6),
    }


def collect_split(runtime_cfg: Mapping[str, Any], splits: Mapping[str, Any], split: str, *, max_steps: int) -> dict[str, np.ndarray]:
    split_cfg = mapping(splits.get(split))
    task = str(runtime_cfg.get("task", "repeat_first_medium"))
    seed_start = int(split_cfg.get("seed_start", 10000))
    seed_count = int(split_cfg.get("seed_count", 128))
    tokens: list[list[int]] = []
    labels: list[list[int]] = []
    contamination_rows: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + seed_count):
        token_row, label_row, contexts = collect_sequence(task=task, seed=seed, max_steps=max_steps)
        tokens.append(token_row)
        labels.append(label_row)
        contamination_rows.extend(contexts)
    return {
        "tokens": np.asarray(tokens, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "contamination": contamination_summary(contamination_rows),
    }


def collect_sequence(*, task: str, seed: int, max_steps: int) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    env = make_repeat_first_env(task)
    try:
        obs, _ = env.reset(seed=int(seed))
        target = int_value(obs)
        tokens = [target]
        labels = [target]
        contexts = [actor_context(obs=target, previous_obs=None, previous_action="suit_0")]
        previous_obs = target
        previous_action = "suit_0"
        for _ in range(1, int(max_steps)):
            obs, _, terminated, truncated, _ = env.step(0)
            current = int_value(obs)
            tokens.append(current)
            labels.append(target)
            contexts.append(actor_context(obs=current, previous_obs=previous_obs, previous_action=previous_action))
            previous_obs = current
            previous_action = "suit_0"
            if bool(terminated or truncated):
                break
        while len(tokens) < int(max_steps):
            tokens.append(tokens[-1])
            labels.append(target)
            contexts.append(actor_context(obs=tokens[-1], previous_obs=previous_obs, previous_action=previous_action))
        return tokens[: int(max_steps)], labels[: int(max_steps)], contexts[: int(max_steps)]
    finally:
        env.close()


def make_repeat_first_env(task: str):
    try:
        from popgym.envs.repeat_first import RepeatFirstEasy, RepeatFirstHard, RepeatFirstMedium
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install popgym to train the RepeatFirst recurrent baseline.") from exc
    mapping_by_task = {
        "repeat_first_easy": RepeatFirstEasy,
        "repeat_first_medium": RepeatFirstMedium,
        "repeat_first_hard": RepeatFirstHard,
    }
    if task not in mapping_by_task:
        raise ValueError(f"Unknown POPGym RepeatFirst task: {task}")
    return mapping_by_task[task]()


def actor_context(*, obs: int, previous_obs: int | None, previous_action: str) -> dict[str, Any]:
    return {
        "popgym_observation": int(obs),
        "previous_popgym_observation": None if previous_obs is None else int(previous_obs),
        "previous_action": str(previous_action),
    }


def contamination_summary(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for idx, context in enumerate(contexts):
        scan = scan_actor_context(context)
        for failure in scan.get("failures", []):
            failures.append({"row": idx, **dict(failure)})
    return {"passed": not failures, "failure_count": len(failures), "failures": failures}


def combine_contamination(datasets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for split, data in datasets.items():
        scan = mapping(data.get("contamination"))
        for failure in scan.get("failures", []):
            failures.append({"split": split, **dict(failure)})
    return {"passed": not failures, "failure_count": len(failures), "failures": failures}


def decision_from_metrics(
    *,
    metrics_by_split: Mapping[str, Mapping[str, Any]],
    contamination: Mapping[str, Any],
    h_lstm: Mapping[str, Any],
    min_short_success: float,
    competence_max_gap: int,
) -> tuple[str, list[str]]:
    _ = competence_max_gap
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count_nonzero"]
    dev = mapping(metrics_by_split.get("dev"))
    test = mapping(metrics_by_split.get("test"))
    reasons = []
    if float(dev.get("short_success_rate", 0.0)) < float(min_short_success):
        reasons.append(f"dev_short_success_rate<{min_short_success}")
    if float(test.get("short_success_rate", 0.0)) < float(min_short_success):
        reasons.append(f"test_short_success_rate<{min_short_success}")
    if reasons:
        return NO_GO_POPGYM_RECURRENT_LOW_COMPETENCE, reasons
    if h_lstm.get("status") == "measured":
        return GO_POPGYM_RECURRENT_H_LSTM_MEASURED, []
    return GO_POPGYM_RECURRENT_H_LSTM_NOT_OBSERVED, ["falloff_not_observed_within_sweep"]


def write_popgym_recurrent_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    checkpoint = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = save_popgym_recurrent_checkpoint(checkpoint, model=result["model"], manifest=result["checkpoint_manifest"])
    paths = {
        "popgym_recurrent_summary_json": root / "popgym_recurrent_summary.json",
        "popgym_recurrent_summary_md": root / "popgym_recurrent_summary.md",
        "metrics_by_split": root / "metrics_by_split.json",
        "metrics_by_gap": root / "metrics_by_gap.json",
        "config_manifest": root / "config_manifest.json",
        "checkpoint_manifest": checkpoint / "manifest.json",
        "contamination_scan": root / "contamination_scan.json",
        "training_log": root / "training_log.jsonl",
    }
    paths["popgym_recurrent_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["popgym_recurrent_summary_md"].write_text(format_summary_markdown(result["summary"]), encoding="utf-8")
    paths["metrics_by_split"].write_text(json.dumps(result["metrics_by_split"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics_by_gap"].write_text(json.dumps(result["metrics_by_gap"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["config_manifest"].write_text(json.dumps(result["config_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(paths["training_log"], result["training_log"])
    paths.update({f"checkpoint_{key}": Path(value) for key, value in checkpoint_paths.items()})
    return {key: str(value) for key, value in paths.items()}


def format_summary_markdown(summary: Mapping[str, Any]) -> str:
    h_lstm = mapping(summary.get("h_lstm"))
    lines = [
        "# POPGym Recurrent Baseline Summary",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Task: `{mapping(summary.get('runtime')).get('task')}`",
        f"- H_lstm status: `{h_lstm.get('status')}`",
        f"- H_lstm: `{h_lstm.get('h_lstm')}`",
        f"- Contamination failures: `{mapping(summary.get('contamination')).get('failure_count')}`",
        "",
        "## Splits",
        "",
        "| Split | Sequences | Success | Short Success |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, row in mapping(summary.get("metrics_by_split")).items():
        metrics = mapping(row)
        lines.append(
            f"| {split} | {metrics.get('sequence_count')} | {metrics.get('success_rate')} | {metrics.get('short_success_rate')} |"
        )
    lines.extend(["", "## Portability Targets", ""])
    for target in summary.get("portability_targets", ()):
        lines.append(f"- `{target}`")
    return "\n".join(lines) + "\n"


def portability_targets() -> list[str]:
    return [
        "popgym_repeat_first_gru_v0",
        "popgym_sequence_memory_gru_future",
        "visual_memory_maze_recurrent_future",
    ]


def split_manifest(splits: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): dict(mapping(value)) for key, value in splits.items()}


def set_determinism(seed: int) -> None:
    torch, _ = require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def int_value(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/popgym_recurrent_repeat_first.yaml")
    parser.add_argument("--out", default="runs/popgym_recurrent_repeat_first")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_popgym_recurrent_training(config)
    artifacts = write_popgym_recurrent_artifacts(result, args.out)
    print(json.dumps({"decision": result["summary"]["decision"], "artifacts": artifacts, "out": str(Path(args.out).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
