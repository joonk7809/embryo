"""Train and sweep a recurrent POPGym Autoencode baseline."""

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
from embryo.memory.popgym_autoencode import OrderedSequenceMemory
from embryo.models.popgym_associative_memory import (
    ExplicitPositionDeltaMemoryPolicy,
    content_corrupt_suits,
    order_corrupt_read_positions,
    save_popgym_associative_checkpoint,
    shuffled_binding_suits,
)
from embryo.models.popgym_recurrent import RepeatFirstGRUPolicy, require_torch, save_popgym_recurrent_checkpoint
from embryo.run.train_popgym_recurrent import extract_h_lstm, mapping, set_determinism


GO_POPGYM_AUTOENCODE_H_LSTM_MEASURED = "GO_popgym_autoencode_h_lstm_measured"
GO_POPGYM_AUTOENCODE_H_LSTM_NOT_OBSERVED = "GO_popgym_autoencode_h_lstm_not_observed"
NO_GO_POPGYM_AUTOENCODE_LOW_COMPETENCE = "NO_GO_popgym_autoencode_low_competence"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"
GO_POPGYM_AUTOENCODE_MEMORY_CURVE_SUPPORTED = "GO_popgym_autoencode_memory_curve_supported"
GO_POPGYM_AUTOENCODE_LEARNED_MEMORY_SUPPORTED = "GO_popgym_autoencode_learned_memory_supported"
NO_GO_POPGYM_AUTOENCODE_LEARNED_MEMORY_LOW_RECALL = "NO_GO_popgym_autoencode_learned_memory_low_recall"
NO_GO_POPGYM_AUTOENCODE_LEARNED_MEMORY_CONTROL_FAILURE = "NO_GO_popgym_autoencode_learned_memory_control_failure"

MODE_PLAY = 0
MODE_WATCH = 1
SUIT_COUNT = 4
IGNORE_INDEX = -100
AUTOENCODE_MEMORY_ARMS = (
    "autoencode_memory_off",
    "autoencode_memory_clean",
    "autoencode_memory_shuffled",
    "autoencode_memory_content_corrupt",
    "autoencode_memory_order_corrupt",
)
AUTOENCODE_LEARNED_MEMORY_ARMS = (
    "autoencode_memory_off",
    "autoencode_hand_coded_memory_clean",
    "autoencode_learned_memory_clean",
    "autoencode_learned_memory_shuffled",
    "autoencode_learned_memory_content_corrupt",
    "autoencode_learned_memory_order_corrupt",
)


def run_popgym_autoencode_training(config: Mapping[str, Any]) -> dict[str, Any]:
    torch, _ = require_torch()
    training_cfg = mapping(config.get("training"))
    model_cfg = mapping(config.get("model"))
    evaluation_cfg = mapping(config.get("evaluation"))
    tasks_cfg = mapping(config.get("tasks"))
    set_determinism(int(training_cfg.get("seed", 20260608)))

    train_tasks = tuple(str(item) for item in tasks_cfg.get("train", ("autoencode_easy",)))
    eval_tasks = tuple(str(item) for item in tasks_cfg.get("eval", ("autoencode_easy", "autoencode_medium", "autoencode_hard")))
    train_data = collect_tasks(
        tasks=train_tasks,
        seed_start=int(training_cfg.get("seed_start", 10000)),
        seed_count=int(training_cfg.get("seed_count", 512)),
    )
    model = RepeatFirstGRUPolicy.build(
        action_count=SUIT_COUNT,
        token_count=8,
        embedding_dim=int(model_cfg.get("embedding_dim", 16)),
        hidden_size=int(model_cfg.get("hidden_size", 32)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 0.01)))
    train_log = train_autoencode_model(
        model,
        optimizer,
        train_data,
        epochs=int(training_cfg.get("epochs", 30)),
        batch_size=int(training_cfg.get("batch_size", 64)),
    )
    metrics_by_task = {}
    metrics_by_gap = {}
    split_cfg = mapping(config.get("splits"))
    for task in eval_tasks:
        task_split = mapping(split_cfg.get(task))
        data = collect_tasks(
            tasks=(task,),
            seed_start=int(task_split.get("seed_start", 12000)),
            seed_count=int(task_split.get("seed_count", 128)),
        )
        rows = evaluate_autoencode_sequences(model, data)
        metrics_by_task[task] = task_metrics(
            rows,
            sequence_count=int(data["tokens"].shape[0]),
            short_gap_max=int(evaluation_cfg.get("competence_max_gap", 4)),
        )
        metrics_by_gap[task] = gap_metrics(rows)
    h_lstm_by_task = {
        task: extract_h_lstm(
            rows,
            chance_success_rate=float(evaluation_cfg.get("chance_success_rate", 0.25)),
            tolerance=float(evaluation_cfg.get("falloff_tolerance", 0.05)),
            tail_bins=int(evaluation_cfg.get("falloff_tail_bins", 8)),
        )
        for task, rows in metrics_by_gap.items()
    }
    memory_curve = autoencode_memory_curve(model, config, baseline_gap_metrics_by_task=metrics_by_gap)
    learned_memory = train_and_evaluate_learned_memory(model, config)
    contamination = combine_contamination(train_data, metrics_by_task)
    decision, reasons = autoencode_decision(
        metrics_by_task=metrics_by_task,
        h_lstm_by_task=h_lstm_by_task,
        contamination=contamination,
        min_easy_success=float(evaluation_cfg.get("min_easy_success_rate", 0.90)),
    )
    manifest = {
        "model_type": "popgym_autoencode_gru_v0",
        "token_count": 8,
        "action_count": SUIT_COUNT,
        "embedding_dim": int(model_cfg.get("embedding_dim", 16)),
        "hidden_size": int(model_cfg.get("hidden_size", 32)),
        "weights": "model.pt",
        "decision": decision,
        "decision_reasons": reasons,
        "training_boundary": "supervised_task_native_recurrent_autoencode_baseline",
    }
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "popgym_autoencode_recurrent_h_lstm",
        "tasks": {"train": list(train_tasks), "eval": list(eval_tasks)},
        "model": manifest,
        "metrics_by_task": metrics_by_task,
        "h_lstm_by_task": h_lstm_by_task,
        "memory_curve": memory_curve,
        "learned_memory_curve": learned_memory["curve"],
        "contamination": contamination,
        "selection_rule": "baseline_breaks_first_no_memory_arms",
    }
    return {
        "summary": summary,
        "metrics_by_task": metrics_by_task,
        "metrics_by_gap": metrics_by_gap,
        "memory_curve": memory_curve,
        "learned_memory_curve": learned_memory["curve"],
        "learned_memory_training_log": learned_memory["training_log"],
        "learned_memory_checkpoint_manifest": learned_memory["manifest"],
        "learned_memory_model": learned_memory["model"],
        "training_log": train_log,
        "config_manifest": json_safe(config),
        "checkpoint_manifest": manifest,
        "contamination": contamination,
        "model": model,
    }


def train_autoencode_model(model: Any, optimizer: Any, data: Mapping[str, np.ndarray], *, epochs: int, batch_size: int) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    tokens = torch.tensor(data["tokens"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    masks = torch.tensor(data["masks"], dtype=torch.bool)
    row_count = int(tokens.shape[0])
    log: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(row_count)
        losses: list[float] = []
        for start in range(0, row_count, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens[idx])
            loss = torch.nn.functional.cross_entropy(logits[masks[idx]], labels[idx][masks[idx]])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            metrics = task_metrics(evaluate_autoencode_sequences(model, data), sequence_count=row_count, short_gap_max=4)
            log.append({"epoch": epoch + 1, "loss": round(sum(losses) / len(losses), 6), "play_success_rate": metrics["play_success_rate"]})
    return log


def collect_tasks(*, tasks: Sequence[str], seed_start: int, seed_count: int) -> dict[str, Any]:
    rows = [collect_autoencode_sequence(task=task, seed=seed) for task in tasks for seed in range(seed_start, seed_start + seed_count)]
    max_len = max(len(row["tokens"]) for row in rows)
    tokens = np.zeros((len(rows), max_len), dtype=np.int64)
    labels = np.full((len(rows), max_len), IGNORE_INDEX, dtype=np.int64)
    masks = np.zeros((len(rows), max_len), dtype=bool)
    gaps = np.zeros((len(rows), max_len), dtype=np.int64)
    task_names = []
    contexts = []
    for idx, row in enumerate(rows):
        length = len(row["tokens"])
        tokens[idx, :length] = row["tokens"]
        labels[idx, :length] = row["labels"]
        masks[idx, :length] = row["masks"]
        gaps[idx, :length] = row["gaps"]
        task_names.append(row["task"])
        contexts.extend(row["contexts"])
    return {
        "tokens": tokens,
        "labels": labels,
        "masks": masks,
        "gaps": gaps,
        "task_names": task_names,
        "watch_suits": [tuple(row["watch_suits"]) for row in rows],
        "contamination": contamination_summary(contexts),
    }


def collect_autoencode_sequence(*, task: str, seed: int) -> dict[str, Any]:
    env = make_autoencode_env(task)
    try:
        obs, _ = env.reset(seed=int(seed))
        total_cards = int((int(env.max_episode_length) + 1) // 2)
        seen: list[int] = []
        tokens: list[int] = []
        labels: list[int] = []
        masks: list[bool] = []
        gaps: list[int] = []
        contexts: list[dict[str, Any]] = []
        previous_obs = None
        previous_action = "suit_0"
        final_play_cue_added = False
        watch_suits: list[int] = []
        done = False
        while not done:
            mode, suit = parse_obs(obs)
            tokens.append(autoencode_token(mode, suit))
            contexts.append(actor_context(mode=mode, suit=suit, previous_obs=previous_obs, previous_action=previous_action))
            if mode == MODE_WATCH:
                seen.append(suit)
                watch_suits.append(suit)
                labels.append(IGNORE_INDEX)
                masks.append(False)
                gaps.append(0)
                action = 0
            else:
                if not final_play_cue_added and len(seen) < total_cards:
                    seen.append(suit)
                    watch_suits.append(suit)
                    final_play_cue_added = True
                target = seen.pop()
                labels.append(target)
                masks.append(True)
                gaps.append(total_cards - len(seen))
                action = 0
            previous_obs = (mode, suit)
            obs, _, terminated, truncated, _ = env.step(action)
            previous_action = f"suit_{action}"
            done = bool(terminated or truncated)
        return {"task": task, "tokens": tokens, "labels": labels, "masks": masks, "gaps": gaps, "watch_suits": watch_suits, "contexts": contexts}
    finally:
        env.close()


def evaluate_autoencode_sequences(model: Any, data: Mapping[str, Any]) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    model.eval()
    tokens = torch.tensor(data["tokens"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    masks = torch.tensor(data["masks"], dtype=torch.bool)
    gaps = np.asarray(data["gaps"])
    task_names = list(data["task_names"])
    with torch.no_grad():
        logits, _ = model(tokens)
    predicted = logits.argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    for seq_idx in range(tokens.shape[0]):
        for tick in range(tokens.shape[1]):
            if not bool(masks[seq_idx, tick]):
                continue
            label = int(labels[seq_idx, tick].item())
            pred = int(predicted[seq_idx, tick].item())
            rows.append(
                {
                    "task": task_names[seq_idx],
                    "sequence_index": int(seq_idx),
                    "tick": int(tick),
                    "gap": int(gaps[seq_idx, tick]),
                    "target": label,
                    "predicted": pred,
                    "success": pred == label,
                }
            )
    return rows


def task_metrics(rows: Sequence[Mapping[str, Any]], *, sequence_count: int, short_gap_max: int) -> dict[str, Any]:
    total = len(rows)
    success = sum(int(row["success"]) for row in rows)
    short_rows = [row for row in rows if int(row["gap"]) <= int(short_gap_max)]
    short_success = sum(int(row["success"]) for row in short_rows)
    return {
        "sequence_count": int(sequence_count),
        "play_tick_count": total,
        "success_count": success,
        "play_success_rate": round(success / total, 6) if total else 0.0,
        "short_gap_max": int(short_gap_max),
        "short_success_rate": round(short_success / len(short_rows), 6) if short_rows else 0.0,
    }


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


def autoencode_decision(
    *,
    metrics_by_task: Mapping[str, Mapping[str, Any]],
    h_lstm_by_task: Mapping[str, Mapping[str, Any]],
    contamination: Mapping[str, Any],
    min_easy_success: float,
) -> tuple[str, list[str]]:
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count_nonzero"]
    easy = mapping(metrics_by_task.get("autoencode_easy"))
    if float(easy.get("short_success_rate", 0.0)) < float(min_easy_success):
        return NO_GO_POPGYM_AUTOENCODE_LOW_COMPETENCE, [f"autoencode_easy_short_success_rate<{min_easy_success}"]
    measured = [task for task, row in h_lstm_by_task.items() if row.get("status") == "measured"]
    if measured:
        return GO_POPGYM_AUTOENCODE_H_LSTM_MEASURED, [f"measured_tasks={','.join(sorted(measured))}"]
    return GO_POPGYM_AUTOENCODE_H_LSTM_NOT_OBSERVED, ["falloff_not_observed_within_sweep"]


def autoencode_memory_curve(model: Any, config: Mapping[str, Any], *, baseline_gap_metrics_by_task: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = mapping(config.get("memory_eval"))
    if not cfg:
        return {"enabled": False, "status": "disabled"}
    task = str(cfg.get("task", "autoencode_easy"))
    gaps = tuple(int(value) for value in cfg.get("gaps", (10, 20, 27, 40, 52)))
    seed_start = int(cfg.get("seed_start", 12000))
    seed_count = int(cfg.get("seed_count", 128))
    data = collect_tasks(tasks=(task,), seed_start=seed_start, seed_count=seed_count)
    baseline_rows = evaluate_autoencode_sequences(model, data)
    baseline_by_key = {(int(row["sequence_index"]), int(row["gap"])): int(row["predicted"]) for row in baseline_rows}
    rows: list[dict[str, Any]] = []
    for seq_idx, suits in enumerate(data["watch_suits"]):
        memory = OrderedSequenceMemory.from_observed_suits(suits)
        labels_by_gap = sequence_labels_by_gap(data, seq_idx)
        for gap in gaps:
            target = labels_by_gap.get(gap)
            if target is None:
                for arm in AUTOENCODE_MEMORY_ARMS:
                    rows.append(memory_curve_row(task, seq_idx, gap, arm, None, None, memory, available=False))
                continue
            actions = {
                "autoencode_memory_off": baseline_by_key.get((seq_idx, gap)),
                "autoencode_memory_clean": memory.recall_reverse(gap),
                "autoencode_memory_shuffled": memory.recall_shuffled(gap, suit_count=SUIT_COUNT),
                "autoencode_memory_content_corrupt": memory.recall_content_corrupt(gap, suit_count=SUIT_COUNT),
                "autoencode_memory_order_corrupt": memory.recall_order_corrupt(gap),
            }
            for arm, action in actions.items():
                rows.append(memory_curve_row(task, seq_idx, gap, arm, target, action, memory, available=True))
    by_gap = memory_curve_by_gap(rows, gaps=gaps)
    by_arm = memory_curve_by_arm(rows)
    reconciliation = baseline_reconciliation(
        by_gap,
        list(mapping(baseline_gap_metrics_by_task).get(task, ())) if baseline_gap_metrics_by_task else (),
        gaps=gaps,
        comparable_seed_block=memory_eval_matches_split(config, task=task, seed_start=seed_start, seed_count=seed_count),
    )
    return {
        "enabled": True,
        "status": GO_POPGYM_AUTOENCODE_MEMORY_CURVE_SUPPORTED,
        "task": task,
        "seed_start": seed_start,
        "seed_count": seed_count,
        "gaps": list(gaps),
        "arms": list(AUTOENCODE_MEMORY_ARMS),
        "note": "Clean/order/content memory actions use only ordered suits observed during Autoencode watch phase.",
        "baseline_reconciliation": reconciliation,
        "summary_by_arm": by_arm,
        "summary_by_gap": by_gap,
        "rows": rows,
    }


def sequence_labels_by_gap(data: Mapping[str, Any], seq_idx: int) -> dict[int, int]:
    labels = np.asarray(data["labels"])[int(seq_idx)]
    masks = np.asarray(data["masks"])[int(seq_idx)]
    gaps = np.asarray(data["gaps"])[int(seq_idx)]
    return {int(gap): int(label) for label, mask, gap in zip(labels, masks, gaps) if bool(mask)}


def memory_curve_row(
    task: str,
    sequence_index: int,
    gap: int,
    arm: str,
    target: int | None,
    action: int | None,
    memory: OrderedSequenceMemory,
    *,
    available: bool,
) -> dict[str, Any]:
    return {
        "task": task,
        "sequence_index": int(sequence_index),
        "gap": int(gap),
        "arm": arm,
        "available": bool(available),
        "target_eval_only": target,
        "action": action,
        "success_eval_only": bool(available and target is not None and action == target),
        "memory_content_hash": memory.content_hash,
        "memory_length": len(memory.suits),
    }


def memory_curve_by_gap(rows: Sequence[Mapping[str, Any]], *, gaps: Sequence[int], arms: Sequence[str] = AUTOENCODE_MEMORY_ARMS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for gap in gaps:
        gap_rows = [row for row in rows if int(row["gap"]) == int(gap)]
        arm_metrics: dict[str, Any] = {}
        for arm in arms:
            arm_rows = [row for row in gap_rows if str(row["arm"]) == arm]
            available = [row for row in arm_rows if bool(row["available"])]
            arm_metrics[arm] = {
                "count": len(arm_rows),
                "available_count": len(available),
                "success_rate": round(bool_rate(row["success_eval_only"] for row in available), 6),
            }
        result[str(gap)] = arm_metrics
    return result


def memory_curve_by_arm(rows: Sequence[Mapping[str, Any]], *, arms: Sequence[str] = AUTOENCODE_MEMORY_ARMS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [row for row in rows if str(row["arm"]) == arm]
        available = [row for row in arm_rows if bool(row["available"])]
        result[arm] = {
            "count": len(arm_rows),
            "available_count": len(available),
            "success_rate": round(bool_rate(row["success_eval_only"] for row in available), 6),
        }
    return result


def baseline_reconciliation(
    memory_by_gap: Mapping[str, Any],
    baseline_gap_rows: Sequence[Mapping[str, Any]],
    *,
    gaps: Sequence[int],
    comparable_seed_block: bool,
) -> dict[str, Any]:
    baseline_by_gap = {int(row["gap"]): row for row in baseline_gap_rows}
    rows: list[dict[str, Any]] = []
    for gap in gaps:
        memory_off = mapping(mapping(memory_by_gap.get(str(gap))).get("autoencode_memory_off"))
        baseline = mapping(baseline_by_gap.get(int(gap)))
        memory_rate = memory_off.get("success_rate")
        baseline_rate = baseline.get("success_rate")
        delta = None if memory_rate is None or baseline_rate is None else round(abs(float(memory_rate) - float(baseline_rate)), 12)
        rows.append(
            {
                "gap": int(gap),
                "memory_off_success_rate": memory_rate,
                "baseline_eval_success_rate": baseline_rate,
                "memory_off_count": int(memory_off.get("available_count", 0)),
                "baseline_eval_count": int(baseline.get("count", 0)),
                "absolute_delta": delta,
            }
        )
    if not comparable_seed_block:
        status = "not_comparable_seed_block"
    elif not rows or any(row["baseline_eval_success_rate"] is None for row in rows):
        status = "missing_baseline_gap"
    elif all(float(row["absolute_delta"]) <= 1e-9 for row in rows if row["absolute_delta"] is not None):
        status = "matched"
    else:
        status = "mismatch"
    return {"status": status, "rows": rows}


def train_and_evaluate_learned_memory(recurrent_model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = mapping(config.get("learned_memory"))
    if not cfg or cfg.get("enabled", True) is False:
        return {
            "curve": {"enabled": False, "status": "disabled"},
            "training_log": [],
            "manifest": {"enabled": False},
            "model": None,
        }
    torch, _ = require_torch()
    task = str(cfg.get("task", mapping(config.get("memory_eval")).get("task", "autoencode_easy")))
    gaps = tuple(int(value) for value in cfg.get("gaps", mapping(config.get("memory_eval")).get("gaps", (1, 2, 3, 4, 10, 20, 27, 40, 52))))
    train_data = collect_tasks(
        tasks=(task,),
        seed_start=int(cfg.get("seed_start", mapping(config.get("training")).get("seed_start", 10000))),
        seed_count=int(cfg.get("seed_count", mapping(config.get("training")).get("seed_count", 512))),
    )
    eval_data = collect_tasks(
        tasks=(task,),
        seed_start=int(cfg.get("eval_seed_start", mapping(config.get("memory_eval")).get("seed_start", 12000))),
        seed_count=int(cfg.get("eval_seed_count", mapping(config.get("memory_eval")).get("seed_count", 128))),
    )
    sequence_length = fixed_watch_length(train_data)
    memory_model = ExplicitPositionDeltaMemoryPolicy.build(
        max_positions=int(cfg.get("max_positions", sequence_length)),
        suit_count=SUIT_COUNT,
        value_dim=int(cfg.get("value_dim", 8)),
        write_eta=float(cfg.get("write_eta", 1.0)),
    )
    optimizer = torch.optim.Adam(memory_model.parameters(), lr=float(cfg.get("learning_rate", 0.05)))
    training_log = train_learned_memory_model(
        memory_model,
        optimizer,
        train_data,
        epochs=int(cfg.get("epochs", 30)),
        batch_size=int(cfg.get("batch_size", 64)),
    )
    curve = learned_memory_curve(memory_model, recurrent_model, eval_data, task=task, gaps=gaps)
    decision, reasons = learned_memory_decision(
        curve,
        min_clean_success=float(cfg.get("min_clean_success_rate", 0.95)),
        max_content_corrupt_success=float(cfg.get("max_content_corrupt_success_rate", 0.05)),
        max_chance_control_success=float(cfg.get("max_chance_control_success_rate", 0.35)),
    )
    curve["status"] = decision
    curve["decision_reasons"] = reasons
    manifest = {
        "model_type": "popgym_autoencode_explicit_position_delta_memory_v0",
        "task": task,
        "suit_count": SUIT_COUNT,
        "max_positions": int(cfg.get("max_positions", sequence_length)),
        "value_dim": int(cfg.get("value_dim", 8)),
        "write_eta": float(cfg.get("write_eta", 1.0)),
        "weights": "model.pt",
        "decision": decision,
        "decision_reasons": reasons,
        "position_key": "explicit_agent_owned_sequence_index",
        "write_rule": "delta_rule_one_hot_position",
        "training_boundary": "supervised_learned_associative_recall_memory",
    }
    return {"curve": curve, "training_log": training_log, "manifest": manifest, "model": memory_model}


def train_learned_memory_model(model: Any, optimizer: Any, data: Mapping[str, Any], *, epochs: int, batch_size: int) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    arrays = learned_memory_arrays(data)
    watch_suits = torch.tensor(arrays["watch_suits"], dtype=torch.long)
    read_positions = torch.tensor(arrays["read_positions"], dtype=torch.long)
    targets = torch.tensor(arrays["targets"], dtype=torch.long)
    row_count = int(watch_suits.shape[0])
    log: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(row_count)
        losses: list[float] = []
        for start in range(0, row_count, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(watch_suits[idx], read_positions[idx])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[idx].reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            pred = learned_memory_predictions(model, arrays, arm="autoencode_learned_memory_clean")
            success = np.asarray(pred) == arrays["targets"]
            log.append({"epoch": epoch + 1, "loss": round(sum(losses) / len(losses), 6), "clean_success_rate": round(float(success.mean()), 6)})
    return log


def learned_memory_curve(model: Any, recurrent_model: Any, data: Mapping[str, Any], *, task: str, gaps: Sequence[int]) -> dict[str, Any]:
    arrays = learned_memory_arrays(data, gaps=gaps)
    baseline_rows = evaluate_autoencode_sequences(recurrent_model, data)
    baseline_by_key = {(int(row["sequence_index"]), int(row["gap"])): int(row["predicted"]) for row in baseline_rows}
    learned_predictions = {
        arm: learned_memory_predictions(model, arrays, arm=arm)
        for arm in (
            "autoencode_learned_memory_clean",
            "autoencode_learned_memory_shuffled",
            "autoencode_learned_memory_content_corrupt",
            "autoencode_learned_memory_order_corrupt",
        )
    }
    rows: list[dict[str, Any]] = []
    for seq_idx, suits in enumerate(data["watch_suits"]):
        memory = OrderedSequenceMemory.from_observed_suits(suits)
        for gap_idx, gap in enumerate(arrays["gaps"]):
            target = int(arrays["targets"][seq_idx, gap_idx])
            actions = {
                "autoencode_memory_off": baseline_by_key.get((seq_idx, int(gap))),
                "autoencode_hand_coded_memory_clean": memory.recall_reverse(int(gap)),
                "autoencode_learned_memory_clean": int(learned_predictions["autoencode_learned_memory_clean"][seq_idx, gap_idx]),
                "autoencode_learned_memory_shuffled": int(learned_predictions["autoencode_learned_memory_shuffled"][seq_idx, gap_idx]),
                "autoencode_learned_memory_content_corrupt": int(learned_predictions["autoencode_learned_memory_content_corrupt"][seq_idx, gap_idx]),
                "autoencode_learned_memory_order_corrupt": int(learned_predictions["autoencode_learned_memory_order_corrupt"][seq_idx, gap_idx]),
            }
            for arm, action in actions.items():
                rows.append(
                    {
                        "task": task,
                        "sequence_index": int(seq_idx),
                        "gap": int(gap),
                        "arm": arm,
                        "available": action is not None,
                        "target_eval_only": target,
                        "action": action,
                        "success_eval_only": bool(action is not None and int(action) == target),
                        "memory_length": int(arrays["sequence_length"]),
                    }
                )
    return {
        "enabled": True,
        "status": "computed",
        "task": task,
        "seed_count": int(arrays["sequence_count"]),
        "gaps": [int(gap) for gap in arrays["gaps"]],
        "arms": list(AUTOENCODE_LEARNED_MEMORY_ARMS),
        "note": "Learned memory uses explicit agent-owned sequence indices and a delta-rule fast store; no surprise-gated write selection is claimed.",
        "summary_by_arm": memory_curve_by_arm(rows, arms=AUTOENCODE_LEARNED_MEMORY_ARMS),
        "summary_by_gap": memory_curve_by_gap(rows, gaps=gaps, arms=AUTOENCODE_LEARNED_MEMORY_ARMS),
        "rows": rows,
    }


def learned_memory_arrays(data: Mapping[str, Any], *, gaps: Sequence[int] | None = None) -> dict[str, Any]:
    lengths = [len(suits) for suits in data["watch_suits"]]
    if not lengths:
        raise ValueError("learned memory data has no sequences")
    if len(set(lengths)) != 1:
        raise ValueError("learned memory v0 requires fixed-length Autoencode sequences")
    sequence_length = int(lengths[0])
    gap_values = tuple(range(1, sequence_length + 1)) if gaps is None else tuple(int(gap) for gap in gaps)
    watch_suits = np.asarray(data["watch_suits"], dtype=np.int64)
    read_positions = np.zeros((len(watch_suits), len(gap_values)), dtype=np.int64)
    targets = np.zeros((len(watch_suits), len(gap_values)), dtype=np.int64)
    for seq_idx in range(len(watch_suits)):
        labels_by_gap = sequence_labels_by_gap(data, seq_idx)
        for gap_idx, gap in enumerate(gap_values):
            if gap not in labels_by_gap:
                raise ValueError(f"missing label for gap {gap}")
            read_positions[seq_idx, gap_idx] = sequence_length - int(gap)
            targets[seq_idx, gap_idx] = int(labels_by_gap[gap])
    return {
        "watch_suits": watch_suits,
        "read_positions": read_positions,
        "targets": targets,
        "gaps": gap_values,
        "sequence_length": sequence_length,
        "sequence_count": len(watch_suits),
    }


def learned_memory_predictions(model: Any, arrays: Mapping[str, Any], *, arm: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    watch_suits = torch.tensor(arrays["watch_suits"], dtype=torch.long)
    gaps = torch.tensor([int(gap) for gap in arrays["gaps"]], dtype=torch.long).unsqueeze(0).expand(int(arrays["sequence_count"]), -1)
    if arm == "autoencode_learned_memory_order_corrupt":
        read_positions = order_corrupt_read_positions(gaps=gaps)
    else:
        read_positions = torch.tensor(arrays["read_positions"], dtype=torch.long)
    write_suits = None
    if arm == "autoencode_learned_memory_content_corrupt":
        write_suits = content_corrupt_suits(watch_suits, suit_count=SUIT_COUNT)
    elif arm == "autoencode_learned_memory_shuffled":
        write_suits = shuffled_binding_suits(watch_suits)
    with torch.no_grad():
        logits = model(watch_suits, read_positions, write_suits=write_suits)
    return logits.argmax(dim=-1).detach().cpu().numpy()


def learned_memory_decision(
    curve: Mapping[str, Any],
    *,
    min_clean_success: float,
    max_content_corrupt_success: float,
    max_chance_control_success: float,
) -> tuple[str, list[str]]:
    by_arm = mapping(curve.get("summary_by_arm"))
    clean = float(mapping(by_arm.get("autoencode_learned_memory_clean")).get("success_rate", 0.0))
    content = float(mapping(by_arm.get("autoencode_learned_memory_content_corrupt")).get("success_rate", 1.0))
    shuffled = float(mapping(by_arm.get("autoencode_learned_memory_shuffled")).get("success_rate", 1.0))
    order = float(mapping(by_arm.get("autoencode_learned_memory_order_corrupt")).get("success_rate", 1.0))
    if clean < float(min_clean_success):
        return NO_GO_POPGYM_AUTOENCODE_LEARNED_MEMORY_LOW_RECALL, [f"clean_success_rate<{min_clean_success}"]
    failures = []
    if content > float(max_content_corrupt_success):
        failures.append(f"content_corrupt_success_rate>{max_content_corrupt_success}")
    if shuffled > float(max_chance_control_success):
        failures.append(f"shuffled_success_rate>{max_chance_control_success}")
    if order > float(max_chance_control_success):
        failures.append(f"order_corrupt_success_rate>{max_chance_control_success}")
    if failures:
        return NO_GO_POPGYM_AUTOENCODE_LEARNED_MEMORY_CONTROL_FAILURE, failures
    return GO_POPGYM_AUTOENCODE_LEARNED_MEMORY_SUPPORTED, ["learned_memory_matches_fixed_gap_control_signature"]


def fixed_watch_length(data: Mapping[str, Any]) -> int:
    lengths = {len(suits) for suits in data["watch_suits"]}
    if len(lengths) != 1:
        raise ValueError("expected fixed-length watch_suits")
    return int(next(iter(lengths)))


def memory_eval_matches_split(config: Mapping[str, Any], *, task: str, seed_start: int, seed_count: int) -> bool:
    split = mapping(mapping(config.get("splits")).get(task))
    return int(split.get("seed_start", -1)) == int(seed_start) and int(split.get("seed_count", -1)) == int(seed_count)


def bool_rate(values) -> float:  # noqa: ANN001
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def make_autoencode_env(task: str):
    try:
        from popgym.envs.autoencode import AutoencodeEasy, AutoencodeHard, AutoencodeMedium
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install popgym to train the Autoencode recurrent baseline.") from exc
    envs = {
        "autoencode_easy": AutoencodeEasy,
        "autoencode_medium": AutoencodeMedium,
        "autoencode_hard": AutoencodeHard,
    }
    if task not in envs:
        raise ValueError(f"Unknown POPGym Autoencode task: {task}")
    return envs[task]()


def parse_obs(obs: Any) -> tuple[int, int]:
    mode, suit = obs
    if hasattr(mode, "item"):
        mode = mode.item()
    if hasattr(suit, "item"):
        suit = suit.item()
    return int(mode), int(suit)


def autoencode_token(mode: int, suit: int) -> int:
    return int(mode) * SUIT_COUNT + int(suit)


def actor_context(*, mode: int, suit: int, previous_obs: tuple[int, int] | None, previous_action: str) -> dict[str, Any]:
    return {
        "popgym_observation": {"mode": int(mode), "suit": int(suit)},
        "previous_popgym_observation": None
        if previous_obs is None
        else {"mode": int(previous_obs[0]), "suit": int(previous_obs[1])},
        "previous_action": str(previous_action),
    }


def contamination_summary(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for idx, context in enumerate(contexts):
        scan = scan_actor_context(context)
        for failure in scan.get("failures", []):
            failures.append({"row": idx, **dict(failure)})
    return {"passed": not failures, "failure_count": len(failures), "failures": failures}


def combine_contamination(train_data: Mapping[str, Any], metrics_by_task: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _ = metrics_by_task
    contamination = mapping(train_data.get("contamination"))
    return {
        "passed": int(contamination.get("failure_count", 0)) == 0,
        "failure_count": int(contamination.get("failure_count", 0)),
        "failures": list(contamination.get("failures", ())),
    }


def write_popgym_autoencode_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    checkpoint = root / "checkpoint"
    learned_checkpoint = root / "learned_memory_checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = save_popgym_recurrent_checkpoint(checkpoint, model=result["model"], manifest=result["checkpoint_manifest"])
    paths = {
        "popgym_autoencode_summary_json": root / "popgym_autoencode_summary.json",
        "popgym_autoencode_summary_md": root / "popgym_autoencode_summary.md",
        "metrics_by_task": root / "metrics_by_task.json",
        "metrics_by_gap": root / "metrics_by_gap.json",
        "memory_curve": root / "memory_curve.json",
        "learned_memory_curve": root / "learned_memory_curve.json",
        "config_manifest": root / "config_manifest.json",
        "checkpoint_manifest": checkpoint / "manifest.json",
        "contamination_scan": root / "contamination_scan.json",
        "training_log": root / "training_log.jsonl",
        "learned_memory_training_log": root / "learned_memory_training_log.jsonl",
    }
    paths["popgym_autoencode_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["popgym_autoencode_summary_md"].write_text(format_summary_markdown(result["summary"]), encoding="utf-8")
    paths["metrics_by_task"].write_text(json.dumps(result["metrics_by_task"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics_by_gap"].write_text(json.dumps(result["metrics_by_gap"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["memory_curve"].write_text(json.dumps(result["memory_curve"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["learned_memory_curve"].write_text(json.dumps(result["learned_memory_curve"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["config_manifest"].write_text(json.dumps(result["config_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(paths["training_log"], result["training_log"])
    write_jsonl(paths["learned_memory_training_log"], result["learned_memory_training_log"])
    paths.update({f"checkpoint_{key}": Path(value) for key, value in checkpoint_paths.items()})
    if result.get("learned_memory_model") is not None:
        learned_checkpoint_paths = save_popgym_associative_checkpoint(
            learned_checkpoint,
            model=result["learned_memory_model"],
            manifest=result["learned_memory_checkpoint_manifest"],
        )
        paths.update({f"learned_memory_checkpoint_{key}": Path(value) for key, value in learned_checkpoint_paths.items()})
    return {key: str(value) for key, value in paths.items()}


def format_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# POPGym Autoencode Recurrent Baseline Summary",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Selection rule: `{summary.get('selection_rule')}`",
        f"- Contamination failures: `{mapping(summary.get('contamination')).get('failure_count')}`",
        "",
        "## Tasks",
        "",
        "| Task | Play success | H_lstm status | H_lstm |",
        "| --- | ---: | --- | ---: |",
    ]
    for task, metrics in mapping(summary.get("metrics_by_task")).items():
        h_lstm = mapping(mapping(summary.get("h_lstm_by_task")).get(task))
        lines.append(
            f"| {task} | {metrics.get('play_success_rate')} "
            f"(short {metrics.get('short_success_rate')}) | {h_lstm.get('status')} | {h_lstm.get('h_lstm')} |"
        )
    curve = mapping(summary.get("memory_curve"))
    if bool(curve.get("enabled", False)):
        lines.extend(
            [
                "",
                "## Fixed-Gap Memory Curve",
                "",
                f"- Status: `{curve.get('status')}`",
                f"- Task: `{curve.get('task')}`",
                f"- Gaps: `{curve.get('gaps')}`",
                "",
                "| Gap | Off | Clean | Shuffled | Content Corrupt | Order Corrupt |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        by_gap = mapping(curve.get("summary_by_gap"))
        for gap in curve.get("gaps", ()):
            row = mapping(by_gap.get(str(gap)))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(gap),
                        str(mapping(row.get("autoencode_memory_off")).get("success_rate")),
                        str(mapping(row.get("autoencode_memory_clean")).get("success_rate")),
                        str(mapping(row.get("autoencode_memory_shuffled")).get("success_rate")),
                        str(mapping(row.get("autoencode_memory_content_corrupt")).get("success_rate")),
                        str(mapping(row.get("autoencode_memory_order_corrupt")).get("success_rate")),
                    ]
                )
                + " |"
            )
        reconciliation = mapping(curve.get("baseline_reconciliation"))
        if reconciliation:
            lines.extend(
                [
                    "",
                    "## Baseline Reconciliation",
                    "",
                    f"- Status: `{reconciliation.get('status')}`",
                    "",
                    "| Gap | Memory-off | Recurrent eval | Delta |",
                    "| ---: | ---: | ---: | ---: |",
                ]
            )
            for row in reconciliation.get("rows", ()):
                item = mapping(row)
                lines.append(
                    f"| {item.get('gap')} | {item.get('memory_off_success_rate')} | "
                    f"{item.get('baseline_eval_success_rate')} | {item.get('absolute_delta')} |"
                )
    learned_curve = mapping(summary.get("learned_memory_curve"))
    if bool(learned_curve.get("enabled", False)):
        lines.extend(
            [
                "",
                "## Learned Memory Curve",
                "",
                f"- Status: `{learned_curve.get('status')}`",
                f"- Task: `{learned_curve.get('task')}`",
                f"- Gaps: `{learned_curve.get('gaps')}`",
                "",
                "| Gap | Off | Hand-Coded | Learned | Shuffled | Content Corrupt | Order Corrupt |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        learned_by_gap = mapping(learned_curve.get("summary_by_gap"))
        for gap in learned_curve.get("gaps", ()):
            row = mapping(learned_by_gap.get(str(gap)))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(gap),
                        str(mapping(row.get("autoencode_memory_off")).get("success_rate")),
                        str(mapping(row.get("autoencode_hand_coded_memory_clean")).get("success_rate")),
                        str(mapping(row.get("autoencode_learned_memory_clean")).get("success_rate")),
                        str(mapping(row.get("autoencode_learned_memory_shuffled")).get("success_rate")),
                        str(mapping(row.get("autoencode_learned_memory_content_corrupt")).get("success_rate")),
                        str(mapping(row.get("autoencode_learned_memory_order_corrupt")).get("success_rate")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--config", default="configs/popgym_autoencode_recurrent.yaml")
    parser.add_argument("--out", default="runs/popgym_autoencode_recurrent")
    args = parser.parse_args(argv)
    result = run_popgym_autoencode_training(load_config(args.config))
    artifacts = write_popgym_autoencode_artifacts(result, args.out)
    print(json.dumps({"decision": result["summary"]["decision"], "artifacts": artifacts, "out": str(Path(args.out).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
