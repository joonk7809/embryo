"""Confirm capacity pressure before learned write-selection."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.memory.distractor_stream import (
    BudgetedFactMemory,
    StreamEpisode,
    StreamFact,
    bayes_admission_indices,
    make_distractor_stream_episode,
    random_admission_indices,
    single_feature_admission_indices,
)


GO_DISTRACTOR_STREAM_PRESSURE_CONFIRMED = "GO_distractor_stream_pressure_confirmed"
GO_DISTRACTOR_STREAM_OBSERVABLE_HEADROOM_CONFIRMED = "GO_distractor_stream_observable_headroom_confirmed"
NO_GO_ORACLE_PRESSURE_FAILURE = "NO_GO_oracle_pressure_failure"
NO_GO_STORE_EVERYTHING_NOT_DEGRADED = "NO_GO_store_everything_not_degraded"
NO_GO_OBSERVABLE_HEADROOM_MISSING = "NO_GO_observable_headroom_missing"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"

ARMS = ("oracle_selection", "bayes_observable_selection", "store_everything_fifo", "random_admission")
PHASE1_ARMS = (*ARMS, "best_single_feature_gate")


def run_distractor_stream_pressure_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    task_cfg = mapping(config.get("task"))
    protocol_cfg = mapping(config.get("protocol"))
    seed_start = int(task_cfg.get("seed_start", 13000))
    seed_count = int(task_cfg.get("seed_count", 64))
    relevant_count = int(task_cfg.get("relevant_count", 8))
    key_space = int(task_cfg.get("key_space", 4096))
    value_space = int(task_cfg.get("value_space", 16))
    cue_model = str(task_cfg.get("cue_model", "symmetric_label_accuracy"))
    cue_true_positive_rate = optional_float(task_cfg.get("cue_true_positive_rate"))
    cue_false_positive_rate = optional_float(task_cfg.get("cue_false_positive_rate"))
    feature_count = int(task_cfg.get("feature_count", 4))
    feature_true_positive_rate = optional_float(task_cfg.get("feature_true_positive_rate"))
    feature_false_positive_rate = optional_float(task_cfg.get("feature_false_positive_rate"))
    distractor_ratios = int_list(task_cfg.get("distractor_ratios"), default=15)
    budgets = int_list(task_cfg.get("budgets"), default=relevant_count)
    p_cues = float_list(task_cfg.get("p_cues"), default=0.5)
    tuning_seed_start = int(task_cfg.get("tuning_seed_start", seed_start))
    tuning_seed_count = int(task_cfg.get("tuning_seed_count", seed_count))
    single_features_by_p_cue = tune_single_feature_gates(
        seed_start=tuning_seed_start,
        seed_count=tuning_seed_count,
        relevant_count=relevant_count,
        distractor_ratios=distractor_ratios,
        budgets=budgets,
        p_cues=p_cues,
        key_space=key_space,
        value_space=value_space,
        cue_model=cue_model,
        cue_true_positive_rate=cue_true_positive_rate,
        cue_false_positive_rate=cue_false_positive_rate,
        feature_count=feature_count,
        feature_true_positive_rate=feature_true_positive_rate,
        feature_false_positive_rate=feature_false_positive_rate,
    )

    episodes: list[dict[str, Any]] = []
    contamination_failures: list[dict[str, Any]] = []
    for distractor_ratio in distractor_ratios:
        for budget in budgets:
            for p_cue in p_cues:
                for seed in range(seed_start, seed_start + seed_count):
                    episode = make_distractor_stream_episode(
                        seed=seed,
                        relevant_count=relevant_count,
                        budget=budget,
                        distractor_ratio=distractor_ratio,
                        p_cue=p_cue,
                        cue_model=cue_model,
                        cue_true_positive_rate=cue_true_positive_rate,
                        cue_false_positive_rate=cue_false_positive_rate,
                        feature_count=feature_count,
                        feature_true_positive_rate=feature_tp_rate(p_cue, feature_true_positive_rate),
                        feature_false_positive_rate=feature_fp_rate(p_cue, feature_false_positive_rate),
                        key_space=key_space,
                        value_space=value_space,
                    )
                    for arm in PHASE1_ARMS:
                        row, failures = run_pressure_arm(
                            episode,
                            arm=arm,
                            single_feature_index=single_features_by_p_cue.get(p_cue, 0),
                        )
                        episodes.append(row)
                        contamination_failures.extend(failures)

    metrics_by_arm = summarize_by_arm(episodes)
    curve = summarize_curve(episodes)
    pairwise = pairwise_metrics(episodes)
    contamination = {"passed": not contamination_failures, "failure_count": len(contamination_failures), "failures": contamination_failures}
    pressure_decision, pressure_reasons = decide_distractor_stream_pressure(
        episodes,
        contamination=contamination,
        protocol_cfg=protocol_cfg,
    )
    headroom_decision, headroom_reasons = decide_observable_headroom(
        episodes,
        contamination=contamination,
        protocol_cfg=protocol_cfg,
        pressure_decision=pressure_decision,
    )
    summary = {
        "decision": headroom_decision,
        "decision_reasons": headroom_reasons,
        "phase0_decision": pressure_decision,
        "phase0_decision_reasons": pressure_reasons,
        "phase1_decision": headroom_decision,
        "phase1_decision_reasons": headroom_reasons,
        "objective": "distractor_stream_capacity_pressure_phase0",
        "boundary": "pressure_confirmation_no_learned_gate_no_policy_training",
        "task": {
            "seed_start": seed_start,
            "seed_count": seed_count,
            "relevant_count": relevant_count,
            "distractor_ratios": distractor_ratios,
            "budgets": budgets,
            "p_cues": p_cues,
            "cue_model": cue_model,
            "cue_definition": cue_definition(
                cue_model=cue_model,
                cue_true_positive_rate=cue_true_positive_rate,
                cue_false_positive_rate=cue_false_positive_rate,
            ),
            "cue_true_positive_rate": cue_true_positive_rate,
            "cue_false_positive_rate": cue_false_positive_rate,
            "feature_count": feature_count,
            "feature_true_positive_rate": feature_true_positive_rate,
            "feature_false_positive_rate": feature_false_positive_rate,
            "feature_definition": feature_definition(
                feature_true_positive_rate=feature_true_positive_rate,
                feature_false_positive_rate=feature_false_positive_rate,
            ),
            "observable_feature_model": "independent_weak_binary_features_conditioned_on_eval_only_relevance",
            "relevant_count_mode": "fixed_exact",
            "key_space": key_space,
            "value_space": value_space,
        },
        "protocol": {
            "arms": list(PHASE1_ARMS),
            "eviction": "fifo",
            "random_admission": "exactly_budget_random_indices",
            "bayes_selection": "top_budget_by_known_observable_feature_likelihood_model",
            "single_feature_selection": "best_feature_index_on_tuning_seed_block",
            "tuning_seed_start": tuning_seed_start,
            "tuning_seed_count": tuning_seed_count,
            "phase": "pressure_confirmation_plus_best_single_feature_reference",
            **json_safe(protocol_cfg),
        },
        "best_single_feature_by_p_cue": {f"{key:.6g}": value for key, value in sorted(single_features_by_p_cue.items())},
        "metrics_by_arm": metrics_by_arm,
        "curve": curve,
        "pairwise": pairwise,
        "fifo_random_diagnostic": fifo_random_diagnostic(episodes, protocol_cfg=protocol_cfg),
        "observable_headroom": observable_headroom(episodes),
        "contamination": contamination,
    }
    return {
        "summary": summary,
        "metrics_by_arm": metrics_by_arm,
        "episodes": episodes,
        "config_manifest": json_safe(config),
        "contamination": contamination,
    }


def tune_single_feature_gates(
    *,
    seed_start: int,
    seed_count: int,
    relevant_count: int,
    distractor_ratios: Sequence[int],
    budgets: Sequence[int],
    p_cues: Sequence[float],
    key_space: int,
    value_space: int,
    cue_model: str,
    cue_true_positive_rate: float | None,
    cue_false_positive_rate: float | None,
    feature_count: int,
    feature_true_positive_rate: float | None,
    feature_false_positive_rate: float | None,
) -> dict[float, int]:
    feature_indices: dict[float, int] = {}
    for p_cue in p_cues:
        scored: list[tuple[float, int]] = []
        for feature_index in range(int(feature_count)):
            rows: list[dict[str, Any]] = []
            for distractor_ratio in distractor_ratios:
                for budget in budgets:
                    for seed in range(int(seed_start), int(seed_start) + int(seed_count)):
                        episode = make_distractor_stream_episode(
                            seed=seed,
                            relevant_count=relevant_count,
                            budget=budget,
                            distractor_ratio=distractor_ratio,
                            p_cue=p_cue,
                            cue_model=cue_model,
                            cue_true_positive_rate=cue_true_positive_rate,
                            cue_false_positive_rate=cue_false_positive_rate,
                            feature_count=feature_count,
                            feature_true_positive_rate=feature_tp_rate(p_cue, feature_true_positive_rate),
                            feature_false_positive_rate=feature_fp_rate(p_cue, feature_false_positive_rate),
                            key_space=key_space,
                            value_space=value_space,
                        )
                        row, _ = run_pressure_arm(episode, arm="best_single_feature_gate", single_feature_index=feature_index)
                        rows.append(row)
            scored.append((mean(row["recall_success_rate"] for row in rows), feature_index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        feature_indices[float(p_cue)] = scored[0][1]
    return feature_indices


def run_pressure_arm(episode: StreamEpisode, *, arm: str, single_feature_index: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    memory = BudgetedFactMemory(budget=episode.budget)
    failures: list[dict[str, Any]] = []
    random_indices = random_admission_indices(seed=episode.seed, stream_length=len(episode.facts), budget=episode.budget) if arm == "random_admission" else frozenset()
    bayes_indices = bayes_admission_indices(episode) if arm == "bayes_observable_selection" else frozenset()
    single_feature_indices = (
        single_feature_admission_indices(episode, feature_index=single_feature_index)
        if arm == "best_single_feature_gate"
        else frozenset()
    )
    for fact in episode.facts:
        context = deployable_fact_context(fact)
        scan = scan_actor_context(context)
        for failure in scan.get("failures", []):
            failures.append({"seed": episode.seed, "arm": arm, "stream_index": fact.stream_index, **dict(failure)})
        if should_admit(
            fact,
            arm=arm,
            random_indices=random_indices,
            bayes_indices=bayes_indices,
            single_feature_indices=single_feature_indices,
        ):
            memory.admit(fact)
    correct = 0
    misses = 0
    for query in episode.queries:
        recalled = memory.recall(query.key)
        if recalled is None:
            misses += 1
        elif recalled.value == query.value:
            correct += 1
    stored = memory.entries()
    relevant_keys = {fact.key for fact in episode.queries}
    relevant_stored = sum(1 for fact in stored if fact.key in relevant_keys)
    return (
        {
            "seed": episode.seed,
            "arm": arm,
            "distractor_ratio": episode.distractor_ratio,
            "budget": episode.budget,
            "p_cue": episode.p_cue,
            "cue_model": episode.cue_model,
            "cue_true_positive_rate": episode.cue_true_positive_rate,
            "cue_false_positive_rate": episode.cue_false_positive_rate,
            "feature_count": episode.feature_count,
            "feature_true_positive_rate": episode.feature_true_positive_rate,
            "feature_false_positive_rate": episode.feature_false_positive_rate,
            "single_feature_index": int(single_feature_index) if arm == "best_single_feature_gate" else None,
            "relevant_count_mode": episode.relevant_count_mode,
            "stream_length": len(episode.facts),
            "relevant_count": len(episode.queries),
            "admission_rate": round(memory.admitted_count / len(episode.facts), 6),
            "admitted_count": memory.admitted_count,
            "occupancy": len(stored),
            "recall_success_rate": round(correct / len(episode.queries), 6),
            "recall_success_count": correct,
            "recall_miss_count": misses,
            "retention_precision": round(relevant_stored / len(stored), 6) if stored else 0.0,
            "retention_recall": round(relevant_stored / len(episode.queries), 6),
            "memory_content_hash": memory.content_hash(),
        },
        failures,
    )


def should_admit(
    fact: StreamFact,
    *,
    arm: str,
    random_indices: frozenset[int],
    bayes_indices: frozenset[int],
    single_feature_indices: frozenset[int],
) -> bool:
    if arm == "oracle_selection":
        return bool(fact.relevant_eval_only)
    if arm == "bayes_observable_selection":
        return int(fact.stream_index) in bayes_indices
    if arm == "store_everything_fifo":
        return True
    if arm == "random_admission":
        return int(fact.stream_index) in random_indices
    if arm == "best_single_feature_gate":
        return int(fact.stream_index) in single_feature_indices
    raise ValueError(f"Unknown distractor-stream arm: {arm}")


def deployable_fact_context(fact: StreamFact) -> dict[str, Any]:
    context = fact.actor_view()
    context["previous_action"] = "admit_or_skip"
    return context


def summarize_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in PHASE1_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        result[arm] = {
            "episode_count": len(arm_rows),
            "mean_recall_success_rate": round(mean(row["recall_success_rate"] for row in arm_rows), 6),
            "mean_retention_precision": round(mean(row["retention_precision"] for row in arm_rows), 6),
            "mean_retention_recall": round(mean(row["retention_recall"] for row in arm_rows), 6),
            "mean_admission_rate": round(mean(row["admission_rate"] for row in arm_rows), 6),
            "mean_occupancy": round(mean(row["occupancy"] for row in arm_rows), 6),
        }
    return result


def summarize_curve(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cell in sorted({cell_key(row) for row in rows}):
        cell_rows = [row for row in rows if cell_key(row) == cell]
        metrics = summarize_by_arm(cell_rows)
        delta = paired_delta_ci(cell_rows, left_arm="oracle_selection", right_arm="store_everything_fifo", metric="recall_success_rate")
        result[cell] = {
            "distractor_ratio": int(cell_rows[0]["distractor_ratio"]),
            "budget": int(cell_rows[0]["budget"]),
            "p_cue": float(cell_rows[0]["p_cue"]),
            "cue_model": str(cell_rows[0].get("cue_model", "")),
            "cue_true_positive_rate": float(cell_rows[0].get("cue_true_positive_rate", 0.0)),
            "cue_false_positive_rate": float(cell_rows[0].get("cue_false_positive_rate", 0.0)),
            "stream_length": int(cell_rows[0]["stream_length"]),
            "metrics_by_arm": metrics,
            "oracle_minus_fifo_recall": delta,
            "bayes_minus_single_feature_recall": paired_delta_ci(cell_rows, left_arm="bayes_observable_selection", right_arm="best_single_feature_gate", metric="recall_success_rate"),
            "oracle_minus_bayes_recall": paired_delta_ci(cell_rows, left_arm="oracle_selection", right_arm="bayes_observable_selection", metric="recall_success_rate"),
            "single_feature_minus_fifo_recall": paired_delta_ci(cell_rows, left_arm="best_single_feature_gate", right_arm="store_everything_fifo", metric="recall_success_rate"),
        }
    return result


def pairwise_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "oracle_minus_fifo_recall": paired_delta_ci(rows, left_arm="oracle_selection", right_arm="store_everything_fifo", metric="recall_success_rate"),
        "oracle_minus_random_recall": paired_delta_ci(rows, left_arm="oracle_selection", right_arm="random_admission", metric="recall_success_rate"),
        "fifo_minus_random_recall": paired_delta_ci(rows, left_arm="store_everything_fifo", right_arm="random_admission", metric="recall_success_rate"),
        "bayes_minus_single_feature_recall": paired_delta_ci(rows, left_arm="bayes_observable_selection", right_arm="best_single_feature_gate", metric="recall_success_rate"),
        "oracle_minus_bayes_recall": paired_delta_ci(rows, left_arm="oracle_selection", right_arm="bayes_observable_selection", metric="recall_success_rate"),
        "single_feature_minus_fifo_recall": paired_delta_ci(rows, left_arm="best_single_feature_gate", right_arm="store_everything_fifo", metric="recall_success_rate"),
    }


def fifo_random_diagnostic(rows: Sequence[Mapping[str, Any]], *, protocol_cfg: Mapping[str, Any]) -> dict[str, Any]:
    primary_rows = primary_cell_rows(rows, protocol_cfg)
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in primary_rows:
        if row["arm"] in {"store_everything_fifo", "random_admission"}:
            by_seed.setdefault(int(row["seed"]), {})[str(row["arm"])] = row
    same_episode_recall = 0
    different_episode_recall = 0
    same_memory_hash = 0
    fifo_success_total = 0
    random_success_total = 0
    for pair in by_seed.values():
        if "store_everything_fifo" not in pair or "random_admission" not in pair:
            continue
        fifo = pair["store_everything_fifo"]
        random = pair["random_admission"]
        fifo_success_total += int(fifo["recall_success_count"])
        random_success_total += int(random["recall_success_count"])
        if float(fifo["recall_success_rate"]) == float(random["recall_success_rate"]):
            same_episode_recall += 1
        else:
            different_episode_recall += 1
        if str(fifo["memory_content_hash"]) == str(random["memory_content_hash"]):
            same_memory_hash += 1
    return {
        "primary_seed_count": len(by_seed),
        "same_episode_recall_count": same_episode_recall,
        "different_episode_recall_count": different_episode_recall,
        "same_memory_hash_count": same_memory_hash,
        "fifo_success_total": fifo_success_total,
        "random_success_total": random_success_total,
        "interpretation": "Relevant positions are randomized, so FIFO's retained suffix is position-uniform in expectation; equality of aggregate recall is not code-path sharing when hashes differ.",
    }


def observable_headroom(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for p_cue in sorted({float(row["p_cue"]) for row in rows}):
        p_rows = [row for row in rows if float(row["p_cue"]) == p_cue]
        result[f"{p_cue:.6g}"] = {
            "bayes_minus_single_feature_recall": paired_delta_ci(p_rows, left_arm="bayes_observable_selection", right_arm="best_single_feature_gate", metric="recall_success_rate"),
            "oracle_minus_bayes_recall": paired_delta_ci(p_rows, left_arm="oracle_selection", right_arm="bayes_observable_selection", metric="recall_success_rate"),
            "single_feature_minus_fifo_recall": paired_delta_ci(p_rows, left_arm="best_single_feature_gate", right_arm="store_everything_fifo", metric="recall_success_rate"),
            "mean_by_arm": summarize_by_arm(p_rows),
        }
    return result


def cue_definition(
    *,
    cue_model: str,
    cue_true_positive_rate: float | None,
    cue_false_positive_rate: float | None,
) -> str:
    if cue_model == "symmetric_label_accuracy":
        return "For each p_cue, P(cue=1|relevant)=p_cue and P(cue=1|distractor)=1-p_cue."
    if cue_model == "explicit_rates":
        return f"P(cue=1|relevant)={cue_true_positive_rate}; P(cue=1|distractor)={cue_false_positive_rate}."
    return str(cue_model)


def feature_definition(
    *,
    feature_true_positive_rate: float | None,
    feature_false_positive_rate: float | None,
) -> str:
    if feature_true_positive_rate is None and feature_false_positive_rate is None:
        return "For each p_cue and each weak feature, P(feature=1|relevant)=p_cue and P(feature=1|distractor)=1-p_cue."
    return f"For each weak feature, P(feature=1|relevant)={feature_true_positive_rate}; P(feature=1|distractor)={feature_false_positive_rate}."


def decide_observable_headroom(
    rows: Sequence[Mapping[str, Any]],
    *,
    contamination: Mapping[str, Any],
    protocol_cfg: Mapping[str, Any],
    pressure_decision: str,
) -> tuple[str, list[str]]:
    if pressure_decision != GO_DISTRACTOR_STREAM_PRESSURE_CONFIRMED:
        return pressure_decision, ["phase0_pressure_not_confirmed"]
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count>0"]
    primary_rows = primary_cell_rows(rows, protocol_cfg)
    gap = paired_delta_ci(
        primary_rows,
        left_arm="bayes_observable_selection",
        right_arm="best_single_feature_gate",
        metric="recall_success_rate",
    )
    min_gap = float(protocol_cfg.get("min_bayes_single_feature_gap", 0.10))
    if float(gap["mean"]) < min_gap or float(gap["ci95_low"]) <= 0.0:
        return NO_GO_OBSERVABLE_HEADROOM_MISSING, [
            f"bayes_minus_single_feature={float(gap['mean']):.6g}<min_bayes_single_feature_gap={min_gap:.6g}"
            if float(gap["mean"]) < min_gap
            else f"bayes_minus_single_feature_ci95_low={float(gap['ci95_low']):.6g}<=0"
        ]
    return GO_DISTRACTOR_STREAM_OBSERVABLE_HEADROOM_CONFIRMED, [
        f"bayes_minus_single_feature={float(gap['mean']):.6g}",
        f"bayes_minus_single_feature_ci95_low={float(gap['ci95_low']):.6g}",
    ]


def decide_distractor_stream_pressure(
    rows: Sequence[Mapping[str, Any]],
    *,
    contamination: Mapping[str, Any],
    protocol_cfg: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count>0"]
    primary_rows = primary_cell_rows(rows, protocol_cfg)
    oracle = [row for row in primary_rows if row["arm"] == "oracle_selection"]
    oracle_recall = mean(row["recall_success_rate"] for row in oracle)
    min_oracle = float(protocol_cfg.get("min_oracle_recall", 0.90))
    if oracle_recall < min_oracle:
        return NO_GO_ORACLE_PRESSURE_FAILURE, [f"oracle_recall={oracle_recall:.6g}<min_oracle_recall={min_oracle:.6g}"]
    gap = paired_delta_ci(primary_rows, left_arm="oracle_selection", right_arm="store_everything_fifo", metric="recall_success_rate")
    min_gap = float(protocol_cfg.get("min_oracle_fifo_gap", 0.30))
    if float(gap["mean"]) < min_gap or float(gap["ci95_low"]) <= 0.0:
        return NO_GO_STORE_EVERYTHING_NOT_DEGRADED, [
            f"oracle_minus_fifo={float(gap['mean']):.6g}<min_oracle_fifo_gap={min_gap:.6g}"
            if float(gap["mean"]) < min_gap
            else f"oracle_minus_fifo_ci95_low={float(gap['ci95_low']):.6g}<=0"
        ]
    return GO_DISTRACTOR_STREAM_PRESSURE_CONFIRMED, [
        f"oracle_recall={oracle_recall:.6g}",
        f"oracle_minus_fifo={float(gap['mean']):.6g}",
        f"oracle_minus_fifo_ci95_low={float(gap['ci95_low']):.6g}",
    ]


def primary_cell_rows(rows: Sequence[Mapping[str, Any]], protocol_cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ratios = sorted({int(row["distractor_ratio"]) for row in rows})
    budgets = sorted({int(row["budget"]) for row in rows})
    p_cues = sorted({float(row["p_cue"]) for row in rows})
    ratio = int(protocol_cfg.get("primary_distractor_ratio", max(ratios)))
    budget = int(protocol_cfg.get("primary_budget", min(budgets)))
    p_cue = float(protocol_cfg.get("primary_p_cue", p_cues[0]))
    return [
        row
        for row in rows
        if int(row["distractor_ratio"]) == ratio and int(row["budget"]) == budget and float(row["p_cue"]) == p_cue
    ]


def paired_delta_ci(rows: Sequence[Mapping[str, Any]], *, left_arm: str, right_arm: str, metric: str) -> dict[str, Any]:
    by_key: dict[tuple[int, int, int, float], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["distractor_ratio"]), int(row["budget"]), float(row["p_cue"]))
        by_key.setdefault(key, {})[str(row["arm"])] = row
    deltas = [
        float(pair[left_arm][metric]) - float(pair[right_arm][metric])
        for pair in by_key.values()
        if left_arm in pair and right_arm in pair
    ]
    if not deltas:
        return {"count": 0, "mean": 0.0, "ci95": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    value = mean(deltas)
    ci = ci95(deltas)
    return {
        "count": len(deltas),
        "mean": round(value, 6),
        "ci95": round(ci, 6),
        "ci95_low": round(value - ci, 6),
        "ci95_high": round(value + ci, 6),
    }


def write_distractor_stream_pressure_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "distractor_stream_pressure_summary.json"
    metrics_path = out_path / "metrics_by_arm.json"
    episodes_path = out_path / "distractor_stream_pressure_episodes.jsonl"
    manifest_path = out_path / "config_manifest.json"
    contamination_path = out_path / "contamination_scan.json"
    summary_path.write_text(json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(result["metrics_by_arm"], indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(result["config_manifest"], indent=2, sort_keys=True), encoding="utf-8")
    contamination_path.write_text(json.dumps(result["contamination"], indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(episodes_path, result["episodes"])
    return {
        "summary_json": str(summary_path),
        "metrics_by_arm": str(metrics_path),
        "episodes": str(episodes_path),
        "config_manifest": str(manifest_path),
        "contamination_scan": str(contamination_path),
    }


def cell_key(row: Mapping[str, Any]) -> str:
    return f"R={int(row['distractor_ratio'])}|B={int(row['budget'])}|p_cue={float(row['p_cue']):.3f}"


def mean(values: Sequence[Any]) -> float:
    clean = [float(value) for value in values]
    return sum(clean) / len(clean) if clean else 0.0


def ci95(values: Sequence[float]) -> float:
    clean = [float(value) for value in values]
    if len(clean) <= 1:
        return 0.0
    avg = mean(clean)
    variance = sum((value - avg) ** 2 for value in clean) / (len(clean) - 1)
    return 1.96 * math.sqrt(variance / len(clean))


def int_list(value: Any, *, default: int) -> list[int]:
    if value is None:
        return [int(default)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def float_list(value: Any, *, default: float) -> list[float]:
    if value is None:
        return [float(default)]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def feature_tp_rate(p_cue: float, configured: float | None) -> float:
    return float(p_cue) if configured is None else float(configured)


def feature_fp_rate(p_cue: float, configured: float | None) -> float:
    return 1.0 - float(p_cue) if configured is None else float(configured)


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distractor_stream_pressure.yaml")
    parser.add_argument("--out", default="runs/distractor_stream_pressure")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_distractor_stream_pressure_probe(config)
    artifacts = write_distractor_stream_pressure_artifacts(result, args.out)
    print(json.dumps({"decision": result["summary"]["decision"], "out": str(Path(args.out).resolve()), "artifacts": artifacts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
