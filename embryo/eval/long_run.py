"""Long-run protocol metrics.

These helpers summarize persistent-world episodes without treating reward as an
actor signal. Runtime outcomes and backend info are kept under `eval_only` in
the artifact schema.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from embryo.eval.memory_grounded_score import ScoreConfig, compute_memory_grounded_scores


GO_LONG_RUN_PROTOCOL_SUPPORTED = "GO_long_run_protocol_supported"
NO_GO_PROTOCOL_ARTIFACT_FAILURE = "NO_GO_protocol_artifact_failure"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"
NOT_EVALUABLE_NO_MEMORY_ANCHORS = "NOT_EVALUABLE_no_memory_anchors"
NOT_EVALUABLE_RUNTIME_UNAVAILABLE = "NOT_EVALUABLE_runtime_unavailable"
GO_MEMORY_EVALUABLE_SEED_BLOCK = "GO_memory_evaluable_seed_block"
NOT_EVALUABLE_INSUFFICIENT_ANCHOR_COVERAGE = "NOT_EVALUABLE_insufficient_anchor_coverage"
GO_WATER_RECALL_EVALUABLE = "GO_water_recall_evaluable"
GO_BENCH_RECALL_EVALUABLE = "GO_bench_recall_evaluable"
GO_PASSIVE_MATCH_EVALUABLE = "GO_passive_match_evaluable"
GO_POPGYM_REPEAT_FIRST_EVALUABLE = "GO_popgym_repeat_first_evaluable"
INCONCLUSIVE_LOW_ERE = "INCONCLUSIVE_LOW_ERE"

FORWARD_CLEAN_ARM = "query_memory_clean"
FORWARD_CORRUPTED_CONTROL_ARMS = (
    "query_memory_shuffled",
    "query_memory_stale",
    "query_memory_wrong_binding",
)
FORWARD_CONTROL_ARMS = (
    *FORWARD_CORRUPTED_CONTROL_ARMS,
    "no_memory",
)
WATER_RECALL_CLEAN_ARM = "water_recall_clean"
WATER_RECALL_CONTROL_ARMS = (
    "water_recall_off",
    "water_recall_shuffled",
    "water_recall_stale",
    "water_recall_wrong_binding",
)
BENCH_RECALL_CLEAN_ARM = "bench_recall_clean"
BENCH_RECALL_CONTROL_ARMS = (
    "bench_recall_off",
    "bench_recall_shuffled",
    "bench_recall_stale",
    "bench_recall_wrong_binding",
)
PASSIVE_MATCH_CLEAN_ARM = "passive_match_clean"
PASSIVE_MATCH_CONTROL_ARMS = (
    "passive_match_off",
    "passive_match_shuffled",
    "passive_match_stale",
    "passive_match_wrong_binding",
)
POPGYM_REPEAT_FIRST_CLEAN_ARM = "popgym_repeat_first_clean"
POPGYM_REPEAT_FIRST_CONTROL_ARMS = (
    "popgym_repeat_first_off",
    "popgym_repeat_first_shuffled",
    "popgym_repeat_first_stale",
    "popgym_repeat_first_wrong_binding",
)


def episode_record(
    *,
    actor: Mapping[str, Any],
    ticks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a namespaced episode artifact row."""
    metric_values = episode_metrics(actor=actor, ticks=ticks)
    eval_values = eval_only_episode_fields(ticks)
    contamination = contamination_summary(ticks)
    return {
        "actor": dict(actor),
        "metrics": metric_values,
        "eval_only": eval_values,
        "contamination": contamination,
    }


def episode_metrics(*, actor: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    horizon = int(actor.get("horizon", 0))
    actions = [str(nested(row, "actor", "action", "")) for row in ticks]
    movement_actions = [action for action in actions if is_movement_action(action)]
    done = bool(ticks and nested(ticks[-1], "eval_only", "done_eval_only", False))
    action_entropy = normalized_entropy(actions)
    movement_entropy = normalized_entropy(movement_actions)
    return {
        "survival_steps": len(ticks),
        "reached_horizon": len(ticks) >= horizon,
        "done": done,
        "death_cause": death_cause(ticks),
        "valid_action_rate": round(bool_rate(nested(row, "metrics", "valid_action", False) for row in ticks), 4),
        "invalid_or_unknown_action_rate": round(
            bool_rate(nested(row, "metrics", "invalid_or_unknown_action", False) for row in ticks),
            4,
        ),
        "repeated_action_loop_rate": round(
            bool_rate(nested(row, "metrics", "repeated_action_loop", False) for row in ticks),
            4,
        ),
        "max_stuck_duration": max_stuck_duration(actions),
        "action_entropy_proxy": None if action_entropy is None else round(action_entropy, 4),
        "movement_entropy_proxy": None if movement_entropy is None else round(movement_entropy, 4),
        "movement_entropy_status": "computed" if movement_entropy is not None else "not_inferable",
        "event_self_trigger_rate": round(
            bool_rate(nested(row, "metrics", "event_self_triggered", False) for row in ticks),
            4,
        ),
        "memory_anchor_count": sum(int(bool(nested(row, "metrics", "memory_anchor_critical", False))) for row in ticks),
        "query_used_cached_fact_rate": round(
            bool_rate(nested(row, "metrics", "query_used_cached_fact", False) for row in ticks),
            4,
        ),
        "contamination_failure_count": sum(int(nested(row, "contamination", "failure_count", 0)) for row in ticks),
    }


def eval_only_episode_fields(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    achievements: set[str] = set()
    health_values: list[float] = []
    food_values: list[float] = []
    drink_values: list[float] = []
    final_inventory = None
    for row in ticks:
        eval_row = row.get("eval_only", {})
        if not isinstance(eval_row, Mapping):
            continue
        achievements.update(str(item) for item in eval_row.get("achievements_eval_only", ()) if item is not None)
        append_numeric(health_values, eval_row.get("health_eval_only"))
        append_numeric(food_values, eval_row.get("food_eval_only"))
        append_numeric(drink_values, eval_row.get("drink_eval_only"))
        if eval_row.get("backend_inventory_eval_only") is not None:
            final_inventory = eval_row.get("backend_inventory_eval_only")
    return {
        "reward_sum_eval_only": round(
            sum(numeric(nested(row, "eval_only", "reward_delta_eval_only", 0.0)) for row in ticks),
            4,
        ),
        "achievement_count_eval_only": len(achievements),
        "unique_achievements_eval_only": sorted(achievements),
        "health_trend_eval_only": trend(health_values),
        "food_trend_eval_only": trend(food_values),
        "drink_trend_eval_only": trend(drink_values),
        "backend_inventory_eval_only": final_inventory,
    }


def summarize_long_run_protocol(
    *,
    protocol_manifest: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    ticks: Sequence[Mapping[str, Any]],
    contamination: Mapping[str, Any],
    runtime_unavailable: Mapping[str, Any] | None = None,
    artifact_failure: bool = False,
    deterministic_replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the top-level long-run summary."""
    memory = memory_grounded_protocol_score(ticks, enabled=metrics_enabled(protocol_manifest, "memory_grounded_score"))
    coverage = anchor_coverage_summary(ticks, protocol_manifest=protocol_manifest)
    determinism = dict(deterministic_replay or {"enabled": False, "passed": None})
    memory_evaluability = memory_evaluability_decision(coverage)
    decision = decision_from_protocol(
        contamination=contamination,
        memory=memory,
        runtime_unavailable=runtime_unavailable,
        artifact_failure=artifact_failure,
        deterministic_replay=determinism,
    )
    return {
        "decision": decision,
        "memory_evaluability_decision": memory_evaluability,
        "protocol": public_protocol_summary(protocol_manifest),
        "episode_count": len(episodes),
        "tick_count": len(ticks),
        "summary_by_arm": summarize_by_key(episodes, "arm"),
        "summary_by_horizon": summarize_by_key(episodes, "horizon"),
        "memory_grounded_score": memory,
        "anchor_coverage": coverage,
        "passive_match": passive_match_protocol_summary(ticks, protocol_manifest=protocol_manifest),
        "popgym_repeat_first": popgym_repeat_first_protocol_summary(ticks, protocol_manifest=protocol_manifest),
        "water_recall": water_recall_protocol_summary(ticks, protocol_manifest=protocol_manifest),
        "bench_recall": bench_recall_protocol_summary(ticks, protocol_manifest=protocol_manifest),
        "deterministic_replay": determinism,
        "contamination": dict(contamination),
        "runtime_unavailable": runtime_unavailable,
        "artifact_failure": bool(artifact_failure),
    }


def memory_grounded_protocol_score(ticks: Sequence[Mapping[str, Any]], *, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "disabled"}
    score_rows = memory_score_rows(ticks)
    if not score_rows:
        return {"enabled": True, "status": NOT_EVALUABLE_NO_MEMORY_ANCHORS, "anchor_count": 0}
    result = compute_memory_grounded_scores(
        score_rows,
        config=ScoreConfig(clean_arm=FORWARD_CLEAN_ARM, control_arms=FORWARD_CONTROL_ARMS, followthrough_window=8),
    )
    status = "computed"
    if result.get("anchor_count", 0) == 0:
        status = NOT_EVALUABLE_NO_MEMORY_ANCHORS
    return {
        "enabled": True,
        "status": status,
        "anchor_count": int(result.get("anchor_count", 0)),
        "score": result,
    }


def memory_score_rows(ticks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ticks:
        actor = row.get("actor", {})
        metrics = row.get("metrics", {})
        eval_only = row.get("eval_only", {})
        if not isinstance(actor, Mapping) or not isinstance(metrics, Mapping) or not isinstance(eval_only, Mapping):
            continue
        arm = str(actor.get("arm", ""))
        if arm not in {FORWARD_CLEAN_ARM, *FORWARD_CONTROL_ARMS}:
            continue
        action = str(actor.get("action", ""))
        progress = metrics.get("memory_followthrough_delta")
        if progress is None:
            progress = eval_only.get("reward_delta_eval_only", 0.0)
        anchor_critical = bool(metrics.get("resource_memory_critical", False) or metrics.get("memory_anchor_critical", False))
        route_preserved = bool(metrics.get("resource_route_preserved", False) or metrics.get("memory_anchor_critical", False))
        rows.append(
            {
                "arm": arm,
                "seed": int(actor.get("seed", 0)),
                "episode_id": str(actor.get("episode_id", "episode")),
                "episode_index": int(actor.get("episode_index", 0)),
                "tick": int(actor.get("tick", 0)),
                "action": action,
                "selected_action": {
                    "action_name": action,
                    "invalid_or_unknown": bool(metrics.get("invalid_or_unknown_action", False)),
                },
                "resource_memory_critical": anchor_critical,
                "resource_route_preserved": route_preserved,
                "memory_anchor_critical": bool(metrics.get("memory_anchor_critical", False)),
                "anchor_family": str(metrics.get("anchor_family", "")),
                "anchor_fact_age": int(metrics.get("anchor_fact_age", 0)),
                "anchor_currently_visible": bool(metrics.get("anchor_currently_visible", False)),
                "query_used_cached_fact": bool(metrics.get("query_used_cached_fact", False)),
                "fallback_triggered": bool(metrics.get("fallback_triggered", False)),
                "event_self_triggered": bool(metrics.get("event_self_triggered", False)),
                "repeated_action_loop": bool(metrics.get("repeated_action_loop", False)),
                "diagnostic_progress_delta_teacher_only": numeric(progress),
                "action_entropy_proxy": numeric(metrics.get("action_entropy_proxy", 0.0)),
                "exploration_bin": metrics.get("exploration_bin"),
            }
        )
    return rows


def passive_match_protocol_summary(
    ticks: Sequence[Mapping[str, Any]],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not passive_match_present(protocol_manifest, ticks):
        return {"enabled": False, "status": "disabled"}
    rows = [
        row
        for row in ticks
        if isinstance(row.get("actor", {}), Mapping)
        and isinstance(row.get("metrics", {}), Mapping)
        and str(row["actor"].get("arm", "")) in {PASSIVE_MATCH_CLEAN_ARM, *PASSIVE_MATCH_CONTROL_ARMS}
    ]
    by_key = {
        (
            str(row["actor"].get("arm", "")),
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
            int(row["actor"].get("tick", 0)),
        ): row
        for row in rows
    }
    clean_ere_rows = [row for row in rows if str(row["actor"].get("arm", "")) == PASSIVE_MATCH_CLEAN_ARM and bool(row["metrics"].get("passive_match_ere", False))]
    episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in rows
        if str(row["actor"].get("arm", "")) == PASSIVE_MATCH_CLEAN_ARM
    }
    ere_episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in clean_ere_rows
    }
    requirements = passive_match_requirements(protocol_manifest)
    ere_episode_rate = round(len(ere_episode_keys) / len(episode_keys), 4) if episode_keys else 0.0
    status = GO_PASSIVE_MATCH_EVALUABLE
    if len(clean_ere_rows) < int(requirements["min_ere_count"]) or ere_episode_rate < float(requirements["min_ere_episode_rate"]):
        status = INCONCLUSIVE_LOW_ERE
    return {
        "enabled": True,
        "status": status,
        "requirements": requirements,
        "clean_ere_count": len(clean_ere_rows),
        "clean_ere_episode_count": len(ere_episode_keys),
        "clean_episode_count": len(episode_keys),
        "clean_ere_episode_rate": ere_episode_rate,
        "summary_by_arm": passive_match_summary_by_arm(rows),
        "contrasts": passive_match_contrasts(clean_ere_rows, by_key),
        "per_seed": passive_match_per_seed(rows),
    }


def passive_match_present(protocol_manifest: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]]) -> bool:
    arms = set(str(arm) for arm in protocol_manifest.get("arms", ()))
    if PASSIVE_MATCH_CLEAN_ARM in arms or any(arm in arms for arm in PASSIVE_MATCH_CONTROL_ARMS):
        return True
    return any(bool(nested(row, "metrics", "passive_match_ere", False)) for row in ticks)


def passive_match_requirements(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_manifest.get("protocol", {})
    passive = protocol.get("passive_match", {}) if isinstance(protocol, Mapping) else {}
    passive = passive if isinstance(passive, Mapping) else {}
    return {
        "min_ere_count": int(passive.get("min_ere_count", 30)),
        "min_ere_episode_rate": float(passive.get("min_ere_episode_rate", 0.5)),
        "recall_window": int(passive.get("recall_window", 1)),
    }


def passive_match_summary_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["actor"].get("arm", "unknown"))].append(row)
    return {arm: passive_match_arm_summary(items) for arm, items in sorted(grouped.items())}


def passive_match_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ere_rows = [row for row in rows if bool(row["metrics"].get("passive_match_ere", False))]
    final_rows = [row for row in rows if row.get("eval_only", {}).get("passive_match_success_eval_only") is not None]
    actions = [str(row["actor"].get("action", "")) for row in ere_rows]
    return {
        "tick_count": len(rows),
        "ere_count": len(ere_rows),
        "cue_visible_count": sum(int(bool(row["metrics"].get("passive_match_cue_visible", False))) for row in rows),
        "choice_visible_count": sum(int(bool(row["metrics"].get("passive_match_choice_visible", False))) for row in rows),
        "recall_active_rate": round(bool_rate(row["metrics"].get("passive_match_recall_active", False) for row in ere_rows), 4),
        "recall_consistent_action_rate": round(
            bool_rate(row["metrics"].get("passive_match_recall_consistent_action", False) for row in ere_rows),
            4,
        ),
        "success_rate_eval_only": round(bool_rate(row["eval_only"].get("passive_match_success_eval_only", False) for row in final_rows), 4),
        "mean_reward_on_final_eval_only": round(mean(row["eval_only"].get("reward_delta_eval_only", 0.0) for row in final_rows), 4),
        "mean_fact_age_on_ere": round(mean(row["metrics"].get("passive_match_fact_age") for row in ere_rows), 4),
        "action_distribution_on_ere": dict(sorted(Counter(actions).items())),
    }


def passive_match_contrasts(
    clean_ere_rows: Sequence[Mapping[str, Any]],
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    clean_consistency = bool_rate(nested(row, "metrics", "passive_match_recall_consistent_action", False) for row in clean_ere_rows)
    clean_success = bool_rate(nested(row, "eval_only", "passive_match_success_eval_only", False) for row in clean_ere_rows)
    for arm in PASSIVE_MATCH_CONTROL_ARMS:
        comparable: list[Mapping[str, Any]] = []
        for row in clean_ere_rows:
            actor = row["actor"]
            key = (
                arm,
                int(actor.get("seed", 0)),
                str(actor.get("episode_id", actor.get("episode_index", "0"))),
                int(actor.get("tick", 0)),
            )
            if key in by_key:
                comparable.append(by_key[key])
        control_consistency = bool_rate(nested(row, "metrics", "passive_match_recall_consistent_action", False) for row in comparable)
        control_success = bool_rate(nested(row, "eval_only", "passive_match_success_eval_only", False) for row in comparable)
        result[arm] = {
            "comparable_ere_count": len(comparable),
            "clean_recall_consistent_action_rate": round(clean_consistency, 4),
            "control_recall_consistent_action_rate": round(control_consistency, 4),
            "clean_minus_control_recall_consistent_action_rate": round(clean_consistency - control_consistency, 4),
            "clean_success_rate_eval_only": round(clean_success, 4),
            "control_success_rate_eval_only": round(control_success, 4),
            "clean_minus_control_success_rate_eval_only": round(clean_success - control_success, 4),
        }
    return result


def passive_match_per_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["actor"].get("seed", 0))].append(row)
    return {
        str(seed): {
            "clean_ere_count": sum(
                int(str(row["actor"].get("arm", "")) == PASSIVE_MATCH_CLEAN_ARM and bool(row["metrics"].get("passive_match_ere", False)))
                for row in items
            ),
            "cue_visible_count": sum(int(bool(row["metrics"].get("passive_match_cue_visible", False))) for row in items),
            "choice_visible_count": sum(int(bool(row["metrics"].get("passive_match_choice_visible", False))) for row in items),
        }
        for seed, items in sorted(grouped.items())
    }


def popgym_repeat_first_protocol_summary(
    ticks: Sequence[Mapping[str, Any]],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not popgym_repeat_first_present(protocol_manifest, ticks):
        return {"enabled": False, "status": "disabled"}
    rows = [
        row
        for row in ticks
        if isinstance(row.get("actor", {}), Mapping)
        and isinstance(row.get("metrics", {}), Mapping)
        and str(row["actor"].get("arm", "")) in {POPGYM_REPEAT_FIRST_CLEAN_ARM, *POPGYM_REPEAT_FIRST_CONTROL_ARMS}
    ]
    by_key = {
        (
            str(row["actor"].get("arm", "")),
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
            int(row["actor"].get("tick", 0)),
        ): row
        for row in rows
    }
    clean_ere_rows = [
        row
        for row in rows
        if str(row["actor"].get("arm", "")) == POPGYM_REPEAT_FIRST_CLEAN_ARM
        and bool(row["metrics"].get("popgym_repeat_first_ere", False))
    ]
    episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in rows
        if str(row["actor"].get("arm", "")) == POPGYM_REPEAT_FIRST_CLEAN_ARM
    }
    ere_episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in clean_ere_rows
    }
    requirements = popgym_repeat_first_requirements(protocol_manifest)
    ere_episode_rate = round(len(ere_episode_keys) / len(episode_keys), 4) if episode_keys else 0.0
    status = GO_POPGYM_REPEAT_FIRST_EVALUABLE
    if len(clean_ere_rows) < int(requirements["min_ere_count"]) or ere_episode_rate < float(requirements["min_ere_episode_rate"]):
        status = INCONCLUSIVE_LOW_ERE
    return {
        "enabled": True,
        "status": status,
        "requirements": requirements,
        "clean_ere_count": len(clean_ere_rows),
        "clean_ere_episode_count": len(ere_episode_keys),
        "clean_episode_count": len(episode_keys),
        "clean_ere_episode_rate": ere_episode_rate,
        "summary_by_arm": popgym_repeat_first_summary_by_arm(rows),
        "contrasts": popgym_repeat_first_contrasts(clean_ere_rows, by_key),
        "per_seed": popgym_repeat_first_per_seed(rows),
    }


def popgym_repeat_first_present(protocol_manifest: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]]) -> bool:
    arms = set(str(arm) for arm in protocol_manifest.get("arms", ()))
    if POPGYM_REPEAT_FIRST_CLEAN_ARM in arms or any(arm in arms for arm in POPGYM_REPEAT_FIRST_CONTROL_ARMS):
        return True
    return any(bool(nested(row, "metrics", "popgym_repeat_first_ere", False)) for row in ticks)


def popgym_repeat_first_requirements(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_manifest.get("protocol", {})
    repeat = protocol.get("popgym_repeat_first", {}) if isinstance(protocol, Mapping) else {}
    repeat = repeat if isinstance(repeat, Mapping) else {}
    return {
        "min_ere_count": int(repeat.get("min_ere_count", 30)),
        "min_ere_episode_rate": float(repeat.get("min_ere_episode_rate", 0.5)),
        "recall_window": int(repeat.get("recall_window", 1)),
    }


def popgym_repeat_first_summary_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["actor"].get("arm", "unknown"))].append(row)
    return {arm: popgym_repeat_first_arm_summary(items) for arm, items in sorted(grouped.items())}


def popgym_repeat_first_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ere_rows = [row for row in rows if bool(row["metrics"].get("popgym_repeat_first_ere", False))]
    actions = [str(row["actor"].get("action", "")) for row in ere_rows]
    return {
        "tick_count": len(rows),
        "ere_count": len(ere_rows),
        "recall_active_rate": round(bool_rate(row["metrics"].get("popgym_repeat_first_recall_active", False) for row in ere_rows), 4),
        "recall_consistent_action_rate": round(
            bool_rate(row["metrics"].get("popgym_repeat_first_recall_consistent_action", False) for row in ere_rows),
            4,
        ),
        "success_rate_eval_only": round(bool_rate(row["eval_only"].get("popgym_success_eval_only", False) for row in ere_rows), 4),
        "mean_reward_eval_only": round(mean(row["eval_only"].get("reward_delta_eval_only", 0.0) for row in ere_rows), 4),
        "mean_fact_age_on_ere": round(mean(row["metrics"].get("popgym_repeat_first_fact_age") for row in ere_rows), 4),
        "current_match_rate_on_ere": round(bool_rate(row["metrics"].get("popgym_repeat_first_current_matches_target", False) for row in ere_rows), 4),
        "action_distribution_on_ere": dict(sorted(Counter(actions).items())),
    }


def popgym_repeat_first_contrasts(
    clean_ere_rows: Sequence[Mapping[str, Any]],
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    clean_consistency = bool_rate(nested(row, "metrics", "popgym_repeat_first_recall_consistent_action", False) for row in clean_ere_rows)
    clean_success = bool_rate(nested(row, "eval_only", "popgym_success_eval_only", False) for row in clean_ere_rows)
    for arm in POPGYM_REPEAT_FIRST_CONTROL_ARMS:
        comparable: list[Mapping[str, Any]] = []
        for row in clean_ere_rows:
            actor = row["actor"]
            key = (
                arm,
                int(actor.get("seed", 0)),
                str(actor.get("episode_id", actor.get("episode_index", "0"))),
                int(actor.get("tick", 0)),
            )
            if key in by_key:
                comparable.append(by_key[key])
        control_consistency = bool_rate(nested(row, "metrics", "popgym_repeat_first_recall_consistent_action", False) for row in comparable)
        control_success = bool_rate(nested(row, "eval_only", "popgym_success_eval_only", False) for row in comparable)
        result[arm] = {
            "comparable_ere_count": len(comparable),
            "clean_recall_consistent_action_rate": round(clean_consistency, 4),
            "control_recall_consistent_action_rate": round(control_consistency, 4),
            "clean_minus_control_recall_consistent_action_rate": round(clean_consistency - control_consistency, 4),
            "clean_success_rate_eval_only": round(clean_success, 4),
            "control_success_rate_eval_only": round(control_success, 4),
            "clean_minus_control_success_rate_eval_only": round(clean_success - control_success, 4),
        }
    return result


def popgym_repeat_first_per_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["actor"].get("seed", 0))].append(row)
    return {
        str(seed): {
            "clean_ere_count": sum(
                int(
                    str(row["actor"].get("arm", "")) == POPGYM_REPEAT_FIRST_CLEAN_ARM
                    and bool(row["metrics"].get("popgym_repeat_first_ere", False))
                )
                for row in items
            ),
            "query_tick_count": sum(int(row["eval_only"].get("popgym_query_tick_eval_only") is not None) for row in items),
        }
        for seed, items in sorted(grouped.items())
    }


def water_recall_protocol_summary(
    ticks: Sequence[Mapping[str, Any]],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not water_recall_present(protocol_manifest, ticks):
        return {"enabled": False, "status": "disabled"}
    rows = [
        row
        for row in ticks
        if isinstance(row.get("actor", {}), Mapping)
        and isinstance(row.get("metrics", {}), Mapping)
        and str(row["actor"].get("arm", "")) in {WATER_RECALL_CLEAN_ARM, *WATER_RECALL_CONTROL_ARMS}
    ]
    by_key = {
        (
            str(row["actor"].get("arm", "")),
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
            int(row["actor"].get("tick", 0)),
        ): row
        for row in rows
    }
    clean_ere_rows = [row for row in rows if str(row["actor"].get("arm", "")) == WATER_RECALL_CLEAN_ARM and bool(row["metrics"].get("water_recall_ere", False))]
    episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in rows
        if str(row["actor"].get("arm", "")) == WATER_RECALL_CLEAN_ARM
    }
    ere_episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in clean_ere_rows
    }
    requirements = water_recall_requirements(protocol_manifest)
    ere_episode_rate = round(len(ere_episode_keys) / len(episode_keys), 4) if episode_keys else 0.0
    status = GO_WATER_RECALL_EVALUABLE
    if len(clean_ere_rows) < int(requirements["min_ere_count"]) or ere_episode_rate < float(requirements["min_ere_episode_rate"]):
        status = INCONCLUSIVE_LOW_ERE
    return {
        "enabled": True,
        "status": status,
        "requirements": requirements,
        "clean_ere_count": len(clean_ere_rows),
        "clean_ere_episode_count": len(ere_episode_keys),
        "clean_episode_count": len(episode_keys),
        "clean_ere_episode_rate": ere_episode_rate,
        "summary_by_arm": water_recall_summary_by_arm(rows),
        "contrasts": water_recall_contrasts(clean_ere_rows, by_key),
        "per_seed": water_recall_per_seed(rows),
    }


def water_recall_present(protocol_manifest: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]]) -> bool:
    arms = set(str(arm) for arm in protocol_manifest.get("arms", ()))
    if WATER_RECALL_CLEAN_ARM in arms or any(arm in arms for arm in WATER_RECALL_CONTROL_ARMS):
        return True
    return any(bool(nested(row, "metrics", "water_recall_ere", False)) for row in ticks)


def water_recall_requirements(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_manifest.get("protocol", {})
    water = protocol.get("water_recall", {}) if isinstance(protocol, Mapping) else {}
    water = water if isinstance(water, Mapping) else {}
    return {
        "min_ere_count": int(water.get("min_ere_count", 30)),
        "min_ere_episode_rate": float(water.get("min_ere_episode_rate", 0.5)),
        "recall_window": int(water.get("recall_window", 4)),
    }


def water_recall_summary_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["actor"].get("arm", "unknown"))].append(row)
    return {arm: water_recall_arm_summary(items) for arm, items in sorted(grouped.items())}


def water_recall_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ere_rows = [row for row in rows if bool(row["metrics"].get("water_recall_ere", False))]
    visible_rows = [row for row in rows if bool(row["metrics"].get("water_visible", False))]
    action_values = [str(row["actor"].get("action", "")) for row in ere_rows]
    return {
        "tick_count": len(rows),
        "ere_count": len(ere_rows),
        "water_visible_count": len(visible_rows),
        "water_visible_rate": round(len(visible_rows) / len(rows), 4) if rows else 0.0,
        "recall_active_rate": round(bool_rate(row["metrics"].get("water_recall_active", False) for row in ere_rows), 4),
        "recall_consistent_action_rate": round(
            bool_rate(row["metrics"].get("water_recall_consistent_action", False) for row in ere_rows),
            4,
        ),
        "mean_fact_age_on_ere": round(mean(row["metrics"].get("water_fact_age") for row in ere_rows), 4),
        "action_distribution_on_ere": dict(sorted(Counter(action_values).items())),
    }


def water_recall_contrasts(
    clean_ere_rows: Sequence[Mapping[str, Any]],
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    clean_rate = bool_rate(nested(row, "metrics", "water_recall_consistent_action", False) for row in clean_ere_rows)
    for arm in WATER_RECALL_CONTROL_ARMS:
        comparable: list[Mapping[str, Any]] = []
        for row in clean_ere_rows:
            actor = row["actor"]
            key = (
                arm,
                int(actor.get("seed", 0)),
                str(actor.get("episode_id", actor.get("episode_index", "0"))),
                int(actor.get("tick", 0)),
            )
            if key in by_key:
                comparable.append(by_key[key])
        control_rate = bool_rate(nested(row, "metrics", "water_recall_consistent_action", False) for row in comparable)
        result[arm] = {
            "comparable_ere_count": len(comparable),
            "clean_recall_consistent_action_rate": round(clean_rate, 4),
            "control_recall_consistent_action_rate": round(control_rate, 4),
            "clean_minus_control_recall_consistent_action_rate": round(clean_rate - control_rate, 4),
        }
    return result


def water_recall_per_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["actor"].get("seed", 0))].append(row)
    return {
        str(seed): {
            "clean_ere_count": sum(
                int(str(row["actor"].get("arm", "")) == WATER_RECALL_CLEAN_ARM and bool(row["metrics"].get("water_recall_ere", False)))
                for row in items
            ),
            "water_visible_count": sum(int(bool(row["metrics"].get("water_visible", False))) for row in items),
            "water_need_active_count": sum(int(bool(row["metrics"].get("water_need_active", False))) for row in items),
        }
        for seed, items in sorted(grouped.items())
    }


def bench_recall_protocol_summary(
    ticks: Sequence[Mapping[str, Any]],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not bench_recall_present(protocol_manifest, ticks):
        return {"enabled": False, "status": "disabled"}
    rows = [
        row
        for row in ticks
        if isinstance(row.get("actor", {}), Mapping)
        and isinstance(row.get("metrics", {}), Mapping)
        and str(row["actor"].get("arm", "")) in {BENCH_RECALL_CLEAN_ARM, *BENCH_RECALL_CONTROL_ARMS}
    ]
    by_key = {
        (
            str(row["actor"].get("arm", "")),
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
            int(row["actor"].get("tick", 0)),
        ): row
        for row in rows
    }
    clean_ere_rows = [row for row in rows if str(row["actor"].get("arm", "")) == BENCH_RECALL_CLEAN_ARM and bool(row["metrics"].get("bench_recall_ere", False))]
    episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in rows
        if str(row["actor"].get("arm", "")) == BENCH_RECALL_CLEAN_ARM
    }
    ere_episode_keys = {
        (
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
        )
        for row in clean_ere_rows
    }
    requirements = bench_recall_requirements(protocol_manifest)
    ere_episode_rate = round(len(ere_episode_keys) / len(episode_keys), 4) if episode_keys else 0.0
    status = GO_BENCH_RECALL_EVALUABLE
    if len(clean_ere_rows) < int(requirements["min_ere_count"]) or ere_episode_rate < float(requirements["min_ere_episode_rate"]):
        status = INCONCLUSIVE_LOW_ERE
    return {
        "enabled": True,
        "status": status,
        "requirements": requirements,
        "clean_ere_count": len(clean_ere_rows),
        "clean_ere_episode_count": len(ere_episode_keys),
        "clean_episode_count": len(episode_keys),
        "clean_ere_episode_rate": ere_episode_rate,
        "summary_by_arm": bench_recall_summary_by_arm(rows),
        "contrasts": bench_recall_contrasts(clean_ere_rows, by_key),
        "per_seed": bench_recall_per_seed(rows),
    }


def bench_recall_present(protocol_manifest: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]]) -> bool:
    arms = set(str(arm) for arm in protocol_manifest.get("arms", ()))
    if BENCH_RECALL_CLEAN_ARM in arms or any(arm in arms for arm in BENCH_RECALL_CONTROL_ARMS):
        return True
    return any(bool(nested(row, "metrics", "bench_recall_ere", False)) for row in ticks)


def bench_recall_requirements(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_manifest.get("protocol", {})
    bench = protocol.get("bench_recall", {}) if isinstance(protocol, Mapping) else {}
    bench = bench if isinstance(bench, Mapping) else {}
    return {
        "min_ere_count": int(bench.get("min_ere_count", 30)),
        "min_ere_episode_rate": float(bench.get("min_ere_episode_rate", 0.5)),
        "recall_window": int(bench.get("recall_window", 4)),
    }


def bench_recall_summary_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["actor"].get("arm", "unknown"))].append(row)
    return {arm: bench_recall_arm_summary(items) for arm, items in sorted(grouped.items())}


def bench_recall_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ere_rows = [row for row in rows if bool(row["metrics"].get("bench_recall_ere", False))]
    action_values = [str(row["actor"].get("action", "")) for row in ere_rows]
    base_actions = [str(row["metrics"].get("bench_base_crafting_action", "")) for row in ere_rows if row["metrics"].get("bench_base_crafting_action")]
    return {
        "tick_count": len(rows),
        "ere_count": len(ere_rows),
        "bench_fact_present_count": sum(int(bool(row["metrics"].get("bench_fact_present", False))) for row in rows),
        "bench_need_active_count": sum(int(bool(row["metrics"].get("bench_need_active", False))) for row in rows),
        "repeat_place_suppressed_count": sum(int(bool(row["metrics"].get("bench_repeat_place_suppressed", False))) for row in rows),
        "recall_active_rate": round(bool_rate(row["metrics"].get("bench_recall_active", False) for row in ere_rows), 4),
        "recall_consistent_action_rate": round(
            bool_rate(row["metrics"].get("bench_recall_consistent_action", False) for row in ere_rows),
            4,
        ),
        "mean_fact_age_on_ere": round(mean(row["metrics"].get("bench_fact_age") for row in ere_rows), 4),
        "action_distribution_on_ere": dict(sorted(Counter(action_values).items())),
        "base_crafting_action_distribution_on_ere": dict(sorted(Counter(base_actions).items())),
    }


def bench_recall_contrasts(
    clean_ere_rows: Sequence[Mapping[str, Any]],
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    clean_rate = bool_rate(nested(row, "metrics", "bench_recall_consistent_action", False) for row in clean_ere_rows)
    for arm in BENCH_RECALL_CONTROL_ARMS:
        comparable: list[Mapping[str, Any]] = []
        for row in clean_ere_rows:
            actor = row["actor"]
            key = (
                arm,
                int(actor.get("seed", 0)),
                str(actor.get("episode_id", actor.get("episode_index", "0"))),
                int(actor.get("tick", 0)),
            )
            if key in by_key:
                comparable.append(by_key[key])
        control_rate = bool_rate(nested(row, "metrics", "bench_recall_consistent_action", False) for row in comparable)
        result[arm] = {
            "comparable_ere_count": len(comparable),
            "clean_recall_consistent_action_rate": round(clean_rate, 4),
            "control_recall_consistent_action_rate": round(control_rate, 4),
            "clean_minus_control_recall_consistent_action_rate": round(clean_rate - control_rate, 4),
        }
    return result


def bench_recall_per_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["actor"].get("seed", 0))].append(row)
    return {
        str(seed): {
            "clean_ere_count": sum(
                int(str(row["actor"].get("arm", "")) == BENCH_RECALL_CLEAN_ARM and bool(row["metrics"].get("bench_recall_ere", False)))
                for row in items
            ),
            "bench_fact_present_count": sum(int(bool(row["metrics"].get("bench_fact_present", False))) for row in items),
            "bench_need_active_count": sum(int(bool(row["metrics"].get("bench_need_active", False))) for row in items),
            "repeat_place_suppressed_count": sum(int(bool(row["metrics"].get("bench_repeat_place_suppressed", False))) for row in items),
        }
        for seed, items in sorted(grouped.items())
    }


def contamination_summary(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[Mapping[str, Any]] = []
    for row in ticks:
        scan = row.get("contamination", {})
        if not isinstance(scan, Mapping):
            continue
        failures.extend(scan.get("failures", ()) if isinstance(scan.get("failures", ()), Sequence) else ())
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": [dict(item) for item in failures],
    }


def anchor_coverage_summary(
    ticks: Sequence[Mapping[str, Any]],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize whether the seed block creates fair memory-anchor tests."""
    rows = [row for row in ticks if isinstance(row.get("actor", {}), Mapping) and isinstance(row.get("metrics", {}), Mapping)]
    by_key = {
        (
            str(row["actor"].get("arm", "")),
            int(row["actor"].get("seed", 0)),
            str(row["actor"].get("episode_id", row["actor"].get("episode_index", "0"))),
            int(row["actor"].get("tick", 0)),
        ): row
        for row in rows
    }
    seeds = configured_seeds(protocol_manifest, rows)
    per_seed = {
        int(seed): {
            "all_arm_anchor_opportunity_count": 0,
            "clean_arm_anchor_opportunity_count": 0,
            "cached_anchor_use_count": 0,
            "evaluable_anchor_count": 0,
        }
        for seed in seeds
    }
    requirements = anchor_coverage_requirements(protocol_manifest)
    all_arm_anchor_opportunity_count = 0
    clean_arm_anchor_opportunity_count = 0
    cached_anchor_use_count = 0
    evaluable_anchor_count = 0
    for row in rows:
        actor = row["actor"]
        metrics = row["metrics"]
        seed = int(actor.get("seed", 0))
        arm = str(actor.get("arm", ""))
        per_seed.setdefault(
            seed,
            {
                "all_arm_anchor_opportunity_count": 0,
                "clean_arm_anchor_opportunity_count": 0,
                "cached_anchor_use_count": 0,
                "evaluable_anchor_count": 0,
            },
        )
        anchor_opportunity = bool(metrics.get("anchor_currently_visible", False) or metrics.get("resource_memory_critical", False))
        if anchor_opportunity:
            all_arm_anchor_opportunity_count += 1
            per_seed[seed]["all_arm_anchor_opportunity_count"] += 1
        if arm != FORWARD_CLEAN_ARM:
            continue
        if anchor_opportunity:
            clean_arm_anchor_opportunity_count += 1
            per_seed[seed]["clean_arm_anchor_opportunity_count"] += 1
        cached_anchor_use = bool(metrics.get("query_used_cached_fact", False))
        if cached_anchor_use:
            cached_anchor_use_count += 1
            per_seed[seed]["cached_anchor_use_count"] += 1
        if cached_anchor_use and controls_are_comparable(by_key, actor, FORWARD_CORRUPTED_CONTROL_ARMS):
            evaluable_anchor_count += 1
            per_seed[seed]["evaluable_anchor_count"] += 1
    evaluable_seeds = [seed for seed, row in per_seed.items() if int(row["evaluable_anchor_count"]) > 0]
    seed_count = len(per_seed)
    return {
        "all_arm_anchor_opportunity_count": all_arm_anchor_opportunity_count,
        "clean_arm_anchor_opportunity_count": clean_arm_anchor_opportunity_count,
        "cached_anchor_use_count": cached_anchor_use_count,
        "evaluable_anchor_count": evaluable_anchor_count,
        "evaluable_seed_count": len(evaluable_seeds),
        "evaluable_seed_rate": round(len(evaluable_seeds) / seed_count, 4) if seed_count else 0.0,
        "seed_count": seed_count,
        "requirements": requirements,
        "per_seed": {str(seed): row for seed, row in sorted(per_seed.items())},
    }


def anchor_coverage_requirements(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_manifest.get("protocol", {})
    protocol = protocol if isinstance(protocol, Mapping) else {}
    return {
        "min_evaluable_anchor_count": int(protocol.get("min_evaluable_anchor_count", 1)),
        "min_evaluable_seed_rate": float(protocol.get("min_evaluable_seed_rate", 0.5)),
    }


def configured_seeds(protocol_manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[int]:
    runtime = protocol_manifest.get("runtime", {})
    if isinstance(runtime, Mapping) and runtime.get("seeds"):
        return [int(seed) for seed in runtime["seeds"]]
    return sorted({int(row["actor"].get("seed", 0)) for row in rows})


def controls_are_comparable(
    by_key: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    actor: Mapping[str, Any],
    control_arms: Sequence[str] = FORWARD_CONTROL_ARMS,
) -> bool:
    seed = int(actor.get("seed", 0))
    episode = str(actor.get("episode_id", actor.get("episode_index", "0")))
    tick = int(actor.get("tick", 0))
    return all((arm, seed, episode, tick) in by_key for arm in control_arms)


def memory_evaluability_decision(coverage: Mapping[str, Any]) -> str:
    requirements = coverage.get("requirements", {})
    requirements = requirements if isinstance(requirements, Mapping) else {}
    if int(coverage.get("evaluable_anchor_count", 0)) < int(requirements.get("min_evaluable_anchor_count", 1)):
        return NOT_EVALUABLE_INSUFFICIENT_ANCHOR_COVERAGE
    if float(coverage.get("evaluable_seed_rate", 0.0)) < float(requirements.get("min_evaluable_seed_rate", 0.5)):
        return NOT_EVALUABLE_INSUFFICIENT_ANCHOR_COVERAGE
    return GO_MEMORY_EVALUABLE_SEED_BLOCK


def deterministic_replay_summary_from_digest(
    primary_ticks: Sequence[Mapping[str, Any]],
    *,
    repeat_digest: str,
    repeat_tick_count: int,
    enabled: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
    primary_digest = replay_digest(primary_ticks)
    passed = primary_digest == repeat_digest and len(primary_ticks) == int(repeat_tick_count)
    return {
        "enabled": bool(enabled),
        "mode": "fresh_process",
        "digest_scope": "full_tick",
        "passed": False if error else passed,
        "primary_digest": primary_digest,
        "repeat_digest": repeat_digest,
        "primary_tick_count": len(primary_ticks),
        "repeat_tick_count": int(repeat_tick_count),
        "error": error,
    }


def replay_digest(ticks: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "actor": row.get("actor", {}),
            "metrics": row.get("metrics", {}),
            "eval_only": row.get("eval_only", {}),
            "contamination": row.get("contamination", {}),
        }
        for row in ticks
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def combine_contamination(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[Mapping[str, Any]] = []
    for episode in episodes:
        contamination = episode.get("contamination", {})
        if not isinstance(contamination, Mapping):
            continue
        failures.extend(contamination.get("failures", ()) if isinstance(contamination.get("failures", ()), Sequence) else ())
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": [dict(item) for item in failures],
    }


def decision_from_protocol(
    *,
    contamination: Mapping[str, Any],
    memory: Mapping[str, Any],
    runtime_unavailable: Mapping[str, Any] | None,
    artifact_failure: bool,
    deterministic_replay: Mapping[str, Any],
) -> str:
    if runtime_unavailable:
        return NOT_EVALUABLE_RUNTIME_UNAVAILABLE
    if artifact_failure:
        return NO_GO_PROTOCOL_ARTIFACT_FAILURE
    if deterministic_replay.get("enabled") and deterministic_replay.get("passed") is False:
        return NO_GO_PROTOCOL_ARTIFACT_FAILURE
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE
    return GO_LONG_RUN_PROTOCOL_SUPPORTED


def summarize_by_key(episodes: Sequence[Mapping[str, Any]], actor_key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        actor = episode.get("actor", {})
        if isinstance(actor, Mapping):
            grouped[str(actor.get(actor_key, "unknown"))].append(episode)
    return {key: aggregate_episodes(rows) for key, rows in sorted(grouped.items())}


def aggregate_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [episode.get("metrics", {}) for episode in episodes if isinstance(episode.get("metrics", {}), Mapping)]
    eval_rows = [episode.get("eval_only", {}) for episode in episodes if isinstance(episode.get("eval_only", {}), Mapping)]
    return {
        "episode_count": len(episodes),
        "mean_survival_steps": round(mean(row.get("survival_steps") for row in metrics), 4),
        "death_rate": round(bool_rate(bool(row.get("done", False)) and not bool(row.get("reached_horizon", False)) for row in metrics), 4),
        "mean_valid_action_rate": round(mean(row.get("valid_action_rate") for row in metrics), 4),
        "mean_invalid_or_unknown_action_rate": round(mean(row.get("invalid_or_unknown_action_rate") for row in metrics), 4),
        "mean_repeated_action_loop_rate": round(mean(row.get("repeated_action_loop_rate") for row in metrics), 4),
        "mean_event_self_trigger_rate": round(mean(row.get("event_self_trigger_rate") for row in metrics), 4),
        "memory_anchor_count": sum(int(row.get("memory_anchor_count", 0)) for row in metrics),
        "mean_query_used_cached_fact_rate": round(mean(row.get("query_used_cached_fact_rate") for row in metrics), 4),
        "mean_action_entropy_proxy": round(mean(row.get("action_entropy_proxy") for row in metrics), 4),
        "reward_sum_eval_only": round(sum(numeric(row.get("reward_sum_eval_only")) for row in eval_rows), 4),
        "contamination_failure_count": sum(int(row.get("contamination_failure_count", 0)) for row in metrics),
    }


def format_long_run_summary_markdown(summary: Mapping[str, Any]) -> str:
    protocol = summary.get("protocol", {})
    protocol = protocol if isinstance(protocol, Mapping) else {}
    lines = [
        "# Long-Run Protocol Summary",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Memory evaluability: `{summary.get('memory_evaluability_decision')}`",
        f"- Runtime: `{protocol.get('runtime')}`",
        f"- Actor: `{protocol.get('actor')}`",
        f"- Actor policy mode: `{protocol.get('actor_policy_mode')}`",
        f"- Memory residual bias: `{protocol.get('actor_memory_residual_bias')}`",
        f"- Split: `{protocol.get('split')}`",
        f"- Fact surface: `{protocol.get('fact_surface')}`",
        f"- Horizons: `{protocol.get('horizons')}`",
        f"- Episode count: `{summary.get('episode_count')}`",
        f"- Tick count: `{summary.get('tick_count')}`",
        f"- Deterministic replay: `{determinism_status(summary.get('deterministic_replay'))}`",
        "",
        "## Arms",
        "",
    ]
    by_arm = summary.get("summary_by_arm", {})
    if isinstance(by_arm, Mapping):
        lines.extend(
            [
                "| Arm | Episodes | Mean survival | Valid action rate | Event self-trigger | Loop rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for arm, row in by_arm.items():
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(arm),
                        str(row.get("episode_count")),
                        str(row.get("mean_survival_steps")),
                        str(row.get("mean_valid_action_rate")),
                        str(row.get("mean_event_self_trigger_rate")),
                        str(row.get("mean_repeated_action_loop_rate")),
                    ]
                )
                + " |"
            )
    memory = summary.get("memory_grounded_score", {})
    memory = memory if isinstance(memory, Mapping) else {}
    lines.extend(
        [
            "",
            "## Memory-Grounded Score",
            "",
            f"- Status: `{memory.get('status')}`",
            f"- Anchor count: `{memory.get('anchor_count')}`",
            "",
        ]
    )
    coverage = summary.get("anchor_coverage", {})
    coverage = coverage if isinstance(coverage, Mapping) else {}
    lines.extend(
        [
            "## Anchor Coverage",
            "",
            f"- All-arm anchor opportunities: `{coverage.get('all_arm_anchor_opportunity_count')}`",
            f"- Clean-arm anchor opportunities: `{coverage.get('clean_arm_anchor_opportunity_count')}`",
            f"- Cached anchor uses: `{coverage.get('cached_anchor_use_count')}`",
            f"- Evaluable anchors: `{coverage.get('evaluable_anchor_count')}`",
            f"- Evaluable seed rate: `{coverage.get('evaluable_seed_rate')}`",
            f"- Coverage requirements: `{coverage.get('requirements')}`",
            "",
        ]
    )
    water = summary.get("water_recall", {})
    water = water if isinstance(water, Mapping) else {}
    if bool(water.get("enabled", False)):
        lines.extend(
            [
                "## Water Recall",
                "",
                f"- Status: `{water.get('status')}`",
                f"- Clean ERE count: `{water.get('clean_ere_count')}`",
                f"- Clean ERE episode rate: `{water.get('clean_ere_episode_rate')}`",
                f"- Requirements: `{water.get('requirements')}`",
                "",
            ]
        )
        contrasts = water.get("contrasts", {})
        if isinstance(contrasts, Mapping):
            lines.extend(
                [
                    "| Control | Comparable EREs | Clean rate | Control rate | Delta |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for arm, row in contrasts.items():
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(arm),
                            str(row.get("comparable_ere_count")),
                            str(row.get("clean_recall_consistent_action_rate")),
                            str(row.get("control_recall_consistent_action_rate")),
                            str(row.get("clean_minus_control_recall_consistent_action_rate")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    passive = summary.get("passive_match", {})
    passive = passive if isinstance(passive, Mapping) else {}
    if bool(passive.get("enabled", False)):
        lines.extend(
            [
                "## Passive Match",
                "",
                f"- Status: `{passive.get('status')}`",
                f"- Clean ERE count: `{passive.get('clean_ere_count')}`",
                f"- Clean ERE episode rate: `{passive.get('clean_ere_episode_rate')}`",
                f"- Requirements: `{passive.get('requirements')}`",
                "",
            ]
        )
        contrasts = passive.get("contrasts", {})
        if isinstance(contrasts, Mapping):
            lines.extend(
                [
                    "| Control | Comparable EREs | Clean success | Control success | Delta |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for arm, row in contrasts.items():
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(arm),
                            str(row.get("comparable_ere_count")),
                            str(row.get("clean_success_rate_eval_only")),
                            str(row.get("control_success_rate_eval_only")),
                            str(row.get("clean_minus_control_success_rate_eval_only")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    repeat = summary.get("popgym_repeat_first", {})
    repeat = repeat if isinstance(repeat, Mapping) else {}
    if bool(repeat.get("enabled", False)):
        lines.extend(
            [
                "## POPGym RepeatFirst",
                "",
                f"- Status: `{repeat.get('status')}`",
                f"- Clean ERE count: `{repeat.get('clean_ere_count')}`",
                f"- Clean ERE episode rate: `{repeat.get('clean_ere_episode_rate')}`",
                f"- Requirements: `{repeat.get('requirements')}`",
                "",
            ]
        )
        contrasts = repeat.get("contrasts", {})
        if isinstance(contrasts, Mapping):
            lines.extend(
                [
                    "| Control | Comparable EREs | Clean success | Control success | Delta |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for arm, row in contrasts.items():
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(arm),
                            str(row.get("comparable_ere_count")),
                            str(row.get("clean_success_rate_eval_only")),
                            str(row.get("control_success_rate_eval_only")),
                            str(row.get("clean_minus_control_success_rate_eval_only")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    bench = summary.get("bench_recall", {})
    bench = bench if isinstance(bench, Mapping) else {}
    if bool(bench.get("enabled", False)):
        lines.extend(
            [
                "## Bench Recall",
                "",
                f"- Status: `{bench.get('status')}`",
                f"- Clean ERE count: `{bench.get('clean_ere_count')}`",
                f"- Clean ERE episode rate: `{bench.get('clean_ere_episode_rate')}`",
                f"- Requirements: `{bench.get('requirements')}`",
                "",
            ]
        )
        contrasts = bench.get("contrasts", {})
        if isinstance(contrasts, Mapping):
            lines.extend(
                [
                    "| Control | Comparable EREs | Clean rate | Control rate | Delta |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for arm, row in contrasts.items():
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(arm),
                            str(row.get("comparable_ere_count")),
                            str(row.get("clean_recall_consistent_action_rate")),
                            str(row.get("control_recall_consistent_action_rate")),
                            str(row.get("clean_minus_control_recall_consistent_action_rate")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def public_protocol_summary(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = protocol_manifest.get("runtime", {})
    protocol = protocol_manifest.get("protocol", {})
    fact_surface = protocol_manifest.get("fact_surface", {})
    actor = protocol_manifest.get("actor", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    protocol = protocol if isinstance(protocol, Mapping) else {}
    fact_surface = fact_surface if isinstance(fact_surface, Mapping) else {}
    actor = actor if isinstance(actor, Mapping) else {}
    return {
        "runtime": runtime.get("name"),
        "runtime_task": runtime.get("task"),
        "actor": actor.get("name"),
        "actor_checkpoint": actor.get("checkpoint"),
        "actor_repo_path": actor.get("repo_path"),
        "actor_policy_mode": actor.get("policy_mode"),
        "actor_memory_residual_bias": actor.get("memory_residual_bias"),
        "split": runtime.get("split"),
        "seed_start": runtime.get("seed_start"),
        "seed_count": runtime.get("seed_count"),
        "seeds": runtime.get("seeds"),
        "horizons": protocol.get("horizons"),
        "max_episodes_per_seed": protocol.get("max_episodes_per_seed"),
        "determinism_check": protocol.get("determinism_check"),
        "min_evaluable_anchor_count": protocol.get("min_evaluable_anchor_count"),
        "min_evaluable_seed_rate": protocol.get("min_evaluable_seed_rate"),
        "water_recall": protocol.get("water_recall"),
        "bench_recall": protocol.get("bench_recall"),
        "passive_match": protocol.get("passive_match"),
        "popgym_repeat_first": protocol.get("popgym_repeat_first"),
        "deterministic_backend_patch": runtime.get("deterministic_backend_patch"),
        "backend_determinism_patch": runtime.get("backend_determinism_patch"),
        "fact_surface": fact_surface.get("name"),
        "fact_surface_checkpoint": fact_surface.get("checkpoint"),
        "fact_surface_threshold": fact_surface.get("threshold"),
        "arms": protocol_manifest.get("arms", ()),
    }


def determinism_status(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "disabled"
    mode = value.get("mode", "unknown")
    passed = value.get("passed")
    if passed is None:
        return f"{mode}: not_checked"
    return f"{mode}: {'passed' if passed else 'failed'}"


def metrics_enabled(protocol_manifest: Mapping[str, Any], key: str) -> bool:
    metrics = protocol_manifest.get("metrics", {})
    return bool(metrics.get(key, False)) if isinstance(metrics, Mapping) else False


def nested(row: Mapping[str, Any], namespace: str, key: str, default: Any = None) -> Any:
    payload = row.get(namespace, {})
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return default


def death_cause(ticks: Sequence[Mapping[str, Any]]) -> str | None:
    if not ticks:
        return None
    value = nested(ticks[-1], "eval_only", "death_cause_eval_only")
    return None if value in ("", None) else str(value)


def bool_rate(values) -> float:  # noqa: ANN001
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def mean(values) -> float:  # noqa: ANN001
    nums = [numeric(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else 0.0


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def append_numeric(values: list[float], value: Any) -> None:
    if value is None:
        return
    values.append(numeric(value))


def trend(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return round(float(values[-1]) - float(values[0]), 4)


def normalized_entropy(values: Sequence[str]) -> float | None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def is_movement_action(action: str) -> bool:
    lowered = action.lower()
    return lowered.startswith("move") or lowered.startswith("turn")


def max_stuck_duration(actions: Sequence[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for action in actions:
        if action == previous:
            current += 1
        else:
            current = 1
            previous = action
        longest = max(longest, current)
    return longest
