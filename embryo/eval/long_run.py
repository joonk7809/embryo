"""Long-run protocol metrics.

These helpers summarize persistent-world episodes without treating reward as an
actor signal. Runtime outcomes and backend info are kept under `eval_only` in
the artifact schema.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from embryo.eval.memory_grounded_score import ScoreConfig, compute_memory_grounded_scores


GO_LONG_RUN_PROTOCOL_SUPPORTED = "GO_long_run_protocol_supported"
NO_GO_PROTOCOL_ARTIFACT_FAILURE = "NO_GO_protocol_artifact_failure"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"
NOT_EVALUABLE_NO_MEMORY_ANCHORS = "NOT_EVALUABLE_no_memory_anchors"
NOT_EVALUABLE_RUNTIME_UNAVAILABLE = "NOT_EVALUABLE_runtime_unavailable"

FORWARD_CLEAN_ARM = "query_memory_clean"
FORWARD_CONTROL_ARMS = (
    "query_memory_wrong_binding",
    "query_memory_shuffled",
    "query_memory_stale",
    "no_memory",
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
) -> dict[str, Any]:
    """Build the top-level long-run summary."""
    memory = memory_grounded_protocol_score(ticks, enabled=metrics_enabled(protocol_manifest, "memory_grounded_score"))
    decision = decision_from_protocol(
        contamination=contamination,
        memory=memory,
        runtime_unavailable=runtime_unavailable,
        artifact_failure=artifact_failure,
    )
    return {
        "decision": decision,
        "protocol": public_protocol_summary(protocol_manifest),
        "episode_count": len(episodes),
        "tick_count": len(ticks),
        "summary_by_arm": summarize_by_key(episodes, "arm"),
        "summary_by_horizon": summarize_by_key(episodes, "horizon"),
        "memory_grounded_score": memory,
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
                "resource_memory_critical": bool(metrics.get("resource_memory_critical", False)),
                "resource_route_preserved": bool(metrics.get("resource_route_preserved", False)),
                "fallback_triggered": bool(metrics.get("fallback_triggered", False)),
                "event_self_triggered": bool(metrics.get("event_self_triggered", False)),
                "repeated_action_loop": bool(metrics.get("repeated_action_loop", False)),
                "diagnostic_progress_delta_teacher_only": numeric(eval_only.get("reward_delta_eval_only", 0.0)),
                "action_entropy_proxy": numeric(metrics.get("action_entropy_proxy", 0.0)),
                "exploration_bin": metrics.get("exploration_bin"),
            }
        )
    return rows


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
) -> str:
    if runtime_unavailable:
        return NOT_EVALUABLE_RUNTIME_UNAVAILABLE
    if artifact_failure:
        return NO_GO_PROTOCOL_ARTIFACT_FAILURE
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE
    if memory.get("enabled") and memory.get("status") == NOT_EVALUABLE_NO_MEMORY_ANCHORS:
        return NOT_EVALUABLE_NO_MEMORY_ANCHORS
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
        f"- Runtime: `{protocol.get('runtime')}`",
        f"- Split: `{protocol.get('split')}`",
        f"- Horizons: `{protocol.get('horizons')}`",
        f"- Episode count: `{summary.get('episode_count')}`",
        f"- Tick count: `{summary.get('tick_count')}`",
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
    return "\n".join(lines) + "\n"


def public_protocol_summary(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = protocol_manifest.get("runtime", {})
    protocol = protocol_manifest.get("protocol", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    protocol = protocol if isinstance(protocol, Mapping) else {}
    return {
        "runtime": runtime.get("name"),
        "split": runtime.get("split"),
        "seed_start": runtime.get("seed_start"),
        "seed_count": runtime.get("seed_count"),
        "seeds": runtime.get("seeds"),
        "horizons": protocol.get("horizons"),
        "max_episodes_per_seed": protocol.get("max_episodes_per_seed"),
        "arms": protocol_manifest.get("arms", ()),
    }


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
