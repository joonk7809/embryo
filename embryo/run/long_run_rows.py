"""Tick-row construction for the long-run protocol."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from embryo.eval.contamination import scan_actor_context
from embryo.run.long_run_arms import EpisodeState, action_is_valid, numeric
from embryo.runtimes.base import RuntimeSpec


def build_tick_row(
    *,
    actor: Mapping[str, Any],
    tick: int,
    spec: RuntimeSpec,
    pre_observation: Mapping[str, Any],
    post_step: Any,
    decision: Mapping[str, Any],
    state: EpisodeState,
) -> dict[str, Any]:
    action = str(decision["action"])
    actor_context = actor_context_from_decision(pre_observation, decision)
    contamination = scan_actor_context(actor_context)
    valid_action = action_is_valid(spec, action)
    row_actor = dict(actor)
    row_actor.update(
        {
            "tick": int(tick),
            "action": action,
            "route_mode": str(decision.get("route_mode", "")),
        }
    )
    return {
        "actor": row_actor,
        "metrics": {
            "valid_action": valid_action,
            "invalid_or_unknown_action": bool(decision.get("invalid_or_unknown_action", False)) or not valid_action,
            "repeated_action_loop": bool(decision.get("repeated_action_loop", False)),
            "event_self_triggered": bool(decision.get("event_self_triggered", False)),
            "resource_memory_critical": bool(decision.get("resource_memory_critical", False)),
            "memory_anchor_critical": bool(decision.get("memory_anchor_critical", False)),
            "anchor_family": str(decision.get("anchor_family", "")),
            "anchor_fact_age": int(decision.get("anchor_fact_age", 0)),
            "anchor_currently_visible": bool(decision.get("anchor_currently_visible", False)),
            "query_used_cached_fact": bool(decision.get("query_used_cached_fact", False)),
            "memory_followthrough_delta": decision.get("memory_followthrough_delta"),
            "resource_route_preserved": bool(decision.get("resource_route_preserved", False)),
            "fallback_triggered": bool(decision.get("fallback_triggered", False)),
            "query_content_hash": str(decision.get("query_content_hash", "")),
            "action_entropy_proxy": action_entropy_with_candidate(state.recent_actions, action),
            "exploration_bin": exploration_bin(state.recent_actions, action),
            "effective_candidate_score": decision.get("effective_candidate_score"),
            "effective_cache_age": decision.get("effective_cache_age"),
            "effective_invalidated": decision.get("effective_invalidated"),
            "effective_center_patch_hash": decision.get("effective_center_patch_hash"),
            "effective_fact_value": decision.get("effective_fact_value"),
            "effective_memory_arm": decision.get("effective_memory_arm", None),
        },
        "eval_only": eval_only_tick_fields(post_step),
        "contamination": contamination,
    }


def actor_context_from_decision(observation: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_rgb_frame": observation.get("raw_rgb_frame", "<rgb_frame>"),
        "previous_rgb_frame": observation.get("previous_rgb_frame", "<previous_rgb_frame>"),
        "previous_action": observation.get("previous_action", "noop"),
        "facing_candidate_v1": bool(decision.get("fact_value", False)),
        "query_content": str(decision.get("query_content_hash", "")),
        "query_content_hash": str(decision.get("query_content_hash", "")),
        "query_cache_age": int(decision.get("cache_age", 0)),
        "short_ttl_stale_state": str(decision.get("cache_age", 0)),
    }


def eval_only_tick_fields(step: Any) -> dict[str, Any]:
    info = step.info if isinstance(step.info, Mapping) else {}
    achievements = achievements_from_info(info)
    return {
        "reward_delta_eval_only": numeric(step.reward),
        "done_eval_only": bool(step.done),
        "death_cause_eval_only": death_cause_from_info(info, done=bool(step.done)),
        "achievements_eval_only": achievements,
        "health_eval_only": optional_numeric(info.get("health")),
        "food_eval_only": optional_numeric(info.get("food")),
        "drink_eval_only": optional_numeric(info.get("drink")),
        "backend_inventory_eval_only": json_safe(info.get("inventory")) if "inventory" in info else None,
    }


def action_entropy_with_candidate(recent_actions: Sequence[str], action: str) -> float:
    actions = [*recent_actions, action]
    if len(set(actions)) <= 1:
        return 0.0
    counts = {item: actions.count(item) for item in set(actions)}
    total = len(actions)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(entropy / math.log(len(counts)), 4)


def exploration_bin(recent_actions: Sequence[str], action: str) -> int:
    return min(4, int(action_entropy_with_candidate(recent_actions, action) * 5))


def achievements_from_info(info: Mapping[str, Any]) -> list[str]:
    raw = info.get("achievements", info.get("achievement"))
    if isinstance(raw, Mapping):
        return sorted(str(key) for key, value in raw.items() if bool(value))
    if isinstance(raw, (list, tuple, set)):
        return sorted(str(item) for item in raw if item is not None)
    if raw:
        return [str(raw)]
    return []


def death_cause_from_info(info: Mapping[str, Any], *, done: bool) -> str | None:
    if not done:
        return None
    for key in ("death_cause", "death_reason", "cause_of_death"):
        if info.get(key):
            return str(info[key])
    return None


def optional_numeric(value: Any) -> float | None:
    if value is None:
        return None
    return numeric(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


