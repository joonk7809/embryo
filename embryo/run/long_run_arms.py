"""Long-run protocol arm policies and deployable action helpers."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from collections.abc import Mapping
from typing import Any

from embryo.memory.freshness import FreshnessState
from embryo.memory.queries import build_event_query, build_resource_query, combine_queries
from embryo.models import RuleRouterModel, ThresholdFactWriter
from embryo.runtimes.base import RuntimeSpec


DEFAULT_ARMS = (
    "no_memory",
    "query_memory_clean",
    "query_memory_shuffled",
    "query_memory_stale",
    "query_memory_wrong_binding",
    "random_valid_action",
    "reference_exploration_sweep",
)


class EpisodeState:
    def __init__(self, *, seed: int, arm: str, horizon: int) -> None:
        self.arm = arm
        self.rng = random.Random(f"{seed}:{arm}:{horizon}")
        self.previous_action: str | None = None
        self.repeat_count = 0
        self.last_route = "hold"
        self.recent_actions: deque[str] = deque(maxlen=8)
        self.visual_cache_hash: str | None = None
        self.visual_cache_age = 0
        self.visual_cache_score = 0.0
        self.visual_cache_ttl = 12
        self.sweep_index = 0
        self.low_delta_count = 0

    def observe(self, action: str, route: str) -> None:
        if action == self.previous_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.previous_action = action
        self.last_route = route
        self.recent_actions.append(action)

    def prepare_visual_memory(self, features: Mapping[str, Any]) -> dict[str, Any]:
        visible = bool(features.get("visual_anchor_visible", False))
        patch_hash = str(features.get("center_patch_hash", ""))
        if visible and patch_hash:
            self.visual_cache_hash = patch_hash
            self.visual_cache_age = 0
            self.visual_cache_score = numeric(features.get("center_salience_score", features.get("candidate_score", 0.0)))
        elif self.visual_cache_hash is not None:
            self.visual_cache_age += 1
        cache_valid = self.visual_cache_hash is not None and self.visual_cache_age <= self.visual_cache_ttl
        return {
            "anchor_family": str(features.get("visual_anchor_family", "")),
            "anchor_currently_visible": visible,
            "anchor_fact_age": int(self.visual_cache_age if self.visual_cache_hash is not None else 0),
            "cached_fact_hash": self.visual_cache_hash or "",
            "cached_fact_score": self.visual_cache_score,
            "cache_valid": cache_valid,
            "query_used_cached_fact": bool(cache_valid and not visible),
        }


def select_action_for_arm(spec: RuntimeSpec, observation: Mapping[str, Any], state: EpisodeState, arm: str) -> dict[str, Any]:
    if arm == "no_memory":
        return no_query_decision(spec, observation, state, arm, spec.noop_action)
    if arm == "random_valid_action":
        return no_query_decision(spec, observation, state, arm, state.rng.choice(tuple(spec.action_names)))
    if arm == "reference_exploration_sweep":
        return reference_exploration_action(spec, observation, state)

    base_features = deployable_features(observation, repeat_count=state.repeat_count, last_route=state.last_route)
    memory = state.prepare_visual_memory(base_features)
    if not bool(memory.get("query_used_cached_fact", False)):
        return matched_exploration_fallback(spec, observation, state)
    features = features_for_memory_arm(base_features, memory, arm)
    fact_writer = ThresholdFactWriter()
    router = RuleRouterModel()
    fact = fact_writer.fact(features)
    freshness = FreshnessState(
        cache_age=int(features.get("cache_age", 0)),
        invalidated=bool(features.get("invalidated", False)),
        visual_change_conflict=bool(features.get("visual_change_conflict", False)),
    )
    resource_query = build_resource_query([fact], cache_age=freshness.cache_age, fresh=not freshness.invalidated)
    event_query = build_event_query([fact], cache_age=freshness.cache_age)
    combined_query = combine_queries(resource_query, event_query)
    router_output = router.predict(
        {
            "facing_candidate": bool(fact.value),
            "failed_action_event": bool(features.get("failed_action_event", False)),
            "cache_age": freshness.cache_age,
            "invalidated": freshness.invalidated,
            "visual_change_conflict": freshness.visual_change_conflict,
            "cooldown": int(features.get("cooldown", 0)),
            "repeat_count": int(features.get("repeat_count", 0)),
            "last_route": str(features.get("last_route", "hold")),
        }
    )
    action = action_for_route(spec, router_output.route_mode)
    event_self_triggered = state.last_route == "event_fallback" and router_output.route_mode == "event_fallback" and not bool(
        features.get("failed_action_event", False)
    )
    query_hash = query_hash_for_visual_memory(combined_query.content_hash, memory=memory, arm=arm, used_cached=bool(features.get("query_used_cached_fact")))
    return {
        "action": action,
        "route_mode": router_output.route_mode,
        "query_content_hash": query_hash,
        "cache_age": freshness.cache_age,
        "resource_memory_critical": resource_memory_critical(observation),
        **memory_decision_fields(memory, used_cached=bool(features.get("query_used_cached_fact")), arm=arm, action=action, resource_action=spec.resource_action),
        "resource_route_preserved": bool(router_output.resource_route_preserved),
        "fallback_triggered": bool(router_output.event_fallback_gate),
        "event_self_triggered": bool(event_self_triggered),
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": bool(fact.value),
        "invalid_or_unknown_action": not action_is_valid(spec, action),
        "effective_invalidated": bool(freshness.invalidated),
        "effective_center_patch_hash": str(features.get("center_patch_hash", "")),
        "effective_candidate_score": numeric(features.get("candidate_score", 0.0)),
        "effective_cache_age": int(freshness.cache_age),
        "effective_fact_value": bool(fact.value),
        "effective_memory_arm": arm,
    }


def reference_exploration_action(spec: RuntimeSpec, observation: Mapping[str, Any], state: EpisodeState) -> dict[str, Any]:
    cycle = reference_action_cycle(spec)
    failed = bool(observation.get("failed_action_event", False))
    visible = bool(observation.get("visual_anchor_visible", False))
    changed = bool(observation.get("visual_change_event", False))
    previous_action = str(observation.get("previous_action", spec.noop_action))
    low_delta = previous_action != spec.noop_action and not changed and numeric(observation.get("rgb_delta_score", 0.0)) <= 0.003
    state.low_delta_count = state.low_delta_count + 1 if low_delta else 0

    if state.repeat_count >= 3 or failed or state.low_delta_count >= 2:
        state.sweep_index += 1
    if visible and action_is_valid(spec, spec.resource_action):
        action = spec.resource_action
    else:
        action = cycle[state.sweep_index % len(cycle)]
        state.sweep_index += 1
    return no_query_decision(spec, observation, state, "reference_exploration_sweep", action)


def matched_exploration_fallback(spec: RuntimeSpec, observation: Mapping[str, Any], state: EpisodeState) -> dict[str, Any]:
    decision = reference_exploration_action(spec, observation, state)
    decision["route_mode"] = "matched_exploration_fallback"
    return decision


def reference_action_cycle(spec: RuntimeSpec) -> tuple[str, ...]:
    preferred = (
        spec.resource_action,
        "move_up",
        "move_right",
        "move_down",
        "move_left",
        "do",
        spec.fallback_action,
        spec.noop_action,
    )
    actions = []
    for action in preferred:
        if action not in actions and action_is_valid(spec, action):
            actions.append(action)
    return tuple(actions or (spec.noop_action,))


def no_query_decision(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: EpisodeState,
    arm: str,
    action: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "route_mode": arm,
        "query_content_hash": "",
        "cache_age": 0,
        "resource_memory_critical": resource_memory_critical(observation),
        **memory_decision_fields(visual_memory_empty(observation), used_cached=False),
        "resource_route_preserved": False,
        "fallback_triggered": False,
        "event_self_triggered": False,
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": False,
        "invalid_or_unknown_action": not action_is_valid(spec, action),
    }


def deployable_features(observation: Mapping[str, Any], *, repeat_count: int, last_route: str) -> dict[str, Any]:
    return {
        "raw_rgb_frame": observation.get("raw_rgb_frame"),
        "previous_rgb_frame": observation.get("previous_rgb_frame"),
        "previous_action": observation.get("previous_action", "noop"),
        "candidate_score": numeric(observation.get("candidate_score", 0.0)),
        "center_salience_score": numeric(observation.get("center_salience_score", observation.get("candidate_score", 0.0))),
        "center_patch_hash": str(observation.get("center_patch_hash", "")),
        "visual_anchor_visible": bool(observation.get("visual_anchor_visible", False)),
        "visual_anchor_family": str(observation.get("visual_anchor_family", "")),
        "failed_action_event": bool(observation.get("failed_action_event", False)),
        "cache_age": int(observation.get("cache_age", 0) or 0),
        "invalidated": bool(observation.get("invalidated", False)),
        "visual_change_conflict": bool(observation.get("visual_change_conflict", False)),
        "repeat_count": int(repeat_count),
        "last_route": str(last_route),
    }


def features_for_memory_arm(features: Mapping[str, Any], memory: Mapping[str, Any], arm: str) -> dict[str, Any]:
    row = dict(features)
    used_cached = bool(memory.get("query_used_cached_fact", False))
    row["query_used_cached_fact"] = used_cached

    if arm not in {"query_memory_clean", "query_memory_shuffled", "query_memory_stale", "query_memory_wrong_binding"}:
        raise ValueError(f"Unknown long-run arm: {arm}")
    if used_cached:
        row["cache_age"] = int(memory.get("anchor_fact_age", 0))
        row["center_patch_hash"] = str(memory.get("cached_fact_hash", ""))
        if arm == "query_memory_clean":
            row.update(candidate_score=max(0.75, numeric(memory.get("cached_fact_score", 0.0))), failed_action_event=False)
        elif arm == "query_memory_stale":
            row.update(candidate_score=max(0.75, numeric(memory.get("cached_fact_score", 0.0))), cache_age=99, invalidated=True)
        else:
            salt = "shuffled" if arm == "query_memory_shuffled" else "wrong_binding"
            row.update(candidate_score=0.0, center_patch_hash=corrupted_hash(str(memory.get("cached_fact_hash", "")), salt))
    return row


def visual_memory_empty(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_family": str(observation.get("visual_anchor_family", "")),
        "anchor_currently_visible": bool(observation.get("visual_anchor_visible", False)),
        "anchor_fact_age": 0,
        "cached_fact_hash": "",
        "cached_fact_score": 0.0,
        "cache_valid": False,
        "query_used_cached_fact": False,
    }


def memory_decision_fields(
    memory: Mapping[str, Any],
    *,
    used_cached: bool,
    arm: str = "",
    action: str = "",
    resource_action: str = "",
) -> dict[str, Any]:
    family = str(memory.get("anchor_family", ""))
    memory_anchor = bool(used_cached and arm == "query_memory_clean" and family)
    followthrough = None
    if family:
        followthrough = 1.0 if memory_anchor and action == resource_action else 0.0
    return {
        "anchor_family": family,
        "anchor_fact_age": int(memory.get("anchor_fact_age", 0)),
        "anchor_currently_visible": bool(memory.get("anchor_currently_visible", False)),
        "query_used_cached_fact": bool(used_cached),
        "memory_anchor_critical": memory_anchor,
        "memory_followthrough_delta": followthrough,
    }


def query_hash_for_visual_memory(base_hash: str, *, memory: Mapping[str, Any], arm: str, used_cached: bool) -> str:
    if not used_cached:
        return base_hash
    payload = "|".join(
        [
            str(memory.get("anchor_family", "")),
            str(memory.get("cached_fact_hash", "")),
            str(memory.get("anchor_fact_age", 0)),
            arm,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def corrupted_hash(value: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]


def action_for_route(spec: RuntimeSpec, route: str) -> str:
    if route == "resource":
        return spec.resource_action
    if route == "event_fallback":
        return spec.fallback_action
    return spec.noop_action


def action_is_valid(spec: RuntimeSpec, action: str) -> bool:
    return action in spec.action_names or action in spec.action_map


def resource_memory_critical(observation: Mapping[str, Any]) -> bool:
    if "resource_memory_critical" in observation:
        return bool(observation["resource_memory_critical"])
    if observation.get("visual_anchor_family"):
        return False
    return numeric(observation.get("candidate_score", 0.0)) >= 0.5


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def optional_bool(value: Any) -> Any:
    if value is None:
        return None
    else:
        return bool(value)

def optional_value(value: Any) -> Any:
    if value is None:
        return None
    else:
        return value

def optional_int(value: int) -> Any:
    if value is None:
        return None
    else:
        return int(value)
