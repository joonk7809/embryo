"""Long-run protocol arm policies and deployable action helpers."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from collections.abc import Mapping
from typing import Any

from embryo.memory.freshness import FreshnessState
from embryo.memory.queries import build_event_query, build_resource_query, combine_queries
from embryo.memory.typed_recall import (
    CRAFTING_BENCH_RESOURCE,
    PassiveCueMemory,
    PlacedLandmarkMemory,
    ResourceMemoryEntry,
    ResourceNeedBelief,
    ResourceRecallMemory,
    action_for_bearing,
    passive_match_action_for_side,
    passive_match_choice_visible,
    passive_match_cue_fact_from_observation,
    water_bearing_fact_from_observation,
)
from embryo.models.actors import ActorDecision
from embryo.models.memory_residual_policy import MemoryResidualInput, MemoryResidualPolicy
from embryo.models import RuleRouterModel, ThresholdFactWriter
from embryo.run.popgym_repeat_first_arms import (
    POPGYM_REPEAT_FIRST_ARMS,
    make_repeat_first_memory,
    select_action_with_popgym_repeat_first,
)
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

MEMORY_RESIDUAL_BIAS = 8.0
MEMORY_ARMS = {
    "query_memory_clean",
    "query_memory_shuffled",
    "query_memory_stale",
    "query_memory_wrong_binding",
    "collapsed_corrupt_memory",
}
WATER_RECALL_CLEAN_ARM = "water_recall_clean"
WATER_RECALL_OFF_ARM = "water_recall_off"
WATER_RECALL_ARMS = {
    WATER_RECALL_OFF_ARM,
    WATER_RECALL_CLEAN_ARM,
    "water_recall_shuffled",
    "water_recall_stale",
    "water_recall_wrong_binding",
}
WATER_RECALL_CONTROL_ARMS = (
    WATER_RECALL_OFF_ARM,
    "water_recall_shuffled",
    "water_recall_stale",
    "water_recall_wrong_binding",
)
WATER_RECALL_BEARINGS = ("left", "right", "up", "down", "center")
BENCH_RECALL_CLEAN_ARM = "bench_recall_clean"
BENCH_RECALL_OFF_ARM = "bench_recall_off"
BENCH_RECALL_ARMS = {
    BENCH_RECALL_OFF_ARM,
    BENCH_RECALL_CLEAN_ARM,
    "bench_recall_shuffled",
    "bench_recall_stale",
    "bench_recall_wrong_binding",
}
BENCH_RECALL_CONTROL_ARMS = (
    BENCH_RECALL_OFF_ARM,
    "bench_recall_shuffled",
    "bench_recall_stale",
    "bench_recall_wrong_binding",
)
CRAFTING_INTENT_ACTIONS = {
    "place_table",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
}
PASSIVE_MATCH_CLEAN_ARM = "passive_match_clean"
PASSIVE_MATCH_OFF_ARM = "passive_match_off"
PASSIVE_MATCH_ARMS = {
    PASSIVE_MATCH_OFF_ARM,
    PASSIVE_MATCH_CLEAN_ARM,
    "passive_match_shuffled",
    "passive_match_stale",
    "passive_match_wrong_binding",
}
PASSIVE_MATCH_CONTROL_ARMS = (
    PASSIVE_MATCH_OFF_ARM,
    "passive_match_shuffled",
    "passive_match_stale",
    "passive_match_wrong_binding",
)


class EpisodeState:
    def __init__(
        self,
        *,
        seed: int,
        arm: str,
        horizon: int,
        water_recall_config: Mapping[str, Any] | None = None,
        bench_recall_config: Mapping[str, Any] | None = None,
        passive_match_config: Mapping[str, Any] | None = None,
        popgym_repeat_first_config: Mapping[str, Any] | None = None,
    ) -> None:
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
        water_cfg = water_recall_config if isinstance(water_recall_config, Mapping) else {}
        self.water_recall_ttl = int(water_cfg.get("ttl", 384))
        self.water_recall_h_lstm = int(water_cfg.get("h_lstm", 32))
        self.water_need_belief = ResourceNeedBelief(need_after_steps=int(water_cfg.get("need_after_steps", 160)))
        self.water_recall_memory = ResourceRecallMemory(ttl=self.water_recall_ttl)
        bench_cfg = bench_recall_config if isinstance(bench_recall_config, Mapping) else {}
        self.bench_recall_ttl = int(bench_cfg.get("ttl", 768))
        self.bench_recall_h_lstm = int(bench_cfg.get("h_lstm", 16))
        self.bench_recall_memory = PlacedLandmarkMemory(resource_type=CRAFTING_BENCH_RESOURCE, ttl=self.bench_recall_ttl)
        passive_cfg = passive_match_config if isinstance(passive_match_config, Mapping) else {}
        self.passive_match_ttl = int(passive_cfg.get("ttl", 1024))
        self.passive_match_h_lstm = int(passive_cfg.get("h_lstm", 8))
        self.passive_match_memory = PassiveCueMemory(ttl=self.passive_match_ttl)
        repeat_cfg = popgym_repeat_first_config if isinstance(popgym_repeat_first_config, Mapping) else {}
        self.popgym_repeat_first_ttl = int(repeat_cfg.get("ttl", 1024))
        self.popgym_repeat_first_h_lstm = int(repeat_cfg.get("h_lstm", 4))
        self.popgym_repeat_first_memory = make_repeat_first_memory(ttl=self.popgym_repeat_first_ttl)

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
    if arm in POPGYM_REPEAT_FIRST_ARMS:
        return select_action_with_popgym_repeat_first(spec, observation, state, arm, None, None, MEMORY_RESIDUAL_BIAS)
    if arm in PASSIVE_MATCH_ARMS:
        return select_action_with_passive_match(spec, observation, state, arm, None, None)
    if arm in WATER_RECALL_ARMS:
        return select_action_with_water_recall(spec, observation, state, arm, None, None)
    if arm in BENCH_RECALL_ARMS:
        return select_action_with_bench_recall(spec, observation, state, arm, None, None)
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
    target_action = memory_residual_target_action(spec, arm=arm, effective_content_hash=str(features.get("center_patch_hash", "")))
    if bool(fact.value) and not bool(router_output.stale_invalidated) and target_action is not None:
        action = target_action
    else:
        action = action_for_route(spec, router_output.route_mode)
    event_self_triggered = state.last_route == "event_fallback" and router_output.route_mode == "event_fallback" and not bool(
        features.get("failed_action_event", False)
    )
    query_hash = query_hash_for_visual_memory(
        combined_query.content_hash,
        memory=memory,
        arm=arm,
        used_cached=bool(features.get("query_used_cached_fact")),
        effective_content_hash=str(features.get("center_patch_hash", "")),
    )
    return {
        "action": action,
        "route_mode": router_output.route_mode,
        "query_content_hash": query_hash,
        "cache_age": freshness.cache_age,
        "resource_memory_critical": resource_memory_critical(observation),
        **memory_decision_fields(memory, used_cached=bool(features.get("query_used_cached_fact")), arm=arm, action=action, resource_action=spec.resource_action),
        "resource_route_preserved": bool(router_output.resource_route_preserved and action == spec.resource_action),
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
        "memory_residual_target_action": target_action if not bool(router_output.stale_invalidated) else None,
    }


def select_action_with_actor_sidecar(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: EpisodeState,
    arm: str,
    base_decision: ActorDecision,
    residual_policy: MemoryResidualPolicy,
    memory_residual_bias: float = MEMORY_RESIDUAL_BIAS,
) -> dict[str, Any]:
    if arm in {"random_valid_action", "reference_exploration_sweep"}:
        return select_action_for_arm(spec, observation, state, arm)

    if arm in WATER_RECALL_ARMS:
        return select_action_with_water_recall(spec, observation, state, arm, base_decision, residual_policy, memory_residual_bias)

    if arm in BENCH_RECALL_ARMS:
        return select_action_with_bench_recall(spec, observation, state, arm, base_decision, residual_policy, memory_residual_bias)

    if arm in PASSIVE_MATCH_ARMS:
        return select_action_with_passive_match(spec, observation, state, arm, base_decision, residual_policy, memory_residual_bias)

    if arm in POPGYM_REPEAT_FIRST_ARMS:
        return select_action_with_popgym_repeat_first(spec, observation, state, arm, base_decision, residual_policy, memory_residual_bias)

    if arm == "no_memory":
        return actor_passthrough_decision(spec, observation, state, arm, base_decision)

    base_features = deployable_features(observation, repeat_count=state.repeat_count, last_route=state.last_route)
    memory = state.prepare_visual_memory(base_features)
    if not bool(memory.get("query_used_cached_fact", False)):
        return actor_passthrough_decision(spec, observation, state, arm, base_decision, memory=memory, route_mode="actor_passthrough")

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
    residual = residual_for_memory_arm(
        spec,
        arm=arm,
        fact_value=bool(fact.value),
        stale_invalidated=bool(router_output.stale_invalidated),
        effective_content_hash=str(features.get("center_patch_hash", "")),
        memory_residual_bias=memory_residual_bias,
    )
    residual_decision = residual_policy.apply(base_decision, residual, action_order=spec.action_names)
    action = residual_decision.action
    used_cached = bool(features.get("query_used_cached_fact", False))
    event_self_triggered = state.last_route == "event_fallback" and router_output.route_mode == "event_fallback" and not bool(
        features.get("failed_action_event", False)
    )
    return {
        "action": action,
        "route_mode": "actor_memory_sidecar" if residual_decision.active else "actor_passthrough",
        "query_content_hash": query_hash_for_visual_memory(
            combined_query.content_hash,
            memory=memory,
            arm=arm,
            used_cached=used_cached,
            effective_content_hash=str(features.get("center_patch_hash", "")),
        ),
        "cache_age": freshness.cache_age,
        "resource_memory_critical": resource_memory_critical(observation),
        **memory_decision_fields(memory, used_cached=used_cached, arm=arm, action=action, resource_action=spec.resource_action),
        "resource_route_preserved": bool(action == spec.resource_action),
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
        "base_actor_action": base_decision.action,
        "memory_sidecar_active": residual_decision.active,
        "memory_sidecar_changed_action": residual_decision.changed_action,
        "memory_residual_bias_count": residual_decision.applied_bias_count,
        "memory_residual_target_action": residual_decision.metadata.get("target_action"),
    }


def select_action_with_passive_match(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: EpisodeState,
    arm: str,
    base_decision: ActorDecision | None,
    residual_policy: MemoryResidualPolicy | None,
    memory_residual_bias: float = MEMORY_RESIDUAL_BIAS,
) -> dict[str, Any]:
    _ = residual_policy
    _ = memory_residual_bias
    base_action = str(base_decision.action) if base_decision is not None else passive_match_floor_action(spec)
    passive = prepare_passive_match(observation, state, spec, arm)
    target_action = passive.get("passive_match_effective_target_action")
    active = bool(passive.get("passive_match_recall_active", False))
    action = str(target_action) if active and isinstance(target_action, str) else base_action
    clean_target = passive.get("passive_match_target_action")
    return {
        "action": action,
        "route_mode": "passive_match_recall" if active else "passive_match_floor",
        "query_content_hash": str(passive.get("passive_match_query_content_hash", "")),
        "cache_age": int(passive.get("passive_match_fact_age", 0)),
        "resource_memory_critical": bool(passive.get("passive_match_ere", False)),
        **memory_decision_fields(visual_memory_empty(observation), used_cached=False),
        "resource_route_preserved": bool(passive.get("passive_match_ere", False) and action == clean_target),
        "fallback_triggered": False,
        "event_self_triggered": False,
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": bool(passive.get("passive_match_cue_visible", False)),
        "invalid_or_unknown_action": not action_is_valid(spec, action),
        "base_actor_action": base_action,
        "memory_sidecar_active": active,
        "memory_sidecar_changed_action": action != base_action,
        "memory_residual_bias_count": int(active),
        "memory_residual_target_action": target_action,
        **passive,
        "passive_match_recall_consistent_action": bool(passive.get("passive_match_ere", False) and action == clean_target),
    }


def prepare_passive_match(observation: Mapping[str, Any], state: EpisodeState, spec: RuntimeSpec, arm: str) -> dict[str, Any]:
    fact = passive_match_cue_fact_from_observation(observation)
    cue_visible = bool(fact.value)
    choice_visible = passive_match_choice_visible(observation)
    state.passive_match_memory.update(fact)
    entry = state.passive_match_memory.entry()
    age = int(entry.age) if entry is not None else 0
    clean_target = passive_match_action_for_side(entry.side, spec.action_names) if entry is not None else None
    ere = bool(choice_visible and entry is not None and not cue_visible and age >= state.passive_match_h_lstm and clean_target is not None)
    effective_side = passive_match_effective_side(entry.side, arm) if ere and entry is not None else None
    effective_target = passive_match_action_for_side(effective_side, spec.action_names) if effective_side else None
    if arm in {PASSIVE_MATCH_OFF_ARM, "passive_match_stale"}:
        effective_target = None
    active = bool(ere and effective_target is not None and arm not in {PASSIVE_MATCH_OFF_ARM, "passive_match_stale"})
    query_hash = passive_match_query_hash(arm=arm, entry=entry, choice_visible=choice_visible, effective_side=effective_side)
    return {
        "passive_match_cue_visible": cue_visible,
        "passive_match_choice_visible": choice_visible,
        "passive_match_fact_present": entry is not None,
        "passive_match_fact_age": age,
        "passive_match_fact_side": entry.side if entry is not None else "",
        "passive_match_fact_content_hash": entry.content_hash if entry is not None else "",
        "passive_match_h_lstm": int(state.passive_match_h_lstm),
        "passive_match_ttl": int(state.passive_match_ttl),
        "passive_match_ere": ere,
        "passive_match_recall_active": active,
        "passive_match_recall_reason": passive_match_reason(arm=arm, ere=ere, choice_visible=choice_visible, entry_present=entry is not None),
        "passive_match_target_action": clean_target,
        "passive_match_effective_side": effective_side,
        "passive_match_effective_target_action": effective_target,
        "passive_match_query_content_hash": query_hash,
        "effective_memory_arm": arm,
        "effective_cache_age": age,
        "effective_center_patch_hash": entry.content_hash if entry is not None else "",
        "effective_candidate_score": 1.0 if entry is not None else 0.0,
        "effective_fact_value": cue_visible,
        "effective_invalidated": bool(arm == "passive_match_stale" and entry is not None),
    }


def passive_match_floor_action(spec: RuntimeSpec) -> str:
    return "choose_left" if "choose_left" in spec.action_names else spec.noop_action


def passive_match_effective_side(side: str, arm: str) -> str | None:
    if arm == PASSIVE_MATCH_CLEAN_ARM:
        return side
    if arm in {"passive_match_shuffled", "passive_match_wrong_binding"}:
        return "right" if side == "left" else "left"
    return None


def passive_match_reason(*, arm: str, ere: bool, choice_visible: bool, entry_present: bool) -> str:
    if arm == PASSIVE_MATCH_OFF_ARM:
        return "off"
    if not choice_visible:
        return "not_choice"
    if not entry_present:
        return "not_found"
    if not ere:
        return "not_evaluable"
    if arm == "passive_match_stale":
        return "stale_control_inactive"
    if arm == "passive_match_wrong_binding":
        return "wrong_binding_control"
    if arm == "passive_match_shuffled":
        return "shuffled_control"
    return "recall_choice"


def passive_match_query_hash(
    *,
    arm: str,
    entry: Any,
    choice_visible: bool,
    effective_side: str | None,
) -> str:
    if entry is None:
        return ""
    payload = {
        "arm": arm,
        "side": entry.side,
        "content_hash": entry.content_hash,
        "age": entry.age,
        "choice_visible": bool(choice_visible),
        "effective_side": effective_side,
    }
    return hashlib.sha1(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]


def select_action_with_water_recall(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: EpisodeState,
    arm: str,
    base_decision: ActorDecision | None,
    residual_policy: MemoryResidualPolicy | None,
    memory_residual_bias: float = MEMORY_RESIDUAL_BIAS,
) -> dict[str, Any]:
    base_action = str(base_decision.action) if base_decision is not None else spec.noop_action
    water = prepare_water_recall(observation, state, spec, arm)
    active = bool(water["water_recall_active"])
    target_action = water.get("water_recall_effective_target_action")

    if base_decision is not None and residual_policy is not None:
        residual = MemoryResidualInput(
            active=active and isinstance(target_action, str),
            action_biases={str(target_action): float(memory_residual_bias)} if active and isinstance(target_action, str) else {},
            content_hash=str(water.get("water_fact_content_hash", "")),
            freshness_age=int(water.get("water_fact_age", 0)),
            metadata={
                "target_action": target_action,
                "clean_target_action": water.get("water_recall_target_action"),
                "water_recall_arm": arm,
            },
        )
        residual_decision = residual_policy.apply(base_decision, residual, action_order=spec.action_names)
        action = residual_decision.action
        sidecar_active = residual_decision.active
        changed = residual_decision.changed_action
        bias_count = residual_decision.applied_bias_count
    else:
        action = str(target_action) if active and isinstance(target_action, str) else base_action
        sidecar_active = active
        changed = action != base_action
        bias_count = int(active)

    query_hash = str(water.get("water_query_content_hash", ""))
    clean_target = water.get("water_recall_target_action")
    return {
        "action": action,
        "route_mode": "water_recall_sidecar" if sidecar_active else "actor_passthrough",
        "query_content_hash": query_hash,
        "cache_age": int(water.get("water_fact_age", 0)),
        "resource_memory_critical": bool(water.get("water_recall_ere", False)),
        **memory_decision_fields(visual_memory_empty(observation), used_cached=False),
        "resource_route_preserved": bool(water.get("water_recall_ere", False) and action == clean_target),
        "fallback_triggered": False,
        "event_self_triggered": False,
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": bool(water.get("water_visible", False)),
        "invalid_or_unknown_action": not action_is_valid(spec, action),
        "base_actor_action": base_action,
        "memory_sidecar_active": sidecar_active,
        "memory_sidecar_changed_action": changed,
        "memory_residual_bias_count": bias_count,
        "memory_residual_target_action": target_action,
        **water,
        "water_recall_consistent_action": bool(water.get("water_recall_ere", False) and action == clean_target),
    }


def prepare_water_recall(observation: Mapping[str, Any], state: EpisodeState, spec: RuntimeSpec, arm: str) -> dict[str, Any]:
    fact = water_bearing_fact_from_observation(observation)
    water_visible = bool(fact.value)
    visible_resources = ("water",) if water_visible else ()
    need_active = state.water_need_belief.update(
        previous_action=str(observation.get("previous_action", state.previous_action or spec.noop_action)),
        visible_resources=visible_resources,
    )
    state.water_recall_memory.update([fact])
    entry = state.water_recall_memory.entry("water")
    clean_target_action = water_clean_target_action(entry, spec)
    age = int(entry.age) if entry is not None else 0
    within_ttl = entry is not None and age <= state.water_recall_ttl
    ere = bool(need_active and within_ttl and not water_visible and age >= state.water_recall_h_lstm and clean_target_action is not None)
    effective_bearing = water_effective_bearing(entry, arm, state) if ere and entry is not None else None
    effective_target = action_for_bearing(effective_bearing, action_names=spec.action_names) if effective_bearing else None
    if arm in {WATER_RECALL_OFF_ARM, "water_recall_stale"}:
        effective_target = None
    active = bool(ere and effective_target is not None and arm not in {WATER_RECALL_OFF_ARM, "water_recall_stale"})
    reason = water_recall_reason(arm=arm, ere=ere, need_active=need_active, visible=water_visible, entry=entry)
    query_hash = water_query_hash(
        arm=arm,
        entry=entry,
        need_active=need_active,
        water_visible=water_visible,
        effective_bearing=effective_bearing,
    )
    return {
        "water_visible": water_visible,
        "water_bearing": str(fact.metadata.get("bearing", "unknown")),
        "water_fact_present": entry is not None,
        "water_fact_age": age,
        "water_fact_content_hash": entry.content_hash if entry is not None else "",
        "water_fact_bearing": entry.bearing if entry is not None else "",
        "water_need_active": bool(need_active),
        "water_need_steps_since_resolution": int(state.water_need_belief.steps_since_resolution),
        "water_recall_h_lstm": int(state.water_recall_h_lstm),
        "water_recall_ttl": int(state.water_recall_ttl),
        "water_recall_ere": ere,
        "water_recall_active": active,
        "water_recall_reason": reason,
        "water_recall_target_action": clean_target_action,
        "water_recall_effective_bearing": effective_bearing,
        "water_recall_effective_target_action": effective_target,
        "water_query_content_hash": query_hash,
        "effective_memory_arm": arm,
        "effective_cache_age": age,
        "effective_center_patch_hash": entry.content_hash if entry is not None else "",
        "effective_candidate_score": entry.confidence if entry is not None else 0.0,
        "effective_fact_value": water_visible,
        "effective_invalidated": bool(arm == "water_recall_stale" and entry is not None),
    }


def water_clean_target_action(entry: ResourceMemoryEntry | None, spec: RuntimeSpec) -> str | None:
    if entry is None:
        return None
    return action_for_bearing(entry.bearing, action_names=spec.action_names)


def water_effective_bearing(entry: ResourceMemoryEntry | None, arm: str, state: EpisodeState) -> str | None:
    if entry is None or arm in {WATER_RECALL_OFF_ARM, "water_recall_stale"}:
        return None
    if arm == WATER_RECALL_CLEAN_ARM:
        return entry.bearing
    if arm == "water_recall_wrong_binding":
        return wrong_water_bearing(entry.bearing)
    if arm == "water_recall_shuffled":
        candidates = [bearing for bearing in WATER_RECALL_BEARINGS if bearing != entry.bearing]
        return state.rng.choice(tuple(candidates)) if candidates else entry.bearing
    return None


def wrong_water_bearing(bearing: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
        "center": "left",
    }.get(str(bearing), "left")


def water_recall_reason(
    *,
    arm: str,
    ere: bool,
    need_active: bool,
    visible: bool,
    entry: ResourceMemoryEntry | None,
) -> str:
    if arm == WATER_RECALL_OFF_ARM:
        return "off"
    if not need_active:
        return "not_needed"
    if visible:
        return "water_visible"
    if entry is None:
        return "not_found"
    if not ere:
        return "not_evaluable"
    if arm == "water_recall_stale":
        return "stale_control_inactive"
    if arm == "water_recall_wrong_binding":
        return "wrong_binding_control"
    if arm == "water_recall_shuffled":
        return "shuffled_control"
    return "recall_navigation"


def water_query_hash(
    *,
    arm: str,
    entry: ResourceMemoryEntry | None,
    need_active: bool,
    water_visible: bool,
    effective_bearing: str | None,
) -> str:
    if entry is None:
        return ""
    payload = {
        "arm": arm,
        "resource_type": entry.resource_type,
        "content_hash": entry.content_hash,
        "age": entry.age,
        "need_active": bool(need_active),
        "water_visible": bool(water_visible),
        "effective_bearing": effective_bearing,
    }
    return hashlib.sha1(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]


def select_action_with_bench_recall(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: EpisodeState,
    arm: str,
    base_decision: ActorDecision | None,
    residual_policy: MemoryResidualPolicy | None,
    memory_residual_bias: float = MEMORY_RESIDUAL_BIAS,
) -> dict[str, Any]:
    base_action = str(base_decision.action) if base_decision is not None else spec.noop_action
    bench = prepare_bench_recall(observation, state, spec, arm, base_action=base_action)
    active = bool(bench["bench_recall_active"])
    target_action = bench.get("bench_recall_effective_target_action")

    if base_decision is not None and residual_policy is not None:
        residual = MemoryResidualInput(
            active=active and isinstance(target_action, str),
            action_biases={str(target_action): float(memory_residual_bias)} if active and isinstance(target_action, str) else {},
            content_hash=str(bench.get("bench_fact_content_hash", "")),
            freshness_age=int(bench.get("bench_fact_age", 0)),
            metadata={
                "target_action": target_action,
                "clean_target_action": bench.get("bench_recall_target_action"),
                "bench_recall_arm": arm,
            },
        )
        residual_decision = residual_policy.apply(base_decision, residual, action_order=spec.action_names)
        action = residual_decision.action
        sidecar_active = residual_decision.active
        changed = residual_decision.changed_action
        bias_count = residual_decision.applied_bias_count
    else:
        action = str(target_action) if active and isinstance(target_action, str) else base_action
        sidecar_active = active
        changed = action != base_action
        bias_count = int(active)

    clean_target = bench.get("bench_recall_target_action")
    if (
        arm == BENCH_RECALL_CLEAN_ARM
        and not sidecar_active
        and bool(bench.get("bench_fact_present", False))
        and base_action == "place_table"
    ):
        action = spec.noop_action
        sidecar_active = True
        changed = action != base_action
        bias_count = 1

    return {
        "action": action,
        "route_mode": "bench_recall_sidecar" if sidecar_active else "actor_passthrough",
        "query_content_hash": str(bench.get("bench_query_content_hash", "")),
        "cache_age": int(bench.get("bench_fact_age", 0)),
        "resource_memory_critical": bool(bench.get("bench_recall_ere", False)),
        **memory_decision_fields(visual_memory_empty(observation), used_cached=False),
        "resource_route_preserved": bool(bench.get("bench_recall_ere", False) and action == clean_target),
        "fallback_triggered": False,
        "event_self_triggered": False,
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": bool(bench.get("bench_fact_present", False)),
        "invalid_or_unknown_action": not action_is_valid(spec, action),
        "base_actor_action": base_action,
        "memory_sidecar_active": sidecar_active,
        "memory_sidecar_changed_action": changed,
        "memory_residual_bias_count": bias_count,
        "memory_residual_target_action": target_action,
        **bench,
        "bench_recall_consistent_action": bool(bench.get("bench_recall_ere", False) and action == clean_target),
        "bench_repeat_place_suppressed": bool(base_action == "place_table" and action != "place_table" and bench.get("bench_fact_present", False)),
    }


def prepare_bench_recall(
    observation: Mapping[str, Any],
    state: EpisodeState,
    spec: RuntimeSpec,
    arm: str,
    *,
    base_action: str,
) -> dict[str, Any]:
    previous_action = str(observation.get("previous_action", state.previous_action or spec.noop_action))
    state.bench_recall_memory.update_from_previous_action(
        previous_action,
        placed_action="place_table",
        failed_action_event=bool(observation.get("failed_action_event", False)),
    )
    entry = state.bench_recall_memory.entry()
    need_active = crafting_intent_active(base_action)
    clean_target_action = bench_clean_target_action(entry, spec)
    age = int(entry.age) if entry is not None else 0
    offscreen = bool(entry is not None and entry.bearing != "center")
    within_ttl = entry is not None and age <= state.bench_recall_ttl
    ere = bool(need_active and within_ttl and offscreen and age >= state.bench_recall_h_lstm and clean_target_action is not None)
    effective_bearing = bench_effective_bearing(entry, arm, state) if ere and entry is not None else None
    effective_target = action_for_bearing(effective_bearing, action_names=spec.action_names) if effective_bearing else None
    if arm in {BENCH_RECALL_OFF_ARM, "bench_recall_stale"}:
        effective_target = None
    active = bool(ere and effective_target is not None and arm not in {BENCH_RECALL_OFF_ARM, "bench_recall_stale"})
    query_hash = bench_query_hash(
        arm=arm,
        entry=entry,
        need_active=need_active,
        effective_bearing=effective_bearing,
        base_action=base_action,
    )
    return {
        "bench_fact_present": entry is not None,
        "bench_fact_age": age,
        "bench_fact_content_hash": entry.content_hash if entry is not None else "",
        "bench_fact_bearing": entry.bearing if entry is not None else "",
        "bench_need_active": bool(need_active),
        "bench_base_crafting_action": base_action if need_active else "",
        "bench_offscreen": offscreen,
        "bench_recall_h_lstm": int(state.bench_recall_h_lstm),
        "bench_recall_ttl": int(state.bench_recall_ttl),
        "bench_recall_ere": ere,
        "bench_recall_active": active,
        "bench_recall_reason": bench_recall_reason(arm=arm, ere=ere, need_active=need_active, entry=entry),
        "bench_recall_target_action": clean_target_action,
        "bench_recall_effective_bearing": effective_bearing,
        "bench_recall_effective_target_action": effective_target,
        "bench_query_content_hash": query_hash,
        "effective_memory_arm": arm,
        "effective_cache_age": age,
        "effective_center_patch_hash": entry.content_hash if entry is not None else "",
        "effective_candidate_score": entry.confidence if entry is not None else 0.0,
        "effective_fact_value": entry is not None,
        "effective_invalidated": bool(arm == "bench_recall_stale" and entry is not None),
    }


def crafting_intent_active(action: str) -> bool:
    return str(action) in CRAFTING_INTENT_ACTIONS or str(action).startswith("make_")


def bench_clean_target_action(entry: ResourceMemoryEntry | None, spec: RuntimeSpec) -> str | None:
    if entry is None:
        return None
    if entry.bearing == "center":
        return spec.noop_action if spec.noop_action in spec.action_names else None
    return action_for_bearing(entry.bearing, action_names=spec.action_names)


def bench_effective_bearing(entry: ResourceMemoryEntry | None, arm: str, state: EpisodeState) -> str | None:
    if entry is None or arm in {BENCH_RECALL_OFF_ARM, "bench_recall_stale"}:
        return None
    if arm == BENCH_RECALL_CLEAN_ARM:
        return entry.bearing
    if arm == "bench_recall_wrong_binding":
        return wrong_water_bearing(entry.bearing)
    if arm == "bench_recall_shuffled":
        candidates = [bearing for bearing in WATER_RECALL_BEARINGS if bearing != entry.bearing]
        return state.rng.choice(tuple(candidates)) if candidates else entry.bearing
    return None


def bench_recall_reason(*, arm: str, ere: bool, need_active: bool, entry: ResourceMemoryEntry | None) -> str:
    if arm == BENCH_RECALL_OFF_ARM:
        return "off"
    if not need_active:
        return "not_needed"
    if entry is None:
        return "not_placed"
    if entry.bearing == "center":
        return "bench_center"
    if not ere:
        return "not_evaluable"
    if arm == "bench_recall_stale":
        return "stale_control_inactive"
    if arm == "bench_recall_wrong_binding":
        return "wrong_binding_control"
    if arm == "bench_recall_shuffled":
        return "shuffled_control"
    return "recall_navigation"


def bench_query_hash(
    *,
    arm: str,
    entry: ResourceMemoryEntry | None,
    need_active: bool,
    effective_bearing: str | None,
    base_action: str,
) -> str:
    if entry is None:
        return ""
    payload = {
        "arm": arm,
        "resource_type": entry.resource_type,
        "content_hash": entry.content_hash,
        "age": entry.age,
        "need_active": bool(need_active),
        "effective_bearing": effective_bearing,
        "base_action": base_action,
    }
    return hashlib.sha1(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]


def actor_passthrough_decision(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: EpisodeState,
    arm: str,
    base_decision: ActorDecision,
    *,
    memory: Mapping[str, Any] | None = None,
    route_mode: str = "actor_passthrough",
) -> dict[str, Any]:
    memory_fields = visual_memory_empty(observation) if memory is None else memory
    action = str(base_decision.action)
    return {
        "action": action,
        "route_mode": route_mode,
        "query_content_hash": "",
        "cache_age": 0,
        "resource_memory_critical": resource_memory_critical(observation),
        **memory_decision_fields(memory_fields, used_cached=False, arm=arm, action=action, resource_action=spec.resource_action),
        "resource_route_preserved": False,
        "fallback_triggered": False,
        "event_self_triggered": False,
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": False,
        "invalid_or_unknown_action": not action_is_valid(spec, action),
        "base_actor_action": base_decision.action,
        "memory_sidecar_active": False,
        "memory_sidecar_changed_action": False,
        "memory_residual_bias_count": 0,
        "memory_residual_target_action": None,
    }


def residual_for_memory_arm(
    spec: RuntimeSpec,
    *,
    arm: str,
    fact_value: bool,
    stale_invalidated: bool = False,
    effective_content_hash: str = "",
    memory_residual_bias: float = MEMORY_RESIDUAL_BIAS,
) -> MemoryResidualInput:
    target_action = memory_residual_target_action(spec, arm=arm, effective_content_hash=effective_content_hash)
    metadata = {"target_action": target_action, "stale_invalidated": bool(stale_invalidated)}
    if not fact_value or stale_invalidated or target_action is None:
        return MemoryResidualInput(active=False, content_hash=effective_content_hash, metadata=metadata)
    return MemoryResidualInput(
        active=True,
        action_biases={target_action: float(memory_residual_bias)},
        content_hash=effective_content_hash,
        metadata=metadata,
    )


def memory_residual_target_action(spec: RuntimeSpec, *, arm: str, effective_content_hash: str = "") -> str | None:
    if arm == "query_memory_clean":
        return spec.resource_action
    if arm == "query_memory_shuffled":
        return deterministic_corrupt_action(spec, effective_content_hash)
    if arm in {"query_memory_wrong_binding", "collapsed_corrupt_memory"}:
        return spec.fallback_action
    return None


def deterministic_corrupt_action(spec: RuntimeSpec, effective_content_hash: str) -> str:
    candidates = [action for action in spec.action_names if action not in {spec.resource_action, spec.noop_action}]
    if not candidates:
        candidates = [action for action in spec.action_names if action != spec.resource_action]
    if not candidates:
        return spec.noop_action
    digest = hashlib.sha1(str(effective_content_hash).encode("utf-8")).hexdigest()
    return candidates[int(digest[:8], 16) % len(candidates)]


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

    if arm not in MEMORY_ARMS:
        raise ValueError(f"Unknown long-run arm: {arm}")
    if used_cached:
        row["cache_age"] = int(memory.get("anchor_fact_age", 0))
        row["center_patch_hash"] = str(memory.get("cached_fact_hash", ""))
        if arm == "query_memory_clean":
            row.update(candidate_score=max(0.75, numeric(memory.get("cached_fact_score", 0.0))), failed_action_event=False)
        elif arm == "query_memory_shuffled":
            row.update(
                candidate_score=max(0.75, numeric(memory.get("cached_fact_score", 0.0))),
                center_patch_hash=corrupted_hash(str(memory.get("cached_fact_hash", "")), "shuffled"),
                failed_action_event=False,
            )
        elif arm == "query_memory_stale":
            row.update(candidate_score=max(0.75, numeric(memory.get("cached_fact_score", 0.0))), cache_age=99, invalidated=True)
        elif arm in {"query_memory_wrong_binding", "collapsed_corrupt_memory"}:
            row.update(candidate_score=max(0.75, numeric(memory.get("cached_fact_score", 0.0))), failed_action_event=False)
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


def query_hash_for_visual_memory(
    base_hash: str,
    *,
    memory: Mapping[str, Any],
    arm: str,
    used_cached: bool,
    effective_content_hash: str = "",
) -> str:
    if not used_cached:
        return base_hash
    payload = "|".join(
        [
            str(base_hash),
            str(memory.get("anchor_family", "")),
            str(memory.get("cached_fact_hash", "")),
            str(effective_content_hash),
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
