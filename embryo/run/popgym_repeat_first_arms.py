"""POPGym RepeatFirst long-run arm behavior."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from embryo.memory.popgym_repeat_first import (
    RepeatFirstMemory,
    repeat_first_action_for_suit,
    repeat_first_observed_suit,
    repeat_first_shuffled_suit,
    repeat_first_target_fact_from_observation,
    repeat_first_wrong_suit,
)
from embryo.models.actors import ActorDecision
from embryo.models.memory_residual_policy import MemoryResidualInput, MemoryResidualPolicy
from embryo.runtimes.base import RuntimeSpec


POPGYM_REPEAT_FIRST_CLEAN_ARM = "popgym_repeat_first_clean"
POPGYM_REPEAT_FIRST_OFF_ARM = "popgym_repeat_first_off"
POPGYM_REPEAT_FIRST_ARMS = {
    POPGYM_REPEAT_FIRST_OFF_ARM,
    POPGYM_REPEAT_FIRST_CLEAN_ARM,
    "popgym_repeat_first_shuffled",
    "popgym_repeat_first_stale",
    "popgym_repeat_first_wrong_binding",
}
POPGYM_REPEAT_FIRST_CONTROL_ARMS = (
    POPGYM_REPEAT_FIRST_OFF_ARM,
    "popgym_repeat_first_shuffled",
    "popgym_repeat_first_stale",
    "popgym_repeat_first_wrong_binding",
)


def select_action_with_popgym_repeat_first(
    spec: RuntimeSpec,
    observation: Mapping[str, Any],
    state: Any,
    arm: str,
    base_decision: ActorDecision | None,
    residual_policy: MemoryResidualPolicy | None,
    memory_residual_bias: float,
) -> dict[str, Any]:
    base_action = str(base_decision.action) if base_decision is not None else repeat_first_floor_action(spec)
    repeat = prepare_popgym_repeat_first(observation, state, spec, arm)
    target_action = repeat.get("popgym_repeat_first_effective_target_action")
    active = bool(repeat.get("popgym_repeat_first_recall_active", False))

    if base_decision is not None and residual_policy is not None and active and isinstance(target_action, str):
        residual = MemoryResidualInput(
            active=True,
            action_biases={target_action: float(memory_residual_bias)},
            content_hash=str(repeat.get("popgym_repeat_first_fact_content_hash", "")),
            freshness_age=int(repeat.get("popgym_repeat_first_fact_age", 0)),
            metadata={"target_action": target_action, "popgym_repeat_first_arm": arm},
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

    clean_target = repeat.get("popgym_repeat_first_target_action")
    ere = bool(repeat.get("popgym_repeat_first_ere", False))
    return {
        "action": action,
        "route_mode": "popgym_repeat_first_recall" if sidecar_active else "popgym_repeat_first_floor",
        "query_content_hash": str(repeat.get("popgym_repeat_first_query_content_hash", "")),
        "cache_age": int(repeat.get("popgym_repeat_first_fact_age", 0)),
        "resource_memory_critical": ere,
        "anchor_family": "",
        "anchor_fact_age": 0,
        "anchor_currently_visible": False,
        "query_used_cached_fact": False,
        "memory_anchor_critical": False,
        "memory_followthrough_delta": None,
        "resource_route_preserved": bool(ere and action == clean_target),
        "fallback_triggered": False,
        "event_self_triggered": False,
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": bool(repeat.get("popgym_repeat_first_current_matches_target", False)),
        "invalid_or_unknown_action": action not in spec.action_names and action not in spec.action_map,
        "base_actor_action": base_action,
        "memory_sidecar_active": sidecar_active,
        "memory_sidecar_changed_action": changed,
        "memory_residual_bias_count": bias_count,
        "memory_residual_target_action": target_action,
        **repeat,
        "popgym_repeat_first_recall_consistent_action": bool(ere and action == clean_target),
    }


def prepare_popgym_repeat_first(observation: Mapping[str, Any], state: Any, spec: RuntimeSpec, arm: str) -> dict[str, Any]:
    fact = repeat_first_target_fact_from_observation(observation)
    current_suit = repeat_first_observed_suit(observation)
    state.popgym_repeat_first_memory.update(fact)
    entry = state.popgym_repeat_first_memory.entry()
    age = int(entry.age) if entry is not None else 0
    clean_target = repeat_first_action_for_suit(entry.suit, spec.action_names) if entry is not None else None
    current_matches_target = bool(entry is not None and current_suit == entry.suit)
    visible_query_shortcut = bool(entry is not None and age == 0)
    ere = bool(entry is not None and age >= state.popgym_repeat_first_h_lstm and not current_matches_target and clean_target is not None)
    effective_suit = popgym_repeat_first_effective_suit(entry.suit, arm, len(spec.action_names), entry.content_hash) if ere and entry is not None else None
    effective_target = repeat_first_action_for_suit(effective_suit, spec.action_names) if effective_suit is not None else None
    if arm in {POPGYM_REPEAT_FIRST_OFF_ARM, "popgym_repeat_first_stale"}:
        effective_target = None
    active = bool(ere and effective_target is not None and arm not in {POPGYM_REPEAT_FIRST_OFF_ARM, "popgym_repeat_first_stale"})
    query_hash = popgym_repeat_first_query_hash(
        arm=arm,
        entry=entry,
        current_suit=current_suit,
        effective_suit=effective_suit,
    )
    return {
        "popgym_repeat_first_current_suit": current_suit,
        "popgym_repeat_first_fact_present": entry is not None,
        "popgym_repeat_first_fact_age": age,
        "popgym_repeat_first_fact_suit": entry.suit if entry is not None else None,
        "popgym_repeat_first_fact_content_hash": entry.content_hash if entry is not None else "",
        "popgym_repeat_first_h_lstm": int(state.popgym_repeat_first_h_lstm),
        "popgym_repeat_first_ttl": int(state.popgym_repeat_first_ttl),
        "popgym_repeat_first_current_matches_target": current_matches_target,
        "popgym_repeat_first_visible_query_shortcut": visible_query_shortcut,
        "popgym_repeat_first_ere": ere,
        "popgym_repeat_first_recall_active": active,
        "popgym_repeat_first_recall_reason": popgym_repeat_first_reason(
            arm=arm,
            ere=ere,
            entry_present=entry is not None,
            current_matches_target=current_matches_target,
        ),
        "popgym_repeat_first_target_action": clean_target,
        "popgym_repeat_first_effective_suit": effective_suit,
        "popgym_repeat_first_effective_target_action": effective_target,
        "popgym_repeat_first_query_content_hash": query_hash,
        "effective_memory_arm": arm,
        "effective_cache_age": age,
        "effective_center_patch_hash": entry.content_hash if entry is not None else "",
        "effective_candidate_score": 1.0 if entry is not None else 0.0,
        "effective_fact_value": current_matches_target,
        "effective_invalidated": bool(arm == "popgym_repeat_first_stale" and entry is not None),
    }


def make_repeat_first_memory(*, ttl: int) -> RepeatFirstMemory:
    return RepeatFirstMemory(ttl=ttl)


def repeat_first_floor_action(spec: RuntimeSpec) -> str:
    return "suit_0" if "suit_0" in spec.action_names else spec.noop_action


def popgym_repeat_first_effective_suit(suit: int, arm: str, action_count: int, content_hash: str) -> int | None:
    if arm == POPGYM_REPEAT_FIRST_CLEAN_ARM:
        return int(suit)
    if arm == "popgym_repeat_first_wrong_binding":
        return repeat_first_wrong_suit(int(suit), action_count)
    if arm == "popgym_repeat_first_shuffled":
        return repeat_first_shuffled_suit(int(suit), action_count, content_hash)
    return None


def popgym_repeat_first_reason(*, arm: str, ere: bool, entry_present: bool, current_matches_target: bool) -> str:
    if arm == POPGYM_REPEAT_FIRST_OFF_ARM:
        return "off"
    if not entry_present:
        return "not_found"
    if current_matches_target:
        return "current_observation_matches_target"
    if not ere:
        return "not_evaluable"
    if arm == "popgym_repeat_first_stale":
        return "stale_control_inactive"
    if arm == "popgym_repeat_first_wrong_binding":
        return "wrong_binding_control"
    if arm == "popgym_repeat_first_shuffled":
        return "shuffled_control"
    return "recall_target"


def popgym_repeat_first_query_hash(
    *,
    arm: str,
    entry: Any,
    current_suit: int | None,
    effective_suit: int | None,
) -> str:
    if entry is None:
        return ""
    payload = {
        "arm": arm,
        "target_suit": entry.suit,
        "content_hash": entry.content_hash,
        "age": entry.age,
        "current_suit": current_suit,
        "effective_suit": effective_suit,
    }
    return hashlib.sha1(str(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
