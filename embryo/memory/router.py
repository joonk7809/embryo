"""Freshness-aware router core."""

from __future__ import annotations

from dataclasses import dataclass

from embryo.memory.freshness import FreshnessState, short_ttl_valid


@dataclass(frozen=True)
class RouterState:
    cooldown: int = 0
    repeat_count: int = 0
    last_route: str = "hold"
    event_self_triggered: bool = False


@dataclass(frozen=True)
class RouterDecision:
    route: str
    resource_route_preserved: bool
    event_fallback_gate: bool
    repeat_loop_breaker_gate: bool
    stale_invalidated: bool
    event_self_triggered: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "resource_route_preserved": self.resource_route_preserved,
            "event_fallback_gate": self.event_fallback_gate,
            "repeat_loop_breaker_gate": self.repeat_loop_breaker_gate,
            "stale_invalidated": self.stale_invalidated,
            "event_self_triggered": self.event_self_triggered,
        }


def route_with_freshness(
    *,
    facing_candidate: bool,
    failed_action_event: bool,
    freshness: FreshnessState,
    state: RouterState | None = None,
    repeat_cap: int = 2,
) -> RouterDecision:
    """Route using the stabilized resource/fallback contract."""
    router_state = state or RouterState()
    fresh = short_ttl_valid(freshness)
    repeat_breaker = router_state.repeat_count >= repeat_cap
    event_gate = bool(failed_action_event and router_state.cooldown <= 0)
    if facing_candidate and fresh:
        route = "resource"
    elif event_gate or repeat_breaker:
        route = "event_fallback"
    else:
        route = "hold"
    event_self_triggered = router_state.last_route == "event_fallback" and route == "event_fallback" and not failed_action_event
    return RouterDecision(
        route=route,
        resource_route_preserved=bool(facing_candidate and fresh and route == "resource"),
        event_fallback_gate=event_gate,
        repeat_loop_breaker_gate=repeat_breaker,
        stale_invalidated=not fresh,
        event_self_triggered=event_self_triggered,
    )


def event_failed_action_fallback_cooldown(
    *,
    facing_candidate: bool,
    failed_action_event: bool,
    cooldown: int,
    freshness: FreshnessState,
) -> str:
    """Compatibility wrapper returning only the route label."""
    return route_with_freshness(
        facing_candidate=facing_candidate,
        failed_action_event=failed_action_event,
        freshness=freshness,
        state=RouterState(cooldown=cooldown),
    ).route
