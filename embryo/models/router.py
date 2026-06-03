"""Router/freshness model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from embryo.memory.freshness import FreshnessState
from embryo.memory.router import RouterDecision, RouterState, route_with_freshness


@dataclass(frozen=True)
class RouterOutput:
    route_mode: str
    event_fallback_gate: bool
    stale_invalidated: bool
    resource_route_preserved: bool
    repeat_loop_breaker_gate: bool

    @classmethod
    def from_decision(cls, decision: RouterDecision) -> "RouterOutput":
        return cls(
            route_mode=decision.route,
            event_fallback_gate=decision.event_fallback_gate,
            stale_invalidated=decision.stale_invalidated,
            resource_route_preserved=decision.resource_route_preserved,
            repeat_loop_breaker_gate=decision.repeat_loop_breaker_gate,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "route_mode": self.route_mode,
            "event_fallback_gate": self.event_fallback_gate,
            "stale_invalidated": self.stale_invalidated,
            "resource_route_preserved": self.resource_route_preserved,
            "repeat_loop_breaker_gate": self.repeat_loop_breaker_gate,
        }


class RouterModel:
    """Interface for learned router/freshness schedulers."""

    def predict(self, features: Mapping[str, Any]) -> RouterOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class RuleRouterModel(RouterModel):
    """Reference router that wraps the frozen freshness-aware route contract."""

    repeat_cap: int = 2

    def predict(self, features: Mapping[str, Any]) -> RouterOutput:
        freshness = FreshnessState(
            cache_age=int(features.get("cache_age", 0)),
            invalidated=bool(features.get("invalidated", False)),
            visual_change_conflict=bool(features.get("visual_change_conflict", False)),
        )
        state = RouterState(
            cooldown=int(features.get("cooldown", 0)),
            repeat_count=int(features.get("repeat_count", 0)),
            last_route=str(features.get("last_route", "hold")),
            event_self_triggered=bool(features.get("event_self_triggered", False)),
        )
        decision = route_with_freshness(
            facing_candidate=bool(features.get("facing_candidate", False)),
            failed_action_event=bool(features.get("failed_action_event", False)),
            freshness=freshness,
            state=state,
            repeat_cap=self.repeat_cap,
        )
        return RouterOutput.from_decision(decision)
