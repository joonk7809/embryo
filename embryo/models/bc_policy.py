"""Behavior-cloned action policy interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from embryo.core.types import ActionDecision
from embryo.models.router import RouterOutput


class BCPolicy:
    """Interface for supervised action policies."""

    def act(self, features: Mapping[str, Any]) -> ActionDecision:
        raise NotImplementedError


@dataclass(frozen=True)
class RouteBCPolicy(BCPolicy):
    """Reference BC policy that maps router modes to valid actions."""

    action_by_route: dict[str, str] = field(
        default_factory=lambda: {
            "resource": "move_forward",
            "event_fallback": "turn_left",
            "hold": "noop",
        }
    )
    source: str = "route_bc_policy"

    def act(self, features: Mapping[str, Any]) -> ActionDecision:
        route = _route_from_features(features)
        action = self.action_by_route.get(route)
        if action is None:
            return ActionDecision(action="noop", source=self.source, invalid_or_unknown=True, metadata={"route": route})
        return ActionDecision(action=action, source=self.source, metadata={"route": route})


def _route_from_features(features: Mapping[str, Any]) -> str:
    if "route_mode" in features:
        return str(features["route_mode"])
    if "route" in features:
        return str(features["route"])
    router_output = features.get("router_output")
    if isinstance(router_output, RouterOutput):
        return router_output.route_mode
    if isinstance(router_output, Mapping):
        return str(router_output.get("route_mode", router_output.get("route", "hold")))
    return "hold"
