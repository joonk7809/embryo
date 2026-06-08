"""Frozen actor contracts for policy providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ActorDecision:
    """A policy action with optional action logits."""

    action: str
    logits: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FrozenActor(Protocol):
    """Protocol for actors that are evaluated without training updates."""

    def reset(self, *, seed: int | None = None) -> None:
        ...

    def act(self, observation: Mapping[str, Any]) -> ActorDecision:
        ...

    def close(self) -> None:
        ...


def action_from_logits(logits: Mapping[str, float], *, action_order: Sequence[str], fallback: str) -> str:
    """Return the highest-logit action with stable tie-breaking."""
    best_action = fallback
    best_value = None
    for action in action_order:
        if action not in logits:
            continue
        value = float(logits[action])
        if best_value is None or value > best_value:
            best_action = action
            best_value = value
    return best_action


def decision_action_is_valid(decision: ActorDecision, action_names: Sequence[str]) -> bool:
    """Check actor output against a public action vocabulary."""
    return decision.action in set(action_names)
