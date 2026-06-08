"""Bounded memory residuals for frozen actor logits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from embryo.models.actors import ActorDecision, action_from_logits


@dataclass(frozen=True)
class MemoryResidualInput:
    """Memory-side proposal expressed as action-logit biases."""

    active: bool
    action_biases: Mapping[str, float] = field(default_factory=dict)
    content_hash: str = ""
    freshness_age: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryResidualDecision:
    """Final action after applying a bounded memory residual."""

    action: str
    base_action: str
    base_logits: Mapping[str, float]
    residual_logits: Mapping[str, float]
    final_logits: Mapping[str, float]
    active: bool
    changed_action: bool
    applied_bias_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MemoryResidualPolicy:
    """Apply bounded action-logit residuals without owning actor state."""

    def __init__(self, *, strength: float = 1.0, max_abs_bias: float = 2.0) -> None:
        if strength < 0:
            raise ValueError("strength must be non-negative")
        if max_abs_bias < 0:
            raise ValueError("max_abs_bias must be non-negative")
        self.strength = float(strength)
        self.max_abs_bias = float(max_abs_bias)

    def apply(
        self,
        base: ActorDecision,
        memory: MemoryResidualInput,
        *,
        action_order: Sequence[str],
    ) -> MemoryResidualDecision:
        base_logits = {str(action): float(value) for action, value in base.logits.items()}
        if not memory.active or not base_logits:
            return self._unchanged(base, base_logits, memory)

        residual_logits = self._bounded_residuals(memory.action_biases, allowed=tuple(base_logits))
        final_logits = {action: value + residual_logits.get(action, 0.0) for action, value in base_logits.items()}
        final_action = action_from_logits(final_logits, action_order=action_order, fallback=base.action)
        return MemoryResidualDecision(
            action=final_action,
            base_action=base.action,
            base_logits=base_logits,
            residual_logits=residual_logits,
            final_logits=final_logits,
            active=bool(residual_logits),
            changed_action=final_action != base.action,
            applied_bias_count=len(residual_logits),
            metadata={
                "content_hash": memory.content_hash,
                "freshness_age": int(memory.freshness_age),
                **dict(memory.metadata),
            },
        )

    def _bounded_residuals(self, biases: Mapping[str, float], *, allowed: Sequence[str]) -> dict[str, float]:
        allowed_set = set(allowed)
        residuals: dict[str, float] = {}
        for action, raw_bias in biases.items():
            if action not in allowed_set:
                continue
            bias = max(-self.max_abs_bias, min(self.max_abs_bias, float(raw_bias)))
            residuals[str(action)] = bias * self.strength
        return residuals

    @staticmethod
    def _unchanged(base: ActorDecision, base_logits: Mapping[str, float], memory: MemoryResidualInput) -> MemoryResidualDecision:
        return MemoryResidualDecision(
            action=base.action,
            base_action=base.action,
            base_logits=base_logits,
            residual_logits={},
            final_logits=base_logits,
            active=False,
            changed_action=False,
            applied_bias_count=0,
            metadata={"content_hash": memory.content_hash, "freshness_age": int(memory.freshness_age), **dict(memory.metadata)},
        )
