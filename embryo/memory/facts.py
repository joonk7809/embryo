"""Deployable fact surfaces."""

from __future__ import annotations

from collections.abc import Iterable

from embryo.core.types import Fact


FACING_CANDIDATE_V1 = "facing_candidate_v1"
EVENT_FAILURE = "event_failure"
VISUAL_CHANGE = "visual_change"


def facing_candidate_fact(
    value: bool,
    *,
    confidence: float | None = None,
    provenance: str = "rgb_delta_prev_action",
    direction_bin: str | None = None,
) -> Fact:
    """Create the sparse visual fact used by the Crafter memory contract."""
    metadata = {}
    if direction_bin is not None:
        metadata["direction_bin"] = direction_bin
    return Fact(name=FACING_CANDIDATE_V1, value=bool(value), confidence=confidence, provenance=provenance, metadata=metadata)


def event_failure_fact(
    value: bool,
    *,
    confidence: float | None = None,
    provenance: str = "previous_action_rgb_delta",
    event_type: str = "failed_action",
) -> Fact:
    """Create a deployable event/failure fact."""
    return Fact(name=EVENT_FAILURE, value=bool(value), confidence=confidence, provenance=provenance, metadata={"event_type": event_type})


def visual_change_fact(
    value: bool,
    *,
    confidence: float | None = None,
    provenance: str = "rgb_delta",
) -> Fact:
    """Create a deployable visual-change fact."""
    return Fact(name=VISUAL_CHANGE, value=bool(value), confidence=confidence, provenance=provenance)


def fact_by_name(facts: Iterable[Fact], name: str) -> Fact | None:
    for fact in facts:
        if fact.name == name:
            return fact
    return None


def fact_bool(facts: Iterable[Fact], name: str, *, default: bool = False) -> bool:
    fact = fact_by_name(facts, name)
    return default if fact is None else bool(fact.value)
