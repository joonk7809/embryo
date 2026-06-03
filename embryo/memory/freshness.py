"""Freshness and stale-cache helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreshnessState:
    cache_age: int
    invalidated: bool = False
    visual_change_conflict: bool = False


def short_ttl_valid(state: FreshnessState, *, ttl: int = 4) -> bool:
    """Return whether cached query content is valid under a short TTL."""
    return not state.invalidated and not state.visual_change_conflict and state.cache_age <= ttl


def advance_freshness(
    state: FreshnessState,
    *,
    visual_change: bool = False,
    content_changed: bool = False,
    ttl: int = 4,
) -> FreshnessState:
    """Advance cache age and invalidate on deployable content conflict."""
    age = 0 if content_changed else state.cache_age + 1
    conflict = bool(visual_change and not content_changed and state.cache_age > 0)
    return FreshnessState(cache_age=age, invalidated=state.invalidated or age > ttl or conflict, visual_change_conflict=conflict)


def reset_freshness() -> FreshnessState:
    return FreshnessState(cache_age=0, invalidated=False, visual_change_conflict=False)
