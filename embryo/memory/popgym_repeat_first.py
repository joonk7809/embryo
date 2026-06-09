"""Deployable fact surface for POPGym RepeatFirst."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from embryo.core.types import Fact


POPGYM_REPEAT_FIRST_TARGET = "popgym_repeat_first_target_v0"
REFERENCE_POPGYM_REPEAT_FIRST_PROVENANCE = "popgym_repeat_first_observation_v0"


@dataclass(frozen=True)
class RepeatFirstEntry:
    suit: int
    age: int
    content_hash: str


class RepeatFirstMemory:
    """Store the first observed suit in a RepeatFirst episode."""

    def __init__(self, *, ttl: int = 1024) -> None:
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
        self.ttl = int(ttl)
        self._entry: RepeatFirstEntry | None = None

    def update(self, fact: Fact) -> None:
        if self._entry is not None:
            aged = RepeatFirstEntry(suit=self._entry.suit, age=self._entry.age + 1, content_hash=self._entry.content_hash)
            self._entry = aged if aged.age <= self.ttl else None
            return
        if fact.name != POPGYM_REPEAT_FIRST_TARGET or not bool(fact.value):
            return
        suit = int(fact.metadata["suit"])
        payload = {"suit": suit, "provenance": fact.provenance}
        content_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        self._entry = RepeatFirstEntry(suit=suit, age=0, content_hash=content_hash)

    def entry(self) -> RepeatFirstEntry | None:
        return self._entry


def repeat_first_target_fact_from_observation(observation: Mapping[str, Any]) -> Fact:
    suit = repeat_first_observed_suit(observation)
    return Fact(
        name=POPGYM_REPEAT_FIRST_TARGET,
        value=suit is not None,
        confidence=1.0 if suit is not None else 0.0,
        provenance=REFERENCE_POPGYM_REPEAT_FIRST_PROVENANCE,
        metadata={"suit": -1 if suit is None else int(suit)},
    )


def repeat_first_observed_suit(observation: Mapping[str, Any]) -> int | None:
    value = observation.get("popgym_observation")
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def repeat_first_action_for_suit(suit: int, action_names: Sequence[str]) -> str | None:
    action = f"suit_{int(suit)}"
    return action if action in action_names else None


def repeat_first_suit_from_action(action: str) -> int | None:
    prefix = "suit_"
    if not str(action).startswith(prefix):
        return None
    try:
        return int(str(action)[len(prefix) :])
    except ValueError:
        return None


def repeat_first_wrong_suit(suit: int, action_count: int) -> int:
    return (int(suit) + 1) % max(1, int(action_count))


def repeat_first_shuffled_suit(suit: int, action_count: int, content_hash: str) -> int:
    count = max(1, int(action_count))
    candidates = [idx for idx in range(count) if idx != int(suit)]
    if not candidates:
        return int(suit)
    digest = hashlib.sha1(f"shuffle:{content_hash}".encode("utf-8")).hexdigest()
    return candidates[int(digest[:8], 16) % len(candidates)]
