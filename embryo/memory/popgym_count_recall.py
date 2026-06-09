"""Agent-owned count memory for POPGym CountRecall."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


REFERENCE_POPGYM_COUNT_RECALL_PROVENANCE = "popgym_count_recall_observation_v0"


@dataclass(frozen=True)
class CountRecallObservation:
    value: int
    query: int


class CountRecallMemory:
    """Store observed CountRecall values as deployable count state."""

    def __init__(self, *, symbol_count: int) -> None:
        if symbol_count <= 0:
            raise ValueError("symbol_count must be positive")
        self.symbol_count = int(symbol_count)
        self._counts = [0 for _ in range(self.symbol_count)]
        self._snapshots: list[list[int]] = []

    def update(self, observation: Mapping[str, Any]) -> CountRecallObservation:
        parsed = count_recall_observation(observation)
        self._counts[parsed.value] += 1
        self._snapshots.append(list(self._counts))
        return parsed

    def counts(self) -> list[int]:
        return list(self._counts)

    def stale_counts(self, *, lag: int) -> list[int]:
        if not self._snapshots:
            return [0 for _ in range(self.symbol_count)]
        idx = max(0, len(self._snapshots) - 1 - max(0, int(lag)))
        return list(self._snapshots[idx])

    def answer(self, query: int, *, counts: Sequence[int] | None = None) -> int:
        source = self._counts if counts is None else counts
        return int(source[int(query) % self.symbol_count])


def count_recall_observation(observation: Mapping[str, Any]) -> CountRecallObservation:
    raw = observation.get("popgym_observation")
    if raw is None:
        raise ValueError("missing popgym_observation")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("CountRecall observation must be [value, query]")
    value = int(raw[0])
    query = int(raw[1])
    return CountRecallObservation(value=value, query=query)


def count_recall_content_hash(*, counts: Sequence[int], query: int, arm: str, effective_query: int, stale_lag: int = 0) -> str:
    payload = {
        "arm": str(arm),
        "counts": [int(value) for value in counts],
        "effective_query": int(effective_query),
        "query": int(query),
        "stale_lag": int(stale_lag),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def count_recall_wrong_binding_query(query: int, symbol_count: int) -> int:
    return (int(query) + 1) % max(1, int(symbol_count))


def count_recall_shuffled_query(query: int, symbol_count: int, *, seed: int) -> int:
    count = max(1, int(symbol_count))
    candidates = [idx for idx in range(count) if idx != int(query)]
    if not candidates:
        return int(query)
    digest = hashlib.sha1(f"count-recall-shuffle:{int(seed)}:{int(query)}".encode("utf-8")).hexdigest()
    return candidates[int(digest[:8], 16) % len(candidates)]
