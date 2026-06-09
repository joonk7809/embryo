"""Agent-owned object-location memory for the gridworld kitchen probe."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


GRIDWORLD_OBJECT_LOCATION_PROVENANCE = "gridworld_visible_object_location_v0"


@dataclass(frozen=True)
class GridworldLocationEntry:
    object_id: str
    drawer_id: int
    position: tuple[int, int]
    sequence_index: int
    content_hash: str


class GridworldObjectLocationMemory:
    """Store visible object-to-drawer facts from reveal observations."""

    def __init__(self) -> None:
        self._history: dict[str, list[GridworldLocationEntry]] = {}
        self._sequence_index = 0

    def update(self, observation: Mapping[str, Any]) -> GridworldLocationEntry | None:
        parsed = gridworld_location_from_observation(observation, sequence_index=self._sequence_index)
        if parsed is None:
            return None
        self._history.setdefault(parsed.object_id, []).append(parsed)
        self._sequence_index += 1
        return parsed

    def latest(self, object_id: str) -> GridworldLocationEntry | None:
        entries = self._history.get(str(object_id), [])
        return entries[-1] if entries else None

    def oldest(self, object_id: str) -> GridworldLocationEntry | None:
        entries = self._history.get(str(object_id), [])
        return entries[0] if entries else None

    def latest_other(self, object_id: str) -> GridworldLocationEntry | None:
        candidates = [entries[-1] for key, entries in sorted(self._history.items()) if key != str(object_id) and entries]
        return candidates[0] if candidates else None

    def shuffled_other(self, object_id: str, *, seed: int) -> GridworldLocationEntry | None:
        candidates = [entries[-1] for key, entries in sorted(self._history.items()) if key != str(object_id) and entries]
        if not candidates:
            return None
        digest = hashlib.sha1(f"gridworld-shuffle:{int(seed)}:{str(object_id)}".encode("utf-8")).hexdigest()
        return candidates[int(digest[:8], 16) % len(candidates)]


def gridworld_location_from_observation(observation: Mapping[str, Any], *, sequence_index: int) -> GridworldLocationEntry | None:
    object_id = observation.get("gridworld_visible_object")
    drawer_id = observation.get("gridworld_visible_drawer_id")
    position = observation.get("gridworld_visible_drawer_position")
    if object_id is None or drawer_id is None or position is None:
        return None
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        raise ValueError("gridworld_visible_drawer_position must be [x, y]")
    payload = {
        "object_id": str(object_id),
        "drawer_id": int(drawer_id),
        "position": [int(position[0]), int(position[1])],
        "provenance": GRIDWORLD_OBJECT_LOCATION_PROVENANCE,
        "sequence_index": int(sequence_index),
    }
    content_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return GridworldLocationEntry(
        object_id=str(object_id),
        drawer_id=int(drawer_id),
        position=(int(position[0]), int(position[1])),
        sequence_index=int(sequence_index),
        content_hash=content_hash,
    )
