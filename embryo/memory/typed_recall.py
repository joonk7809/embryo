"""Typed deployable recall surfaces for resource-return probes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from embryo.core.types import Fact


LOCALIZED_RESOURCE_SEEN = "localized_resource_seen_v1"
REFERENCE_WATER_RGB_PROVENANCE = "reference_rgb_water_bearing_v0"
REFERENCE_BENCH_PLACEMENT_PROVENANCE = "reference_place_table_event_v0"
CRAFTING_BENCH_RESOURCE = "crafting_bench"
PASSIVE_MATCH_CUE_SEEN = "passive_match_cue_seen_v0"
REFERENCE_PASSIVE_MATCH_RGB_PROVENANCE = "reference_passive_match_rgb_v0"


@dataclass(frozen=True)
class ResourceMemoryEntry:
    resource_type: str
    bearing: str
    confidence: float
    age: int
    last_seen_tick: int
    content_hash: str


@dataclass(frozen=True)
class ResourceRecallDecision:
    active: bool
    action: str | None
    reason: str
    entry: ResourceMemoryEntry | None = None


@dataclass(frozen=True)
class PassiveCueEntry:
    side: str
    age: int
    content_hash: str


class PassiveCueMemory:
    """Small typed store for passive visual match cue facts."""

    def __init__(self, *, ttl: int = 1024) -> None:
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
        self.ttl = int(ttl)
        self._entry: PassiveCueEntry | None = None

    def update(self, fact: Fact) -> None:
        if self._entry is not None:
            aged = PassiveCueEntry(side=self._entry.side, age=self._entry.age + 1, content_hash=self._entry.content_hash)
            self._entry = aged if aged.age <= self.ttl else None
        if fact.name != PASSIVE_MATCH_CUE_SEEN or not bool(fact.value):
            return
        side = str(fact.metadata.get("side", ""))
        if side not in {"left", "right"}:
            return
        payload = {"side": side, "provenance": fact.provenance}
        content_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        self._entry = PassiveCueEntry(side=side, age=0, content_hash=content_hash)

    def entry(self) -> PassiveCueEntry | None:
        return self._entry


class ResourceNeedBelief:
    """Reference deployable need clock for resource-return probes.

    The clock is intentionally simple: it uses elapsed actor ticks, previous
    action, and previously visible typed resource facts. It does not read
    backend inventory.
    """

    def __init__(self, *, need_after_steps: int = 160) -> None:
        if need_after_steps < 0:
            raise ValueError("need_after_steps must be non-negative")
        self.need_after_steps = int(need_after_steps)
        self.steps_since_resolution = 0
        self.previous_visible_resource: str | None = None

    def update(self, *, previous_action: str, visible_resources: Iterable[str]) -> bool:
        current_visible = tuple(str(item) for item in visible_resources)
        if str(previous_action) == "do" and self.previous_visible_resource:
            self.steps_since_resolution = 0
        else:
            self.steps_since_resolution += 1
        self.previous_visible_resource = current_visible[0] if current_visible else None
        return self.need_active

    @property
    def need_active(self) -> bool:
        return self.steps_since_resolution >= self.need_after_steps


class PlacedLandmarkMemory:
    """Agent-owned relative memory for a landmark the actor placed."""

    def __init__(self, *, resource_type: str, ttl: int = 768) -> None:
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
        self.resource_type = str(resource_type)
        self.ttl = int(ttl)
        self.age = 0
        self.rel_x = 0
        self.rel_y = 0
        self._tick = 0
        self._present = False
        self._content_hash = ""

    def update_from_previous_action(
        self,
        previous_action: str,
        *,
        placed_action: str,
        failed_action_event: bool = False,
    ) -> None:
        self._tick += 1
        action = str(previous_action)
        if action == placed_action and not failed_action_event:
            self._present = True
            self.age = 0
            self.rel_x = 0
            self.rel_y = 0
            self._content_hash = hashlib.sha1(f"{self.resource_type}:{self._tick}".encode("utf-8")).hexdigest()[:16]
            return
        if not self._present:
            return
        self.age += 1
        if self.age > self.ttl:
            self._present = False
            return
        if action == "move_left":
            self.rel_x += 1
        elif action == "move_right":
            self.rel_x -= 1
        elif action == "move_up":
            self.rel_y += 1
        elif action == "move_down":
            self.rel_y -= 1

    def entry(self) -> ResourceMemoryEntry | None:
        if not self._present:
            return None
        bearing = bearing_from_displacement(self.rel_x, self.rel_y)
        return ResourceMemoryEntry(
            resource_type=self.resource_type,
            bearing=bearing,
            confidence=1.0,
            age=self.age,
            last_seen_tick=self._tick - self.age,
            content_hash=self._content_hash,
        )


class ResourceRecallMemory:
    """Small explicit store for typed resource facts.

    Relevance gates such as `need_resource` must come from deployable belief
    state supplied by the caller; this class does not read simulator inventory.
    """

    def __init__(self, *, ttl: int = 256) -> None:
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
        self.ttl = int(ttl)
        self._tick = 0
        self._entries: dict[str, ResourceMemoryEntry] = {}

    def update(self, facts: Iterable[Fact]) -> None:
        self._tick += 1
        self._entries = {
            key: ResourceMemoryEntry(
                resource_type=entry.resource_type,
                bearing=entry.bearing,
                confidence=entry.confidence,
                age=entry.age + 1,
                last_seen_tick=entry.last_seen_tick,
                content_hash=entry.content_hash,
            )
            for key, entry in self._entries.items()
            if entry.age + 1 <= self.ttl
        }
        for fact in facts:
            entry = resource_entry_from_fact(fact, tick=self._tick)
            if entry is not None:
                self._entries[entry.resource_type] = entry

    def recall(self, resource_type: str, *, need_resource: bool, action_names: Sequence[str]) -> ResourceRecallDecision:
        if not need_resource:
            return ResourceRecallDecision(active=False, action=None, reason="not_needed")
        entry = self._entries.get(str(resource_type))
        if entry is None:
            return ResourceRecallDecision(active=False, action=None, reason="not_found")
        if entry.age > self.ttl:
            return ResourceRecallDecision(active=False, action=None, reason="stale", entry=entry)
        action = action_for_bearing(entry.bearing, action_names=action_names)
        if action is None:
            return ResourceRecallDecision(active=False, action=None, reason="no_valid_action", entry=entry)
        return ResourceRecallDecision(active=True, action=action, reason="recall_navigation", entry=entry)

    def entry(self, resource_type: str) -> ResourceMemoryEntry | None:
        return self._entries.get(str(resource_type))


def localized_resource_fact(
    resource_type: str,
    *,
    visible: bool,
    bearing: str = "unknown",
    confidence: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    provenance: str = REFERENCE_WATER_RGB_PROVENANCE,
) -> Fact:
    metadata = {
        "resource_type": str(resource_type),
        "bearing": str(bearing),
        "offset_x": round(float(offset_x), 4),
        "offset_y": round(float(offset_y), 4),
    }
    return Fact(
        name=LOCALIZED_RESOURCE_SEEN,
        value=bool(visible),
        confidence=round(clamp01(confidence), 6),
        provenance=provenance,
        metadata=metadata,
    )


def passive_match_cue_fact_from_observation(observation: Mapping[str, Any] | Any) -> Fact:
    frame = observation.get("raw_rgb_frame") if isinstance(observation, Mapping) else observation
    image = rgb_array(frame)
    side = passive_match_cue_side(image)
    return Fact(
        name=PASSIVE_MATCH_CUE_SEEN,
        value=side is not None,
        confidence=1.0 if side is not None else 0.0,
        provenance=REFERENCE_PASSIVE_MATCH_RGB_PROVENANCE,
        metadata={"side": side or "unknown"},
    )


def passive_match_cue_side(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    left = image[4:12, 1:5]
    right = image[4:12, 11:15]
    left_red = float(left[..., 0].mean() - np.maximum(left[..., 1], left[..., 2]).mean())
    right_green = float(right[..., 1].mean() - np.maximum(right[..., 0], right[..., 2]).mean())
    if left_red >= 0.35 and left_red > right_green:
        return "left"
    if right_green >= 0.35 and right_green > left_red:
        return "right"
    return None


def passive_match_choice_visible(observation: Mapping[str, Any] | Any) -> bool:
    frame = observation.get("raw_rgb_frame") if isinstance(observation, Mapping) else observation
    image = rgb_array(frame)
    if image is None:
        return False
    left = image[5:11, 1:5]
    right = image[5:11, 11:15]
    return bool(float(left.mean()) >= 0.65 and float(right.mean()) >= 0.65)


def passive_match_action_for_side(side: str, action_names: Sequence[str]) -> str | None:
    action = {"left": "choose_left", "right": "choose_right"}.get(str(side))
    return action if action in action_names else None


def water_bearing_fact_from_observation(observation: Mapping[str, Any] | Any) -> Fact:
    frame = observation.get("raw_rgb_frame") if isinstance(observation, Mapping) else observation
    image = rgb_array(frame)
    if image is None:
        return localized_resource_fact("water", visible=False)
    mask = water_like_mask(image)
    min_pixels = max(12, int(round(image.shape[0] * image.shape[1] * 0.01)))
    if int(mask.sum()) < min_pixels:
        return localized_resource_fact("water", visible=False)
    ys, xs = np.nonzero(mask)
    height, width = image.shape[:2]
    offset_x = float((xs.mean() - (width - 1) / 2.0) / max(1.0, width / 2.0))
    offset_y = float((ys.mean() - (height - 1) / 2.0) / max(1.0, height / 2.0))
    confidence = float(mask.mean())
    return localized_resource_fact(
        "water",
        visible=True,
        bearing=bearing_from_offset(offset_x, offset_y),
        confidence=confidence,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def water_like_mask(image: np.ndarray) -> np.ndarray:
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    mask = (blue >= 0.35) & ((blue - np.maximum(red, green)) >= 0.08)
    hud_rows = max(2, int(round(image.shape[0] * 0.12)))
    if hud_rows > 0:
        mask[-hud_rows:, :] = False
    return mask


def bearing_from_offset(offset_x: float, offset_y: float, *, center_tolerance: float = 0.2) -> str:
    x = float(offset_x)
    y = float(offset_y)
    if abs(x) <= center_tolerance and abs(y) <= center_tolerance:
        return "center"
    if abs(x) >= abs(y):
        return "right" if x > 0 else "left"
    return "down" if y > 0 else "up"


def bearing_from_displacement(rel_x: int, rel_y: int) -> str:
    if rel_x == 0 and rel_y == 0:
        return "center"
    if abs(rel_x) >= abs(rel_y):
        return "right" if rel_x > 0 else "left"
    return "down" if rel_y > 0 else "up"


def action_for_bearing(bearing: str, *, action_names: Sequence[str]) -> str | None:
    mapping = {
        "left": "move_left",
        "right": "move_right",
        "up": "move_up",
        "down": "move_down",
        "center": "do",
    }
    action = mapping.get(str(bearing))
    return action if action in action_names else None


def resource_entry_from_fact(fact: Fact, *, tick: int) -> ResourceMemoryEntry | None:
    if fact.name != LOCALIZED_RESOURCE_SEEN or not bool(fact.value):
        return None
    resource_type = str(fact.metadata.get("resource_type", ""))
    bearing = str(fact.metadata.get("bearing", "unknown"))
    if not resource_type or bearing == "unknown":
        return None
    payload = {
        "resource_type": resource_type,
        "bearing": bearing,
        "confidence": fact.confidence,
        "metadata": dict(fact.metadata),
        "provenance": fact.provenance,
    }
    content_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return ResourceMemoryEntry(
        resource_type=resource_type,
        bearing=bearing,
        confidence=clamp01(0.0 if fact.confidence is None else float(fact.confidence)),
        age=0,
        last_seen_tick=int(tick),
        content_hash=content_hash,
    )


def rgb_array(observation: Any) -> np.ndarray | None:
    if observation is None:
        return None
    try:
        image = np.asarray(observation)
    except Exception:  # noqa: BLE001
        return None
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    image = image.astype(np.float32, copy=False)
    if float(np.nanmax(image)) > 1.0:
        image = image / 255.0
    return np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
