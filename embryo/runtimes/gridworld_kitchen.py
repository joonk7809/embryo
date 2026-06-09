"""Gridworld kitchen abstraction for goal-injection diagnostics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


ACTIONS = ("up", "down", "left", "right", "open")
OBJECT_IDS = ("scissors", "spoon", "tape", "key")


@dataclass(frozen=True)
class Drawer:
    drawer_id: int
    position: tuple[int, int]


@dataclass(frozen=True)
class KitchenEpisode:
    seed: int
    width: int
    height: int
    drawers: tuple[Drawer, ...]
    object_drawers: dict[str, int]
    stale_drawer_id: int
    target_object: str
    start_position: tuple[int, int]

    @property
    def target_drawer_id(self) -> int:
        return int(self.object_drawers[self.target_object])

    def drawer_position(self, drawer_id: int) -> tuple[int, int]:
        return self.drawers[int(drawer_id)].position


class GoalConditionedNavigator:
    """Frozen shortest-path navigator over deployable grid observations."""

    def action(self, position: tuple[int, int], goal_position: tuple[int, int]) -> str:
        x, y = position
        gx, gy = goal_position
        if x < gx:
            return "right"
        if x > gx:
            return "left"
        if y < gy:
            return "down"
        if y > gy:
            return "up"
        return "open"


def make_kitchen_episode(
    *,
    seed: int,
    width: int = 7,
    height: int = 7,
    drawer_count: int = 12,
    object_count: int = 4,
) -> KitchenEpisode:
    if drawer_count < 4:
        raise ValueError("drawer_count must be at least 4")
    objects = OBJECT_IDS[: int(object_count)]
    if not objects:
        raise ValueError("object_count must be positive")
    positions = drawer_positions(width=width, height=height)
    if int(drawer_count) > len(positions):
        raise ValueError("drawer_count exceeds available perimeter positions")
    drawers = tuple(Drawer(drawer_id=idx, position=positions[idx]) for idx in range(int(drawer_count)))
    rng = random.Random(int(seed))
    drawer_ids = list(range(int(drawer_count)))
    rng.shuffle(drawer_ids)
    object_drawers = {obj: int(drawer_ids[idx]) for idx, obj in enumerate(objects)}
    target_object = objects[int(seed) % len(objects)]
    stale_drawer_id = next(drawer for drawer in drawer_ids[len(objects) :] if drawer != object_drawers[target_object])
    return KitchenEpisode(
        seed=int(seed),
        width=int(width),
        height=int(height),
        drawers=drawers,
        object_drawers=object_drawers,
        stale_drawer_id=int(stale_drawer_id),
        target_object=str(target_object),
        start_position=(int(width) // 2, int(height) // 2),
    )


def drawer_positions(*, width: int, height: int) -> tuple[tuple[int, int], ...]:
    positions: list[tuple[int, int]] = []
    for x in range(int(width)):
        positions.append((x, 0))
    for y in range(1, int(height)):
        positions.append((int(width) - 1, y))
    for x in range(int(width) - 2, -1, -1):
        positions.append((x, int(height) - 1))
    for y in range(int(height) - 2, 0, -1):
        positions.append((0, y))
    return tuple(positions)


def move(position: tuple[int, int], action: str, *, width: int, height: int) -> tuple[int, int]:
    x, y = position
    if action == "up":
        y -= 1
    elif action == "down":
        y += 1
    elif action == "left":
        x -= 1
    elif action == "right":
        x += 1
    return max(0, min(int(width) - 1, x)), max(0, min(int(height) - 1, y))


def deployable_reveal_observation(episode: KitchenEpisode) -> dict[str, Any]:
    drawer_id = episode.target_drawer_id
    return deployable_object_location_observation(
        object_id=episode.target_object,
        drawer_id=drawer_id,
        position=episode.drawer_position(drawer_id),
        phase="actual",
    )


def deployable_reveal_observations(episode: KitchenEpisode) -> list[dict[str, Any]]:
    rows = [
        deployable_object_location_observation(
            object_id=episode.target_object,
            drawer_id=episode.stale_drawer_id,
            position=episode.drawer_position(episode.stale_drawer_id),
            phase="stale_prior",
        )
    ]
    for object_id, drawer_id in sorted(episode.object_drawers.items()):
        rows.append(
            deployable_object_location_observation(
                object_id=object_id,
                drawer_id=drawer_id,
                position=episode.drawer_position(drawer_id),
                phase="actual",
            )
        )
    return rows


def deployable_object_location_observation(*, object_id: str, drawer_id: int, position: tuple[int, int], phase: str) -> dict[str, Any]:
    return {
        "gridworld_visible_object": str(object_id),
        "gridworld_visible_drawer_id": int(drawer_id),
        "gridworld_visible_drawer_position": list(position),
        "gridworld_reveal_phase": str(phase),
    }


def deployable_need_observation(episode: KitchenEpisode, position: tuple[int, int], *, previous_action: str) -> dict[str, Any]:
    return {
        "gridworld_need_object": episode.target_object,
        "gridworld_current_position": list(position),
        "previous_action": str(previous_action),
    }
