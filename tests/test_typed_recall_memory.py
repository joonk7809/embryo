from __future__ import annotations

import unittest

import numpy as np

from embryo.memory.typed_recall import (
    LOCALIZED_RESOURCE_SEEN,
    ResourceRecallMemory,
    water_bearing_fact_from_observation,
)


ACTION_NAMES = ("noop", "move_left", "move_right", "move_up", "move_down", "do")


def water_frame(*, x0: int, y0: int, size: int = 4) -> np.ndarray:
    frame = np.zeros((16, 16, 3), dtype=np.uint8) + 40
    frame[y0 : y0 + size, x0 : x0 + size] = np.array([20, 80, 230], dtype=np.uint8)
    return frame


class TypedRecallMemoryTests(unittest.TestCase):
    def test_water_fact_encodes_typed_relative_bearing_from_rgb(self) -> None:
        left = water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=1, y0=6)})
        right = water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=11, y0=6)})

        self.assertEqual(left.name, LOCALIZED_RESOURCE_SEEN)
        self.assertTrue(left.value)
        self.assertEqual(left.metadata["resource_type"], "water")
        self.assertEqual(left.metadata["bearing"], "left")
        self.assertEqual(right.metadata["bearing"], "right")
        self.assertGreater(left.confidence or 0.0, 0.0)

    def test_water_fact_ignores_forbidden_eval_only_fields(self) -> None:
        frame = water_frame(x0=6, y0=1)
        base = water_bearing_fact_from_observation({"raw_rgb_frame": frame})
        with_forbidden = water_bearing_fact_from_observation(
            {
                "raw_rgb_frame": frame,
                "reward": 1.0,
                "done": True,
                "info": {"inventory": {"drink": 0}, "semantic": [[1]], "player_pos": [3, 4]},
                "seed": 10000,
            }
        )

        self.assertEqual(base.value, with_forbidden.value)
        self.assertEqual(base.metadata, with_forbidden.metadata)
        self.assertEqual(base.confidence, with_forbidden.confidence)

    def test_recall_action_depends_on_stored_bearing_and_need(self) -> None:
        memory = ResourceRecallMemory(ttl=256)
        memory.update([water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=1, y0=6)})])
        for _ in range(200):
            memory.update([])

        not_needed = memory.recall("water", need_resource=False, action_names=ACTION_NAMES)
        needed = memory.recall("water", need_resource=True, action_names=ACTION_NAMES)

        self.assertFalse(not_needed.active)
        self.assertEqual(not_needed.reason, "not_needed")
        self.assertTrue(needed.active)
        self.assertEqual(needed.action, "move_left")
        self.assertEqual(needed.entry.bearing, "left")  # type: ignore[union-attr]
        self.assertGreater(needed.entry.age, 100)  # type: ignore[union-attr]

    def test_recall_does_not_collapse_to_fixed_move_up(self) -> None:
        left_memory = ResourceRecallMemory(ttl=256)
        right_memory = ResourceRecallMemory(ttl=256)
        up_memory = ResourceRecallMemory(ttl=256)
        left_memory.update([water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=1, y0=6)})])
        right_memory.update([water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=11, y0=6)})])
        up_memory.update([water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=6, y0=1)})])

        self.assertEqual(left_memory.recall("water", need_resource=True, action_names=ACTION_NAMES).action, "move_left")
        self.assertEqual(right_memory.recall("water", need_resource=True, action_names=ACTION_NAMES).action, "move_right")
        self.assertEqual(up_memory.recall("water", need_resource=True, action_names=ACTION_NAMES).action, "move_up")

    def test_stale_resource_memory_is_not_actionable(self) -> None:
        memory = ResourceRecallMemory(ttl=2)
        memory.update([water_bearing_fact_from_observation({"raw_rgb_frame": water_frame(x0=1, y0=6)})])
        memory.update([])
        memory.update([])
        fresh = memory.recall("water", need_resource=True, action_names=ACTION_NAMES)
        memory.update([])
        stale = memory.recall("water", need_resource=True, action_names=ACTION_NAMES)

        self.assertTrue(fresh.active)
        self.assertFalse(stale.active)
        self.assertEqual(stale.reason, "not_found")


if __name__ == "__main__":
    unittest.main()
