from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from embryo.runtimes.crafter.adapter import CrafterRuntime, deterministic_object_key
from embryo.runtimes.crafter.features import VISUAL_ANCHOR_FAMILY, extract_deployable_rgb_features


class CrafterFeatureTests(unittest.TestCase):
    def test_rgb_features_are_deployable_and_hashed(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[8:12, 8:12] = np.array([220, 40, 40], dtype=np.uint8)

        features = extract_deployable_rgb_features(image)

        self.assertEqual(features["visual_anchor_family"], VISUAL_ANCHOR_FAMILY)
        self.assertGreater(features["center_salience_score"], 0.0)
        self.assertTrue(features["center_patch_hash"])
        self.assertEqual(features["candidate_score"], features["center_salience_score"])
        self.assertFalse(features["failed_action_event"])

    def test_low_delta_after_action_is_deployable_failed_action_event(self) -> None:
        image = np.full((20, 20, 3), 80, dtype=np.uint8)

        features = extract_deployable_rgb_features(image, previous_observation=image.copy(), previous_action="move_up")

        self.assertEqual(features["rgb_delta_score"], 0.0)
        self.assertFalse(features["visual_change_event"])
        self.assertTrue(features["failed_action_event"])

    def test_crafter_runtime_passes_seed_to_env_constructor(self) -> None:
        original = sys.modules.get("crafter")
        constructed: list[int | None] = []

        class FakeEnv:
            action_names = ("noop", "move_up", "move_left")

            def __init__(self, *, seed=None):  # noqa: ANN001
                constructed.append(seed)

            def reset(self):
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def step(self, action):  # noqa: ANN001
                _ = action
                return np.zeros((8, 8, 3), dtype=np.uint8), 0.0, False, {}

            def close(self):
                pass

        sys.modules["crafter"] = types.SimpleNamespace(Env=FakeEnv)
        try:
            runtime = CrafterRuntime(seed=123)
            runtime.reset(seed=456)
            runtime.close()
        finally:
            if original is None:
                sys.modules.pop("crafter", None)
            else:
                sys.modules["crafter"] = original

        self.assertEqual(constructed, [123, 456])

    def test_crafter_runtime_patches_balance_object_order(self) -> None:
        original = sys.modules.get("crafter")

        class FakeEnv:
            action_names = ("noop", "move_up", "move_left")

            def __init__(self, *, seed=None):  # noqa: ANN001
                _ = seed

            def reset(self):
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def step(self, action):  # noqa: ANN001
                _ = action
                return np.zeros((8, 8, 3), dtype=np.uint8), 0.0, False, {}

            def close(self):
                pass

            def _balance_object(self):  # noqa: ANN001
                pass

        sys.modules["crafter"] = types.SimpleNamespace(Env=FakeEnv)
        try:
            runtime = CrafterRuntime(seed=123)
            patched = runtime._env._balance_object  # noqa: SLF001
            runtime.close()
        finally:
            if original is None:
                sys.modules.pop("crafter", None)
            else:
                sys.modules["crafter"] = original

        self.assertIs(patched.__self__, runtime._env)  # noqa: SLF001
        self.assertIsNot(patched.__func__, FakeEnv._balance_object)

    def test_crafter_runtime_can_leave_backend_unpatched(self) -> None:
        original = sys.modules.get("crafter")

        class FakeEnv:
            action_names = ("noop", "move_up", "move_left")

            def __init__(self, *, seed=None):  # noqa: ANN001
                _ = seed

            def reset(self):
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def step(self, action):  # noqa: ANN001
                _ = action
                return np.zeros((8, 8, 3), dtype=np.uint8), 0.0, False, {}

            def close(self):
                pass

            def _balance_object(self):  # noqa: ANN001
                pass

        sys.modules["crafter"] = types.SimpleNamespace(Env=FakeEnv)
        try:
            runtime = CrafterRuntime(seed=123, deterministic_backend_patch=False)
            balance = runtime._env._balance_object  # noqa: SLF001
            runtime.close()
        finally:
            if original is None:
                sys.modules.pop("crafter", None)
            else:
                sys.modules["crafter"] = original

        self.assertIs(balance.__func__, FakeEnv._balance_object)

    def test_deterministic_object_key_sorts_by_position_then_class(self) -> None:
        class ObjA:
            def __init__(self, pos) -> None:  # noqa: ANN001
                self.pos = np.array(pos)

        class ObjB(ObjA):
            pass

        rows = [ObjB((2, 1)), ObjB((1, 1)), ObjA((1, 1))]

        ordered = sorted(rows, key=deterministic_object_key)

        self.assertEqual([tuple(row.pos) for row in ordered], [(1, 1), (1, 1), (2, 1)])
        self.assertEqual([row.__class__.__name__ for row in ordered], ["ObjA", "ObjB", "ObjB"])


if __name__ == "__main__":
    unittest.main()
