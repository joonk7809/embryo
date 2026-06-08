from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from embryo.datasets.crafter_human import iter_demo_transitions, load_demo_npz, summarize_human_dataset
from embryo.runtimes.crafter.gym_env import action_name_from_index, resize_rgb_frame, training_reward
from embryo.train.crafter_survival import bc_split_indices, histogram, load_bc_arrays, run_setup_check, summarize_rollouts


class CrafterSurvivalTrainingTests(unittest.TestCase):
    def test_resize_rgb_frame_returns_uint8_rgb(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.float32)
        frame[2:6, 2:6, 0] = 1.0

        resized = resize_rgb_frame(frame, (4, 4))

        self.assertEqual(resized.shape, (4, 4, 3))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertGreater(int(resized[..., 0].max()), 0)

    def test_training_reward_uses_native_or_small_alive_bonus(self) -> None:
        self.assertEqual(training_reward(1.0, done=False, reward_mode="native", alive_bonus=0.1), 1.0)
        self.assertEqual(training_reward(1.0, done=False, reward_mode="native_plus_alive", alive_bonus=0.1), 1.1)
        self.assertEqual(training_reward(1.0, done=True, reward_mode="native_plus_alive", alive_bonus=0.1), 1.0)

    def test_action_name_from_index_rejects_out_of_range(self) -> None:
        self.assertEqual(action_name_from_index(("noop", "move_up"), 1), "move_up")
        with self.assertRaises(ValueError):
            action_name_from_index(("noop",), 3)

    def test_human_npz_loader_emits_deployable_transitions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode.npz"
            frames = np.zeros((4, 6, 6, 3), dtype=np.uint8)
            actions = np.array([0, 1, 2], dtype=np.int64)
            rewards = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            dones = np.array([False, False, True])
            np.savez(path, observations=frames, actions=actions, rewards=rewards, dones=dones)

            episode = load_demo_npz(path)
            rows = list(iter_demo_transitions(tmpdir))
            summary = summarize_human_dataset(tmpdir)
            bc_frames, bc_actions, bc_summary = load_bc_arrays(tmpdir)

        self.assertEqual(episode.transition_count, 3)
        self.assertEqual(episode.frames.shape[0], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0].keys()), ("raw_rgb_frame", "action_index", "episode_index", "tick", "source", "reward_eval_only", "done_eval_only"))
        self.assertEqual(summary["transition_count"], 3)
        self.assertEqual(summary["action_histogram"], {"0": 1, "1": 1, "2": 1})
        self.assertEqual(bc_frames.shape, (3, 6, 6, 3))
        self.assertEqual(bc_actions.tolist(), [0, 1, 2])
        self.assertEqual(bc_summary["transition_count"], 3)

    def test_bc_split_is_deterministic_and_nonempty(self) -> None:
        train_a, dev_a = bc_split_indices(10, validation_fraction=0.2, seed=7)
        train_b, dev_b = bc_split_indices(10, validation_fraction=0.2, seed=7)

        self.assertEqual(train_a.tolist(), train_b.tolist())
        self.assertEqual(dev_a.tolist(), dev_b.tolist())
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(dev_a), 2)
        self.assertEqual(histogram([0, 1, 1]), {"0": 1, "1": 2})

    def test_rollout_summary_tracks_survival_progression_metrics(self) -> None:
        summary = summarize_rollouts(
            [
                {"survival_steps": 10, "done": True, "reward_sum_eval_only": 2.0, "achievement_count_eval_only": 1, "loop_rate": 0.2, "action_entropy": 0.5},
                {"survival_steps": 20, "done": False, "reward_sum_eval_only": 4.0, "achievement_count_eval_only": 3, "loop_rate": 0.0, "action_entropy": 1.0},
            ]
        )

        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["mean_survival_steps"], 15.0)
        self.assertEqual(summary["death_rate"], 0.5)
        self.assertEqual(summary["mean_achievement_count_eval_only"], 2.0)

    def test_setup_check_writes_dependency_status(self) -> None:
        with TemporaryDirectory() as tmpdir:
            result = run_setup_check({"dataset": {"human_root": str(Path(tmpdir) / "missing")}}, tmpdir)
            payload = json.loads(Path(result["paths"]["setup_status"]).read_text(encoding="utf-8"))

        self.assertIn("decision", payload)
        self.assertIn("dependencies", payload)
        self.assertEqual(payload["training_boundary"], "crafter_survival_actor_no_memory")


if __name__ == "__main__":
    unittest.main()
