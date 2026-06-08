"""Human Crafter demonstration loading for survival-agent pretraining."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FRAME_KEYS = ("observations", "observation", "images", "image", "obs", "frames", "frame")
ACTION_KEYS = ("actions", "action")
REWARD_KEYS = ("rewards", "reward")
DONE_KEYS = ("dones", "done", "terminals", "terminal")


@dataclass(frozen=True)
class CrafterDemoEpisode:
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray | None
    dones: np.ndarray | None
    source: str

    @property
    def transition_count(self) -> int:
        return int(len(self.actions))


def discover_demo_files(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file() and path.suffix == ".npz":
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.npz"))


def load_demo_npz(path: str | Path) -> CrafterDemoEpisode:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        frames = np.asarray(required_array(payload, FRAME_KEYS, source))
        actions = np.asarray(required_array(payload, ACTION_KEYS, source), dtype=np.int64).reshape(-1)
        rewards = optional_array(payload, REWARD_KEYS)
        dones = optional_array(payload, DONE_KEYS)
    frames = align_frames_to_actions(frames, actions, source)
    rewards = align_optional_vector(rewards, actions, "rewards", source)
    dones = align_optional_vector(dones, actions, "dones", source)
    return CrafterDemoEpisode(frames=frames, actions=actions, rewards=rewards, dones=dones, source=str(source))


def iter_demo_episodes(root: str | Path) -> Iterator[CrafterDemoEpisode]:
    for path in discover_demo_files(root):
        yield load_demo_npz(path)


def iter_demo_transitions(root: str | Path, *, max_examples: int | None = None) -> Iterator[dict[str, Any]]:
    emitted = 0
    for episode_index, episode in enumerate(iter_demo_episodes(root)):
        for tick, action in enumerate(episode.actions):
            if max_examples is not None and emitted >= int(max_examples):
                return
            row: dict[str, Any] = {
                "raw_rgb_frame": episode.frames[tick],
                "action_index": int(action),
                "episode_index": int(episode_index),
                "tick": int(tick),
                "source": episode.source,
            }
            if episode.rewards is not None:
                row["reward_eval_only"] = float(episode.rewards[tick])
            if episode.dones is not None:
                row["done_eval_only"] = bool(episode.dones[tick])
            emitted += 1
            yield row


def summarize_human_dataset(root: str | Path, *, max_examples: int | None = None) -> dict[str, Any]:
    files = discover_demo_files(root)
    action_counts: Counter[int] = Counter()
    transition_count = 0
    episode_count = 0
    frame_shape = None
    for episode in iter_demo_episodes(root):
        episode_count += 1
        if frame_shape is None:
            frame_shape = list(episode.frames.shape[1:])
        for action in episode.actions:
            if max_examples is not None and transition_count >= int(max_examples):
                break
            action_counts[int(action)] += 1
            transition_count += 1
        if max_examples is not None and transition_count >= int(max_examples):
            break
    return {
        "root": str(root),
        "file_count": len(files),
        "episode_count": episode_count,
        "transition_count": transition_count,
        "frame_shape": frame_shape,
        "action_histogram": {str(key): value for key, value in sorted(action_counts.items())},
    }


def required_array(payload: Mapping[str, Any], keys: tuple[str, ...], source: Path) -> np.ndarray:
    for key in keys:
        if key in payload:
            return np.asarray(payload[key])
    raise ValueError(f"{source} missing required keys; expected one of {keys}")


def optional_array(payload: Mapping[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in payload:
            return np.asarray(payload[key])
    return None


def align_frames_to_actions(frames: np.ndarray, actions: np.ndarray, source: Path) -> np.ndarray:
    if len(frames) == len(actions):
        return frames
    if len(frames) == len(actions) + 1:
        return frames[:-1]
    raise ValueError(f"{source} frame/action length mismatch: {len(frames)} frames, {len(actions)} actions")


def align_optional_vector(values: np.ndarray | None, actions: np.ndarray, name: str, source: Path) -> np.ndarray | None:
    if values is None:
        return None
    vector = np.asarray(values).reshape(-1)
    if len(vector) == len(actions):
        return vector
    if len(vector) == len(actions) + 1:
        return vector[:-1]
    raise ValueError(f"{source} {name}/action length mismatch: {len(vector)} values, {len(actions)} actions")


__all__ = [
    "CrafterDemoEpisode",
    "discover_demo_files",
    "iter_demo_episodes",
    "iter_demo_transitions",
    "load_demo_npz",
    "summarize_human_dataset",
]
