"""Gym-compatible Crafter environment for survival training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from embryo.runtimes.crafter.adapter import CRAFTER_SPEC, CrafterRuntime, CrafterUnavailable


class CrafterTrainingUnavailable(RuntimeError):
    """Raised when optional training dependencies are missing."""


def make_crafter_survival_env(
    *,
    seed: int | None = None,
    deterministic_backend_patch: bool = True,
    frame_size: Sequence[int] = (64, 64),
    max_episode_steps: int | None = None,
    reward_mode: str = "native",
    alive_bonus: float = 0.0,
):
    gym = load_gym_module()
    frame_shape = frame_shape_from_config(frame_size)

    class CrafterSurvivalEnv(gym.Env):  # type: ignore[name-defined]
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.runtime = CrafterRuntime(seed=seed, deterministic_backend_patch=deterministic_backend_patch)
            self.action_names = tuple(self.runtime.spec.action_names)
            self.observation_space = gym.spaces.Box(0, 255, shape=(*frame_shape, 3), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(len(self.action_names))
            self.step_count = 0

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # noqa: ARG002
            self.step_count = 0
            step = self.runtime.reset(seed=seed)
            return observation_frame(step.observation, frame_shape), {}

        def step(self, action: int):
            self.step_count += 1
            action_name = action_name_from_index(self.action_names, action)
            step = self.runtime.step(action_name)
            terminated = bool(step.done)
            truncated = bool(max_episode_steps is not None and self.step_count >= int(max_episode_steps) and not terminated)
            reward = training_reward(step.reward, done=terminated or truncated, reward_mode=reward_mode, alive_bonus=alive_bonus)
            return observation_frame(step.observation, frame_shape), reward, terminated, truncated, dict(step.info or {})

        def close(self) -> None:
            self.runtime.close()

    return CrafterSurvivalEnv()


def load_gym_module():
    try:
        import gymnasium as gym  # type: ignore

        return gym
    except Exception:
        try:
            import gym  # type: ignore

            return gym
        except Exception as exc:
            raise CrafterTrainingUnavailable("Install embryo[train] to use Crafter survival training.") from exc


def frame_shape_from_config(frame_size: Sequence[int]) -> tuple[int, int]:
    if len(tuple(frame_size)) != 2:
        raise ValueError("frame_size must contain [height, width]")
    height, width = (int(value) for value in frame_size)
    if height <= 0 or width <= 0:
        raise ValueError("frame_size values must be positive")
    return height, width


def observation_frame(observation: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    if isinstance(observation, dict):
        observation = observation.get("raw_rgb_frame")
    return resize_rgb_frame(observation, frame_shape)


def resize_rgb_frame(frame: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("expected RGB frame with shape HxWx3")
    image = image[..., :3]
    if image.shape[:2] != frame_shape:
        y_index = np.linspace(0, image.shape[0] - 1, frame_shape[0]).round().astype(np.int64)
        x_index = np.linspace(0, image.shape[1] - 1, frame_shape[1]).round().astype(np.int64)
        image = image[y_index][:, x_index]
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def action_name_from_index(action_names: Sequence[str], action: int) -> str:
    index = int(action)
    if index < 0 or index >= len(action_names):
        raise ValueError(f"action index out of range: {action}")
    return str(action_names[index])


def training_reward(native_reward: float, *, done: bool, reward_mode: str, alive_bonus: float) -> float:
    mode = str(reward_mode)
    reward = float(native_reward)
    if mode == "native":
        return reward
    if mode == "native_plus_alive":
        return reward + (0.0 if done else float(alive_bonus))
    raise ValueError(f"Unsupported reward_mode: {reward_mode}")


def gym_dependency_status() -> dict[str, Any]:
    try:
        gym = load_gym_module()
        return {"available": True, "module": gym.__name__, "version": getattr(gym, "__version__", None)}
    except CrafterTrainingUnavailable as exc:
        return {"available": False, "reason": str(exc)}


__all__ = [
    "CRAFTER_SPEC",
    "CrafterTrainingUnavailable",
    "CrafterUnavailable",
    "action_name_from_index",
    "frame_shape_from_config",
    "gym_dependency_status",
    "make_crafter_survival_env",
    "observation_frame",
    "resize_rgb_frame",
    "training_reward",
]
