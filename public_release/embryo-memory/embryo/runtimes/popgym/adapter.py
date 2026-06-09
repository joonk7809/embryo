"""Optional POPGym runtime adapter."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from embryo.runtimes.base import RuntimeSpec, RuntimeStep, action_to_backend
from embryo.runtimes.registry import register_runtime


POPGYM_REPEAT_FIRST_TASKS = {
    "repeat_first_easy": ("popgym.envs.repeat_first", "RepeatFirstEasy"),
    "repeat_first_medium": ("popgym.envs.repeat_first", "RepeatFirstMedium"),
    "repeat_first_hard": ("popgym.envs.repeat_first", "RepeatFirstHard"),
}


class PopGymUnavailableError(RuntimeError):
    """Raised when the optional POPGym package is unavailable."""


@dataclass
class POPGymRuntimeAdapter:
    """Thin Gymnasium-style adapter for POPGym diagnostic tasks."""

    task: str = "repeat_first_easy"
    max_steps: int | None = None
    seed: int | None = None
    deterministic_backend_patch: Any = None
    _env: Any = field(default=None, init=False, repr=False)
    _spec: RuntimeSpec | None = field(default=None, init=False, repr=False)
    _previous_observation: Any = field(default=None, init=False, repr=False)
    _previous_action: str = field(default="suit_0", init=False)
    _step_count: int = field(default=0, init=False)

    @property
    def spec(self) -> RuntimeSpec:
        if self._spec is None:
            self._env = self._env or self._make_env()
            self._spec = self._spec_from_env(self._env)
        return self._spec

    def reset(self, *, seed: int | None = None) -> RuntimeStep:
        if seed is not None:
            self.seed = int(seed)
        self._env = self._env or self._make_env()
        self._spec = self._spec or self._spec_from_env(self._env)
        self._step_count = 0
        self._previous_action = self.spec.noop_action
        obs, info = self._env.reset(seed=self.seed)
        self._previous_observation = obs
        return RuntimeStep(observation=self._deployable_observation(obs, previous=None), reward=None, done=False, info=dict(info or {}))

    def step(self, action: str) -> RuntimeStep:
        backend_action = self.action_to_backend(action)
        obs, reward, terminated, truncated, info = self._env.step(backend_action)
        self._step_count += 1
        done = bool(terminated or truncated or (self.max_steps is not None and self._step_count >= int(self.max_steps)))
        post_info = dict(info or {})
        post_info.update(
            {
                "popgym_task": self.task,
                "popgym_success": float(reward) > 0.0,
                "popgym_query_tick": True,
            }
        )
        previous = self._previous_observation
        self._previous_observation = obs
        self._previous_action = str(action)
        return RuntimeStep(observation=self._deployable_observation(obs, previous=previous), reward=float(reward), done=done, info=post_info)

    def action_to_backend(self, action: str) -> Any:
        return action_to_backend(self.spec, action)

    def close(self) -> None:
        env = self._env
        if env is not None and hasattr(env, "close"):
            env.close()

    def _make_env(self) -> Any:
        module_name, class_name = POPGYM_REPEAT_FIRST_TASKS.get(self.task, ("", ""))
        if not module_name:
            raise ValueError(f"Unknown POPGym task: {self.task}")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise PopGymUnavailableError("Install the optional popgym package to use popgym_repeat_first.") from exc
        env_class = getattr(module, class_name)
        return env_class()

    def _spec_from_env(self, env: Any) -> RuntimeSpec:
        action_count = discrete_action_count(env.action_space)
        action_names = tuple(f"suit_{idx}" for idx in range(action_count))
        return RuntimeSpec(
            name="popgym_repeat_first",
            suite="popgym",
            observation_keys=("popgym_observation", "previous_popgym_observation", "previous_action"),
            action_names=action_names,
            noop_action=action_names[0],
            resource_action=action_names[0],
            fallback_action=action_names[0],
            action_map={name: idx for idx, name in enumerate(action_names)},
            optional_dependency="popgym",
            description=f"POPGym {self.task} adapter with deployable observation fields.",
        )

    def _deployable_observation(self, observation: Any, *, previous: Any) -> dict[str, Any]:
        return {
            "popgym_observation": json_safe_value(observation),
            "previous_popgym_observation": json_safe_value(previous),
            "previous_action": self._previous_action,
        }


def discrete_action_count(action_space: Any) -> int:
    count = getattr(action_space, "n", None)
    if count is None:
        raise TypeError("POPGym RepeatFirst action space must be discrete.")
    return int(count)


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return json_safe_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


@register_runtime("popgym_repeat_first")
def make_popgym_repeat_first_runtime(**kwargs: Any) -> POPGymRuntimeAdapter:
    return POPGymRuntimeAdapter(**kwargs)
