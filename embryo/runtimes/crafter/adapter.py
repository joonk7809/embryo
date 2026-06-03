"""Crafter runtime adapter.

Crafter is optional. Importing this module is safe without the package
installed; construction fails closed with `CrafterUnavailable`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embryo.runtimes.base import RuntimeSpec, RuntimeStep, action_to_backend
from embryo.runtimes.registry import register_runtime


class CrafterUnavailable(RuntimeError):
    """Raised when Crafter is requested but not installed."""


CRAFTER_SPEC = RuntimeSpec(
    name="crafter_memory",
    suite="crafter",
    observation_keys=("raw_rgb_frame", "previous_rgb_frame", "previous_action"),
    action_names=(
        "noop",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "do",
        "sleep",
        "place_stone",
        "place_table",
        "place_furnace",
        "place_plant",
        "make_wood_pickaxe",
        "make_stone_pickaxe",
        "make_iron_pickaxe",
        "make_wood_sword",
        "make_stone_sword",
        "make_iron_sword",
    ),
    noop_action="noop",
    resource_action="move_up",
    fallback_action="move_left",
    action_map={
        "noop": 0,
        "move_left": 1,
        "move_right": 2,
        "move_up": 3,
        "move_down": 4,
        "do": 5,
        "sleep": 6,
        "place_stone": 7,
        "place_table": 8,
        "place_furnace": 9,
        "place_plant": 10,
        "make_wood_pickaxe": 11,
        "make_stone_pickaxe": 12,
        "make_iron_pickaxe": 13,
        "make_wood_sword": 14,
        "make_stone_sword": 15,
        "make_iron_sword": 16,
    },
    optional_dependency="crafter",
    description="Crafter RGB runtime under the deployable memory contract.",
)


@dataclass
class CrafterRuntime:
    seed: int | None = None
    spec: RuntimeSpec = CRAFTER_SPEC
    _env: Any = field(default=None, init=False, repr=False)
    _previous_observation: Any = field(default=None, init=False, repr=False)
    _previous_action: str = "noop"

    def __post_init__(self) -> None:
        try:
            import crafter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise CrafterUnavailable("Install embryo[crafter] to use the Crafter runtime.") from exc
        self._env = crafter.Env()
        self.spec = self._spec_from_env(self._env)
        if self.seed is not None and hasattr(self._env, "seed"):
            self._env.seed(self.seed)

    def reset(self, *, seed: int | None = None) -> RuntimeStep:
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        observation = self._env.reset()
        self._previous_observation = observation
        self._previous_action = "noop"
        return RuntimeStep(observation=self._deployable_observation(observation))

    def step(self, action: str) -> RuntimeStep:
        backend_action = self.action_to_backend(action)
        raw_step = self._env.step(backend_action)
        if len(raw_step) == 5:
            observation, reward, terminated, truncated, info = raw_step
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = raw_step
        step = RuntimeStep(
            observation=self._deployable_observation(observation),
            reward=float(reward),
            done=bool(done),
            info=dict(info or {}),
        )
        self._previous_observation = observation
        self._previous_action = str(action)
        return step

    def action_to_backend(self, action: str) -> Any:
        if isinstance(action, str) and action.isdigit():
            return int(action)
        return action_to_backend(self.spec, action)

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()

    def _deployable_observation(self, observation: Any) -> dict[str, Any]:
        return {
            "raw_rgb_frame": observation,
            "previous_rgb_frame": self._previous_observation,
            "previous_action": self._previous_action,
        }

    def _spec_from_env(self, env: Any) -> RuntimeSpec:
        action_names = tuple(str(name) for name in getattr(env, "action_names", ())) or CRAFTER_SPEC.action_names
        action_map = {name: index for index, name in enumerate(action_names)}
        return RuntimeSpec(
            name=CRAFTER_SPEC.name,
            suite=CRAFTER_SPEC.suite,
            observation_keys=CRAFTER_SPEC.observation_keys,
            action_names=action_names,
            noop_action="noop" if "noop" in action_map else action_names[0],
            resource_action="move_up" if "move_up" in action_map else action_names[0],
            fallback_action="move_left" if "move_left" in action_map else action_names[0],
            action_map=action_map,
            optional_dependency=CRAFTER_SPEC.optional_dependency,
            description=CRAFTER_SPEC.description,
        )


@register_runtime("crafter_memory")
def make_crafter_runtime(**kwargs) -> CrafterRuntime:  # noqa: ANN003
    return CrafterRuntime(**kwargs)
