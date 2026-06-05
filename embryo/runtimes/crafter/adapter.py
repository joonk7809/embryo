"""Crafter runtime adapter.

Crafter is optional. Importing this module is safe without the package
installed; construction fails closed with `CrafterUnavailable`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import numpy as np

from embryo.runtimes.base import RuntimeSpec, RuntimeStep, action_to_backend
from embryo.runtimes.crafter.features import extract_deployable_rgb_features
from embryo.runtimes.registry import register_runtime


class CrafterUnavailable(RuntimeError):
    """Raised when Crafter is requested but not installed."""


CRAFTER_SPEC = RuntimeSpec(
    name="crafter_memory",
    suite="crafter",
    observation_keys=(
        "raw_rgb_frame",
        "previous_rgb_frame",
        "previous_action",
        "rgb_delta_score",
        "center_salience_score",
        "center_patch_hash",
        "visual_change_event",
        "failed_action_event",
        "visual_anchor_visible",
        "visual_anchor_family",
        "candidate_score",
    ),
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
    deterministic_backend_patch: bool = True
    spec: RuntimeSpec = CRAFTER_SPEC
    _env: Any = field(default=None, init=False, repr=False)
    _previous_observation: Any = field(default=None, init=False, repr=False)
    _previous_action: str = "noop"

    def __post_init__(self) -> None:
        try:
            import crafter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise CrafterUnavailable("Install embryo[crafter] to use the Crafter runtime.") from exc
        self._env = crafter.Env(seed=self.seed) if self.seed is not None else crafter.Env()
        if self.deterministic_backend_patch:
            self._patch_deterministic_balance(self._env)
        self.spec = self._spec_from_env(self._env)
        if self.seed is not None and hasattr(self._env, "seed"):
            self._env.seed(self.seed)

    def reset(self, *, seed: int | None = None) -> RuntimeStep:
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        elif seed is not None and seed != self.seed:
            self.seed = seed
            self.close()
            try:
                import crafter  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency.
                raise CrafterUnavailable("Install embryo[crafter] to use the Crafter runtime.") from exc
            self._env = crafter.Env(seed=seed)
            if self.deterministic_backend_patch:
                self._patch_deterministic_balance(self._env)
            self.spec = self._spec_from_env(self._env)
        observation = self._env.reset()
        self._previous_observation = observation
        self._previous_action = "noop"
        return RuntimeStep(observation=self._deployable_observation(observation, previous_observation=None, previous_action="noop"))

    def step(self, action: str) -> RuntimeStep:
        backend_action = self.action_to_backend(action)
        previous_observation = self._previous_observation
        previous_action = str(action)
        raw_step = self._env.step(backend_action)
        if len(raw_step) == 5:
            observation, reward, terminated, truncated, info = raw_step
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = raw_step
        step = RuntimeStep(
            observation=self._deployable_observation(observation, previous_observation=previous_observation, previous_action=previous_action),
            reward=float(reward),
            done=bool(done),
            info=dict(info or {}),
        )
        self._previous_observation = observation
        self._previous_action = previous_action
        return step

    def action_to_backend(self, action: str) -> Any:
        if isinstance(action, str) and action.isdigit():
            return int(action)
        return action_to_backend(self.spec, action)

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()

    def _deployable_observation(self, observation: Any, *, previous_observation: Any, previous_action: str) -> dict[str, Any]:
        payload = {
            "raw_rgb_frame": observation,
            "previous_rgb_frame": previous_observation,
            "previous_action": previous_action,
        }
        payload.update(
            extract_deployable_rgb_features(
                observation,
                previous_observation=previous_observation,
                previous_action=previous_action,
            )
        )
        return payload

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

    def _patch_deterministic_balance(self, env: Any) -> None:
        if not hasattr(env, "_balance_object"):
            return

        def balance_object(
            env_self,  # noqa: ANN001
            chunk,  # noqa: ANN001
            objs,  # noqa: ANN001
            cls,  # noqa: ANN001
            material,  # noqa: ANN001
            span_dist,  # noqa: ANN001
            despan_dist,  # noqa: ANN001
            spawn_prob,  # noqa: ANN001
            despawn_prob,  # noqa: ANN001
            ctor,  # noqa: ANN001
            target_fn,  # noqa: ANN001
        ) -> None:
            xmin, xmax, ymin, ymax = chunk
            random = env_self._world.random
            creatures = sorted((obj for obj in objs if isinstance(obj, cls)), key=deterministic_object_key)
            mask = env_self._world.mask(*chunk, material)
            target_min, target_max = target_fn(len(creatures), mask.sum())
            if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
                xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
                ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
                xs, ys = xs[mask], ys[mask]
                i = random.randint(0, len(xs))
                pos = np.array((xs[i], ys[i]))
                empty = env_self._world[pos][1] is None
                away = env_self._player.distance(pos) >= span_dist
                if empty and away:
                    env_self._world.add(ctor(pos))
            elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
                obj = creatures[random.randint(0, len(creatures))]
                away = env_self._player.distance(obj.pos) >= despan_dist
                if away:
                    env_self._world.remove(obj)

        env._balance_object = MethodType(balance_object, env)


def deterministic_object_key(obj: Any) -> tuple[int, int, str]:
    pos = getattr(obj, "pos", (0, 0))
    try:
        x = int(pos[0])
        y = int(pos[1])
    except Exception:  # noqa: BLE001
        x = 0
        y = 0
    return x, y, obj.__class__.__name__


@register_runtime("crafter_memory")
def make_crafter_runtime(**kwargs) -> CrafterRuntime:  # noqa: ANN003
    return CrafterRuntime(**kwargs)
