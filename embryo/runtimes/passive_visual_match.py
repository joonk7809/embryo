"""Passive visual match runtime for causal memory sanity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embryo.runtimes.base import RuntimeSpec, RuntimeStep, action_to_backend
from embryo.runtimes.registry import register_runtime


PASSIVE_VISUAL_MATCH_SPEC = RuntimeSpec(
    name="passive_visual_match",
    suite="passive_visual_match",
    observation_keys=("raw_rgb_frame", "previous_rgb_frame", "previous_action"),
    action_names=("noop", "choose_left", "choose_right"),
    noop_action="noop",
    resource_action="choose_left",
    fallback_action="choose_left",
    description="Deterministic cue-delay-choice memory sanity runtime.",
)


@dataclass
class PassiveVisualMatchRuntime:
    """Cue appears once, disappears, then final reward requires matching it."""

    max_steps: int = 32
    seed: int | None = None
    spec: RuntimeSpec = PASSIVE_VISUAL_MATCH_SPEC
    _tick: int = 0
    _cue_side: str = "left"
    _previous_observation: np.ndarray | None = field(default=None, init=False, repr=False)
    _previous_action: str = "noop"
    _closed: bool = False

    def reset(self, *, seed: int | None = None) -> RuntimeStep:
        if seed is not None:
            self.seed = int(seed)
        self._tick = 0
        self._previous_action = "noop"
        self._closed = False
        self._cue_side = cue_side_for_seed(self.seed)
        observation = self._observation()
        self._previous_observation = observation
        return RuntimeStep(observation=self._deployable_observation(observation, previous=None))

    def step(self, action: str) -> RuntimeStep:
        if self._closed:
            raise RuntimeError("PassiveVisualMatchRuntime is closed")
        previous = self._previous_observation
        self._previous_action = str(self.action_to_backend(action))
        correct_action = f"choose_{self._cue_side}"
        choice_tick = self._tick >= self.max_steps - 1
        success = bool(choice_tick and self._previous_action == correct_action)
        reward = 1.0 if success else 0.0
        done = bool(choice_tick)
        self._tick += 1
        observation = self._observation()
        self._previous_observation = observation
        info = {
            "passive_match_phase": self._phase(self._tick - 1),
            "passive_match_cue_side": self._cue_side,
            "passive_match_correct_action": correct_action,
        }
        if choice_tick:
            info["passive_match_success"] = success
        return RuntimeStep(
            observation=self._deployable_observation(observation, previous=previous),
            reward=reward,
            done=done,
            info=info,
        )

    def action_to_backend(self, action: str) -> Any:
        return action_to_backend(self.spec, action)

    def close(self) -> None:
        self._closed = True

    def _deployable_observation(self, observation: np.ndarray, *, previous: np.ndarray | None) -> dict[str, Any]:
        return {
            "raw_rgb_frame": observation,
            "previous_rgb_frame": previous,
            "previous_action": self._previous_action,
        }

    def _observation(self) -> np.ndarray:
        phase = self._phase(self._tick)
        frame = np.zeros((16, 16, 3), dtype=np.uint8) + 24
        if phase == "cue":
            if self._cue_side == "left":
                frame[4:12, 1:5] = np.array([230, 40, 40], dtype=np.uint8)
            else:
                frame[4:12, 11:15] = np.array([40, 210, 80], dtype=np.uint8)
        elif phase == "choice":
            frame[5:11, 1:5] = np.array([220, 220, 220], dtype=np.uint8)
            frame[5:11, 11:15] = np.array([220, 220, 220], dtype=np.uint8)
        return frame

    def _phase(self, tick: int) -> str:
        if tick <= 0:
            return "cue"
        if tick >= self.max_steps - 1:
            return "choice"
        return "delay"


def cue_side_for_seed(seed: int | None) -> str:
    return "left" if int(seed or 0) % 2 == 0 else "right"


@register_runtime("passive_visual_match")
def make_passive_visual_match_runtime(**kwargs) -> PassiveVisualMatchRuntime:  # noqa: ANN003
    return PassiveVisualMatchRuntime(**kwargs)
