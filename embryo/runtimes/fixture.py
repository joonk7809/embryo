"""Fixture runtime for tests and package smoke checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embryo.runtimes.base import RuntimeSpec, RuntimeStep, action_to_backend
from embryo.runtimes.registry import register_runtime


FIXTURE_SPEC = RuntimeSpec(
    name="fixture_memory",
    suite="fixture",
    observation_keys=(
        "raw_rgb_frame",
        "previous_rgb_frame",
        "previous_action",
        "candidate_score",
        "center_salience_score",
        "center_patch_hash",
        "visual_anchor_visible",
        "visual_anchor_family",
        "failed_action_event",
    ),
    action_names=("noop", "move_forward", "turn_left"),
    noop_action="noop",
    resource_action="move_forward",
    fallback_action="turn_left",
    description="Deterministic deployable-memory fixture runtime.",
)


@dataclass
class FixtureRuntime:
    """Small deterministic runtime that mimics deployable observation flow."""

    max_steps: int = 4
    spec: RuntimeSpec = FIXTURE_SPEC
    _tick: int = 0
    _previous_action: str = "noop"
    _closed: bool = False
    _script: tuple[float, ...] = field(default_factory=lambda: (0.9, 0.8, 0.1, 0.0))

    def reset(self, *, seed: int | None = None) -> RuntimeStep:
        self._tick = 0
        self._previous_action = "noop"
        self._closed = False
        return RuntimeStep(observation=self._observation(seed=seed))

    def step(self, action: str) -> RuntimeStep:
        if self._closed:
            raise RuntimeError("FixtureRuntime is closed")
        self._previous_action = str(self.action_to_backend(action))
        self._tick += 1
        reward = 1.0 if self._previous_action == "move_forward" and self._candidate_score() >= 0.5 else 0.0
        done = self._tick >= self.max_steps
        return RuntimeStep(observation=self._observation(), reward=reward, done=done, info={"fixture_tick": self._tick})

    def action_to_backend(self, action: str) -> Any:
        return action_to_backend(self.spec, action)

    def close(self) -> None:
        self._closed = True

    def _candidate_score(self) -> float:
        index = min(self._tick, len(self._script) - 1)
        return float(self._script[index])

    def _observation(self, *, seed: int | None = None) -> dict[str, object]:
        _ = seed
        visible = self._tick <= 1
        salience = self._candidate_score() if visible else 0.0
        return {
            "raw_rgb_frame": f"fixture_rgb_{self._tick}",
            "previous_rgb_frame": f"fixture_rgb_{max(0, self._tick - 1)}",
            "previous_action": self._previous_action,
            "candidate_score": self._candidate_score(),
            "center_salience_score": salience,
            "center_patch_hash": "fixture_anchor_a" if visible else "",
            "visual_anchor_visible": visible,
            "visual_anchor_family": "fixture_visual_anchor_v1",
            "failed_action_event": self._previous_action == "move_forward" and self._candidate_score() < 0.5,
            "tick": self._tick,
        }


@register_runtime("fixture_memory")
def make_fixture_runtime(**kwargs) -> FixtureRuntime:  # noqa: ANN003
    return FixtureRuntime(**kwargs)
