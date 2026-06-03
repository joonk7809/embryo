"""Shared runtime interface.

Runtime adapters are intentionally separate from memory, model, and scoring
code. A runtime owns environment construction and step/reset semantics; the
memory stack only sees deployable observation dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    suite: str
    observation_keys: tuple[str, ...]
    action_names: tuple[str, ...]
    noop_action: str = "noop"
    resource_action: str = "move_forward"
    fallback_action: str = "turn_left"
    action_map: Mapping[str, Any] = field(default_factory=dict)
    optional_dependency: str | None = None
    description: str = ""


@dataclass(frozen=True)
class RuntimeStep:
    observation: dict[str, Any]
    reward: float | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class RuntimeAdapter(Protocol):
    spec: RuntimeSpec

    def reset(self, *, seed: int | None = None) -> RuntimeStep:
        ...

    def step(self, action: str) -> RuntimeStep:
        ...

    def action_to_backend(self, action: str) -> Any:
        ...

    def close(self) -> None:
        ...


def deployable_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a runtime observation into a plain deployable mapping."""
    return dict(payload)


def action_to_backend(spec: RuntimeSpec, action: str) -> Any:
    """Map a public action label to a backend action value."""
    if action in spec.action_map:
        return spec.action_map[action]
    if action in spec.action_names:
        return action
    raise ValueError(f"Action {action!r} is not valid for runtime {spec.name}")
