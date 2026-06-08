"""Actor config helpers for long-run collection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embryo.runtimes.base import RuntimeSpec


INLINE_DIAGNOSTIC_ACTOR = "inline_diagnostic"
DELTA_IRIS_ACTOR = "delta_iris"
DEFAULT_MEMORY_RESIDUAL_BIAS = 8.0


@dataclass(frozen=True)
class LongRunActorConfig:
    """Resolved actor configuration for long-run collection."""

    name: str = INLINE_DIAGNOSTIC_ACTOR
    checkpoint: str = ""
    repo_path: str = ""
    policy_mode: str = "argmax"
    device: str = "cpu"
    memory_residual_bias: float = DEFAULT_MEMORY_RESIDUAL_BIAS


def normalize_actor_config(value: Any) -> LongRunActorConfig:
    """Normalize optional actor config without importing external actors."""
    if value is None or value == "":
        return LongRunActorConfig()
    if isinstance(value, str):
        return LongRunActorConfig(name=value)
    if not isinstance(value, Mapping):
        raise TypeError("actor config must be a mapping, string, or null")
    return LongRunActorConfig(
        name=str(value.get("name", INLINE_DIAGNOSTIC_ACTOR)),
        checkpoint=str(value.get("checkpoint", "")),
        repo_path=str(value.get("repo_path", "")),
        policy_mode=str(value.get("policy_mode", "argmax")),
        device=str(value.get("device", "cpu")),
        memory_residual_bias=float(value.get("memory_residual_bias", DEFAULT_MEMORY_RESIDUAL_BIAS)),
    )


def actor_manifest(config: LongRunActorConfig) -> dict[str, Any]:
    """Public manifest fields for actor provenance."""
    return {
        "name": config.name,
        "checkpoint": config.checkpoint,
        "repo_path": config.repo_path,
        "policy_mode": config.policy_mode,
        "device": config.device,
        "memory_residual_bias": config.memory_residual_bias,
    }


def make_long_run_actor(config: Mapping[str, Any], spec: RuntimeSpec, *, root: str | Path | None = None):
    """Construct an optional frozen actor for a long-run episode."""
    actor = normalize_actor_config(config)
    if actor.name == INLINE_DIAGNOSTIC_ACTOR:
        return None
    if actor.name != DELTA_IRIS_ACTOR:
        raise ValueError(f"Unknown long-run actor: {actor.name}")

    from embryo.models.delta_iris_actor import DeltaIrisActor

    root_path = Path(root) if root is not None else Path.cwd()
    return DeltaIrisActor(
        repo_path=resolve_path(root_path, actor.repo_path),
        checkpoint_path=resolve_path(root_path, actor.checkpoint),
        action_names=spec.action_names,
        policy_mode=actor.policy_mode,
        device=actor.device,
    )


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
