"""Deployable observation contracts for the reproduction subset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal


SurfaceKind = Literal["actor", "query", "router", "diagnostic", "forbidden"]


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    kind: SurfaceKind
    provenance: str
    description: str
    aliases: tuple[str, ...] = ()
    allowed: bool = True
    post_action_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationContract:
    name: str
    version: str
    allowed_actor_inputs: tuple[SurfaceSpec, ...]
    allowed_query_inputs: tuple[SurfaceSpec, ...]
    allowed_router_state: tuple[SurfaceSpec, ...]
    teacher_only_diagnostics: tuple[SurfaceSpec, ...]
    forbidden_inputs: tuple[SurfaceSpec, ...]
    fixed_router_rules: tuple[str, ...] = ()
    fixed_stale_handling_rules: tuple[str, ...] = ()

    @property
    def allowed_names(self) -> frozenset[str]:
        return frozenset(
            spec.name
            for spec in (
                *self.allowed_actor_inputs,
                *self.allowed_query_inputs,
                *self.allowed_router_state,
            )
        )

    @property
    def forbidden_names_and_aliases(self) -> frozenset[str]:
        names: set[str] = set()
        for spec in (*self.teacher_only_diagnostics, *self.forbidden_inputs):
            names.add(spec.name)
            names.update(spec.aliases)
        return frozenset(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "allowed_actor_inputs": [spec.to_dict() for spec in self.allowed_actor_inputs],
            "allowed_query_inputs": [spec.to_dict() for spec in self.allowed_query_inputs],
            "allowed_router_state": [spec.to_dict() for spec in self.allowed_router_state],
            "teacher_only_diagnostics": [spec.to_dict() for spec in self.teacher_only_diagnostics],
            "forbidden_inputs": [spec.to_dict() for spec in self.forbidden_inputs],
            "fixed_router_rules": list(self.fixed_router_rules),
            "fixed_stale_handling_rules": list(self.fixed_stale_handling_rules),
        }


def spec(
    name: str,
    kind: SurfaceKind,
    provenance: str,
    description: str,
    *,
    aliases: Iterable[str] = (),
    allowed: bool = True,
    post_action_only: bool = False,
) -> SurfaceSpec:
    return SurfaceSpec(
        name=name,
        kind=kind,
        provenance=provenance,
        description=description,
        aliases=tuple(aliases),
        allowed=allowed,
        post_action_only=post_action_only,
    )


def reproduction_subset_contract() -> ObservationContract:
    actor_inputs = (
        spec("popgym_observation", "actor", "environment_observation", "Current POPGym observation."),
        spec("previous_popgym_observation", "actor", "prior_environment_observation", "Previous POPGym observation."),
        spec("gridworld_observation", "actor", "gridworld_runtime_observation", "Current gridworld observation."),
        spec("previous_action", "actor", "agent_action_history", "Agent-owned previous action."),
        spec("query_symbol", "actor", "deployable_query_surface", "Symbol queried from agent-owned memory."),
        spec("query_content_hash", "actor", "approved_query_content", "Audit hash of queried content."),
        spec("memory_age", "actor", "agent_owned_memory_state", "Age of cached memory content."),
    )
    diagnostics = (
        spec("reward", "diagnostic", "environment_outcome", "Reward is evaluation only.", aliases=("rewards",), allowed=False, post_action_only=True),
        spec("done", "diagnostic", "environment_outcome", "Done flag is evaluation only.", aliases=("terminal", "terminated", "truncated"), allowed=False, post_action_only=True),
        spec("target", "diagnostic", "environment_label", "Target labels are evaluation only.", aliases=("label", "target_count_eval_only"), allowed=False, post_action_only=True),
        spec("info", "diagnostic", "environment_info", "Backend info is evaluation only.", aliases=("inventory", "semantic", "player_pos", "achievements"), allowed=False, post_action_only=True),
    )
    forbidden = (
        spec("seed_source_metadata", "forbidden", "run_metadata", "Seed/source metadata is not an actor feature.", aliases=("seed", "source_path", "source_metadata"), allowed=False),
        spec("backend_state", "forbidden", "environment_backend", "Backend internals are privileged.", aliases=("world", "backend", "internals", "backend_object_ids"), allowed=False),
    )
    return ObservationContract(
        name="embryo_memory_reproduction_subset_v1",
        version="0.1.0",
        allowed_actor_inputs=actor_inputs,
        allowed_query_inputs=(),
        allowed_router_state=(),
        teacher_only_diagnostics=diagnostics,
        forbidden_inputs=forbidden,
    )


DEFAULT_CONTRACT = reproduction_subset_contract()


def contract_from_mapping(payload: Mapping[str, Any]) -> ObservationContract:
    def rows(key: str) -> tuple[SurfaceSpec, ...]:
        return tuple(SurfaceSpec(**dict(row)) for row in payload.get(key, ()))

    return ObservationContract(
        name=str(payload["name"]),
        version=str(payload["version"]),
        allowed_actor_inputs=rows("allowed_actor_inputs"),
        allowed_query_inputs=rows("allowed_query_inputs"),
        allowed_router_state=rows("allowed_router_state"),
        teacher_only_diagnostics=rows("teacher_only_diagnostics"),
        forbidden_inputs=rows("forbidden_inputs"),
        fixed_router_rules=tuple(str(row) for row in payload.get("fixed_router_rules", ())),
        fixed_stale_handling_rules=tuple(str(row) for row in payload.get("fixed_stale_handling_rules", ())),
    )
