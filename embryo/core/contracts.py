"""Deployable observation contracts.

The forward package keeps contracts as ordinary Python data structures so they
can be tested, serialized, and used by contamination scans without importing
historical experiment scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal


SurfaceKind = Literal["actor", "query", "router", "diagnostic", "forbidden"]


@dataclass(frozen=True)
class SurfaceSpec:
    """A field that may appear in an actor, query, router, or diagnostic surface."""

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
    """A deployable observation contract for a runtime."""

    name: str
    version: str
    allowed_actor_inputs: tuple[SurfaceSpec, ...]
    allowed_query_inputs: tuple[SurfaceSpec, ...]
    allowed_router_state: tuple[SurfaceSpec, ...]
    teacher_only_diagnostics: tuple[SurfaceSpec, ...]
    forbidden_inputs: tuple[SurfaceSpec, ...]
    fixed_router_rules: tuple[str, ...]
    fixed_stale_handling_rules: tuple[str, ...]

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


def crafter_memory_contract() -> ObservationContract:
    """Return the forward Crafter memory contract."""
    actor_inputs = (
        spec("raw_rgb_frame", "actor", "crafter_env_observation_rgb", "Current public RGB observation."),
        spec("previous_rgb_frame", "actor", "prior_raw_rgb_frame", "Previous public RGB frame from the same episode."),
        spec("bounded_recent_rgb_deltas", "actor", "rgb_frame_differences", "Bounded visual deltas derived from public RGB frames."),
        spec("previous_action", "actor", "policy_action_history", "Actor-owned previous action."),
        spec("previous_action_result_event", "actor", "previous_action_and_rgb_delta", "Event inferred from previous action and deployable visual change."),
        spec("facing_candidate_v1", "actor", "rgb_delta_previous_action_fact_writer", "Sparse visual fact used by the memory contract."),
        spec("query_content", "actor", "deployable_query_surface", "Query content derived from deployable facts and events."),
        spec("query_cache_age", "actor", "query_cache_state", "Age of cached query content."),
        spec("short_ttl_stale_state", "actor", "query_cache_and_visual_change", "Short-TTL freshness and invalidation state."),
    )
    query_inputs = (
        spec("resource_candidate_query", "query", "facing_candidate_v1", "Resource-oriented query surface."),
        spec("event_failure_query", "query", "previous_action_result_event", "Event or failure query surface."),
        spec("visual_plus_event_query", "query", "resource_candidate_query_and_event_failure_query", "Combined visual/event query surface."),
        spec("query_content_hash", "query", "approved_query_content", "Trace/audit hash of query content."),
        spec("query_cache_age", "query", "query_cache_state", "Age of cached query content."),
        spec("freshness_validity_bits", "query", "cache_age_and_visual_change", "Freshness and invalidation booleans."),
    )
    router_state = (
        spec("event_failed_action_fallback_cooldown", "router", "previous_action_result_event_and_cooldown_counter", "Stabilized event fallback gate."),
        spec("short_ttl_stale_handling", "router", "query_cache_age_and_visual_change", "Short-TTL stale handling."),
        spec("event_self_trigger_prevention", "router", "router_state_and_visual_change_event", "Prevents recursive event fallback without new visual evidence."),
        spec("repeated_loop_prevention_fallback", "router", "previous_actions_and_visual_change_events", "Bounded fallback for repeated action loops."),
        spec("resource_route_preservation", "router", "facing_candidate_v1", "Preserves resource route on resource-critical ticks."),
    )
    diagnostics = (
        spec("info_inventory", "diagnostic", "environment_info", "Inventory labels and deltas.", aliases=('info["inventory"]', "info.inventory", "inventory"), allowed=False, post_action_only=True),
        spec("info_semantic", "diagnostic", "environment_info", "Semantic labels and retrospective audits.", aliases=('info["semantic"]', "info.semantic", "semantic"), allowed=False, post_action_only=True),
        spec("info_player_pos", "diagnostic", "environment_info", "Player position diagnostics.", aliases=('info["player_pos"]', "info.player_pos", "player_pos"), allowed=False, post_action_only=True),
        spec("reward", "diagnostic", "environment_outcome", "Reward is evaluation only.", aliases=("rewards",), allowed=False, post_action_only=True),
        spec("done", "diagnostic", "environment_outcome", "Done flag is evaluation only.", aliases=("terminal", "terminated", "discount"), allowed=False, post_action_only=True),
        spec("achievements", "diagnostic", "environment_info", "Achievements are evaluation only.", aliases=("achievement",), allowed=False, post_action_only=True),
        spec("diagnostic_progress_score", "diagnostic", "offline_metric", "Memory-grounded and raw progress scores.", aliases=("memory_grounded_score", "raw_progress"), allowed=False, post_action_only=True),
    )
    forbidden = (
        spec("seed_source_metadata", "forbidden", "run_metadata", "Seed/source metadata is not a feature.", aliases=("seed", "source_path", "source_metadata"), allowed=False),
        spec("backend_internals", "forbidden", "environment_backend", "Backend internals are privileged.", aliases=("world", "backend_state", "internals"), allowed=False),
    )
    return ObservationContract(
        name="crafter_memory_v1",
        version="0.1.0",
        allowed_actor_inputs=actor_inputs,
        allowed_query_inputs=query_inputs,
        allowed_router_state=router_state,
        teacher_only_diagnostics=diagnostics,
        forbidden_inputs=forbidden,
        fixed_router_rules=(
            "Use event_failed_action_fallback_cooldown as the stabilized router.",
            "Event fallback requires deployable failed-action or repeated-loop evidence.",
            "Event route cannot self-trigger without new RGB visual-change evidence.",
            "Resource route must be preserved on facing_candidate_v1 resource-critical ticks.",
            "Diagnostic outcome fields cannot affect routing.",
        ),
        fixed_stale_handling_rules=(
            "Use short_ttl_stale_handling for freshness.",
            "Track query cache age explicitly.",
            "Invalidate stale cache when deployable visual-change evidence conflicts with cached content.",
            "Do not let stale content silently override fresh facing_candidate_v1 evidence.",
        ),
    )


DEFAULT_CONTRACT = crafter_memory_contract()


def contract_from_mapping(payload: Mapping[str, Any]) -> ObservationContract:
    """Load a contract from a mapping produced by :meth:`ObservationContract.to_dict`."""

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
