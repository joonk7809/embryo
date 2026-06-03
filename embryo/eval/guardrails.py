"""Guardrail checks for diagnostic replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embryo.eval.memory_grounded_score import action_name


def rate_at_most(value: float, threshold: float) -> bool:
    return float(value) <= float(threshold)


def rate_at_least(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold)


def resource_binding_guardrail(
    ticks: Sequence[Mapping[str, Any]],
    *,
    clean_arm: str = "clean",
    controls: Sequence[str] = ("wrong_binding", "shuffled", "stale"),
    min_divergence: float = 0.25,
    min_route_preservation: float = 0.8,
) -> dict[str, Any]:
    """Check resource-critical divergence and route preservation."""
    by_key = {
        (str(row.get("arm")), int(row.get("seed", 0)), str(row.get("episode_id", row.get("episode_index", "0"))), int(row.get("tick", 0))): row
        for row in ticks
    }
    clean_rows = [
        row
        for row in ticks
        if str(row.get("arm")) == clean_arm and bool(row.get("resource_memory_critical") or _critical_flag(row, "resource_memory_critical"))
    ]
    divergences: dict[str, float] = {}
    for control in controls:
        compared = 0
        divergent = 0
        for row in clean_rows:
            key = (control, int(row.get("seed", 0)), str(row.get("episode_id", row.get("episode_index", "0"))), int(row.get("tick", 0)))
            other = by_key.get(key)
            if other is None:
                continue
            compared += 1
            divergent += int(action_name(row) != action_name(other))
        divergences[control] = round(divergent / compared, 4) if compared else 0.0
    preserved_values = [
        bool(row.get("resource_route_preserved", row.get("route_preserved", False)))
        for row in clean_rows
    ]
    route_preservation = sum(preserved_values) / len(preserved_values) if preserved_values else 0.0
    passed = all(value >= min_divergence for value in divergences.values()) and route_preservation >= min_route_preservation
    return {
        "passed": passed,
        "resource_critical_tick_count": len(clean_rows),
        "resource_critical_divergence_vs_controls": divergences,
        "route_preservation": round(route_preservation, 4),
        "thresholds": {"resource_critical_divergence_min": min_divergence, "route_preservation_min": min_route_preservation},
    }


def event_repetition_guardrail(
    ticks: Sequence[Mapping[str, Any]],
    *,
    clean_arm: str = "clean",
    no_memory_arm: str = "no_memory",
    max_event_self_trigger: float = 0.15,
    max_repeated_excess: float = 0.05,
    max_invalid: float = 0.05,
) -> dict[str, Any]:
    """Check event self-trigger, repeated-loop, and invalid-action rates."""
    clean_rows = [row for row in ticks if str(row.get("arm")) == clean_arm]
    no_memory_rows = [row for row in ticks if str(row.get("arm")) == no_memory_arm]
    event_self_trigger = bool_rate(row.get("event_self_triggered", False) for row in clean_rows)
    repeated = bool_rate(row.get("repeated_action_loop", False) for row in clean_rows)
    no_memory_repeated = bool_rate(row.get("repeated_action_loop", False) for row in no_memory_rows)
    invalid = bool_rate(_invalid(row) for row in clean_rows)
    repeated_excess = repeated - no_memory_repeated
    passed = event_self_trigger <= max_event_self_trigger and repeated_excess <= max_repeated_excess and invalid <= max_invalid
    return {
        "passed": passed,
        "event_self_trigger_rate": round(event_self_trigger, 4),
        "repeated_action_loop_rate": round(repeated, 4),
        "no_memory_repeated_action_loop_rate": round(no_memory_repeated, 4),
        "repeated_action_loop_excess_vs_no_memory": round(repeated_excess, 4),
        "invalid_or_unknown_action_rate": round(invalid, 4),
        "thresholds": {
            "event_self_trigger_rate_max": max_event_self_trigger,
            "repeated_action_loop_excess_vs_no_memory_max": max_repeated_excess,
            "invalid_or_unknown_action_rate_max": max_invalid,
        },
    }


def _critical_flag(row: Mapping[str, Any], name: str) -> bool:
    flags = row.get("critical_flags")
    return bool(flags.get(name)) if isinstance(flags, Mapping) else False


def _invalid(row: Mapping[str, Any]) -> bool:
    action = row.get("selected_action")
    if isinstance(action, Mapping):
        return bool(action.get("invalid_or_unknown"))
    return bool(row.get("invalid_or_unknown_action"))


def bool_rate(values) -> float:  # noqa: ANN001
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0
