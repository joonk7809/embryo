"""Contamination checks for actor/query contexts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embryo.core.contracts import DEFAULT_CONTRACT, ObservationContract


def scan_actor_context(
    context: Mapping[str, object],
    *,
    contract: ObservationContract = DEFAULT_CONTRACT,
    strict_unknown: bool = False,
) -> dict[str, object]:
    """Fail closed if forbidden actor inputs are present.

    The scan checks exact field names, known aliases, and nested dictionaries
    such as ``{"info": {"inventory": ...}}``.
    """
    flat = flatten_keys(context)
    forbidden = sorted(key for key in flat if key in contract.forbidden_names_and_aliases)
    unknown = sorted(key for key in flat if key not in contract.allowed_names and key not in contract.forbidden_names_and_aliases)
    failures = [{"kind": "forbidden_actor_input", "field": key} for key in forbidden]
    if strict_unknown:
        failures.extend({"kind": "unknown_actor_input", "field": key} for key in unknown)
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "forbidden_present": forbidden,
        "unknown_present": unknown,
        "strict_unknown": bool(strict_unknown),
        "contract": contract.name,
    }


def validate_contract(contract: ObservationContract = DEFAULT_CONTRACT) -> dict[str, object]:
    """Validate provenance and separation properties of a contract."""
    failures: list[dict[str, str]] = []
    allowed = (*contract.allowed_actor_inputs, *contract.allowed_query_inputs, *contract.allowed_router_state)
    for row in allowed:
        if not row.provenance:
            failures.append({"kind": "missing_provenance", "field": row.name})
        if not row.allowed:
            failures.append({"kind": "allowed_row_marked_disallowed", "field": row.name})
    for row in (*contract.teacher_only_diagnostics, *contract.forbidden_inputs):
        if row.allowed:
            failures.append({"kind": "forbidden_row_marked_allowed", "field": row.name})
    overlap = contract.allowed_names.intersection(contract.forbidden_names_and_aliases)
    for field in sorted(overlap):
        failures.append({"kind": "allowed_forbidden_overlap", "field": field})
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "allowed_input_count": len(allowed),
        "teacher_or_forbidden_count": len(contract.teacher_only_diagnostics) + len(contract.forbidden_inputs),
        "contract": contract.name,
    }


def flatten_keys(payload: Mapping[str, Any], *, prefix: str = "") -> set[str]:
    """Return dotted keys and common aliases for nested mappings."""
    keys: set[str] = set()
    for raw_key, value in payload.items():
        key = str(raw_key)
        dotted = f"{prefix}.{key}" if prefix else key
        keys.add(dotted)
        if prefix == "info":
            keys.add(f'info["{key}"]')
            keys.add(key)
        if isinstance(value, Mapping):
            keys.update(flatten_keys(value, prefix=dotted))
    return keys
