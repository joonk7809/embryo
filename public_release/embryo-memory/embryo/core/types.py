"""Shared typed records used by the forward testbed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Fact:
    """A deployable memory fact derived from allowed observations."""

    name: str
    value: Any
    provenance: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecision:
    """A policy decision and its provenance."""

    action: str
    source: str
    invalid_or_unknown: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
