"""Query surface builders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from embryo.core.types import Fact
from embryo.memory.facts import EVENT_FAILURE, FACING_CANDIDATE_V1, fact_bool


@dataclass(frozen=True)
class QuerySurface:
    """A deployable query surface built from facts and freshness state."""

    family: str
    content: dict[str, Any]
    provenance: str

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(self.content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_hash"] = self.content_hash
        return payload


def build_resource_query(facts: Iterable[Fact], *, cache_age: int = 0, fresh: bool = True) -> QuerySurface:
    """Build the resource candidate query from deployable facts."""
    facts = tuple(facts)
    return QuerySurface(
        family="resource_candidate_query",
        content={
            "facing_candidate": fact_bool(facts, FACING_CANDIDATE_V1),
            "cache_age": int(cache_age),
            "fresh": bool(fresh),
        },
        provenance="facing_candidate_v1_and_cache_state",
    )


def build_event_query(facts: Iterable[Fact], *, cache_age: int = 0) -> QuerySurface:
    """Build the event/failure query from deployable event facts."""
    facts = tuple(facts)
    return QuerySurface(
        family="event_failure_query",
        content={
            "event_failure": fact_bool(facts, EVENT_FAILURE),
            "cache_age": int(cache_age),
        },
        provenance="previous_action_result_event",
    )


def combine_queries(*queries: QuerySurface) -> QuerySurface:
    """Combine query surfaces while preserving family names and content hashes."""
    return QuerySurface(
        family="visual_plus_event_query",
        content={query.family: query.content for query in queries},
        provenance="+".join(query.provenance for query in queries),
    )


def query_content_hash(facts: Iterable[Fact]) -> str:
    """Stable public string representation for a fact-only query.

    This preserves the original smoke behavior while richer query surfaces use
    `QuerySurface.content_hash`.
    """
    parts = [f"{fact.name}={fact.value}" for fact in sorted(facts, key=lambda item: item.name)]
    return "|".join(parts)
