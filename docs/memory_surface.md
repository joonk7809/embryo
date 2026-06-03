# Memory Surface

Embryo represents memory as a small set of deployable facts, query surfaces, freshness state, and router decisions.

## Facts

- `facing_candidate_v1`: sparse visual candidate fact.
- `event_failure`: action-result event inferred from deployable observations.
- `visual_change`: visual delta event inferred from RGB changes.

## Queries

- `resource_candidate_query`: asks whether the current deployable facts support a resource-facing decision.
- `event_failure_query`: asks whether recent action outcomes suggest recovery.
- `visual_plus_event_query`: combines resource and event surfaces.

## Freshness

The cache tracks age, invalidation, and visual-change conflicts. Short-TTL freshness prevents stale content from silently overriding fresh visual evidence.

## Router

The router preserves fresh resource routes, allows bounded event fallback, and exposes guardrail fields for route preservation, stale invalidation, repeat-loop breaking, and event self-trigger checks.
