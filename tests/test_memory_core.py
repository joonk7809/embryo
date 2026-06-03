from __future__ import annotations

import unittest

from embryo.memory.facts import event_failure_fact, facing_candidate_fact, fact_bool, visual_change_fact
from embryo.memory.freshness import FreshnessState, advance_freshness, reset_freshness, short_ttl_valid
from embryo.memory.queries import build_event_query, build_resource_query, combine_queries, query_content_hash
from embryo.memory.router import RouterState, event_failed_action_fallback_cooldown, route_with_freshness


class MemoryCoreTests(unittest.TestCase):
    def test_fact_helpers_create_deployable_facts(self) -> None:
        facts = [facing_candidate_fact(True, confidence=0.8), event_failure_fact(False), visual_change_fact(True)]

        self.assertTrue(fact_bool(facts, "facing_candidate_v1"))
        self.assertFalse(fact_bool(facts, "event_failure"))
        self.assertEqual(facts[0].provenance, "rgb_delta_prev_action")

    def test_query_surface_hash_is_stable(self) -> None:
        facts = [facing_candidate_fact(True), event_failure_fact(True)]
        resource = build_resource_query(facts, cache_age=1, fresh=True)
        event = build_event_query(facts, cache_age=1)
        combined = combine_queries(resource, event)

        self.assertEqual(resource.family, "resource_candidate_query")
        self.assertEqual(len(combined.content_hash), 16)
        self.assertEqual(combined.content_hash, combine_queries(resource, event).content_hash)
        self.assertIn("facing_candidate_v1=True", query_content_hash(facts))

    def test_freshness_ttl_and_visual_conflict(self) -> None:
        state = reset_freshness()
        self.assertTrue(short_ttl_valid(state))
        aged = advance_freshness(state, visual_change=False, content_changed=False, ttl=0)
        self.assertFalse(short_ttl_valid(aged, ttl=0))
        conflict = advance_freshness(FreshnessState(cache_age=2), visual_change=True, content_changed=False)
        self.assertTrue(conflict.visual_change_conflict)
        self.assertFalse(short_ttl_valid(conflict))

    def test_router_preserves_fresh_resource_route(self) -> None:
        decision = route_with_freshness(
            facing_candidate=True,
            failed_action_event=True,
            freshness=FreshnessState(cache_age=0),
            state=RouterState(cooldown=0),
        )

        self.assertEqual(decision.route, "resource")
        self.assertTrue(decision.resource_route_preserved)
        self.assertEqual(
            event_failed_action_fallback_cooldown(
                facing_candidate=True,
                failed_action_event=True,
                cooldown=0,
                freshness=FreshnessState(cache_age=0),
            ),
            "resource",
        )

    def test_router_fallback_and_self_trigger_detection(self) -> None:
        fallback = route_with_freshness(
            facing_candidate=False,
            failed_action_event=True,
            freshness=FreshnessState(cache_age=0),
            state=RouterState(cooldown=0),
        )
        self.assertEqual(fallback.route, "event_fallback")
        self.assertTrue(fallback.event_fallback_gate)

        self_trigger = route_with_freshness(
            facing_candidate=False,
            failed_action_event=False,
            freshness=FreshnessState(cache_age=0),
            state=RouterState(cooldown=0, repeat_count=3, last_route="event_fallback"),
        )
        self.assertTrue(self_trigger.repeat_loop_breaker_gate)
        self.assertTrue(self_trigger.event_self_triggered)

    def test_stale_resource_does_not_preserve_resource_route(self) -> None:
        decision = route_with_freshness(
            facing_candidate=True,
            failed_action_event=False,
            freshness=FreshnessState(cache_age=10, invalidated=True),
        )

        self.assertEqual(decision.route, "hold")
        self.assertFalse(decision.resource_route_preserved)
        self.assertTrue(decision.stale_invalidated)


if __name__ == "__main__":
    unittest.main()
