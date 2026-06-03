from __future__ import annotations

from pathlib import Path
import unittest

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.memory.facts import facing_candidate_fact
from embryo.memory.freshness import FreshnessState, short_ttl_valid
from embryo.memory.queries import query_content_hash
from embryo.memory.router import event_failed_action_fallback_cooldown


ROOT = Path(__file__).resolve().parents[1]


class EmbryoV1SkeletonTests(unittest.TestCase):
    def test_smoke_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        self.assertTrue(config["smoke"]["enabled"])

    def test_fact_query_and_router_smoke(self) -> None:
        fact = facing_candidate_fact(True, confidence=0.9)
        self.assertEqual(query_content_hash([fact]), "facing_candidate_v1=True")
        route = event_failed_action_fallback_cooldown(
            facing_candidate=True,
            failed_action_event=False,
            cooldown=0,
            freshness=FreshnessState(cache_age=1),
        )
        self.assertEqual(route, "resource")
        self.assertTrue(short_ttl_valid(FreshnessState(cache_age=4)))

    def test_contamination_fails_closed(self) -> None:
        clean = scan_actor_context({"raw_rgb_frame": "fixture", "previous_action": "noop"})
        dirty = scan_actor_context({"reward": 1.0, "raw_rgb_frame": "fixture"})
        self.assertTrue(clean["passed"])
        self.assertFalse(dirty["passed"])
        self.assertEqual(dirty["failure_count"], 1)

    def test_no_legacy_imports_in_forward_package(self) -> None:
        forbidden = ("experiments.", "embryo_arena", "runs.", "artifacts.")
        for path in (ROOT / "embryo").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path} imports or references legacy token {token}")

    def test_public_docs_do_not_use_milestone_codes(self) -> None:
        forbidden = tuple(
            prefix + marker
            for prefix in ("M", "m")
            for marker in ("6.", "7.", "6_", "7_")
        )
        public_paths = [
            ROOT / "README.md",
            *(ROOT / "docs").glob("*.md"),
            *(ROOT / "configs").glob("*.yaml"),
            *(ROOT / "configs").glob("*.md"),
        ]
        for path in public_paths:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path} exposes internal milestone token {token}")


if __name__ == "__main__":
    unittest.main()
