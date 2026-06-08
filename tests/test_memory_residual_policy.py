from __future__ import annotations

import unittest

from embryo.models import (
    ActorDecision,
    MemoryResidualInput,
    MemoryResidualPolicy,
    action_from_logits,
    decision_action_is_valid,
)
from embryo.run.long_run_arms import EpisodeState, select_action_with_actor_sidecar
from embryo.run.long_run_actors import INLINE_DIAGNOSTIC_ACTOR, actor_manifest, normalize_actor_config
from embryo.runtimes.fixture import FIXTURE_SPEC


class MemoryResidualPolicyTests(unittest.TestCase):
    def test_action_from_logits_uses_stable_action_order(self) -> None:
        action = action_from_logits({"left": 1.0, "right": 1.0}, action_order=("right", "left"), fallback="noop")

        self.assertEqual(action, "right")

    def test_inactive_memory_preserves_base_action(self) -> None:
        policy = MemoryResidualPolicy()
        base = ActorDecision(action="move_left", logits={"move_left": 1.0, "move_right": 0.5})

        decision = policy.apply(base, MemoryResidualInput(active=False), action_order=("move_left", "move_right"))

        self.assertEqual(decision.action, "move_left")
        self.assertFalse(decision.active)
        self.assertFalse(decision.changed_action)
        self.assertEqual(decision.applied_bias_count, 0)

    def test_active_memory_can_bias_action_without_owning_actor_state(self) -> None:
        policy = MemoryResidualPolicy(strength=1.0, max_abs_bias=2.0)
        base = ActorDecision(action="move_left", logits={"move_left": 1.0, "move_right": 0.5})
        memory = MemoryResidualInput(active=True, action_biases={"move_right": 1.0}, content_hash="abc", freshness_age=3)

        decision = policy.apply(base, memory, action_order=("move_left", "move_right"))

        self.assertEqual(decision.base_action, "move_left")
        self.assertEqual(decision.action, "move_right")
        self.assertTrue(decision.active)
        self.assertTrue(decision.changed_action)
        self.assertEqual(decision.applied_bias_count, 1)
        self.assertEqual(decision.metadata["content_hash"], "abc")
        self.assertEqual(decision.metadata["freshness_age"], 3)

    def test_residual_bias_is_clamped_and_ignores_unknown_actions(self) -> None:
        policy = MemoryResidualPolicy(strength=1.0, max_abs_bias=0.25)
        base = ActorDecision(action="noop", logits={"noop": 0.0, "do": 0.0})
        memory = MemoryResidualInput(active=True, action_biases={"do": 10.0, "backend_action": 10.0})

        decision = policy.apply(base, memory, action_order=("noop", "do"))

        self.assertEqual(decision.residual_logits, {"do": 0.25})
        self.assertEqual(decision.final_logits["do"], 0.25)
        self.assertNotIn("backend_action", decision.residual_logits)

    def test_actor_decision_validation_uses_public_action_names(self) -> None:
        self.assertTrue(decision_action_is_valid(ActorDecision("do"), ("noop", "do")))
        self.assertFalse(decision_action_is_valid(ActorDecision("backend_action"), ("noop", "do")))

    def test_actor_sidecar_diverges_only_after_cached_memory_use(self) -> None:
        visible = {
            "candidate_score": 0.9,
            "center_salience_score": 0.9,
            "center_patch_hash": "anchor",
            "visual_anchor_visible": True,
            "visual_anchor_family": "fixture_anchor",
            "previous_action": "noop",
        }
        hidden = dict(visible, visual_anchor_visible=False, center_patch_hash="", center_salience_score=0.0, candidate_score=0.0)
        base = ActorDecision(action="noop", logits={"noop": 2.0, "move_forward": 0.0, "turn_left": 0.0})
        residual = MemoryResidualPolicy(strength=1.0, max_abs_bias=8.0)

        def cached_decision(arm: str) -> tuple[dict, dict]:
            state = EpisodeState(seed=1, arm=arm, horizon=4)
            first = select_action_with_actor_sidecar(FIXTURE_SPEC, visible, state, arm, base, residual)
            state.observe(first["action"], first["route_mode"])
            return first, select_action_with_actor_sidecar(FIXTURE_SPEC, hidden, state, arm, base, residual)

        first, clean = cached_decision("query_memory_clean")
        _, shuffled = cached_decision("query_memory_shuffled")
        _, stale = cached_decision("query_memory_stale")
        _, wrong_binding = cached_decision("query_memory_wrong_binding")

        self.assertFalse(first["memory_sidecar_active"])
        self.assertTrue(clean["query_used_cached_fact"])
        self.assertEqual(clean["action"], "move_forward")
        self.assertEqual(clean["memory_residual_target_action"], "move_forward")
        self.assertTrue(clean["memory_sidecar_changed_action"])
        self.assertEqual(shuffled["action"], "turn_left")
        self.assertEqual(shuffled["memory_residual_target_action"], "turn_left")
        self.assertTrue(shuffled["memory_sidecar_changed_action"])
        self.assertEqual(stale["action"], "noop")
        self.assertIsNone(stale["memory_residual_target_action"])
        self.assertFalse(stale["memory_sidecar_active"])
        self.assertTrue(stale["effective_invalidated"])
        self.assertEqual(wrong_binding["action"], "turn_left")
        self.assertEqual(wrong_binding["memory_residual_target_action"], "turn_left")
        self.assertTrue(wrong_binding["memory_sidecar_changed_action"])


class LongRunActorConfigTests(unittest.TestCase):
    def test_default_actor_config_is_inline_diagnostic(self) -> None:
        config = normalize_actor_config(None)

        self.assertEqual(config.name, INLINE_DIAGNOSTIC_ACTOR)
        self.assertEqual(actor_manifest(config)["name"], INLINE_DIAGNOSTIC_ACTOR)

    def test_external_actor_config_preserves_provenance_fields(self) -> None:
        config = normalize_actor_config(
            {
                "name": "delta_iris",
                "repo_path": "runs/external/delta-iris",
                "checkpoint": "runs/external/delta-iris/checkpoints/last.pt",
                "policy_mode": "sampled",
                "memory_residual_bias": 4.0,
            }
        )

        manifest = actor_manifest(config)
        self.assertEqual(manifest["name"], "delta_iris")
        self.assertEqual(manifest["repo_path"], "runs/external/delta-iris")
        self.assertEqual(manifest["checkpoint"], "runs/external/delta-iris/checkpoints/last.pt")
        self.assertEqual(manifest["policy_mode"], "sampled")
        self.assertEqual(manifest["memory_residual_bias"], 4.0)


if __name__ == "__main__":
    unittest.main()
