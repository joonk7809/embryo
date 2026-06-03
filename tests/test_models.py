from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.models import (
    CheckpointManifest,
    RouteBCPolicy,
    RuleRouterModel,
    ThresholdFactWriter,
    build_model_from_manifest,
    load_checkpoint_manifest,
    save_manifest,
    validate_manifest,
)


class ModelInterfaceTests(unittest.TestCase):
    def test_threshold_fact_writer_emits_sparse_fact(self) -> None:
        writer = ThresholdFactWriter(feature_name="candidate_score", threshold=0.7)

        low = writer.predict({"candidate_score": 0.2})
        high_fact = writer.fact({"candidate_score": 0.9, "direction_bin": "front"})

        self.assertFalse(low.predicted)
        self.assertEqual(high_fact.name, "facing_candidate_v1")
        self.assertTrue(high_fact.value)
        self.assertEqual(high_fact.metadata["direction_bin"], "front")

    def test_rule_router_wraps_freshness_contract(self) -> None:
        router = RuleRouterModel()

        resource = router.predict({"facing_candidate": True, "failed_action_event": True, "cache_age": 0})
        stale = router.predict({"facing_candidate": True, "failed_action_event": False, "cache_age": 9, "invalidated": True})

        self.assertEqual(resource.route_mode, "resource")
        self.assertTrue(resource.resource_route_preserved)
        self.assertEqual(stale.route_mode, "hold")
        self.assertTrue(stale.stale_invalidated)

    def test_route_bc_policy_maps_routes_to_actions(self) -> None:
        policy = RouteBCPolicy()

        self.assertEqual(policy.act({"route_mode": "resource"}).action, "move_forward")
        unknown = policy.act({"route_mode": "not_a_route"})
        self.assertTrue(unknown.invalid_or_unknown)

    def test_checkpoint_manifest_round_trips_and_validates_artifacts(self) -> None:
        manifest = CheckpointManifest(
            model_type="threshold_fact_writer",
            model_name="tiny_reference_writer",
            version="1",
            feature_schema=("candidate_score",),
            artifact_paths={"weights": "weights.json"},
            metadata={"parameters": {"feature_name": "candidate_score", "threshold": 0.25}},
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            save_manifest(manifest, manifest_path)
            loaded = load_checkpoint_manifest(manifest_path)

            self.assertEqual(loaded.feature_schema, ("candidate_score",))
            self.assertFalse(validate_manifest(loaded, base_dir=root, require_artifacts=True)["passed"])
            (root / "weights.json").write_text("{}", encoding="utf-8")
            self.assertTrue(validate_manifest(loaded, base_dir=root, require_artifacts=True)["passed"])

    def test_lightweight_factory_builds_reference_models(self) -> None:
        writer = build_model_from_manifest(
            {
                "model_type": "threshold_fact_writer",
                "model_name": "writer",
                "version": "1",
                "feature_schema": ["candidate_score"],
                "metadata": {"parameters": {"threshold": 0.1}},
            }
        )

        self.assertIsInstance(writer, ThresholdFactWriter)
        self.assertTrue(writer.predict({"candidate_score": 0.2}).predicted)


if __name__ == "__main__":
    unittest.main()
