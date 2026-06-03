from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from embryo.eval.traces import read_jsonl
from embryo.models import (
    StackManifest,
    assemble_stack_manifest,
    build_stack_from_manifest,
    load_stack_manifest,
    save_stack_manifest,
    validate_stack_manifest,
)
from embryo.run.replay import DEFAULT_ARMS, run_reference_replay
from embryo.run.train_supervised import train_bc_policy, train_fact_writer, train_router, write_training_artifacts


ROOT = Path(__file__).resolve().parents[1]


class StackManifestTests(unittest.TestCase):
    def test_stack_manifest_round_trip_and_replay(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_component_manifests(root)
            manifest = assemble_stack_manifest(
                fact_writer="fact_writer/checkpoint_manifest.json",
                router="router/checkpoint_manifest.json",
                bc_policy="bc_policy/checkpoint_manifest.json",
            )
            stack_path = root / "stack_manifest.json"
            save_stack_manifest(manifest, stack_path)

            loaded = load_stack_manifest(stack_path)
            validation = validate_stack_manifest(loaded, base_dir=root)
            stack = build_stack_from_manifest(stack_path)
            rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_replay_features.jsonl")
            replay = run_reference_replay(rows, arms=DEFAULT_ARMS, stack=stack, stack_manifest=loaded.to_dict())

        self.assertIsInstance(loaded, StackManifest)
        self.assertTrue(validation["passed"])
        self.assertEqual(replay["summary"]["decision"], "GO_reference_replay_supported")
        self.assertEqual(replay["summary"]["stack_manifest"]["name"], "reference_memory_stack")

    def test_stack_manifest_fails_closed_on_missing_component(self) -> None:
        manifest = assemble_stack_manifest(
            fact_writer="missing_fact.json",
            router="missing_router.json",
            bc_policy="missing_bc.json",
        )
        with TemporaryDirectory() as tmpdir:
            validation = validate_stack_manifest(manifest, base_dir=tmpdir)

        self.assertFalse(validation["passed"])
        self.assertEqual(validation["failure_count"], 3)

    def _write_component_manifests(self, root: Path) -> None:
        fact_rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_fact_writer_dataset.jsonl")
        router_rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_router_dataset.jsonl")
        bc_rows = read_jsonl(ROOT / "data" / "fixtures" / "tiny_bc_dataset.jsonl")
        write_training_artifacts(train_fact_writer(fact_rows), root / "fact_writer")
        write_training_artifacts(train_router(router_rows), root / "router")
        write_training_artifacts(train_bc_policy(bc_rows), root / "bc_policy")


if __name__ == "__main__":
    unittest.main()
