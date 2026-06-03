"""Config-driven supervised memory pipeline.

The pipeline is a small reproducibility harness:

runtime collection -> feature rows -> datasets -> module manifests -> stack replay

It is intentionally runtime-agnostic and does not make a benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.core.config import load_config
from embryo.datasets import build_supervised_datasets, write_dataset_bundle
from embryo.eval.traces import write_jsonl
from embryo.models import assemble_stack_manifest, build_stack_from_manifest, load_stack_manifest, save_stack_manifest, validate_stack_manifest
from embryo.run.collect import collect_feature_rows, collect_runtime_trace
from embryo.run.replay import DEFAULT_ARMS, run_reference_replay, write_replay_artifacts
from embryo.run.train_supervised import train_bc_policy, train_fact_writer, train_router, write_training_artifacts


def run_supervised_memory_pipeline(
    *,
    out: str | Path,
    runtime_name: str = "fixture_memory",
    steps: int = 4,
    seed: int = 0,
    replay_arms: Sequence[str] = DEFAULT_ARMS,
) -> dict[str, Any]:

    root = Path(out)
    paths = pipeline_paths(root)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    trace_rows = collect_runtime_trace(runtime_name, steps=steps, seed=seed)
    feature_rows = collect_feature_rows(trace_rows)
    write_jsonl(paths["trace"], trace_rows)
    write_jsonl(paths["features"], feature_rows)

    dataset_result = build_supervised_datasets(feature_rows, source_name=str(paths["features"]))
    write_dataset_bundle(dataset_result, paths["datasets"])

    fact_result = train_fact_writer(dataset_result.fact_writer_rows)
    router_result = train_router(dataset_result.router_rows)
    bc_result = train_bc_policy(dataset_result.bc_rows)
    write_training_artifacts(fact_result, paths["fact_writer"])
    write_training_artifacts(router_result, paths["router"])
    write_training_artifacts(bc_result, paths["bc_policy"])

    stack_manifest = assemble_stack_manifest(
        fact_writer=relative_component(paths["fact_writer"] / "checkpoint_manifest.json", paths["stack_manifest"]),
        router=relative_component(paths["router"] / "checkpoint_manifest.json", paths["stack_manifest"]),
        bc_policy=relative_component(paths["bc_policy"] / "checkpoint_manifest.json", paths["stack_manifest"]),
        metadata={"pipeline": "supervised_memory"},
    )
    save_stack_manifest(stack_manifest, paths["stack_manifest"])
    stack_validation = validate_stack_manifest(stack_manifest, base_dir=paths["stack_manifest"].parent)
    (paths["stack_validation"]).write_text(json.dumps(stack_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loaded_stack = build_stack_from_manifest(paths["stack_manifest"])
    loaded_manifest = load_stack_manifest(paths["stack_manifest"])
    replay_result = run_reference_replay(
        feature_rows,
        arms=tuple(replay_arms),
        stack=loaded_stack,
        stack_manifest={"path": str(paths["stack_manifest"]), "loaded": loaded_manifest.to_dict(), "validation": stack_validation},
    )
    write_replay_artifacts(replay_result, paths["replay"])

    summary = {
        "decision": decision_from_pipeline(dataset_result.manifest, stack_validation, replay_result["summary"]),
        "runtime": runtime_name,
        "steps_requested": int(steps),
        "seed": int(seed),
        "trace_rows": len(trace_rows),
        "feature_rows": len(feature_rows),
        "dataset_manifest": dataset_result.manifest,
        "training_metrics": {
            "fact_writer": fact_result["metrics"],
            "router": router_result["metrics"],
            "bc_policy": bc_result["metrics"],
        },
        "stack_validation": {"passed": stack_validation["passed"], "failure_count": stack_validation["failure_count"]},
        "replay_decision": replay_result["summary"]["decision"],
        "replay_gaps": replay_result["summary"]["score"]["memory_grounded_gaps_clean_minus_controls"],
        "artifacts": {name: str(path) for name, path in paths.items()},
        "claim_boundary": "diagnostic_pipeline_scaffold",
    }
    write_pipeline_summary(summary, root)
    return summary


def pipeline_paths(root: Path) -> dict[str, Path]:
    return {
        "collection": root / "collection",
        "trace": root / "collection" / "trace.jsonl",
        "features": root / "collection" / "features.jsonl",
        "datasets": root / "datasets",
        "fact_writer": root / "models" / "fact_writer",
        "router": root / "models" / "router",
        "bc_policy": root / "models" / "bc_policy",
        "stack": root / "stack",
        "stack_manifest": root / "stack" / "stack_manifest.json",
        "stack_validation": root / "stack" / "stack_manifest_validation.json",
        "replay": root / "replay",
    }


def relative_component(component_path: Path, stack_path: Path) -> str:
    return os.path.relpath(component_path, stack_path.parent)


def decision_from_pipeline(dataset_manifest: Mapping[str, Any], stack_validation: Mapping[str, Any], replay_summary: Mapping[str, Any]) -> str:
    if not dataset_manifest.get("contamination", {}).get("passed", False):
        return "NO_GO_dataset_contamination_failed"
    if not stack_validation.get("passed", False):
        return "NO_GO_stack_validation_failed"
    if replay_summary.get("decision") != "GO_reference_replay_supported":
        return "NO_GO_replay_failed"
    return "GO_supervised_memory_pipeline_supported"


def write_pipeline_summary(summary: Mapping[str, Any], root: str | Path) -> None:
    target = Path(root) / "pipeline_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    package_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the supervised memory reference pipeline.")
    parser.add_argument("--config", default=str(package_root / "configs" / "pipelines" / "fixture_supervised_memory.yaml"))
    parser.add_argument("--out", default="")
    parser.add_argument("--runtime", default="")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    pipeline_cfg = config.get("pipeline", {}) if isinstance(config.get("pipeline", {}), Mapping) else {}
    out = args.out or str(pipeline_cfg.get("output", package_root / "runs" / "supervised_memory_pipeline"))
    runtime_name = args.runtime or str(pipeline_cfg.get("runtime", "fixture_memory"))
    steps = args.steps if args.steps >= 0 else int(pipeline_cfg.get("steps", 4))
    seed = args.seed if args.seed >= 0 else int(pipeline_cfg.get("seed", 0))
    arms = tuple(str(arm) for arm in pipeline_cfg.get("replay_arms", DEFAULT_ARMS))
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = package_root / out_path
    summary = run_supervised_memory_pipeline(out=out_path, runtime_name=runtime_name, steps=steps, seed=seed, replay_arms=arms)
    print(json.dumps({"out": str(out_path), "decision": summary["decision"], "feature_rows": summary["feature_rows"]}, sort_keys=True))
    return 0
