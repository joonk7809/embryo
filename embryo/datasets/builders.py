"""Runtime-agnostic supervised dataset builders."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import read_jsonl, write_jsonl
from embryo.models import RouteBCPolicy, RuleRouterModel


@dataclass(frozen=True)
class DatasetBuildResult:
    fact_writer_rows: list[dict[str, Any]]
    router_rows: list[dict[str, Any]]
    bc_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def build_supervised_datasets(trace_rows: Sequence[Mapping[str, Any]], *, source_name: str = "trace") -> DatasetBuildResult:
    """Build module-specific supervised datasets from runtime trace rows."""
    fact_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    bc_rows: list[dict[str, Any]] = []
    scans: list[dict[str, Any]] = []
    router = RuleRouterModel()
    policy = RouteBCPolicy()
    for index, row in enumerate(trace_rows):
        obs = observation_from_row(row)
        actor_features = actor_features_from_observation(obs)
        scan = scan_actor_context(actor_features)
        scans.append(scan)

        target_facing = target_facing_candidate(row, obs)
        fact_rows.append({"candidate_score": float(obs.get("candidate_score", 0.0)), "target_facing_candidate": target_facing})

        router_features = {
            "facing_candidate": target_facing,
            "failed_action_event": bool(obs.get("failed_action_event", row.get("failed_action_event", False))),
            "cache_age": int(obs.get("cache_age", row.get("cache_age", 0))),
            "invalidated": bool(obs.get("invalidated", row.get("invalidated", False))),
            "repeat_count": int(obs.get("repeat_count", row.get("repeat_count", 0))),
        }
        target_route = str(row.get("target_route", router.predict(router_features).route_mode))
        router_rows.append({**router_features, "target_route": target_route})

        target_action = str(row.get("target_action", row.get("action", policy.act({"route_mode": target_route}).action)))
        bc_rows.append({"route_mode": target_route, "target_action": target_action})

    failures = [scan for scan in scans if not scan["passed"]]
    manifest = {
        "source": source_name,
        "builder": "supervised_module_dataset_builder",
        "row_count": len(trace_rows),
        "datasets": {
            "fact_writer": {"rows": len(fact_rows), "schema": ["candidate_score", "target_facing_candidate"]},
            "router": {
                "rows": len(router_rows),
                "schema": ["facing_candidate", "failed_action_event", "cache_age", "invalidated", "repeat_count", "target_route"],
            },
            "bc_policy": {"rows": len(bc_rows), "schema": ["route_mode", "target_action"]},
        },
        "label_policy": {
            "fact_writer": "existing_target_or_candidate_score_threshold",
            "router": "existing_target_or_reference_router_contract",
            "bc_policy": "existing_target_or_collected_action_or_reference_route_policy",
        },
        "contamination": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
    }
    return DatasetBuildResult(fact_writer_rows=fact_rows, router_rows=router_rows, bc_rows=bc_rows, manifest=manifest)


def observation_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = row.get("observation")
    if isinstance(observation, Mapping):
        return observation
    return row


def actor_features_from_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    allowed_like = (
        "raw_rgb_frame",
        "previous_rgb_frame",
        "previous_action",
        "previous_action_result_event",
        "facing_candidate_v1",
        "query_content",
        "query_cache_age",
        "short_ttl_stale_state",
    )
    return {key: observation[key] for key in allowed_like if key in observation}


def target_facing_candidate(row: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    for payload in (row, observation):
        if "target_facing_candidate" in payload:
            return bool(payload["target_facing_candidate"])
        if "resource_memory_critical" in payload:
            return bool(payload["resource_memory_critical"])
    return float(observation.get("candidate_score", row.get("candidate_score", 0.0))) >= 0.5


def write_dataset_bundle(result: DatasetBuildResult, out: str | Path) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "fact_writer_dataset.jsonl", result.fact_writer_rows)
    write_jsonl(root / "router_dataset.jsonl", result.router_rows)
    write_jsonl(root / "bc_dataset.jsonl", result.bc_rows)
    (root / "dataset_manifest.json").write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build supervised module datasets from a runtime trace.")
    parser.add_argument("--config", default=str(root / "configs" / "datasets" / "fixture_supervised.yaml"))
    parser.add_argument("--input", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else {}
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset", {}), Mapping) else {}
    input_value = args.input or str(dataset_cfg.get("input", root / "data" / "fixtures" / "tiny_replay_features.jsonl"))
    out_value = args.out or str(dataset_cfg.get("output", root / "runs" / "build_supervised_datasets"))
    input_path = resolve_path(root, input_value)
    out_path = resolve_path(root, out_value)
    result = build_supervised_datasets(read_jsonl(input_path), source_name=str(input_path))
    write_dataset_bundle(result, out_path)
    print(json.dumps({"out": str(out_path), "row_count": result.manifest["row_count"], "contamination": result.manifest["contamination"]}, sort_keys=True))
    return 0


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
