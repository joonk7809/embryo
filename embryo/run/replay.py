"""Reference stack replay.

This module composes the clean package interfaces into a small reproducible
replay path:

visual features -> fact writer -> query/freshness router -> BC policy -> trace

It is a diagnostic replay helper, not a task benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.guardrails import event_repetition_guardrail, resource_binding_guardrail
from embryo.eval.memory_grounded_score import ScoreConfig, compute_memory_grounded_scores, compute_window_sensitivity
from embryo.eval.traces import read_jsonl, write_jsonl
from embryo.memory.freshness import FreshnessState
from embryo.memory.queries import build_event_query, build_resource_query, combine_queries
from embryo.models import (
    BCPolicy,
    RouteBCPolicy,
    RuleRouterModel,
    RouterModel,
    ThresholdFactWriter,
    VisualFactWriter,
    build_stack_from_manifest,
    load_stack_manifest,
    validate_stack_manifest,
)


DEFAULT_ARMS = ("clean", "no_memory", "shuffled", "stale", "wrong_binding")


@dataclass(frozen=True)
class StackModules:
    fact_writer: VisualFactWriter
    router: RouterModel
    policy: BCPolicy


def default_stack() -> StackModules:
    return StackModules(fact_writer=ThresholdFactWriter(), router=RuleRouterModel(), policy=RouteBCPolicy())


def run_reference_replay(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    arms: Iterable[str] = DEFAULT_ARMS,
    stack: StackModules | None = None,
    stack_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the reference stack over deployable feature rows."""
    modules = stack or default_stack()
    ticks: list[dict[str, Any]] = []
    contamination_scans: list[dict[str, Any]] = []
    for arm in arms:
        for source in feature_rows:
            row = replay_tick(source, arm=str(arm), stack=modules)
            ticks.append(row)
            contamination_scans.append(row["actor_contamination"])

    score = compute_memory_grounded_scores(ticks, config=ScoreConfig(followthrough_window=4))
    windows = compute_window_sensitivity(ticks, windows=(4, 8, 16), config=ScoreConfig())
    resource_guardrail = resource_binding_guardrail(ticks)
    event_guardrail = event_repetition_guardrail(ticks)
    contamination_failures = [scan for scan in contamination_scans if not scan["passed"]]
    summary = {
        "decision": decision_from_replay(score, resource_guardrail, event_guardrail, contamination_failures),
        "tick_count": len(ticks),
        "arms": list(arms),
        "stack_manifest": stack_manifest,
        "score": score,
        "window_sensitivity": windows,
        "resource_binding_guardrail": resource_guardrail,
        "event_repetition_guardrail": event_guardrail,
        "contamination": {
            "passed": not contamination_failures,
            "failure_count": len(contamination_failures),
            "failures": contamination_failures,
        },
    }
    return {"summary": summary, "ticks": ticks}


def replay_tick(source: Mapping[str, Any], *, arm: str, stack: StackModules) -> dict[str, Any]:
    features = corrupt_features(source, arm)
    fact = stack.fact_writer.fact(features)
    freshness = FreshnessState(
        cache_age=int(features.get("cache_age", 0)),
        invalidated=bool(features.get("invalidated", False)),
        visual_change_conflict=bool(features.get("visual_change_conflict", False)),
    )
    resource_query = build_resource_query([fact], cache_age=freshness.cache_age, fresh=not freshness.invalidated)
    event_query = build_event_query([fact], cache_age=freshness.cache_age)
    combined_query = combine_queries(resource_query, event_query)
    router_output = stack.router.predict(
        {
            "facing_candidate": bool(fact.value),
            "failed_action_event": bool(features.get("failed_action_event", False)),
            "cache_age": freshness.cache_age,
            "invalidated": freshness.invalidated,
            "visual_change_conflict": freshness.visual_change_conflict,
            "cooldown": int(features.get("cooldown", 0)),
            "repeat_count": int(features.get("repeat_count", 0)),
            "last_route": str(features.get("last_route", "hold")),
        }
    )
    action = stack.policy.act({"router_output": router_output})
    actor_context = actor_context_from_source(source, fact.value, combined_query.content_hash, freshness.cache_age)
    contamination = scan_actor_context(actor_context)
    resource_critical = bool(source.get("resource_memory_critical", source.get("candidate_score", 0.0)))
    progress = progress_for_action(source, action.action, resource_critical=resource_critical)
    return {
        "arm": arm,
        "seed": int(source.get("seed", 0)),
        "episode_id": str(source.get("episode_id", "fixture")),
        "episode_index": int(source.get("episode_index", 0)),
        "tick": int(source.get("tick", 0)),
        "action": action.action,
        "selected_action": {"action_name": action.action, "invalid_or_unknown": action.invalid_or_unknown},
        "route_mode": router_output.route_mode,
        "query_content_hash": combined_query.content_hash,
        "resource_memory_critical": resource_critical,
        "resource_route_preserved": router_output.resource_route_preserved,
        "fallback_triggered": router_output.event_fallback_gate,
        "event_self_triggered": False,
        "repeated_action_loop": bool(features.get("repeat_count", 0) >= 2),
        "diagnostic_progress_delta_teacher_only": progress,
        "actor_contamination": contamination,
    }


def corrupt_features(source: Mapping[str, Any], arm: str) -> dict[str, Any]:
    features = dict(source)
    score = float(features.get("candidate_score", 0.0))
    if arm == "no_memory":
        features.update(candidate_score=0.0, failed_action_event=False)
    elif arm == "shuffled":
        features["candidate_score"] = 1.0 - score
    elif arm == "stale":
        features.update(cache_age=max(99, int(features.get("cache_age", 0))), invalidated=True)
    elif arm == "wrong_binding":
        features["candidate_score"] = 0.0 if score >= 0.5 else 1.0
    elif arm != "clean":
        raise ValueError(f"Unknown replay arm: {arm}")
    return features


def actor_context_from_source(source: Mapping[str, Any], facing_candidate: bool, query_hash: str, cache_age: int) -> dict[str, Any]:
    return {
        "raw_rgb_frame": source.get("raw_rgb_frame", "fixture_rgb"),
        "previous_rgb_frame": source.get("previous_rgb_frame", "fixture_previous_rgb"),
        "previous_action": source.get("previous_action", "noop"),
        "facing_candidate_v1": bool(facing_candidate),
        "query_content": query_hash,
        "query_cache_age": int(cache_age),
    }


def progress_for_action(source: Mapping[str, Any], action: str, *, resource_critical: bool) -> float:
    if not resource_critical:
        return 0.0
    if action != str(source.get("target_action", "move_forward")):
        return 0.0
    return float(source.get("diagnostic_progress_delta_teacher_only", 1.0))


def decision_from_replay(
    score: Mapping[str, Any],
    resource_guardrail: Mapping[str, Any],
    event_guardrail: Mapping[str, Any],
    contamination_failures: Sequence[Mapping[str, Any]],
) -> str:
    gaps = score.get("memory_grounded_gaps_clean_minus_controls", {})
    separates = all(value is not None and float(value) > 0.0 for value in gaps.values())
    if contamination_failures:
        return "NO_GO_contamination_failed"
    if not resource_guardrail.get("passed", False):
        return "NO_GO_resource_guardrail_failed"
    if not event_guardrail.get("passed", False):
        return "NO_GO_event_guardrail_failed"
    if not separates:
        return "NO_GO_clean_memory_not_separated"
    return "GO_reference_replay_supported"


def write_replay_artifacts(result: Mapping[str, Any], out: str | Path) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "replay_summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(root / "replay_ticks.jsonl", result["ticks"])
    (root / "contamination_scan.json").write_text(
        json.dumps(result["summary"]["contamination"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result["summary"].get("stack_manifest"):
        (root / "stack_manifest.json").write_text(
            json.dumps(result["summary"]["stack_manifest"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def arms_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    replay = config.get("replay", {})
    if isinstance(replay, Mapping) and replay.get("arms"):
        return tuple(str(arm) for arm in replay["arms"])
    return DEFAULT_ARMS


def replay_config_value(config: Mapping[str, Any], key: str) -> str:
    replay = config.get("replay", {})
    if isinstance(replay, Mapping) and replay.get(key):
        return str(replay[key])
    return ""


def resolve_path(root: Path, cli_value: str, config_value: str, fallback: Path) -> Path:
    selected = cli_value or config_value
    if not selected:
        return fallback
    path = Path(selected)
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a tiny reference stack replay.")
    parser.add_argument("--config", default=str(root / "configs" / "learned_stack_replay.yaml"))
    parser.add_argument("--input", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--stack-manifest", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    input_path = resolve_path(root, args.input, replay_config_value(config, "input"), root / "data" / "fixtures" / "tiny_replay_features.jsonl")
    out_path = resolve_path(root, args.out, replay_config_value(config, "output"), root / "runs" / "reference_stack_replay")
    stack_manifest_value = args.stack_manifest or replay_config_value(config, "stack_manifest")
    rows = read_jsonl(input_path)
    stack = None
    stack_manifest = None
    if stack_manifest_value:
        stack_manifest_path = resolve_path(root, stack_manifest_value, "", Path(""))
        stack = build_stack_from_manifest(stack_manifest_path)
        loaded = load_stack_manifest(stack_manifest_path)
        validation = validate_stack_manifest(loaded, base_dir=stack_manifest_path.parent)
        stack_manifest = {"path": str(stack_manifest_path), "loaded": loaded.to_dict(), "validation": validation}
    result = run_reference_replay(rows, arms=arms_from_config(config), stack=stack, stack_manifest=stack_manifest)
    write_replay_artifacts(result, out_path)
    print(json.dumps({"out": str(out_path), "decision": result["summary"]["decision"], "tick_count": result["summary"]["tick_count"]}, sort_keys=True))
    return 0
