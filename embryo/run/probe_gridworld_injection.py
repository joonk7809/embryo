"""Probe memory-to-goal injection in a gridworld kitchen abstraction."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.runtimes.gridworld_kitchen import (
    GoalConditionedNavigator,
    KitchenEpisode,
    deployable_need_observation,
    deployable_reveal_observation,
    make_kitchen_episode,
    move,
)


GO_GRIDWORLD_INJECTION_SUPPORTED = "GO_gridworld_injection_supported"
NO_GO_GOAL_NAVIGATOR_FAILURE = "NO_GO_goal_navigator_failure"
NO_GO_MEMORY_INJECTION_FAILURE = "NO_GO_memory_injection_failure"
NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING = "NO_GO_corrupt_controls_not_separating"
NO_GO_NO_TASK_HEADROOM = "NO_GO_no_task_headroom"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"

ARMS = (
    "no_memory_search",
    "memory_clean",
    "memory_shuffled_location",
    "memory_wrong_binding",
    "memory_stale",
    "oracle_goal_eval_only",
)


def run_gridworld_injection_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_cfg = mapping(config.get("runtime"))
    protocol_cfg = mapping(config.get("protocol"))
    seed_start = int(runtime_cfg.get("seed_start", 12000))
    seed_count = int(runtime_cfg.get("seed_count", 64))
    episodes: list[dict[str, Any]] = []
    contamination_failures: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + seed_count):
        episode = make_kitchen_episode(
            seed=seed,
            width=int(runtime_cfg.get("width", 7)),
            height=int(runtime_cfg.get("height", 7)),
            drawer_count=int(runtime_cfg.get("drawer_count", 12)),
            object_count=int(runtime_cfg.get("object_count", 4)),
        )
        contamination_failures.extend(contamination_for_episode(episode))
        for arm in ARMS:
            episodes.append(run_arm_episode(episode, arm=arm, max_steps=int(protocol_cfg.get("max_steps", 128))))
    metrics_by_arm = summarize_by_arm(episodes)
    pairwise = pairwise_metrics(metrics_by_arm)
    contamination = {"passed": not contamination_failures, "failure_count": len(contamination_failures), "failures": contamination_failures}
    decision, reasons = decide_gridworld_injection(metrics_by_arm, contamination=contamination, protocol_cfg=protocol_cfg)
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "gridworld_kitchen_goal_injection",
        "boundary": "within_episode_goal_injection_no_rl_no_cross_episode_memory",
        "runtime": {
            "name": "gridworld_kitchen",
            "width": int(runtime_cfg.get("width", 7)),
            "height": int(runtime_cfg.get("height", 7)),
            "drawer_count": int(runtime_cfg.get("drawer_count", 12)),
            "object_count": int(runtime_cfg.get("object_count", 4)),
            "seed_start": seed_start,
            "seed_count": seed_count,
        },
        "arms": list(ARMS),
        "metrics_by_arm": metrics_by_arm,
        "pairwise": pairwise,
        "contamination": contamination,
    }
    return {
        "summary": summary,
        "metrics_by_arm": metrics_by_arm,
        "episodes": episodes,
        "config_manifest": json_safe(config),
        "contamination": contamination,
    }


def run_arm_episode(episode: KitchenEpisode, *, arm: str, max_steps: int) -> dict[str, Any]:
    navigator = GoalConditionedNavigator()
    position = episode.start_position
    previous_action = "none"
    search_order = deterministic_search_order(episode)
    opened_drawers: list[int] = []
    injected_goal_drawer = None if arm == "no_memory_search" else injected_drawer_for_arm(episode, arm=arm)
    steps = 0
    success = False
    wrong_drawer = False
    path_efficiency = 0.0
    while steps < int(max_steps):
        if arm == "no_memory_search":
            unopened = [drawer for drawer in search_order if drawer not in opened_drawers]
            if not unopened:
                break
            goal_drawer = int(unopened[0])
        else:
            goal_drawer = int(injected_goal_drawer)
        goal_position = episode.drawer_position(goal_drawer)
        action = navigator.action(position, goal_position)
        steps += 1
        if action == "open":
            opened_drawers.append(goal_drawer)
            if goal_drawer == episode.target_drawer_id:
                success = True
                shortest = manhattan(episode.start_position, episode.drawer_position(episode.target_drawer_id)) + 1
                path_efficiency = round(shortest / steps, 6)
                break
            wrong_drawer = True
            if arm != "no_memory_search":
                break
        else:
            position = move(position, action, width=episode.width, height=episode.height)
        previous_action = action
    goal_accuracy = None if arm == "no_memory_search" else bool(int(injected_goal_drawer) == episode.target_drawer_id)
    return {
        "seed": episode.seed,
        "arm": arm,
        "target_object": episode.target_object,
        "target_drawer_id": episode.target_drawer_id,
        "injected_goal_drawer_id": None if arm == "no_memory_search" else int(injected_goal_drawer),
        "goal_injection_accuracy": goal_accuracy,
        "retrieval_success": success,
        "steps_to_retrieve": steps if success else None,
        "episode_steps": steps,
        "wrong_drawer_opened": wrong_drawer,
        "opened_drawers": opened_drawers,
        "path_efficiency": path_efficiency,
        "query_success": arm != "no_memory_search",
        "memory_hit": arm in {"memory_clean", "memory_shuffled_location", "memory_wrong_binding", "memory_stale"},
        "eval_only": {
            "start_position": list(episode.start_position),
            "target_position": list(episode.drawer_position(episode.target_drawer_id)),
        },
    }


def injected_drawer_for_arm(episode: KitchenEpisode, *, arm: str) -> int:
    if arm in {"memory_clean", "oracle_goal_eval_only"}:
        return episode.target_drawer_id
    if arm == "memory_stale":
        return episode.stale_drawer_id
    if arm == "memory_wrong_binding":
        object_ids = sorted(episode.object_drawers)
        other_objects = [obj for obj in object_ids if obj != episode.target_object]
        return int(episode.object_drawers[other_objects[0]])
    if arm == "memory_shuffled_location":
        rng = random.Random(episode.seed + 1729)
        candidates = [drawer.drawer_id for drawer in episode.drawers if drawer.drawer_id != episode.target_drawer_id]
        return int(rng.choice(candidates))
    raise ValueError(f"Arm does not inject a goal drawer: {arm}")


def deterministic_search_order(episode: KitchenEpisode) -> list[int]:
    return sorted((drawer.drawer_id for drawer in episode.drawers), key=lambda drawer_id: (manhattan(episode.start_position, episode.drawer_position(drawer_id)), drawer_id))


def summarize_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if str(row["arm"]) == arm]
        success_rows = [row for row in arm_rows if bool(row["retrieval_success"])]
        goal_rows = [row for row in arm_rows if row["goal_injection_accuracy"] is not None]
        result[arm] = {
            "episode_count": len(arm_rows),
            "retrieval_success_rate": round(rate(row["retrieval_success"] for row in arm_rows), 6),
            "mean_steps_to_retrieve": round(float(np.mean([int(row["steps_to_retrieve"]) for row in success_rows])), 6) if success_rows else None,
            "mean_episode_steps": round(float(np.mean([int(row["episode_steps"]) for row in arm_rows])), 6) if arm_rows else 0.0,
            "wrong_drawer_rate": round(rate(row["wrong_drawer_opened"] for row in arm_rows), 6),
            "goal_injection_accuracy": round(rate(row["goal_injection_accuracy"] for row in goal_rows), 6) if goal_rows else None,
            "query_success_rate": round(rate(row["query_success"] for row in arm_rows), 6),
            "memory_hit_rate": round(rate(row["memory_hit"] for row in arm_rows), 6),
            "mean_path_efficiency": round(float(np.mean([float(row["path_efficiency"]) for row in success_rows])), 6) if success_rows else 0.0,
        }
    return result


def pairwise_metrics(metrics_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    clean = mapping(metrics_by_arm.get("memory_clean"))
    no_memory = mapping(metrics_by_arm.get("no_memory_search"))
    corrupt_arms = ("memory_shuffled_location", "memory_wrong_binding", "memory_stale")
    clean_steps = clean.get("mean_steps_to_retrieve")
    no_memory_steps = no_memory.get("mean_steps_to_retrieve")
    clean_success = float(clean.get("retrieval_success_rate", 0.0))
    return {
        "clean_minus_no_memory_steps": None if clean_steps is None or no_memory_steps is None else round(float(no_memory_steps) - float(clean_steps), 6),
        "clean_minus_no_memory_success": round(clean_success - float(no_memory.get("retrieval_success_rate", 0.0)), 6),
        "clean_minus_corrupt_success": {
            arm: round(clean_success - float(mapping(metrics_by_arm.get(arm)).get("retrieval_success_rate", 0.0)), 6)
            for arm in corrupt_arms
        },
    }


def decide_gridworld_injection(
    metrics_by_arm: Mapping[str, Any],
    *,
    contamination: Mapping[str, Any],
    protocol_cfg: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count_nonzero"]
    oracle = mapping(metrics_by_arm.get("oracle_goal_eval_only"))
    clean = mapping(metrics_by_arm.get("memory_clean"))
    no_memory = mapping(metrics_by_arm.get("no_memory_search"))
    corrupt = [mapping(metrics_by_arm.get(arm)) for arm in ("memory_shuffled_location", "memory_wrong_binding", "memory_stale")]
    if float(oracle.get("retrieval_success_rate", 0.0)) < float(protocol_cfg.get("min_oracle_success_rate", 1.0)):
        return NO_GO_GOAL_NAVIGATOR_FAILURE, ["oracle_goal_success_below_gate"]
    if float(clean.get("retrieval_success_rate", 0.0)) < float(protocol_cfg.get("min_clean_success_rate", 0.90)):
        return NO_GO_MEMORY_INJECTION_FAILURE, ["clean_success_below_gate"]
    if float(clean.get("goal_injection_accuracy", 0.0)) < float(protocol_cfg.get("min_goal_injection_accuracy", 0.90)):
        return NO_GO_MEMORY_INJECTION_FAILURE, ["goal_injection_accuracy_below_gate"]
    step_gain = float(no_memory.get("mean_steps_to_retrieve", 0.0)) - float(clean.get("mean_steps_to_retrieve", 0.0))
    if step_gain < float(protocol_cfg.get("min_clean_step_gain", 4.0)):
        return NO_GO_NO_TASK_HEADROOM, ["clean_step_gain_below_gate"]
    max_corrupt_success = float(protocol_cfg.get("max_corrupt_success_rate", 0.20))
    min_wrong_drawer = float(protocol_cfg.get("min_corrupt_wrong_drawer_rate", 0.80))
    if any(float(row.get("retrieval_success_rate", 0.0)) > max_corrupt_success for row in corrupt):
        return NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING, ["corrupt_success_above_gate"]
    if any(float(row.get("wrong_drawer_rate", 0.0)) < min_wrong_drawer for row in corrupt):
        return NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING, ["corrupt_wrong_drawer_below_gate"]
    return GO_GRIDWORLD_INJECTION_SUPPORTED, ["clean_goal_injection_beats_search_and_corrupt_controls"]


def contamination_for_episode(episode: KitchenEpisode) -> list[dict[str, Any]]:
    contexts = [
        deployable_reveal_observation(episode),
        deployable_need_observation(episode, episode.start_position, previous_action="none"),
    ]
    failures: list[dict[str, Any]] = []
    for idx, context in enumerate(contexts):
        scan = scan_actor_context(context)
        for failure in scan.get("failures", []):
            failures.append({"context_index": idx, **dict(failure)})
    return failures


def manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def rate(values: Sequence[Any]) -> float:
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def write_gridworld_injection_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "gridworld_injection_summary_json": root / "gridworld_injection_summary.json",
        "gridworld_injection_summary_md": root / "gridworld_injection_summary.md",
        "metrics_by_arm": root / "metrics_by_arm.json",
        "episodes": root / "gridworld_injection_episodes.jsonl",
        "config_manifest": root / "config_manifest.json",
        "contamination_scan": root / "contamination_scan.json",
    }
    paths["gridworld_injection_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["gridworld_injection_summary_md"].write_text(format_summary_markdown(result["summary"]), encoding="utf-8")
    paths["metrics_by_arm"].write_text(json.dumps(result["metrics_by_arm"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["config_manifest"].write_text(json.dumps(result["config_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(paths["episodes"], result["episodes"])
    return {key: str(value) for key, value in paths.items()}


def format_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gridworld Kitchen Injection Probe",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Boundary: `{summary.get('boundary')}`",
        f"- Contamination failures: `{mapping(summary.get('contamination')).get('failure_count')}`",
        "",
        "## Arms",
        "",
        "| Arm | Success | Mean steps | Wrong drawer | Goal accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, metrics in mapping(summary.get("metrics_by_arm")).items():
        row = mapping(metrics)
        lines.append(
            f"| {arm} | {row.get('retrieval_success_rate')} | {row.get('mean_steps_to_retrieve')} | "
            f"{row.get('wrong_drawer_rate')} | {row.get('goal_injection_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise",
            "",
            "```json",
            json.dumps(summary.get("pairwise", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gridworld_kitchen_injection.yaml")
    parser.add_argument("--out", default="runs/gridworld_kitchen_injection")
    args = parser.parse_args(argv)
    result = run_gridworld_injection_probe(load_config(args.config))
    artifacts = write_gridworld_injection_artifacts(result, args.out)
    print(json.dumps({"decision": result["summary"]["decision"], "artifacts": artifacts, "out": str(Path(args.out).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
