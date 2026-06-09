"""Probe memory-to-goal injection in a gridworld kitchen abstraction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.memory.gridworld_kitchen import GridworldLocationEntry, GridworldObjectLocationMemory
from embryo.runtimes.gridworld_kitchen import (
    GoalConditionedNavigator,
    KitchenEpisode,
    deployable_need_observation,
    deployable_reveal_observations,
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
    drawer_counts = int_list(runtime_cfg.get("drawer_counts"), default=int(runtime_cfg.get("drawer_count", 12)))
    episodes: list[dict[str, Any]] = []
    contamination_failures: list[dict[str, Any]] = []
    for drawer_count in drawer_counts:
        for seed in range(seed_start, seed_start + seed_count):
            episode = make_kitchen_episode(
                seed=seed,
                width=int(runtime_cfg.get("width", 7)),
                height=int(runtime_cfg.get("height", 7)),
                drawer_count=drawer_count,
                object_count=int(runtime_cfg.get("object_count", 4)),
            )
            contamination_failures.extend(contamination_for_episode(episode))
            for arm in ARMS:
                episodes.append(
                    run_arm_episode(
                        episode,
                        arm=arm,
                        max_steps=int(protocol_cfg.get("max_steps", 512)),
                        fallback_reset_to_start=bool(protocol_cfg.get("fallback_reset_to_start", True)),
                    )
                )
    metrics_by_arm = summarize_by_arm(episodes)
    curve_by_drawer_count = summarize_curve_by_drawer_count(episodes)
    pairwise = pairwise_metrics(metrics_by_arm, curve_by_drawer_count=curve_by_drawer_count)
    contamination = {"passed": not contamination_failures, "failure_count": len(contamination_failures), "failures": contamination_failures}
    decision, reasons = decide_gridworld_injection(
        metrics_by_arm,
        curve_by_drawer_count=curve_by_drawer_count,
        contamination=contamination,
        protocol_cfg=protocol_cfg,
    )
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "gridworld_kitchen_goal_injection",
        "boundary": "within_episode_reveal_write_query_goal_injection_no_rl_no_cross_episode_memory",
        "runtime": {
            "name": "gridworld_kitchen",
            "width": int(runtime_cfg.get("width", 7)),
            "height": int(runtime_cfg.get("height", 7)),
            "drawer_counts": drawer_counts,
            "object_count": int(runtime_cfg.get("object_count", 4)),
            "seed_start": seed_start,
            "seed_count": seed_count,
        },
        "protocol": {
            "fallback_after_failed_memory": "systematic_sweep",
            "fallback_reset_to_start": bool(protocol_cfg.get("fallback_reset_to_start", True)),
            "primary_metric": "drawers_opened_to_success",
            "secondary_metric": "steps_to_retrieve",
        },
        "arms": list(ARMS),
        "metrics_by_arm": metrics_by_arm,
        "curve_by_drawer_count": curve_by_drawer_count,
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


def run_arm_episode(episode: KitchenEpisode, *, arm: str, max_steps: int, fallback_reset_to_start: bool = True) -> dict[str, Any]:
    navigator = GoalConditionedNavigator()
    position = episode.start_position
    previous_action = "none"
    search_order = deterministic_search_order(episode)
    opened_drawers: list[int] = []
    reveal_observations = deployable_reveal_observations(episode)
    memory = build_gridworld_memory(reveal_observations)
    need_observation = deployable_need_observation(episode, episode.start_position, previous_action=previous_action)
    memory_entry = memory_entry_for_arm(memory, episode=episode, arm=arm)
    injected_goal_drawer = None if memory_entry is None else int(memory_entry.drawer_id)
    steps = 0
    success = False
    wrong_drawer = False
    failed_memory_goal = False
    search_mode = arm == "no_memory_search" or memory_entry is None
    memory_goal_committed = False
    path_efficiency = 0.0
    while steps < int(max_steps):
        if search_mode:
            unopened = [drawer for drawer in search_order if drawer not in opened_drawers]
            if not unopened:
                break
            goal_drawer = int(unopened[0])
            using_memory_goal = False
        else:
            goal_drawer = int(injected_goal_drawer)
            using_memory_goal = True
            memory_goal_committed = True
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
            if using_memory_goal:
                failed_memory_goal = True
                search_mode = True
                if fallback_reset_to_start:
                    position = episode.start_position
        else:
            position = move(position, action, width=episode.width, height=episode.height)
        previous_action = action
    goal_accuracy = None if arm == "no_memory_search" or injected_goal_drawer is None else bool(int(injected_goal_drawer) == episode.target_drawer_id)
    drawers_to_success = len(opened_drawers) if success else None
    wrong_drawers_before_success = max(0, len(opened_drawers) - 1) if success else len(opened_drawers)
    return {
        "seed": episode.seed,
        "drawer_count": len(episode.drawers),
        "arm": arm,
        "target_object": episode.target_object,
        "target_drawer_id": episode.target_drawer_id,
        "injected_goal_drawer_id": None if injected_goal_drawer is None else int(injected_goal_drawer),
        "memory_entry_content_hash": "" if memory_entry is None else memory_entry.content_hash,
        "memory_entry_object": "" if memory_entry is None else memory_entry.object_id,
        "memory_entry_sequence_index": None if memory_entry is None else memory_entry.sequence_index,
        "goal_injection_accuracy": goal_accuracy,
        "retrieval_success": success,
        "steps_to_retrieve": steps if success else None,
        "episode_steps": steps,
        "drawers_opened_to_success": drawers_to_success,
        "wrong_drawers_before_success": wrong_drawers_before_success,
        "wrong_drawer_opened": wrong_drawer,
        "failed_memory_goal": failed_memory_goal,
        "memory_goal_committed": memory_goal_committed,
        "opened_drawers": opened_drawers,
        "path_efficiency": path_efficiency,
        "query_success": arm != "no_memory_search" and memory_entry is not None,
        "memory_hit": arm in {"memory_clean", "memory_shuffled_location", "memory_wrong_binding", "memory_stale"},
        "reveal_count": len(reveal_observations),
        "eval_only": {
            "start_position": list(episode.start_position),
            "target_position": list(episode.drawer_position(episode.target_drawer_id)),
            "need_observation_eval_only": need_observation,
        },
    }


def build_gridworld_memory(reveal_observations: Sequence[Mapping[str, Any]]) -> GridworldObjectLocationMemory:
    memory = GridworldObjectLocationMemory()
    for observation in reveal_observations:
        memory.update(observation)
    return memory


def memory_entry_for_arm(memory: GridworldObjectLocationMemory, *, episode: KitchenEpisode, arm: str) -> GridworldLocationEntry | None:
    if arm == "no_memory_search":
        return None
    if arm == "oracle_goal_eval_only":
        return GridworldLocationEntry(
            object_id=episode.target_object,
            drawer_id=episode.target_drawer_id,
            position=episode.drawer_position(episode.target_drawer_id),
            sequence_index=-1,
            content_hash="oracle_goal_eval_only",
        )
    if arm == "memory_clean":
        return memory.latest(episode.target_object)
    if arm == "memory_stale":
        return memory.oldest(episode.target_object)
    if arm == "memory_wrong_binding":
        return memory.latest_other(episode.target_object)
    if arm == "memory_shuffled_location":
        return memory.shuffled_other(episode.target_object, seed=episode.seed)
    raise ValueError(f"Unknown gridworld arm: {arm}")


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
            "mean_drawers_opened_to_success": round(float(np.mean([int(row["drawers_opened_to_success"]) for row in success_rows])), 6) if success_rows else None,
            "mean_wrong_drawers_before_success": round(float(np.mean([int(row["wrong_drawers_before_success"]) for row in success_rows])), 6) if success_rows else None,
            "wrong_drawer_rate": round(rate(row["wrong_drawer_opened"] for row in arm_rows), 6),
            "failed_memory_goal_rate": round(rate(row["failed_memory_goal"] for row in arm_rows), 6),
            "memory_goal_commit_rate": round(rate(row["memory_goal_committed"] for row in arm_rows), 6),
            "goal_injection_accuracy": round(rate(row["goal_injection_accuracy"] for row in goal_rows), 6) if goal_rows else None,
            "query_success_rate": round(rate(row["query_success"] for row in arm_rows), 6),
            "memory_hit_rate": round(rate(row["memory_hit"] for row in arm_rows), 6),
            "mean_path_efficiency": round(float(np.mean([float(row["path_efficiency"]) for row in success_rows])), 6) if success_rows else 0.0,
        }
    return result


def summarize_curve_by_drawer_count(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for drawer_count in sorted({int(row["drawer_count"]) for row in rows}):
        count_rows = [row for row in rows if int(row["drawer_count"]) == drawer_count]
        metrics = summarize_by_arm(count_rows)
        result[str(drawer_count)] = {
            "metrics_by_arm": metrics,
            "pairwise": pairwise_for_metrics(metrics),
        }
    return result


def pairwise_metrics(metrics_by_arm: Mapping[str, Any], *, curve_by_drawer_count: Mapping[str, Any]) -> dict[str, Any]:
    pairwise = pairwise_for_metrics(metrics_by_arm)
    drawer_counts = sorted(int(key) for key in curve_by_drawer_count)
    if drawer_counts:
        first = mapping(mapping(curve_by_drawer_count.get(str(drawer_counts[0]))).get("pairwise"))
        last = mapping(mapping(curve_by_drawer_count.get(str(drawer_counts[-1]))).get("pairwise"))
        pairwise["drawer_count_sweep"] = {
            "drawer_counts": drawer_counts,
            "first_clean_minus_no_memory_drawers": first.get("clean_minus_no_memory_drawers_opened"),
            "last_clean_minus_no_memory_drawers": last.get("clean_minus_no_memory_drawers_opened"),
            "gap_growth_drawers": round(
                float(last.get("clean_minus_no_memory_drawers_opened", 0.0)) - float(first.get("clean_minus_no_memory_drawers_opened", 0.0)),
                6,
            ),
            "last_wrong_binding_minus_no_memory_steps": last.get("wrong_binding_minus_no_memory_steps"),
        }
    return pairwise


def pairwise_for_metrics(metrics_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    clean = mapping(metrics_by_arm.get("memory_clean"))
    no_memory = mapping(metrics_by_arm.get("no_memory_search"))
    corrupt_arms = ("memory_shuffled_location", "memory_wrong_binding", "memory_stale")
    clean_steps = clean.get("mean_steps_to_retrieve")
    no_memory_steps = no_memory.get("mean_steps_to_retrieve")
    clean_drawers = clean.get("mean_drawers_opened_to_success")
    no_memory_drawers = no_memory.get("mean_drawers_opened_to_success")
    wrong_steps = mapping(metrics_by_arm.get("memory_wrong_binding")).get("mean_steps_to_retrieve")
    clean_success = float(clean.get("retrieval_success_rate", 0.0))
    return {
        "clean_minus_no_memory_steps": None if clean_steps is None or no_memory_steps is None else round(float(no_memory_steps) - float(clean_steps), 6),
        "clean_minus_no_memory_drawers_opened": None
        if clean_drawers is None or no_memory_drawers is None
        else round(float(no_memory_drawers) - float(clean_drawers), 6),
        "wrong_binding_minus_no_memory_steps": None
        if wrong_steps is None or no_memory_steps is None
        else round(float(wrong_steps) - float(no_memory_steps), 6),
        "clean_minus_no_memory_success": round(clean_success - float(no_memory.get("retrieval_success_rate", 0.0)), 6),
        "clean_minus_corrupt_success": {
            arm: round(clean_success - float(mapping(metrics_by_arm.get(arm)).get("retrieval_success_rate", 0.0)), 6)
            for arm in corrupt_arms
        },
    }


def decide_gridworld_injection(
    metrics_by_arm: Mapping[str, Any],
    *,
    curve_by_drawer_count: Mapping[str, Any] | None = None,
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
    if float(no_memory.get("retrieval_success_rate", 0.0)) < float(protocol_cfg.get("min_search_success_rate", 1.0)):
        return NO_GO_GOAL_NAVIGATOR_FAILURE, ["systematic_search_success_below_gate"]
    if float(clean.get("retrieval_success_rate", 0.0)) < float(protocol_cfg.get("min_clean_success_rate", 0.90)):
        return NO_GO_MEMORY_INJECTION_FAILURE, ["clean_success_below_gate"]
    if float(clean.get("goal_injection_accuracy", 0.0)) < float(protocol_cfg.get("min_goal_injection_accuracy", 0.90)):
        return NO_GO_MEMORY_INJECTION_FAILURE, ["goal_injection_accuracy_below_gate"]
    if float(clean.get("mean_drawers_opened_to_success", 0.0)) > float(protocol_cfg.get("max_clean_mean_drawers_opened", 1.0)):
        return NO_GO_MEMORY_INJECTION_FAILURE, ["clean_drawers_opened_above_gate"]
    pairwise = pairwise_metrics(metrics_by_arm, curve_by_drawer_count=curve_by_drawer_count or {})
    step_gain = float(pairwise.get("clean_minus_no_memory_steps") or 0.0)
    drawer_gain = float(pairwise.get("clean_minus_no_memory_drawers_opened") or 0.0)
    if step_gain < float(protocol_cfg.get("min_clean_step_gain", 4.0)):
        return NO_GO_NO_TASK_HEADROOM, ["clean_step_gain_below_gate"]
    if drawer_gain < float(protocol_cfg.get("min_clean_drawer_gain", 2.0)):
        return NO_GO_NO_TASK_HEADROOM, ["clean_drawer_gain_below_gate"]
    sweep = mapping(pairwise.get("drawer_count_sweep"))
    if sweep:
        if float(sweep.get("gap_growth_drawers", 0.0)) < float(protocol_cfg.get("min_drawer_gap_growth", 2.0)):
            return NO_GO_NO_TASK_HEADROOM, ["drawer_gap_growth_below_gate"]
        if float(sweep.get("last_wrong_binding_minus_no_memory_steps", 0.0)) < float(protocol_cfg.get("min_wrong_binding_step_cost", 1.0)):
            return NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING, ["wrong_binding_step_cost_below_gate"]
    min_wrong_drawer = float(protocol_cfg.get("min_corrupt_wrong_drawer_rate", 0.80))
    if any(float(row.get("wrong_drawer_rate", 0.0)) < min_wrong_drawer for row in corrupt):
        return NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING, ["corrupt_wrong_drawer_below_gate"]
    return GO_GRIDWORLD_INJECTION_SUPPORTED, ["clean_memory_retrieval_efficiency_scales_with_drawer_count_and_corrupt_controls_commit_wrong_goals"]


def contamination_for_episode(episode: KitchenEpisode) -> list[dict[str, Any]]:
    contexts = [*deployable_reveal_observations(episode), deployable_need_observation(episode, episode.start_position, previous_action="none")]
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


def int_list(value: Any, *, default: int) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return [int(default)]


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
        "| Arm | Success | Drawers opened | Steps | Wrong drawer | Goal accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, metrics in mapping(summary.get("metrics_by_arm")).items():
        row = mapping(metrics)
        lines.append(
            f"| {arm} | {row.get('retrieval_success_rate')} | {row.get('mean_drawers_opened_to_success')} | "
            f"{row.get('mean_steps_to_retrieve')} | "
            f"{row.get('wrong_drawer_rate')} | {row.get('goal_injection_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## Drawer Count Sweep",
            "",
            "| Drawer count | Clean drawers | Search drawers | Gap | Wrong-binding step cost |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for drawer_count, row in mapping(summary.get("curve_by_drawer_count")).items():
        metrics = mapping(row.get("metrics_by_arm"))
        clean = mapping(metrics.get("memory_clean"))
        search = mapping(metrics.get("no_memory_search"))
        pairwise = mapping(row.get("pairwise"))
        lines.append(
            f"| {drawer_count} | {clean.get('mean_drawers_opened_to_success')} | "
            f"{search.get('mean_drawers_opened_to_success')} | "
            f"{pairwise.get('clean_minus_no_memory_drawers_opened')} | "
            f"{pairwise.get('wrong_binding_minus_no_memory_steps')} |"
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
