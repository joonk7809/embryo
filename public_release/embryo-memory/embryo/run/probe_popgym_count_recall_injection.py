"""Probe memory-count injection on POPGym CountRecall."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.memory.popgym_count_recall import (
    CountRecallMemory,
    count_recall_content_hash,
    count_recall_shuffled_query,
    count_recall_wrong_binding_query,
)


GO_POPGYM_COUNT_RECALL_INJECTION_SUPPORTED = "GO_popgym_count_recall_injection_supported"
NOT_EVALUABLE_RUNTIME_UNAVAILABLE = "NOT_EVALUABLE_runtime_unavailable"
NO_GO_ORACLE_FAILURE = "NO_GO_oracle_count_failure"
NO_GO_CLEAN_MEMORY_FAILURE = "NO_GO_clean_count_memory_failure"
NO_GO_NO_MEMORY_NOT_LOW = "NO_GO_no_memory_not_low"
NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING = "NO_GO_corrupt_controls_not_separating"
NO_GO_ACTION_SPACE_MISMATCH = "NO_GO_action_space_mismatch"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"

ARMS = (
    "count_recall_no_memory",
    "count_recall_memory_clean",
    "count_recall_memory_shuffled",
    "count_recall_memory_wrong_binding",
    "count_recall_memory_stale",
    "count_recall_oracle_eval_only",
)
CORRUPT_ARMS = (
    "count_recall_memory_shuffled",
    "count_recall_memory_wrong_binding",
    "count_recall_memory_stale",
)
TASKS = {
    "count_recall_easy": ("popgym.envs.count_recall", "CountRecallEasy"),
    "count_recall_medium": ("popgym.envs.count_recall", "CountRecallMedium"),
    "count_recall_hard": ("popgym.envs.count_recall", "CountRecallHard"),
}


def run_popgym_count_recall_injection_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_cfg = mapping(config.get("runtime"))
    protocol_cfg = mapping(config.get("protocol"))
    task = str(runtime_cfg.get("task", "count_recall_hard"))
    seed_start = int(runtime_cfg.get("seed_start", 12000))
    seed_count = int(runtime_cfg.get("seed_count", 64))
    try:
        probe_env = make_count_recall_env(task)
    except RuntimeError as exc:
        summary = unavailable_summary(config, reason=str(exc), task=task, seed_start=seed_start, seed_count=seed_count)
        return {"summary": summary, "metrics_by_arm": {}, "episodes": [], "ticks": [], "config_manifest": json_safe(config), "contamination": summary["contamination"]}
    action_space_patch = str(getattr(probe_env, "_embryo_action_space_patch", "none"))

    episodes: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    contamination_failures: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + seed_count):
        for arm in ARMS:
            episode, episode_ticks, failures = run_arm_episode(
                task=task,
                seed=seed,
                arm=arm,
                stale_lag=int(protocol_cfg.get("stale_lag", 32)),
            )
            episodes.append(episode)
            ticks.extend(episode_ticks)
            contamination_failures.extend(failures)

    metrics_by_arm = summarize_by_arm(episodes)
    pairwise = pairwise_metrics(metrics_by_arm)
    contamination = {"passed": not contamination_failures, "failure_count": len(contamination_failures), "failures": contamination_failures}
    decision, reasons = decide_popgym_count_recall_injection(metrics_by_arm, contamination=contamination, protocol_cfg=protocol_cfg)
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "popgym_count_recall_memory_count_injection",
        "boundary": "within_episode_count_memory_from_deployable_observations_no_rl_no_training",
        "runtime": {
            "name": "popgym_count_recall",
            "task": task,
            "seed_start": seed_start,
            "seed_count": seed_count,
            "action_space_patch": action_space_patch,
        },
        "arms": list(ARMS),
        "metrics_by_arm": metrics_by_arm,
        "pairwise": pairwise,
        "contamination": contamination,
        "protocol": json_safe(protocol_cfg),
    }
    return {
        "summary": summary,
        "metrics_by_arm": metrics_by_arm,
        "episodes": episodes,
        "ticks": ticks,
        "config_manifest": json_safe(config),
        "contamination": contamination,
    }


def run_arm_episode(*, task: str, seed: int, arm: str, stale_lag: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = make_count_recall_env(task)
    obs, _ = reset_env(env, seed)
    memory = CountRecallMemory(symbol_count=int(env.num_distinct_cards))
    previous_obs = None
    previous_action: int | None = None
    done = False
    tick = 0
    reward_sum = 0.0
    correct_count = 0
    invalid_count = 0
    absolute_errors: list[int] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    while not done:
        context = deployable_count_recall_context(obs, previous_obs=previous_obs, previous_action=previous_action)
        scan = scan_actor_context(context)
        for failure in scan.get("failures", []):
            failures.append({"seed": int(seed), "arm": arm, "tick": tick, **dict(failure)})
        parsed = memory.update(context)
        target_count = int(env.counts[parsed.query])
        decision = decide_count_recall_action(
            arm=arm,
            memory=memory,
            query=parsed.query,
            target_count_eval_only=target_count,
            seed=seed,
            tick=tick,
            stale_lag=stale_lag,
        )
        action = int(decision["action"])
        invalid = action < 0 or action >= int(env.action_space.n)
        next_obs, reward, done, info = step_env(env, action)
        correct = action == target_count
        reward_sum += float(reward)
        correct_count += int(correct)
        invalid_count += int(invalid)
        absolute_errors.append(abs(action - target_count))
        rows.append(
            {
                "seed": int(seed),
                "arm": arm,
                "tick": tick,
                "actor": {
                    "runtime": "popgym_count_recall",
                    "task": task,
                    "arm": arm,
                    "observation": context["popgym_observation"],
                    "previous_observation": context["previous_popgym_observation"],
                    "previous_action": context["previous_action"],
                    "action": action,
                },
                "memory": decision,
                "metrics": {
                    "correct": correct,
                    "absolute_count_error": abs(action - target_count),
                    "invalid_action": invalid,
                    "memory_hit": bool(decision["memory_hit"]),
                    "corrupt_memory": arm in CORRUPT_ARMS,
                },
                "eval_only": {
                    "reward_delta_eval_only": float(reward),
                    "target_count_eval_only": target_count,
                    "backend_counts_eval_only": [int(value) for value in env.counts],
                    "info_keys_eval_only": sorted(str(key) for key in mapping(info)),
                },
            }
        )
        previous_obs = obs
        previous_action = action
        obs = next_obs
        tick += 1

    total_ticks = len(rows)
    episode = {
        "seed": int(seed),
        "arm": arm,
        "task": task,
        "tick_count": total_ticks,
        "reward_sum": round(reward_sum, 8),
        "query_accuracy": correct_count / total_ticks if total_ticks else 0.0,
        "correct_count": correct_count,
        "incorrect_count": total_ticks - correct_count,
        "invalid_action_rate": invalid_count / total_ticks if total_ticks else 0.0,
        "mean_absolute_count_error": sum(absolute_errors) / total_ticks if total_ticks else 0.0,
    }
    return episode, rows, failures


def decide_count_recall_action(
    *,
    arm: str,
    memory: CountRecallMemory,
    query: int,
    target_count_eval_only: int,
    seed: int,
    tick: int,
    stale_lag: int,
) -> dict[str, Any]:
    counts = memory.counts()
    effective_counts = counts
    effective_query = int(query)
    stale_active = False
    if arm == "count_recall_no_memory":
        action = 0
        memory_hit = False
    elif arm == "count_recall_oracle_eval_only":
        action = int(target_count_eval_only)
        memory_hit = False
    elif arm == "count_recall_memory_clean":
        action = memory.answer(query)
        memory_hit = True
    elif arm == "count_recall_memory_wrong_binding":
        effective_query = count_recall_wrong_binding_query(query, memory.symbol_count)
        action = memory.answer(effective_query)
        memory_hit = True
    elif arm == "count_recall_memory_shuffled":
        effective_query = count_recall_shuffled_query(query, memory.symbol_count, seed=seed)
        action = memory.answer(effective_query)
        memory_hit = True
    elif arm == "count_recall_memory_stale":
        effective_counts = memory.stale_counts(lag=stale_lag)
        action = memory.answer(query, counts=effective_counts)
        memory_hit = True
        stale_active = effective_counts != counts
    else:
        raise ValueError(f"Unknown CountRecall arm: {arm}")
    return {
        "action": int(action),
        "query_symbol": int(query),
        "effective_query_symbol": int(effective_query),
        "effective_count": int(action),
        "latest_counts_hash": count_recall_content_hash(counts=counts, query=query, arm="latest", effective_query=query),
        "query_content_hash": count_recall_content_hash(
            counts=effective_counts,
            query=query,
            arm=arm,
            effective_query=effective_query,
            stale_lag=stale_lag if arm == "count_recall_memory_stale" else 0,
        ),
        "memory_hit": bool(memory_hit),
        "stale_invalidated": bool(stale_active),
        "corruption_mode": corruption_mode(arm),
        "provenance": "popgym_count_recall_observation_v0" if memory_hit else "no_memory_or_eval_only",
        "tick": int(tick),
    }


def summarize_by_arm(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in episodes if str(row.get("arm")) == arm]
        result[arm] = {
            "episode_count": len(rows),
            "tick_count": sum(int(row.get("tick_count", 0)) for row in rows),
            "mean_reward_sum": round(mean(row.get("reward_sum") for row in rows), 6),
            "query_accuracy": round(mean(row.get("query_accuracy") for row in rows), 6),
            "mean_absolute_count_error": round(mean(row.get("mean_absolute_count_error") for row in rows), 6),
            "invalid_action_rate": round(mean(row.get("invalid_action_rate") for row in rows), 6),
        }
    return result


def pairwise_metrics(metrics_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    clean = mapping(metrics_by_arm.get("count_recall_memory_clean"))
    clean_accuracy = float(clean.get("query_accuracy", 0.0))
    clean_reward = float(clean.get("mean_reward_sum", 0.0))
    return {
        "clean_minus_no_memory_accuracy": round(clean_accuracy - float(mapping(metrics_by_arm.get("count_recall_no_memory")).get("query_accuracy", 0.0)), 6),
        "clean_minus_oracle_accuracy": round(clean_accuracy - float(mapping(metrics_by_arm.get("count_recall_oracle_eval_only")).get("query_accuracy", 0.0)), 6),
        "clean_minus_corrupt_accuracy": {
            arm: round(clean_accuracy - float(mapping(metrics_by_arm.get(arm)).get("query_accuracy", 0.0)), 6)
            for arm in CORRUPT_ARMS
        },
        "clean_minus_no_memory_reward": round(clean_reward - float(mapping(metrics_by_arm.get("count_recall_no_memory")).get("mean_reward_sum", 0.0)), 6),
        "clean_minus_corrupt_reward": {
            arm: round(clean_reward - float(mapping(metrics_by_arm.get(arm)).get("mean_reward_sum", 0.0)), 6)
            for arm in CORRUPT_ARMS
        },
    }


def decide_popgym_count_recall_injection(
    metrics_by_arm: Mapping[str, Any],
    *,
    contamination: Mapping[str, Any],
    protocol_cfg: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count_nonzero"]
    clean = mapping(metrics_by_arm.get("count_recall_memory_clean"))
    oracle = mapping(metrics_by_arm.get("count_recall_oracle_eval_only"))
    no_memory = mapping(metrics_by_arm.get("count_recall_no_memory"))
    corrupt = [mapping(metrics_by_arm.get(arm)) for arm in CORRUPT_ARMS]
    if float(oracle.get("query_accuracy", 0.0)) < float(protocol_cfg.get("min_oracle_accuracy", 0.99)):
        return NO_GO_ORACLE_FAILURE, ["oracle_accuracy_below_gate"]
    if float(clean.get("query_accuracy", 0.0)) < float(protocol_cfg.get("min_clean_accuracy", 0.99)):
        return NO_GO_CLEAN_MEMORY_FAILURE, ["clean_accuracy_below_gate"]
    if float(no_memory.get("query_accuracy", 0.0)) > float(protocol_cfg.get("max_no_memory_accuracy", 0.15)):
        return NO_GO_NO_MEMORY_NOT_LOW, ["no_memory_accuracy_above_gate"]
    if any(float(row.get("query_accuracy", 0.0)) > float(protocol_cfg.get("max_corrupt_accuracy", 0.35)) for row in corrupt):
        return NO_GO_CORRUPT_CONTROLS_NOT_SEPARATING, ["corrupt_accuracy_above_gate"]
    max_invalid = max(float(mapping(row).get("invalid_action_rate", 0.0)) for row in metrics_by_arm.values()) if metrics_by_arm else 0.0
    if max_invalid > float(protocol_cfg.get("max_invalid_action_rate", 0.0)):
        return NO_GO_ACTION_SPACE_MISMATCH, ["invalid_action_rate_above_gate"]
    return GO_POPGYM_COUNT_RECALL_INJECTION_SUPPORTED, ["clean_count_memory_matches_oracle_and_separates_from_no_memory_and_corrupt_controls"]


def deployable_count_recall_context(obs: Any, *, previous_obs: Any, previous_action: int | None) -> dict[str, Any]:
    return {
        "popgym_observation": json_safe(obs),
        "previous_popgym_observation": None if previous_obs is None else json_safe(previous_obs),
        "previous_action": None if previous_action is None else int(previous_action),
    }


def make_count_recall_env(task: str) -> Any:
    if task not in TASKS:
        raise ValueError(f"Unknown POPGym CountRecall task: {task}")
    module_name, class_name = TASKS[task]
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install popgym to run the CountRecall injection probe.") from exc
    env = getattr(module, class_name)()
    env._embryo_action_space_patch = normalize_count_recall_action_space(env)
    return env


def normalize_count_recall_action_space(env: Any) -> str:
    inclusive_count_actions = int(env.max_card_count) + 1
    declared_actions = int(env.action_space.n)
    if declared_actions >= inclusive_count_actions:
        return "none"
    env.action_space = env.action_space.__class__(inclusive_count_actions)
    return "count_recall_inclusive_max_count_action_space_v0"


def reset_env(env: Any, seed: int) -> tuple[Any, Mapping[str, Any]]:
    out = env.reset(seed=seed)
    return out if isinstance(out, tuple) else (out, {})


def step_env(env: Any, action: int) -> tuple[Any, float, bool, Mapping[str, Any]]:
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), mapping(info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), mapping(info)


def corruption_mode(arm: str) -> str:
    if arm.endswith("_clean"):
        return "clean"
    if arm.endswith("_shuffled"):
        return "shuffled_query_binding"
    if arm.endswith("_wrong_binding"):
        return "wrong_query_binding"
    if arm.endswith("_stale"):
        return "stale_counts"
    if arm.endswith("_no_memory"):
        return "off"
    return "eval_only"


def unavailable_summary(config: Mapping[str, Any], *, reason: str, task: str, seed_start: int, seed_count: int) -> dict[str, Any]:
    contamination = {"passed": True, "failure_count": 0, "failures": []}
    return {
        "decision": NOT_EVALUABLE_RUNTIME_UNAVAILABLE,
        "decision_reasons": [reason],
        "objective": "popgym_count_recall_memory_count_injection",
        "boundary": "within_episode_count_memory_from_deployable_observations_no_rl_no_training",
        "runtime": {
            "name": "popgym_count_recall",
            "task": task,
            "seed_start": seed_start,
            "seed_count": seed_count,
            "action_space_patch": "unavailable",
        },
        "arms": list(ARMS),
        "metrics_by_arm": {},
        "pairwise": {},
        "contamination": contamination,
        "protocol": json_safe(mapping(config.get("protocol"))),
    }


def write_popgym_count_recall_injection_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "popgym_count_recall_summary_json": root / "popgym_count_recall_summary.json",
        "popgym_count_recall_summary_md": root / "popgym_count_recall_summary.md",
        "metrics_by_arm": root / "metrics_by_arm.json",
        "episodes": root / "popgym_count_recall_episodes.jsonl",
        "ticks": root / "popgym_count_recall_ticks.jsonl",
        "config_manifest": root / "config_manifest.json",
        "contamination_scan": root / "contamination_scan.json",
    }
    paths["popgym_count_recall_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["popgym_count_recall_summary_md"].write_text(format_summary_markdown(result["summary"]), encoding="utf-8")
    paths["metrics_by_arm"].write_text(json.dumps(result["metrics_by_arm"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["config_manifest"].write_text(json.dumps(result["config_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(paths["episodes"], result["episodes"])
    write_jsonl(paths["ticks"], result["ticks"])
    return {key: str(value) for key, value in paths.items()}


def format_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# POPGym CountRecall Injection Probe",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Boundary: `{summary.get('boundary')}`",
        f"- Contamination failures: `{mapping(summary.get('contamination')).get('failure_count')}`",
        "",
        "## Arms",
        "",
        "| Arm | Episodes | Accuracy | Reward | Mean absolute error | Invalid action rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, metrics in mapping(summary.get("metrics_by_arm")).items():
        row = mapping(metrics)
        lines.append(
            f"| {arm} | {row.get('episode_count')} | {row.get('query_accuracy')} | "
            f"{row.get('mean_reward_sum')} | {row.get('mean_absolute_count_error')} | {row.get('invalid_action_rate')} |"
        )
    lines.extend(["", "## Pairwise", "", "```json", json.dumps(summary.get("pairwise", {}), indent=2, sort_keys=True), "```"])
    return "\n".join(lines) + "\n"


def mean(values: Sequence[Any]) -> float:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else 0.0


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/popgym_count_recall_injection.yaml")
    parser.add_argument("--out", default="runs/popgym_count_recall_injection")
    args = parser.parse_args(argv)
    result = run_popgym_count_recall_injection_probe(load_config(args.config))
    artifacts = write_popgym_count_recall_injection_artifacts(result, args.out)
    print(json.dumps({"decision": result["summary"]["decision"], "artifacts": artifacts, "out": str(Path(args.out).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
