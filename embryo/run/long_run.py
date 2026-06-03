"""Reusable long-run protocol runner."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.eval.contamination import scan_actor_context
from embryo.eval.long_run import (
    NOT_EVALUABLE_RUNTIME_UNAVAILABLE,
    combine_contamination,
    episode_record,
    format_long_run_summary_markdown,
    summarize_long_run_protocol,
)
from embryo.eval.traces import write_jsonl
from embryo.memory.freshness import FreshnessState
from embryo.memory.queries import build_event_query, build_resource_query, combine_queries
from embryo.models import RuleRouterModel, ThresholdFactWriter
from embryo.runtimes import make_runtime
from embryo.runtimes.base import RuntimeAdapter, RuntimeSpec


DEFAULT_ARMS = (
    "no_memory",
    "query_memory_clean",
    "query_memory_shuffled",
    "query_memory_stale",
    "query_memory_wrong_binding",
    "random_valid_action",
)


def run_long_run_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run fixed-seed long-run episodes and return artifact payloads."""
    manifest = resolve_protocol_manifest(config)
    episodes: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    runtime_unavailable = None

    runtime_cfg = manifest["runtime"]
    protocol_cfg = manifest["protocol"]
    runtime_name = str(runtime_cfg["name"])
    split = str(runtime_cfg["split"])
    detail_ticks = bool(protocol_cfg.get("detail_ticks", False))

    for horizon in protocol_cfg["horizons"]:
        for seed in runtime_cfg["seeds"]:
            for episode_index in range(int(protocol_cfg["max_episodes_per_seed"])):
                for arm in manifest["arms"]:
                    actor = {
                        "runtime": runtime_name,
                        "arm": str(arm),
                        "seed": int(seed),
                        "horizon": int(horizon),
                        "split": split,
                        "episode_index": int(episode_index),
                        "episode_id": episode_id(split=split, seed=int(seed), horizon=int(horizon), episode_index=int(episode_index)),
                    }
                    try:
                        episode_ticks = run_long_run_episode(actor=actor, runtime_name=runtime_name)
                    except Exception as exc:  # noqa: BLE001
                        if runtime_is_unavailable(exc):
                            runtime_unavailable = {"runtime": runtime_name, "reason": str(exc), "decision": NOT_EVALUABLE_RUNTIME_UNAVAILABLE}
                            episode_ticks = []
                        else:
                            raise
                    episodes.append(episode_record(actor=actor, ticks=episode_ticks))
                    ticks.extend(episode_ticks)
                    if runtime_unavailable:
                        break
                if runtime_unavailable:
                    break
            if runtime_unavailable:
                break
        if runtime_unavailable:
            break

    contamination = combine_contamination(episodes)
    summary = summarize_long_run_protocol(
        protocol_manifest=manifest,
        episodes=episodes,
        ticks=ticks,
        contamination=contamination,
        runtime_unavailable=runtime_unavailable,
    )
    return {
        "summary": summary,
        "episodes": episodes,
        "ticks": ticks if detail_ticks else [],
        "protocol_manifest": manifest,
        "contamination": contamination,
    }


def run_long_run_episode(*, actor: Mapping[str, Any], runtime_name: str) -> list[dict[str, Any]]:
    horizon = int(actor["horizon"])
    runtime = make_runtime_for_horizon(runtime_name, horizon=horizon)
    try:
        current = runtime.reset(seed=int(actor["seed"]))
        state = EpisodeState(seed=int(actor["seed"]), arm=str(actor["arm"]), horizon=horizon)
        rows: list[dict[str, Any]] = []
        for tick in range(horizon):
            if current.done:
                break
            decision = select_action_for_arm(runtime.spec, current.observation, state, str(actor["arm"]))
            post_step = runtime.step(decision["action"])
            row = build_tick_row(
                actor=actor,
                tick=tick,
                spec=runtime.spec,
                pre_observation=current.observation,
                post_step=post_step,
                decision=decision,
                state=state,
            )
            rows.append(row)
            state.observe(decision["action"], decision["route_mode"])
            current = post_step
            if post_step.done:
                break
        return rows
    finally:
        runtime.close()


class EpisodeState:
    def __init__(self, *, seed: int, arm: str, horizon: int) -> None:
        self.arm = arm
        self.rng = random.Random(f"{seed}:{arm}:{horizon}")
        self.previous_action: str | None = None
        self.repeat_count = 0
        self.last_route = "hold"
        self.recent_actions: deque[str] = deque(maxlen=8)

    def observe(self, action: str, route: str) -> None:
        if action == self.previous_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.previous_action = action
        self.last_route = route
        self.recent_actions.append(action)


def select_action_for_arm(spec: RuntimeSpec, observation: Mapping[str, Any], state: EpisodeState, arm: str) -> dict[str, Any]:
    if arm == "no_memory":
        action = spec.noop_action
        return {
            "action": action,
            "route_mode": "no_memory",
            "query_content_hash": "",
            "cache_age": 0,
            "resource_memory_critical": resource_memory_critical(observation),
            "resource_route_preserved": False,
            "fallback_triggered": False,
            "event_self_triggered": False,
            "repeated_action_loop": state.repeat_count >= 2,
            "fact_value": False,
            "invalid_or_unknown_action": not action_is_valid(spec, action),
        }

    if arm == "random_valid_action":
        action = state.rng.choice(tuple(spec.action_names))
        return {
            "action": action,
            "route_mode": "random_valid_action",
            "query_content_hash": "",
            "cache_age": 0,
            "resource_memory_critical": resource_memory_critical(observation),
            "resource_route_preserved": False,
            "fallback_triggered": False,
            "event_self_triggered": False,
            "repeated_action_loop": state.repeat_count >= 2,
            "fact_value": False,
            "invalid_or_unknown_action": not action_is_valid(spec, action),
        }

    features = corrupt_features(deployable_features(observation, repeat_count=state.repeat_count, last_route=state.last_route), arm)
    fact_writer = ThresholdFactWriter()
    router = RuleRouterModel()
    fact = fact_writer.fact(features)
    freshness = FreshnessState(
        cache_age=int(features.get("cache_age", 0)),
        invalidated=bool(features.get("invalidated", False)),
        visual_change_conflict=bool(features.get("visual_change_conflict", False)),
    )
    resource_query = build_resource_query([fact], cache_age=freshness.cache_age, fresh=not freshness.invalidated)
    event_query = build_event_query([fact], cache_age=freshness.cache_age)
    combined_query = combine_queries(resource_query, event_query)
    router_output = router.predict(
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
    action = action_for_route(spec, router_output.route_mode)
    event_self_triggered = state.last_route == "event_fallback" and router_output.route_mode == "event_fallback" and not bool(
        features.get("failed_action_event", False)
    )
    return {
        "action": action,
        "route_mode": router_output.route_mode,
        "query_content_hash": combined_query.content_hash,
        "cache_age": freshness.cache_age,
        "resource_memory_critical": resource_memory_critical(observation),
        "resource_route_preserved": bool(router_output.resource_route_preserved),
        "fallback_triggered": bool(router_output.event_fallback_gate),
        "event_self_triggered": bool(event_self_triggered),
        "repeated_action_loop": state.repeat_count >= 2,
        "fact_value": bool(fact.value),
        "invalid_or_unknown_action": not action_is_valid(spec, action),
    }


def build_tick_row(
    *,
    actor: Mapping[str, Any],
    tick: int,
    spec: RuntimeSpec,
    pre_observation: Mapping[str, Any],
    post_step: Any,
    decision: Mapping[str, Any],
    state: EpisodeState,
) -> dict[str, Any]:
    action = str(decision["action"])
    actor_context = actor_context_from_decision(pre_observation, decision)
    contamination = scan_actor_context(actor_context)
    valid_action = action_is_valid(spec, action)
    row_actor = dict(actor)
    row_actor.update(
        {
            "tick": int(tick),
            "action": action,
            "route_mode": str(decision.get("route_mode", "")),
        }
    )
    return {
        "actor": row_actor,
        "metrics": {
            "valid_action": valid_action,
            "invalid_or_unknown_action": bool(decision.get("invalid_or_unknown_action", False)) or not valid_action,
            "repeated_action_loop": bool(decision.get("repeated_action_loop", False)),
            "event_self_triggered": bool(decision.get("event_self_triggered", False)),
            "resource_memory_critical": bool(decision.get("resource_memory_critical", False)),
            "resource_route_preserved": bool(decision.get("resource_route_preserved", False)),
            "fallback_triggered": bool(decision.get("fallback_triggered", False)),
            "query_content_hash": str(decision.get("query_content_hash", "")),
            "action_entropy_proxy": action_entropy_with_candidate(state.recent_actions, action),
            "exploration_bin": exploration_bin(state.recent_actions, action),
        },
        "eval_only": eval_only_tick_fields(post_step),
        "contamination": contamination,
    }


def deployable_features(observation: Mapping[str, Any], *, repeat_count: int, last_route: str) -> dict[str, Any]:
    return {
        "raw_rgb_frame": observation.get("raw_rgb_frame"),
        "previous_rgb_frame": observation.get("previous_rgb_frame"),
        "previous_action": observation.get("previous_action", "noop"),
        "candidate_score": numeric(observation.get("candidate_score", 0.0)),
        "failed_action_event": bool(observation.get("failed_action_event", False)),
        "cache_age": int(observation.get("cache_age", 0) or 0),
        "invalidated": bool(observation.get("invalidated", False)),
        "visual_change_conflict": bool(observation.get("visual_change_conflict", False)),
        "repeat_count": int(repeat_count),
        "last_route": str(last_route),
    }


def corrupt_features(features: Mapping[str, Any], arm: str) -> dict[str, Any]:
    row = dict(features)
    score = numeric(row.get("candidate_score", 0.0))
    if arm == "no_memory":
        row.update(candidate_score=0.0, failed_action_event=False)
    elif arm == "query_memory_clean":
        pass
    elif arm == "query_memory_shuffled":
        row["candidate_score"] = 1.0 - score
    elif arm == "query_memory_stale":
        row.update(cache_age=max(99, int(row.get("cache_age", 0))), invalidated=True)
    elif arm == "query_memory_wrong_binding":
        row["candidate_score"] = 0.0 if score >= 0.5 else 1.0
    else:
        raise ValueError(f"Unknown long-run arm: {arm}")
    return row


def actor_context_from_decision(observation: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_rgb_frame": observation.get("raw_rgb_frame", "<rgb_frame>"),
        "previous_rgb_frame": observation.get("previous_rgb_frame", "<previous_rgb_frame>"),
        "previous_action": observation.get("previous_action", "noop"),
        "facing_candidate_v1": bool(decision.get("fact_value", False)),
        "query_content": str(decision.get("query_content_hash", "")),
        "query_content_hash": str(decision.get("query_content_hash", "")),
        "query_cache_age": int(decision.get("cache_age", 0)),
        "short_ttl_stale_state": str(decision.get("cache_age", 0)),
    }


def eval_only_tick_fields(step: Any) -> dict[str, Any]:
    info = step.info if isinstance(step.info, Mapping) else {}
    achievements = achievements_from_info(info)
    return {
        "reward_delta_eval_only": numeric(step.reward),
        "done_eval_only": bool(step.done),
        "death_cause_eval_only": death_cause_from_info(info, done=bool(step.done)),
        "achievements_eval_only": achievements,
        "health_eval_only": optional_numeric(info.get("health")),
        "food_eval_only": optional_numeric(info.get("food")),
        "drink_eval_only": optional_numeric(info.get("drink")),
        "backend_inventory_eval_only": json_safe(info.get("inventory")) if "inventory" in info else None,
    }


def action_for_route(spec: RuntimeSpec, route: str) -> str:
    if route == "resource":
        return spec.resource_action
    if route == "event_fallback":
        return spec.fallback_action
    return spec.noop_action


def action_is_valid(spec: RuntimeSpec, action: str) -> bool:
    return action in spec.action_names or action in spec.action_map


def resource_memory_critical(observation: Mapping[str, Any]) -> bool:
    if "resource_memory_critical" in observation:
        return bool(observation["resource_memory_critical"])
    return numeric(observation.get("candidate_score", 0.0)) >= 0.5


def action_entropy_with_candidate(recent_actions: Sequence[str], action: str) -> float:
    actions = [*recent_actions, action]
    if len(set(actions)) <= 1:
        return 0.0
    counts = {item: actions.count(item) for item in set(actions)}
    total = len(actions)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(entropy / math.log(len(counts)), 4)


def exploration_bin(recent_actions: Sequence[str], action: str) -> int:
    return min(4, int(action_entropy_with_candidate(recent_actions, action) * 5))


def achievements_from_info(info: Mapping[str, Any]) -> list[str]:
    raw = info.get("achievements", info.get("achievement"))
    if isinstance(raw, Mapping):
        return sorted(str(key) for key, value in raw.items() if bool(value))
    if isinstance(raw, (list, tuple, set)):
        return sorted(str(item) for item in raw if item is not None)
    if raw:
        return [str(raw)]
    return []


def death_cause_from_info(info: Mapping[str, Any], *, done: bool) -> str | None:
    if not done:
        return None
    for key in ("death_cause", "death_reason", "cause_of_death"):
        if info.get(key):
            return str(info[key])
    return None


def optional_numeric(value: Any) -> float | None:
    if value is None:
        return None
    return numeric(value)


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def make_runtime_for_horizon(runtime_name: str, *, horizon: int) -> RuntimeAdapter:
    try:
        return make_runtime(runtime_name, max_steps=horizon)
    except TypeError:
        return make_runtime(runtime_name)


def runtime_is_unavailable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    return "unavailable" in name or "install embryo[crafter]" in str(exc).lower()


def episode_id(*, split: str, seed: int, horizon: int, episode_index: int) -> str:
    return f"{split}:seed-{seed}:horizon-{horizon}:episode-{episode_index}"


def resolve_protocol_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config.get("runtime", {})) if isinstance(config.get("runtime", {}), Mapping) else {}
    protocol = dict(config.get("protocol", {})) if isinstance(config.get("protocol", {}), Mapping) else {}
    metrics = dict(config.get("metrics", {})) if isinstance(config.get("metrics", {}), Mapping) else {}
    seed_start = int(runtime.get("seed_start", 10000))
    seed_count = int(runtime.get("seed_count", 1))
    seeds = tuple(int(seed) for seed in runtime.get("seeds", range(seed_start, seed_start + seed_count)))
    runtime.update(
        {
            "name": str(runtime.get("name", "fixture_memory")),
            "split": str(runtime.get("split", "dev")),
            "seed_start": seed_start,
            "seed_count": seed_count,
            "seeds": list(seeds),
        }
    )
    protocol.update(
        {
            "horizons": [int(value) for value in protocol.get("horizons", (256, 2048))],
            "max_episodes_per_seed": int(protocol.get("max_episodes_per_seed", 1)),
            "detail_ticks": bool(protocol.get("detail_ticks", False)),
        }
    )
    arms = tuple(str(arm) for arm in config.get("arms", DEFAULT_ARMS))
    return {
        "runtime": runtime,
        "protocol": protocol,
        "arms": list(arms),
        "metrics": {
            "survival": bool(metrics.get("survival", True)),
            "valid_actions": bool(metrics.get("valid_actions", True)),
            "loop_rate": bool(metrics.get("loop_rate", True)),
            "event_self_trigger": bool(metrics.get("event_self_trigger", True)),
            "memory_grounded_score": bool(metrics.get("memory_grounded_score", True)),
            "contamination": bool(metrics.get("contamination", True)),
        },
    }


def write_long_run_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "long_run_summary_json": root / "long_run_summary.json",
        "long_run_summary_md": root / "long_run_summary.md",
        "long_run_episodes": root / "long_run_episodes.jsonl",
        "protocol_manifest": root / "protocol_manifest.json",
        "contamination_scan": root / "contamination_scan.json",
    }
    paths["long_run_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["long_run_summary_md"].write_text(format_long_run_summary_markdown(result["summary"]), encoding="utf-8")
    write_jsonl(paths["long_run_episodes"], result["episodes"])
    paths["protocol_manifest"].write_text(json.dumps(result["protocol_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = result.get("protocol_manifest", {})
    protocol = manifest.get("protocol", {}) if isinstance(manifest, Mapping) else {}
    if isinstance(protocol, Mapping) and bool(protocol.get("detail_ticks", False)):
        tick_path = root / "long_run_ticks.jsonl"
        write_jsonl(tick_path, result.get("ticks", ()))
        paths["long_run_ticks"] = tick_path
    return {key: str(path) for key, path in paths.items()}


def config_with_overrides(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from embryo.core.config import load_config

    config = load_config(args.config)
    runtime = dict(config.get("runtime", {}))
    protocol = dict(config.get("protocol", {}))
    if args.runtime:
        runtime["name"] = args.runtime
    if args.split:
        runtime["split"] = args.split
    if args.seed_start is not None:
        runtime["seed_start"] = args.seed_start
    if args.seed_count is not None:
        runtime["seed_count"] = args.seed_count
    if args.seeds:
        runtime["seeds"] = args.seeds
        runtime["seed_count"] = len(args.seeds)
    if args.horizons:
        protocol["horizons"] = args.horizons
    if args.detail_ticks:
        protocol["detail_ticks"] = True
    config["runtime"] = runtime
    config["protocol"] = protocol
    if args.out:
        config["output"] = str(resolve_path(root, args.out))
    return config


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the long-run memory protocol.")
    parser.add_argument("--config", default=str(root / "configs" / "long_run_sanity.yaml"))
    parser.add_argument("--runtime", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--seed-block", dest="split", default="")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--horizon", dest="horizons", type=int, action="append")
    parser.add_argument("--detail-ticks", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    config = config_with_overrides(args, root)
    out = Path(config.get("output", root / "runs" / "long_run_sanity"))
    result = run_long_run_protocol(config)
    paths = write_long_run_artifacts(result, out)
    print(json.dumps({"decision": result["summary"]["decision"], "out": str(out), "artifacts": paths}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
