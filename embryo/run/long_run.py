"""Reusable long-run protocol runner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.eval.long_run import (
    NOT_EVALUABLE_RUNTIME_UNAVAILABLE,
    combine_contamination,
    episode_record,
    format_long_run_summary_markdown,
    summarize_long_run_protocol,
)
from embryo.eval.traces import write_jsonl
from embryo.models.memory_residual_policy import MemoryResidualPolicy
from embryo.run.long_run_actors import make_long_run_actor
from embryo.run.long_run_arms import (
    DEFAULT_ARMS,
    MEMORY_RESIDUAL_BIAS,
    EpisodeState,
    action_is_valid,
    select_action_for_arm,
    select_action_with_actor_sidecar,
)
from embryo.run.long_run_config import CRAFTER_BACKEND_DETERMINISM_PATCH, resolve_protocol_manifest
from embryo.run.long_run_fact_surface import load_fact_surface
from embryo.run.long_run_replay import fresh_process_deterministic_replay_summary
from embryo.run.long_run_rows import build_tick_row
from embryo.runtimes import make_runtime
from embryo.runtimes.base import RuntimeAdapter


def run_long_run_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run fixed-seed long-run episodes and return artifact payloads."""
    manifest = resolve_protocol_manifest(config)
    primary = collect_long_run_pass(manifest)
    deterministic_replay = {"enabled": False, "mode": "disabled", "passed": None}
    if primary["runtime_unavailable"] is None:
        deterministic_replay = fresh_process_deterministic_replay_summary(manifest, primary["ticks"])
    contamination = combine_contamination(primary["episodes"])
    summary = summarize_long_run_protocol(
        protocol_manifest=manifest,
        episodes=primary["episodes"],
        ticks=primary["ticks"],
        contamination=contamination,
        runtime_unavailable=primary["runtime_unavailable"],
        deterministic_replay=deterministic_replay,
    )
    detail_ticks = bool(manifest["protocol"].get("detail_ticks", False))
    return {
        "summary": summary,
        "episodes": primary["episodes"],
        "ticks": primary["ticks"] if detail_ticks else [],
        "protocol_manifest": manifest,
        "contamination": contamination,
    }


def collect_long_run_pass(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run one protocol pass. Reused for deterministic replay checks."""
    episodes: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    runtime_unavailable = None

    runtime_cfg = manifest["runtime"]
    protocol_cfg = manifest["protocol"]
    fact_surface = load_fact_surface(manifest.get("fact_surface", {}), root=Path(__file__).resolve().parents[2])
    runtime_name = str(runtime_cfg["name"])
    split = str(runtime_cfg["split"])

    for horizon in protocol_cfg["horizons"]:
        for seed in runtime_cfg["seeds"]:
            for episode_index in range(int(protocol_cfg["max_episodes_per_seed"])):
                for arm in manifest["arms"]:
                    fact_surface_cfg = manifest.get("fact_surface", {})
                    fact_surface_cfg = fact_surface_cfg if isinstance(fact_surface_cfg, Mapping) else {}
                    actor = {
                        "runtime": runtime_name,
                        "arm": str(arm),
                        "seed": int(seed),
                        "horizon": int(horizon),
                        "split": split,
                        "episode_index": int(episode_index),
                        "episode_id": episode_id(split=split, seed=int(seed), horizon=int(horizon), episode_index=int(episode_index)),
                        "fact_surface": str(fact_surface_cfg.get("name", "reference_rgb_scaffold")),
                        "fact_surface_checkpoint": fact_surface_cfg.get("checkpoint"),
                        "fact_surface_threshold": fact_surface_cfg.get("threshold"),
                    }
                    try:
                        episode_ticks = run_long_run_episode(
                            actor=actor,
                            runtime_config=runtime_cfg,
                            protocol_config=protocol_cfg,
                            fact_surface=fact_surface,
                            actor_config=manifest.get("actor", {}),
                        )
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

    return {"episodes": episodes, "ticks": ticks, "runtime_unavailable": runtime_unavailable}


def run_long_run_episode(
    *,
    actor: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    protocol_config: Mapping[str, Any] | None = None,
    fact_surface: Any,
    actor_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    horizon = int(actor["horizon"])
    runtime = make_runtime_for_horizon(
        str(runtime_config["name"]),
        horizon=horizon,
        seed=int(actor["seed"]),
        deterministic_backend_patch=runtime_config.get("deterministic_backend_patch"),
    )
    frozen_actor = None
    try:
        current = runtime.reset(seed=int(actor["seed"]))
        frozen_actor = make_long_run_actor(actor_config or {}, runtime.spec, root=Path(__file__).resolve().parents[2])
        if frozen_actor is not None:
            frozen_actor.reset(seed=int(actor["seed"]))
        memory_residual_bias = max(0.0, float((actor_config or {}).get("memory_residual_bias", MEMORY_RESIDUAL_BIAS)))
        residual_policy = MemoryResidualPolicy(strength=1.0, max_abs_bias=memory_residual_bias)
        protocol = protocol_config if isinstance(protocol_config, Mapping) else {}
        water_recall_config = protocol.get("water_recall", {}) if isinstance(protocol.get("water_recall", {}), Mapping) else {}
        bench_recall_config = protocol.get("bench_recall", {}) if isinstance(protocol.get("bench_recall", {}), Mapping) else {}
        passive_match_config = protocol.get("passive_match", {}) if isinstance(protocol.get("passive_match", {}), Mapping) else {}
        state = EpisodeState(
            seed=int(actor["seed"]),
            arm=str(actor["arm"]),
            horizon=horizon,
            water_recall_config=water_recall_config,
            bench_recall_config=bench_recall_config,
            passive_match_config=passive_match_config,
        )
        rows: list[dict[str, Any]] = []
        for tick in range(horizon):
            if current.done:
                break
            actor_observation = fact_surface.apply(current.observation)
            if frozen_actor is None:
                decision = select_action_for_arm(runtime.spec, actor_observation, state, str(actor["arm"]))
            else:
                base_decision = frozen_actor.act(actor_observation)
                decision = select_action_with_actor_sidecar(
                    runtime.spec,
                    actor_observation,
                    state,
                    str(actor["arm"]),
                    base_decision,
                    residual_policy,
                    memory_residual_bias=memory_residual_bias,
                )
            post_step = runtime.step(decision["action"])
            row = build_tick_row(
                actor=actor,
                tick=tick,
                spec=runtime.spec,
                pre_observation=actor_observation,
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
        if frozen_actor is not None:
            frozen_actor.close()
        runtime.close()


def make_runtime_for_horizon(
    runtime_name: str,
    *,
    horizon: int,
    seed: int | None = None,
    deterministic_backend_patch: Any = None,
) -> RuntimeAdapter:
    for kwargs in (
        {"max_steps": horizon, "seed": seed, "deterministic_backend_patch": deterministic_backend_patch},
        {"max_steps": horizon, "seed": seed},
        {"seed": seed, "deterministic_backend_patch": deterministic_backend_patch},
        {"max_steps": horizon},
        {"seed": seed},
        {},
    ):
        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            return make_runtime(runtime_name, **clean_kwargs)
        except TypeError:
            continue
    return make_runtime(runtime_name)


def runtime_is_unavailable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    return "unavailable" in name or "install embryo[crafter]" in str(exc).lower()


def episode_id(*, split: str, seed: int, horizon: int, episode_index: int) -> str:
    return f"{split}:seed-{seed}:horizon-{horizon}:episode-{episode_index}"


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
    fact_surface = dict(config.get("fact_surface", {}))
    if args.fact_surface:
        fact_surface["name"] = args.fact_surface
    if args.fact_writer_checkpoint:
        fact_surface["checkpoint"] = args.fact_writer_checkpoint
    if args.fact_surface_threshold is not None:
        fact_surface["threshold"] = args.fact_surface_threshold
    config["runtime"] = runtime
    config["protocol"] = protocol
    if fact_surface:
        config["fact_surface"] = fact_surface
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
    parser.add_argument("--fact-surface", default="")
    parser.add_argument("--fact-writer-checkpoint", default="")
    parser.add_argument("--fact-surface-threshold", type=float, default=None)
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
