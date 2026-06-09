"""Manifest normalization for the long-run protocol runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embryo.run.long_run_actors import actor_manifest, normalize_actor_config
from embryo.run.long_run_arms import DEFAULT_ARMS
from embryo.run.long_run_fact_surface import REFERENCE_RGB_SCAFFOLD


CRAFTER_BACKEND_DETERMINISM_PATCH = "crafter_balance_object_order_v1"


def resolve_protocol_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config.get("runtime", {})) if isinstance(config.get("runtime", {}), Mapping) else {}
    protocol = dict(config.get("protocol", {})) if isinstance(config.get("protocol", {}), Mapping) else {}
    metrics = dict(config.get("metrics", {})) if isinstance(config.get("metrics", {}), Mapping) else {}
    fact_surface = normalize_fact_surface_config(config.get("fact_surface", {}))
    actor = normalize_actor_config(config.get("actor", {}))
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
    runtime.update(normalize_runtime_determinism_config(runtime))
    protocol.update(
        {
            "horizons": [int(value) for value in protocol.get("horizons", (256, 2048))],
            "max_episodes_per_seed": int(protocol.get("max_episodes_per_seed", 1)),
            "detail_ticks": bool(protocol.get("detail_ticks", False)),
            "determinism_check": True,
            "min_evaluable_anchor_count": int(protocol.get("min_evaluable_anchor_count", 1)),
            "min_evaluable_seed_rate": float(protocol.get("min_evaluable_seed_rate", 0.5)),
        }
    )
    arms = tuple(str(arm) for arm in config.get("arms", DEFAULT_ARMS))
    return {
        "runtime": runtime,
        "protocol": protocol,
        "arms": list(arms),
        "fact_surface": fact_surface,
        "actor": actor_manifest(actor),
        "metrics": {
            "survival": bool(metrics.get("survival", True)),
            "valid_actions": bool(metrics.get("valid_actions", True)),
            "loop_rate": bool(metrics.get("loop_rate", True)),
            "event_self_trigger": bool(metrics.get("event_self_trigger", True)),
            "memory_grounded_score": bool(metrics.get("memory_grounded_score", True)),
            "passive_match": bool(metrics.get("passive_match", False)),
            "popgym_repeat_first": bool(metrics.get("popgym_repeat_first", False)),
            "water_recall": bool(metrics.get("water_recall", False)),
            "bench_recall": bool(metrics.get("bench_recall", False)),
            "contamination": bool(metrics.get("contamination", True)),
        },
    }


def normalize_fact_surface_config(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    name = str(raw.get("name", REFERENCE_RGB_SCAFFOLD))
    result: dict[str, Any] = {"name": name}
    if raw.get("checkpoint"):
        result["checkpoint"] = str(raw["checkpoint"])
    if name == "learned_fact_writer_v0" or raw.get("threshold") is not None:
        result["threshold"] = float(raw.get("threshold", 0.5))
    return result


def normalize_runtime_determinism_config(runtime: Mapping[str, Any]) -> dict[str, Any]:
    runtime_name = str(runtime.get("name", ""))
    default_enabled = runtime_name == "crafter_memory"
    enabled = runtime_name == "crafter_memory" and config_bool(runtime.get("deterministic_backend_patch", default_enabled))
    patch = CRAFTER_BACKEND_DETERMINISM_PATCH if runtime_name == "crafter_memory" and enabled else None
    return {
        "deterministic_backend_patch": enabled,
        "backend_determinism_patch": patch,
    }


def config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "none", "disabled"}
    return bool(value)
