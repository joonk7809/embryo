"""Runtime collection entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from typing import Sequence

from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.runtimes import make_runtime, runtime_names
from embryo.runtimes.base import RuntimeSpec


def collect_runtime_trace(runtime_name: str, *, steps: int = 4, seed: int | None = None) -> list[dict[str, object]]:
    runtime = make_runtime(runtime_name)
    rows: list[dict[str, object]] = []
    try:
        current = runtime.reset(seed=seed)
        for tick in range(steps):
            action = scripted_action(runtime.spec, current.observation)
            scan = scan_actor_context(current.observation)
            rows.append(
                {
                    "runtime": runtime.spec.name,
                    "suite": runtime.spec.suite,
                    "runtime_action_names": list(runtime.spec.action_names),
                    "tick": tick,
                    "action": action,
                    "observation": current.observation,
                    "reward": current.reward,
                    "done": current.done,
                    "actor_contamination": scan,
                }
            )
            if current.done:
                break
            current = runtime.step(action)
    finally:
        runtime.close()
    return rows


def scripted_action(spec: RuntimeSpec, observation: Mapping[str, object]) -> str:
    if float(observation.get("candidate_score", 0.0)) >= 0.5:
        return spec.resource_action
    if bool(observation.get("failed_action_event", False)):
        return spec.fallback_action
    return spec.noop_action


def collect_feature_rows(trace_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize runtime traces into replay/dataset-friendly feature rows."""
    rows: list[dict[str, Any]] = []
    for row in trace_rows:
        obs = row.get("observation", {})
        observation = obs if isinstance(obs, Mapping) else {}
        candidate_score = float(observation.get("candidate_score", row.get("candidate_score", 0.0)))
        action = str(row.get("action", observation.get("previous_action", "noop")))
        rows.append(
            {
                "runtime": row.get("runtime", "unknown_runtime"),
                "suite": row.get("suite", "unknown_suite"),
                "episode_id": str(row.get("episode_id", "collected_episode")),
                "episode_index": int(row.get("episode_index", 0)),
                "tick": int(row.get("tick", 0)),
                "raw_rgb_frame": observation.get("raw_rgb_frame"),
                "previous_rgb_frame": observation.get("previous_rgb_frame"),
                "previous_action": observation.get("previous_action", "noop"),
                "candidate_score": candidate_score,
                "failed_action_event": bool(observation.get("failed_action_event", False)),
                "resource_memory_critical": bool(observation.get("resource_memory_critical", candidate_score >= 0.5)),
                "target_action": action,
                "diagnostic_progress_delta_teacher_only": float(row.get("reward") or 0.0),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Collect a tiny runtime trace.")
    parser.add_argument("--runtime", default="fixture_memory", choices=runtime_names())
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(root / "runs" / "runtime_collect" / "trace.jsonl"))
    parser.add_argument("--features-out", default="")
    args = parser.parse_args(argv)
    rows = collect_runtime_trace(args.runtime, steps=args.steps, seed=args.seed)
    write_jsonl(args.out, rows)
    feature_rows = []
    if args.features_out:
        feature_rows = collect_feature_rows(rows)
        write_jsonl(args.features_out, feature_rows)
    failures = sum(1 for row in rows if not row["actor_contamination"]["passed"])
    print(
        json.dumps(
            {
                "runtime": args.runtime,
                "out": args.out,
                "rows": len(rows),
                "features_out": args.features_out,
                "feature_rows": len(feature_rows),
                "contamination_failures": failures,
            },
            sort_keys=True,
        )
    )
    return 0
