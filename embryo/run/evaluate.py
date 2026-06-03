"""Trace evaluation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embryo.eval.guardrails import event_repetition_guardrail, resource_binding_guardrail
from embryo.eval.memory_grounded_score import ScoreConfig, compute_memory_grounded_scores, compute_window_sensitivity
from embryo.eval.traces import read_jsonl


def evaluate_trace(path: str | Path) -> dict[str, object]:
    ticks = read_jsonl(path)
    score = compute_memory_grounded_scores(ticks, config=ScoreConfig())
    return {
        "trace": str(path),
        "tick_count": len(ticks),
        "score": score,
        "window_sensitivity": compute_window_sensitivity(ticks, config=ScoreConfig()),
        "resource_binding_guardrail": resource_binding_guardrail(ticks),
        "event_repetition_guardrail": event_repetition_guardrail(ticks),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate a replay trace.")
    parser.add_argument("--trace", default=str(root / "data" / "fixtures" / "tiny_memory_score_trace.jsonl"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = evaluate_trace(args.trace)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(encoded, encoding="utf-8")
    print(json.dumps({"trace": args.trace, "tick_count": result["tick_count"], "best_arm": result["score"].get("best_arm")}, sort_keys=True))
    return 0
