"""Fresh-process deterministic replay checks for long-run protocols."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.eval.long_run import deterministic_replay_summary_from_digest


def fresh_process_deterministic_replay_summary(
    manifest: Mapping[str, Any],
    primary_ticks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    worker = (
        "import json, sys\n"
        "from embryo.eval.long_run import replay_digest\n"
        "from embryo.run.long_run import collect_long_run_pass\n"
        "manifest = json.load(sys.stdin)\n"
        "result = collect_long_run_pass(manifest)\n"
        "print(json.dumps({"
        "'digest': replay_digest(result['ticks']), "
        "'tick_count': len(result['ticks']), "
        "'runtime_unavailable': result['runtime_unavailable']"
        "}, sort_keys=True))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [sys.executable, "-c", worker],
        input=json.dumps(manifest, sort_keys=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(root),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        return failed_replay_summary(primary_ticks, completed.stderr.strip() or f"worker exited with {completed.returncode}")
    try:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {}
    except (IndexError, json.JSONDecodeError) as exc:
        return failed_replay_summary(primary_ticks, f"worker output parse failure: {exc}")
    if payload.get("runtime_unavailable"):
        return failed_replay_summary(primary_ticks, str(payload["runtime_unavailable"]))
    return deterministic_replay_summary_from_digest(
        primary_ticks,
        repeat_digest=str(payload.get("digest", "")),
        repeat_tick_count=int(payload.get("tick_count", 0)),
    )


def failed_replay_summary(primary_ticks: Sequence[Mapping[str, Any]], error: str) -> dict[str, Any]:
    return deterministic_replay_summary_from_digest(primary_ticks, repeat_digest="", repeat_tick_count=0, error=error)
