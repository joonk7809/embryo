#!/usr/bin/env python
"""Tiny Embryo v1 smoke command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.memory.facts import facing_candidate_fact
from embryo.memory.queries import query_content_hash


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Embryo v1 smoke.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    fact = facing_candidate_fact(True, confidence=1.0)
    scan = scan_actor_context({"raw_rgb_frame": "fixture", "previous_action": "noop"})
    print(json.dumps({"config": config, "query": query_content_hash([fact]), "contamination": scan}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
