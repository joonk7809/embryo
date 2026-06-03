#!/usr/bin/env python
"""Run the reference supervised BC policy trainer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.run.train_supervised import main

if __name__ == "__main__":
    raise SystemExit(main(["--module", "bc_policy", *sys.argv[1:]]))
