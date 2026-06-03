#!/usr/bin/env python
"""Run public package release-readiness checks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.run.release_check import main

if __name__ == "__main__":
    raise SystemExit(main())
