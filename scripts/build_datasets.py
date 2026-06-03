#!/usr/bin/env python
"""Build supervised datasets from a collected runtime trace."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.datasets.builders import main

if __name__ == "__main__":
    raise SystemExit(main())
