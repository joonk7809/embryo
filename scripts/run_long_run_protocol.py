#!/usr/bin/env python
"""Run the long-run memory protocol."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.run.long_run import main


if __name__ == "__main__":
    raise SystemExit(main())
