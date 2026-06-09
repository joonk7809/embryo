#!/usr/bin/env python
"""Run the POPGym CountRecall injection probe."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.run.probe_popgym_count_recall_injection import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
