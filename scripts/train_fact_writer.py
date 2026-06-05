#!/usr/bin/env python
"""Train the offline fact-writer v0 model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.run.train_fact_writer import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
