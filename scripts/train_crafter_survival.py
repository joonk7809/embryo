#!/usr/bin/env python
"""Train or inspect the Crafter survival actor."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.train.crafter_survival import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
