#!/usr/bin/env python
"""Train the POPGym Autoencode recurrent baseline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embryo.run.train_popgym_autoencode import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
