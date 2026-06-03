from __future__ import annotations

from pathlib import Path
import unittest

from embryo.run.release_check import run_release_checks


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCheckTests(unittest.TestCase):
    def test_release_check_passes_current_public_surface(self) -> None:
        result = run_release_checks(ROOT)

        self.assertTrue(result["passed"], result["failures"])


if __name__ == "__main__":
    unittest.main()
