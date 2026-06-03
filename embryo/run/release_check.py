"""Release-readiness checks for the public package surface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence


FORBIDDEN_PUBLIC_TOKENS = ("M" + "6.", "M" + "7.", "m" + "6_", "m" + "7_")
LEGACY_IMPORT_TOKENS = (
    "experiment" + "s.",
    "embryo" + "_arena",
    "run" + "s.",
    "artifact" + "s.",
)
REQUIRED_PATHS = (
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "pyproject.toml",
    "docs/architecture.md",
    "docs/research_boundary.md",
    "docs/runtimes.md",
    "docs/datasets.md",
    "docs/pipelines.md",
    "configs/pipelines/fixture_supervised_memory.yaml",
    ".github/workflows/ci.yml",
)


def run_release_checks(root: str | Path) -> dict[str, object]:
    package_root = Path(root)
    failures: list[dict[str, str]] = []
    for relative in REQUIRED_PATHS:
        if not (package_root / relative).exists():
            failures.append({"kind": "missing_required_path", "path": relative})
    failures.extend(check_public_tokens(package_root))
    failures.extend(check_legacy_imports(package_root))
    return {"passed": not failures, "failure_count": len(failures), "failures": failures}


def check_public_tokens(root: Path) -> list[dict[str, str]]:
    public_paths = [
        root / "README.md",
        root / "CONTRIBUTING.md",
        *(root / "docs").glob("*.md"),
        *(root / "configs").rglob("*.yaml"),
        *(root / "configs").glob("*.md"),
    ]
    failures: list[dict[str, str]] = []
    for path in public_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_PUBLIC_TOKENS:
            if token in text:
                failures.append({"kind": "public_internal_token", "path": str(path.relative_to(root)), "token": token})
    return failures


def check_legacy_imports(root: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for path in iter_python_files(root / "embryo"):
        text = path.read_text(encoding="utf-8")
        for token in LEGACY_IMPORT_TOKENS:
            if token in text:
                failures.append({"kind": "legacy_import_token", "path": str(path.relative_to(root)), "token": token})
    return failures


def iter_python_files(root: Path) -> Iterable[Path]:
    return root.rglob("*.py") if root.exists() else ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Embryo package release readiness.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    result = run_release_checks(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1
