"""Compatibility wrapper for the Crafter runtime.

Prefer `embryo.runtimes.crafter` for new code.
"""

from __future__ import annotations

from embryo.runtimes.crafter import CrafterRuntime, CrafterUnavailable


def make_crafter_env(*, seed: int | None = None) -> CrafterRuntime:
    """Create a Crafter environment if the optional dependency is available."""
    return CrafterRuntime(seed=seed)
