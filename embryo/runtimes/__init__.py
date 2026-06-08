"""Runtime adapters and registry."""

from embryo.runtimes.base import RuntimeAdapter, RuntimeSpec, RuntimeStep
from embryo.runtimes.registry import make_runtime, register_runtime, runtime_names

# Import built-in runtimes for registry side effects. Optional dependencies are
# loaded only when a runtime is constructed.
from embryo.runtimes import fixture as _fixture  # noqa: F401
from embryo.runtimes import passive_visual_match as _passive_visual_match  # noqa: F401
from embryo.runtimes.crafter import adapter as _crafter_adapter  # noqa: F401

__all__ = [
    "RuntimeAdapter",
    "RuntimeSpec",
    "RuntimeStep",
    "make_runtime",
    "register_runtime",
    "runtime_names",
]
