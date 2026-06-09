"""Runtime registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from embryo.core.registry import Registry
from embryo.runtimes.base import RuntimeAdapter


_RUNTIMES = Registry()


def register_runtime(name: str) -> Callable[[Callable[..., RuntimeAdapter]], Callable[..., RuntimeAdapter]]:
    return _RUNTIMES.decorator(name)


def make_runtime(name: str, **kwargs: Any) -> RuntimeAdapter:
    factory = _RUNTIMES.get(name)
    return factory(**kwargs)


def runtime_names() -> tuple[str, ...]:
    return _RUNTIMES.names()
