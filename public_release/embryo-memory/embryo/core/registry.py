"""Small component registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """Map string names to callables or classes."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str, item: Any) -> Any:
        if name in self._items:
            raise KeyError(f"Duplicate registry item: {name}")
        self._items[name] = item
        return item

    def decorator(self, name: str) -> Callable[[Any], Any]:
        def _wrap(item: Any) -> Any:
            return self.register(name, item)

        return _wrap

    def get(self, name: str) -> Any:
        return self._items[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
