"""Ordered deployable sequence memory for POPGym Autoencode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OrderedSequenceMemory:
    """Agent-owned ordered store for observed Autoencode suits."""

    suits: tuple[int, ...]
    content_hash: str

    @classmethod
    def from_observed_suits(cls, suits: Sequence[int]) -> "OrderedSequenceMemory":
        clean = tuple(int(suit) for suit in suits)
        payload = {"ordered_suits": clean, "provenance": "popgym_autoencode_observation_v0"}
        content_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        return cls(suits=clean, content_hash=content_hash)

    def recall_reverse(self, gap: int) -> int | None:
        idx = len(self.suits) - int(gap)
        if idx < 0 or idx >= len(self.suits):
            return None
        return self.suits[idx]

    def recall_content_corrupt(self, gap: int, *, suit_count: int = 4) -> int | None:
        suit = self.recall_reverse(gap)
        if suit is None:
            return None
        return (int(suit) + 1) % int(suit_count)

    def recall_order_corrupt(self, gap: int) -> int | None:
        idx = int(gap) - 1
        if idx < 0 or idx >= len(self.suits):
            return None
        return self.suits[idx]

    def recall_shuffled(self, gap: int, *, suit_count: int = 4) -> int:
        digest = hashlib.sha1(f"{self.content_hash}:{int(gap)}:shuffled".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % int(suit_count)
