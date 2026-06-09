"""Small associative memory modules for POPGym sequence recall."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from embryo.models.popgym_recurrent import require_torch


class ExplicitPositionDeltaMemoryPolicy:
    """Factory for an explicit-position delta-rule memory."""

    @staticmethod
    def build(*, max_positions: int, suit_count: int, value_dim: int, write_eta: float = 1.0):
        torch, nn = require_torch()
        max_positions = int(max_positions)
        suit_count = int(suit_count)
        value_dim = int(value_dim)
        write_eta = float(write_eta)

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.max_positions = max_positions
                self.suit_count = suit_count
                self.value_dim = value_dim
                self.write_eta = write_eta
                self.suit_value = nn.Embedding(suit_count, value_dim)
                self.read_head = nn.Linear(value_dim, suit_count)

            def forward(self, watch_suits, read_positions, write_suits=None):  # noqa: ANN001
                watch_suits = watch_suits.long()
                read_positions = read_positions.long()
                if write_suits is None:
                    write_suits = watch_suits
                write_suits = write_suits.long()
                if int(watch_suits.shape[1]) > self.max_positions:
                    raise ValueError(f"watch length {int(watch_suits.shape[1])} exceeds max_positions={self.max_positions}")
                if bool((read_positions < 0).any() or (read_positions >= self.max_positions).any()):
                    raise ValueError("read_positions must be within max_positions")

                batch_size = int(watch_suits.shape[0])
                slots = []
                for position in range(int(watch_suits.shape[1])):
                    value = self.suit_value(write_suits[:, position])
                    slots.append(self.write_eta * value)
                memory = torch.stack(slots, dim=1)
                if int(memory.shape[1]) < self.max_positions:
                    pad = memory.new_zeros((batch_size, self.max_positions - int(memory.shape[1]), self.value_dim))
                    memory = torch.cat([memory, pad], dim=1)
                expanded = read_positions.unsqueeze(-1).expand(-1, -1, self.value_dim)
                read_values = memory.gather(1, expanded)
                return self.read_head(read_values)

        return _Policy()


def order_corrupt_read_positions(*, gaps: Any):
    torch, _ = require_torch()
    return torch.as_tensor(gaps, dtype=torch.long) - 1


def content_corrupt_suits(watch_suits: Any, *, suit_count: int):
    return (watch_suits.long() + 1) % int(suit_count)


def shuffled_binding_suits(watch_suits: Any):
    return watch_suits.long().roll(shifts=1, dims=1)


def save_popgym_associative_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: Any,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    torch, _ = require_torch()
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    weights = root / "model.pt"
    manifest_path = root / "manifest.json"
    torch.save(model.state_dict(), weights)
    manifest_path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"weights": str(weights), "manifest": str(manifest_path)}
