"""Tiny recurrent actor for POPGym RepeatFirst."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embryo.models.actors import ActorDecision, action_from_logits


def require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install torch to use the POPGym recurrent actor.") from exc
    return torch, nn


class RepeatFirstGRUPolicy:
    """Factory wrapper for the torch GRU module."""

    @staticmethod
    def build(*, action_count: int, embedding_dim: int, hidden_size: int, token_count: int | None = None):
        torch, nn = require_torch()
        vocab_size = int(action_count if token_count is None else token_count)

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, int(embedding_dim))
                self.gru = nn.GRU(int(embedding_dim), int(hidden_size), batch_first=True)
                self.head = nn.Linear(int(hidden_size), int(action_count))

            def forward(self, tokens, hidden=None):  # noqa: ANN001
                embedded = self.embedding(tokens.long())
                output, next_hidden = self.gru(embedded, hidden)
                return self.head(output), next_hidden

        return _Policy()


class VectorObservationGRUPolicy:
    """Factory for recurrent policies over fixed-width discrete observation vectors."""

    @staticmethod
    def build(
        *,
        action_count: int,
        obs_dim: int,
        obs_value_count: int,
        obs_embedding_dim: int,
        prev_action_embedding_dim: int,
        hidden_size: int,
    ):
        torch, nn = require_torch()
        action_count = int(action_count)
        obs_dim = int(obs_dim)

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_count = action_count
                self.obs_dim = obs_dim
                self.obs_embedding = nn.Embedding(int(obs_value_count), int(obs_embedding_dim))
                self.prev_action_embedding = nn.Embedding(action_count + 1, int(prev_action_embedding_dim))
                self.input = nn.Linear(obs_dim * int(obs_embedding_dim) + int(prev_action_embedding_dim), int(hidden_size))
                self.gru = nn.GRU(int(hidden_size), int(hidden_size), batch_first=True)
                self.head = nn.Linear(int(hidden_size), action_count)

            def forward(self, observations, previous_actions, hidden=None):  # noqa: ANN001
                obs = observations.long()
                prev = previous_actions.long()
                embedded_obs = self.obs_embedding(obs).reshape(obs.shape[0], obs.shape[1], -1)
                embedded_prev = self.prev_action_embedding(prev)
                features = torch.relu(self.input(torch.cat([embedded_obs, embedded_prev], dim=-1)))
                output, next_hidden = self.gru(features, hidden)
                return self.head(output), next_hidden

        return _Policy()


class RepeatFirstRecurrentActor:
    """Frozen actor wrapper used by long-run replay."""

    def __init__(self, *, checkpoint_path: str | Path, action_names: Sequence[str], device: str = "cpu") -> None:
        torch, _ = require_torch()
        self.torch = torch
        self.device = torch.device(device)
        checkpoint = load_popgym_recurrent_checkpoint(checkpoint_path, device=self.device)
        manifest = checkpoint["manifest"]
        self.action_names = tuple(str(action) for action in action_names)
        self.model = RepeatFirstGRUPolicy.build(
            action_count=int(manifest["action_count"]),
            embedding_dim=int(manifest["embedding_dim"]),
            hidden_size=int(manifest["hidden_size"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.hidden = None

    def reset(self, *, seed: int | None = None) -> None:
        _ = seed
        self.hidden = None

    def act(self, observation: Mapping[str, Any]) -> ActorDecision:
        value = observation.get("popgym_observation", 0)
        if hasattr(value, "item"):
            value = value.item()
        token = self.torch.tensor([[int(value)]], dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            logits_tensor, self.hidden = self.model(token, self.hidden)
        values = logits_tensor[0, -1].detach().cpu().tolist()
        logits = {action: float(values[idx]) for idx, action in enumerate(self.action_names) if idx < len(values)}
        action = action_from_logits(logits, action_order=self.action_names, fallback=self.action_names[0])
        return ActorDecision(action=action, logits=logits, metadata={"actor": "popgym_recurrent"})

    def close(self) -> None:
        self.hidden = None


def save_popgym_recurrent_checkpoint(
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


def load_popgym_recurrent_checkpoint(checkpoint_path: str | Path, *, device: Any = "cpu") -> dict[str, Any]:
    torch, _ = require_torch()
    root = Path(checkpoint_path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing POPGym recurrent manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = manifest_path.parent / str(manifest.get("weights", "model.pt"))
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    return {"manifest": manifest, "state_dict": state_dict}
