from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import sys


class DeltaIrisUnavailable(RuntimeError):
    """Raised when the external Delta-IRIS actor cannot be loaded."""


def add_delta_iris_src(repo_path:str | Path) -> Path:
    src = Path(repo_path).resolve() / "src"
    if not src.exists():
        raise FileNotFoundError(f"Delta-IRIS src directory not found: {src}")

    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    return src

def import_delta_iris_modules(repo_path: str | Path):
    add_delta_iris_src(repo_path)

    from agent import Agent
    from models.tokenizer import Tokenizer
    from models.world_model import WorldModel
    from models.actor_critic import ActorCritic

    return Agent, Tokenizer, WorldModel, ActorCritic


def load_delta_iris_config(repo_path: str | Path):
    """Load the Delta-IRIS trainer config without constructing a model."""
    config_dir = Path(repo_path).resolve() / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Delta-IRIS config directory not found: {config_dir}")

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except Exception as exc:
        raise DeltaIrisUnavailable("Install Delta-IRIS config dependencies to load its actor config.") from exc

    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    except TypeError:
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass

    try:
        context = initialize_config_dir(config_dir=str(config_dir), version_base=None)
    except TypeError:
        context = initialize_config_dir(config_dir=str(config_dir))

    with context:
        return compose(
            config_name="trainer",
            overrides=[
                "+mode=agent_in_env",
                "+fps=5",
                "+header=0",
                "+save_mode=0",
                "wandb.mode=disabled",
            ],
        )

class DeltaIrisActor:

    def __init__(self, *, repo_path, checkpoint_path, action_names, policy_mode="argmax", device="cpu") -> None:


        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.action_names = tuple(action_names)
        self.policy_mode = policy_mode
        self.device = device
        self.Agent, self.Tokenizer, self.WorldModel, self.ActorCritic = import_delta_iris_modules(self.repo_path)
        self.agent = self._load_agent()

    def _load_config(self):
        return load_delta_iris_config(self.repo_path)

    def _load_agent(self):

        from hydra.utils import instantiate

        cfg = self._load_config()

        cfg.params.tokenizer.num_actions = len(self.action_names)
        cfg.params.world_model.num_actions = len(self.action_names)
        cfg.params.actor_critic.model.num_actions = len(self.action_names)

        tok_cfg = cfg.params.tokenizer
        wm_cfg = cfg.params.world_model
        ac_cfg = cfg.params.actor_critic

        tok_config_object = instantiate(tok_cfg)
        wm_config_object = instantiate(wm_cfg)
        ac_config_object = instantiate(ac_cfg)

        tokenizer = self.Tokenizer(tok_config_object)
        world_model = self.WorldModel(wm_config_object)
        actor_critic = self.ActorCritic(ac_config_object)

        agent = self.Agent(tokenizer, world_model, actor_critic).to(self.device)
        agent.load(self.checkpoint_path, device=self.device, strict=False)
        agent.eval()
        return agent

    def reset(self, *, seed = None):
        if seed is not None:
            import torch

            torch.manual_seed(int(seed))
        self.agent.actor_critic.reset(n=1)

    def _observation_tensor(self, observation):
        import torch
        from einops import rearrange

        frame = observation["raw_rgb_frame"]

        return rearrange(
            torch.FloatTensor(frame).div(255),
            "h w c -> 1 c h w",
        ).to(self.device)

    def act(self, observation):
        import torch
        from torch.distributions.categorical import Categorical

        from embryo.models.actors import ActorDecision

        obs = self._observation_tensor(observation)
        actor_critic = self.agent.actor_critic
        if actor_critic._past_obs is None:
            self.reset()

        with torch.no_grad():
            actor_critic._past_obs.append(obs)
            inputs = actor_critic.model.build_present_input_from_past(actor_critic.past_obs)
            outputs = actor_critic.model(inputs)
            logits_tensor = outputs.logits_actions[:, -1]
            if self.policy_mode in {"sampled", "sample", "sample_temperature_1_seeded"}:
                action_token = Categorical(logits=logits_tensor).sample()
            elif self.policy_mode == "argmax":
                action_token = logits_tensor.argmax(dim=-1)
            else:
                raise ValueError(f"Unknown Delta-IRIS policy_mode: {self.policy_mode}")

        action_index = int(action_token.detach().cpu().reshape(-1)[0])
        logits_values = logits_tensor[0].detach().cpu().tolist()
        if len(logits_values) != len(self.action_names):
            raise ValueError(f"Delta-IRIS emitted {len(logits_values)} logits for {len(self.action_names)} actions")
        return ActorDecision(
            action=str(self.action_names[action_index]),
            logits={action: float(logits_values[index]) for index, action in enumerate(self.action_names)},
            metadata={"actor": "delta_iris", "policy_mode": self.policy_mode},
        )

    def close(self) -> None:
        return None
