"""Crafter survival-agent training lane."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embryo.core.config import load_config
from embryo.datasets.crafter_human import iter_demo_transitions, summarize_human_dataset
from embryo.runtimes.crafter.gym_env import CrafterTrainingUnavailable, make_crafter_survival_env


READY_CRAFTER_SURVIVAL_TRAINING = "READY_crafter_survival_training"
NOT_READY_MISSING_TRAINING_DEPENDENCY = "NOT_READY_missing_training_dependency"
GO_CRAFTER_SURVIVAL_PPO_TRAINED = "GO_crafter_survival_ppo_trained"
GO_CRAFTER_SURVIVAL_BC_TRAINED = "GO_crafter_survival_bc_trained"
GO_CRAFTER_SURVIVAL_EVAL_WRITTEN = "GO_crafter_survival_eval_written"
NOT_EVALUABLE_CHECKPOINT_MISSING = "NOT_EVALUABLE_checkpoint_missing"
NOT_EVALUABLE_HUMAN_DATASET_EMPTY = "NOT_EVALUABLE_human_dataset_empty"


def run_setup_check(config: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    status = dependency_status()
    missing = [name for name, row in status.items() if not bool(row.get("available", False))]
    dataset_cfg = mapping(config.get("dataset"))
    human_root = dataset_cfg.get("human_root")
    human_summary = None
    if human_root:
        human_summary = summarize_human_dataset(human_root, max_examples=optional_int(dataset_cfg.get("max_examples")))
    summary = {
        "decision": NOT_READY_MISSING_TRAINING_DEPENDENCY if missing else READY_CRAFTER_SURVIVAL_TRAINING,
        "missing_dependencies": missing,
        "dependencies": status,
        "human_dataset": human_summary,
        "training_boundary": "crafter_survival_actor_no_memory",
    }
    paths = {"setup_status": root / "setup_status.json"}
    paths["setup_status"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "paths": stringify_paths(paths)}


def inspect_human_dataset(config: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    dataset_cfg = mapping(config.get("dataset"))
    summary = summarize_human_dataset(
        dataset_cfg.get("human_root", "runs/data/crafter_human_dataset"),
        max_examples=optional_int(dataset_cfg.get("max_examples")),
    )
    paths = {"human_dataset_manifest": root / "human_dataset_manifest.json"}
    paths["human_dataset_manifest"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "paths": stringify_paths(paths)}


def train_bc(config: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    dataset_cfg = mapping(config.get("dataset"))
    bc_cfg = mapping(config.get("bc"))
    ppo_cfg = mapping(config.get("ppo"))
    frames, actions, dataset_summary = load_bc_arrays(
        dataset_cfg.get("human_root", "runs/data/crafter_human_dataset"),
        max_examples=optional_int(dataset_cfg.get("max_examples")),
    )
    if len(actions) == 0:
        summary = {
            "decision": NOT_EVALUABLE_HUMAN_DATASET_EMPTY,
            "dataset": dataset_summary,
            "training_boundary": "crafter_survival_actor_bc_from_rgb_actions",
        }
        paths = {"bc_summary": root / "bc_summary.json"}
        paths["bc_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"summary": summary, "paths": stringify_paths(paths)}

    model_class = load_ppo_class(str(ppo_cfg.get("algorithm", "recurrent_ppo")))
    env = make_env_from_config(config, seed=int(bc_cfg.get("seed", ppo_cfg.get("seed", 10000))))
    try:
        model = model_class(
            str(ppo_cfg.get("policy", default_policy_name(str(ppo_cfg.get("algorithm", "recurrent_ppo"))))),
            env,
            learning_rate=float(ppo_cfg.get("learning_rate", 2.5e-4)),
            n_steps=int(ppo_cfg.get("n_steps", 2048)),
            batch_size=int(ppo_cfg.get("batch_size", 256)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
            verbose=0,
        )
        train_indices, dev_indices = bc_split_indices(
            len(actions),
            validation_fraction=float(bc_cfg.get("validation_fraction", 0.1)),
            seed=int(bc_cfg.get("seed", ppo_cfg.get("seed", 10000))),
        )
        history = train_policy_by_bc(
            model,
            frames,
            actions,
            train_indices=train_indices,
            epochs=int(bc_cfg.get("epochs", 5)),
            batch_size=int(bc_cfg.get("batch_size", 128)),
            learning_rate=float(bc_cfg.get("learning_rate", 3e-4)),
            seed=int(bc_cfg.get("seed", ppo_cfg.get("seed", 10000))),
        )
        metrics = {
            "train": bc_action_metrics(model, frames, actions, train_indices, batch_size=int(bc_cfg.get("batch_size", 128))),
            "dev": bc_action_metrics(model, frames, actions, dev_indices, batch_size=int(bc_cfg.get("batch_size", 128))),
        }
        model_path = root / "model.zip"
        model.save(str(model_path))
    finally:
        env.close()

    summary = {
        "decision": GO_CRAFTER_SURVIVAL_BC_TRAINED,
        "algorithm": str(ppo_cfg.get("algorithm", "recurrent_ppo")),
        "policy": str(ppo_cfg.get("policy", default_policy_name(str(ppo_cfg.get("algorithm", "recurrent_ppo"))))),
        "dataset": dataset_summary,
        "metrics": metrics,
        "history": history,
        "training_boundary": "crafter_survival_actor_bc_from_rgb_actions",
    }
    paths = {"model": root / "model.zip", "bc_summary": root / "bc_summary.json"}
    paths["bc_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "paths": stringify_paths(paths)}


def train_ppo(config: Mapping[str, Any], out: str | Path, checkpoint: str | Path = "") -> dict[str, Any]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    ppo_cfg = mapping(config.get("ppo"))
    model_class = load_ppo_class(str(ppo_cfg.get("algorithm", "recurrent_ppo")))
    env = make_env_from_config(config, seed=int(ppo_cfg.get("seed", 10000)))
    try:
        kwargs = {
            "learning_rate": float(ppo_cfg.get("learning_rate", 2.5e-4)),
            "n_steps": int(ppo_cfg.get("n_steps", 2048)),
            "batch_size": int(ppo_cfg.get("batch_size", 256)),
            "gamma": float(ppo_cfg.get("gamma", 0.99)),
            "gae_lambda": float(ppo_cfg.get("gae_lambda", 0.95)),
            "clip_range": float(ppo_cfg.get("clip_range", 0.2)),
            "ent_coef": float(ppo_cfg.get("ent_coef", 0.01)),
            "verbose": 1,
        }
        if ppo_cfg.get("tensorboard_log"):
            kwargs["tensorboard_log"] = str(root / "tb")
        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path and checkpoint_path.exists():
            model = model_class.load(str(checkpoint_path), env=env)
        else:
            model = model_class(
                str(ppo_cfg.get("policy", default_policy_name(str(ppo_cfg.get("algorithm", "recurrent_ppo"))))),
                env,
                **kwargs,
            )
        model.learn(total_timesteps=int(float(ppo_cfg.get("total_timesteps", 1_000_000))))
        model_path = root / "model.zip"
        model.save(str(model_path))
    finally:
        env.close()
    summary = {
        "decision": GO_CRAFTER_SURVIVAL_PPO_TRAINED,
        "algorithm": str(ppo_cfg.get("algorithm", "recurrent_ppo")),
        "policy": str(ppo_cfg.get("policy", default_policy_name(str(ppo_cfg.get("algorithm", "recurrent_ppo"))))),
        "total_timesteps": int(float(ppo_cfg.get("total_timesteps", 1_000_000))),
        "init_checkpoint": str(checkpoint) if checkpoint else None,
        "training_boundary": "crafter_survival_actor_no_memory",
    }
    paths = {"model": root / "model.zip", "train_summary": root / "train_summary.json"}
    paths["train_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "paths": stringify_paths(paths)}


def load_bc_arrays(root: str | Path, *, max_examples: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary = summarize_human_dataset(root, max_examples=max_examples)
    count = int(summary.get("transition_count", 0))
    frame_shape = tuple(summary.get("frame_shape") or ())
    if count == 0 or not frame_shape:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64), summary
    frames = np.empty((count, *frame_shape), dtype=np.uint8)
    actions = np.empty((count,), dtype=np.int64)
    for index, row in enumerate(iter_demo_transitions(root, max_examples=max_examples)):
        frames[index] = np.asarray(row["raw_rgb_frame"], dtype=np.uint8)
        actions[index] = int(row["action_index"])
    return frames, actions, summary


def bc_split_indices(count: int, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(count), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    if len(indices) < 2 or validation_fraction <= 0:
        return indices, np.empty((0,), dtype=np.int64)
    dev_count = max(1, int(round(len(indices) * float(validation_fraction))))
    dev_count = min(dev_count, len(indices) - 1)
    return indices[dev_count:], indices[:dev_count]


def train_policy_by_bc(
    model: Any,
    frames: np.ndarray,
    actions: np.ndarray,
    *,
    train_indices: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch

    rng = np.random.default_rng(int(seed))
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(learning_rate))
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        model.policy.train()
        epoch_indices = np.array(train_indices, copy=True)
        rng.shuffle(epoch_indices)
        losses: list[float] = []
        accuracies: list[float] = []
        for batch_indices in batched_indices(epoch_indices, int(batch_size)):
            log_prob, entropy = policy_log_prob_entropy(model.policy, frames[batch_indices], actions[batch_indices])
            loss = -log_prob.mean() - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            accuracies.append(policy_batch_accuracy(model.policy, frames[batch_indices], actions[batch_indices]))
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": round(mean(losses), 6),
                "mean_batch_accuracy": round(mean(accuracies), 6),
            }
        )
    return history


def bc_action_metrics(model: Any, frames: np.ndarray, actions: np.ndarray, indices: np.ndarray, *, batch_size: int) -> dict[str, Any]:
    import torch

    if len(indices) == 0:
        return {"example_count": 0, "accuracy": 0.0, "negative_log_likelihood": 0.0, "label_action_histogram": {}, "predicted_action_histogram": {}}
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    for batch_indices in batched_indices(indices, int(batch_size)):
        with torch.no_grad():
            log_prob, _entropy = policy_log_prob_entropy(model.policy, frames[batch_indices], actions[batch_indices])
            prediction = policy_predict_actions(model.policy, frames[batch_indices])
        losses.extend((-log_prob.detach().cpu().numpy()).astype(float).tolist())
        predictions.extend(int(value) for value in prediction)
        labels.extend(int(value) for value in actions[batch_indices])
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return {
        "example_count": len(labels),
        "accuracy": round(correct / len(labels), 6) if labels else 0.0,
        "negative_log_likelihood": round(mean(losses), 6),
        "label_action_histogram": histogram(labels),
        "predicted_action_histogram": histogram(predictions),
    }


def policy_log_prob_entropy(policy: Any, observations: np.ndarray, actions: np.ndarray):
    import torch

    obs_tensor, _ = policy.obs_to_tensor(observations)
    action_tensor = torch.as_tensor(actions, device=policy.device, dtype=torch.long)
    if hasattr(policy, "lstm_hidden_state_shape"):
        states = zero_recurrent_states(policy, len(actions))
        episode_starts = torch.ones((len(actions),), device=policy.device)
        _values, log_prob, entropy = policy.evaluate_actions(obs_tensor, action_tensor, states, episode_starts)
        return log_prob, entropy
    _values, log_prob, entropy = policy.evaluate_actions(obs_tensor, action_tensor)
    return log_prob, entropy


def policy_predict_actions(policy: Any, observations: np.ndarray) -> list[int]:
    import torch

    with torch.no_grad():
        obs_tensor, _ = policy.obs_to_tensor(observations)
        if hasattr(policy, "lstm_hidden_state_shape"):
            states = zero_recurrent_states(policy, len(observations))
            episode_starts = torch.ones((len(observations),), device=policy.device)
            distribution, _new_states = policy.get_distribution(obs_tensor, states.pi, episode_starts)
        else:
            distribution = policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs
        return [int(value) for value in probs.argmax(dim=1).detach().cpu().numpy()]


def policy_batch_accuracy(policy: Any, observations: np.ndarray, actions: np.ndarray) -> float:
    predictions = policy_predict_actions(policy, observations)
    if not predictions:
        return 0.0
    return sum(1 for pred, label in zip(predictions, actions) if int(pred) == int(label)) / len(predictions)


def zero_recurrent_states(policy: Any, batch_size: int):
    import torch
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    shape = (int(policy.lstm_hidden_state_shape[0]), int(batch_size), int(policy.lstm_hidden_state_shape[2]))
    pi_hidden = torch.zeros(shape, device=policy.device)
    pi_cell = torch.zeros(shape, device=policy.device)
    vf_hidden = torch.zeros(shape, device=policy.device)
    vf_cell = torch.zeros(shape, device=policy.device)
    return RNNStates((pi_hidden, pi_cell), (vf_hidden, vf_cell))


def batched_indices(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def evaluate_checkpoint(config: Mapping[str, Any], out: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        summary = {"decision": NOT_EVALUABLE_CHECKPOINT_MISSING, "checkpoint": str(checkpoint_path)}
        paths = {"eval_summary": root / "eval_summary.json"}
        paths["eval_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"summary": summary, "paths": stringify_paths(paths)}

    ppo_cfg = mapping(config.get("ppo"))
    model_class = load_ppo_class(str(ppo_cfg.get("algorithm", "recurrent_ppo")))
    model = model_class.load(str(checkpoint_path))
    eval_cfg = mapping(config.get("eval"))
    seed_start = int(eval_cfg.get("seed_start", 11000))
    seed_count = int(eval_cfg.get("seed_count", 16))
    horizon = int(eval_cfg.get("horizon", 2048))
    episodes = [rollout_model(model, config, seed=seed_start + offset, horizon=horizon) for offset in range(seed_count)]
    summary = summarize_rollouts(episodes)
    summary.update({"decision": GO_CRAFTER_SURVIVAL_EVAL_WRITTEN, "checkpoint": str(checkpoint_path)})
    paths = {"eval_summary": root / "eval_summary.json", "eval_rollouts": root / "eval_rollouts.jsonl"}
    paths["eval_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["eval_rollouts"].write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes), encoding="utf-8")
    return {"summary": summary, "paths": stringify_paths(paths)}


def rollout_model(model: Any, config: Mapping[str, Any], *, seed: int, horizon: int) -> dict[str, Any]:
    env = make_env_from_config(config, seed=seed)
    try:
        observation, _ = env.reset(seed=seed)
        state = None
        episode_start = np.ones((1,), dtype=bool)
        actions: list[int] = []
        rewards: list[float] = []
        achievements: set[str] = set()
        done = False
        for _tick in range(horizon):
            action, state = model.predict(observation, state=state, episode_start=episode_start, deterministic=True)
            episode_start[:] = False
            action_index = int(np.asarray(action).reshape(-1)[0])
            observation, reward, terminated, truncated, info = env.step(action_index)
            done = bool(terminated or truncated)
            actions.append(action_index)
            rewards.append(float(reward))
            achievements.update(achievement_names(info))
            if done:
                break
        return {
            "seed": int(seed),
            "survival_steps": len(actions),
            "done": done,
            "reward_sum_eval_only": round(float(sum(rewards)), 6),
            "unique_achievements_eval_only": sorted(achievements),
            "achievement_count_eval_only": len(achievements),
            "loop_rate": repeated_action_loop_rate(actions),
            "action_entropy": action_entropy(actions),
        }
    finally:
        env.close()


def make_env_from_config(config: Mapping[str, Any], *, seed: int | None = None):
    runtime_cfg = mapping(config.get("runtime"))
    ppo_cfg = mapping(config.get("ppo"))
    return make_crafter_survival_env(
        seed=seed,
        deterministic_backend_patch=config_bool(runtime_cfg.get("deterministic_backend_patch", True)),
        frame_size=runtime_cfg.get("frame_size", (64, 64)),
        max_episode_steps=optional_int(runtime_cfg.get("max_episode_steps")),
        reward_mode=str(ppo_cfg.get("reward_mode", "native")),
        alive_bonus=float(ppo_cfg.get("alive_bonus", 0.0)),
    )


def dependency_status() -> dict[str, dict[str, Any]]:
    names = ("crafter", "gymnasium", "stable_baselines3", "sb3_contrib", "torch")
    return {name: module_status(name) for name in names}


def module_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}


def load_ppo_class(algorithm: str):
    if algorithm == "recurrent_ppo":
        try:
            from sb3_contrib import RecurrentPPO  # type: ignore

            return RecurrentPPO
        except Exception as exc:
            raise CrafterTrainingUnavailable("Install embryo[train] for RecurrentPPO training.") from exc
    if algorithm == "ppo":
        try:
            from stable_baselines3 import PPO  # type: ignore

            return PPO
        except Exception as exc:
            raise CrafterTrainingUnavailable("Install embryo[train] for PPO training.") from exc
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def default_policy_name(algorithm: str) -> str:
    return "CnnLstmPolicy" if algorithm == "recurrent_ppo" else "CnnPolicy"


def summarize_rollouts(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    survivals = [numeric(row.get("survival_steps")) for row in episodes]
    rewards = [numeric(row.get("reward_sum_eval_only")) for row in episodes]
    achievements = [int(row.get("achievement_count_eval_only", 0)) for row in episodes]
    return {
        "episode_count": len(episodes),
        "mean_survival_steps": round(mean(survivals), 6),
        "median_survival_steps": round(float(np.median(survivals)), 6) if survivals else 0.0,
        "death_rate": round(sum(1 for row in episodes if bool(row.get("done", False))) / len(episodes), 6) if episodes else 0.0,
        "mean_reward_eval_only": round(mean(rewards), 6),
        "mean_achievement_count_eval_only": round(mean(achievements), 6),
        "mean_loop_rate": round(mean(row.get("loop_rate", 0.0) for row in episodes), 6),
        "mean_action_entropy": round(mean(row.get("action_entropy", 0.0) for row in episodes), 6),
    }


def achievement_names(info: Mapping[str, Any]) -> list[str]:
    raw = info.get("achievements", {})
    if isinstance(raw, Mapping):
        return sorted(str(key) for key, value in raw.items() if bool(value))
    return []


def repeated_action_loop_rate(actions: Sequence[int], *, window: int = 3) -> float:
    if not actions:
        return 0.0
    repeated = 0
    for index, action in enumerate(actions):
        if index >= window - 1 and all(actions[index - offset] == action for offset in range(window)):
            repeated += 1
    return round(repeated / len(actions), 6)


def action_entropy(actions: Sequence[int]) -> float:
    if not actions:
        return 0.0
    counts = Counter(int(action) for action in actions)
    total = float(len(actions))
    entropy = -sum((count / total) * np.log(count / total) for count in counts.values())
    normalizer = np.log(len(counts)) if len(counts) > 1 else 1.0
    return round(float(entropy / normalizer), 6)


def histogram(values: Sequence[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(int(value) for value in values).items())}


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Sequence[Any]) -> float:
    rows = [numeric(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "none", "disabled"}
    return bool(value)


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def resolve_out(config: Mapping[str, Any], root: Path, out: str) -> Path:
    if out:
        path = Path(out)
    else:
        output = mapping(config.get("output"))
        path = Path(str(output.get("path", "runs/crafter_survival")))
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train or inspect a Crafter survival actor.")
    parser.add_argument("--config", default=str(root / "configs" / "crafter_survival.yaml"))
    parser.add_argument("--out", default="")
    parser.add_argument("--mode", choices=("check", "inspect-human", "bc", "ppo", "eval"), default="check")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--human-data-root", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.human_data_root:
        config = dict(config)
        dataset_cfg = dict(mapping(config.get("dataset")))
        dataset_cfg["human_root"] = args.human_data_root
        config["dataset"] = dataset_cfg
    out = resolve_out(config, root, args.out)
    if args.mode == "check":
        result = run_setup_check(config, out)
    elif args.mode == "inspect-human":
        result = inspect_human_dataset(config, out)
    elif args.mode == "bc":
        result = train_bc(config, out)
    elif args.mode == "ppo":
        result = train_ppo(config, out, checkpoint=args.checkpoint)
    else:
        result = evaluate_checkpoint(config, out, args.checkpoint or out / "model.zip")
    print(json.dumps({"decision": result["summary"].get("decision"), "out": str(out), "paths": result["paths"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
