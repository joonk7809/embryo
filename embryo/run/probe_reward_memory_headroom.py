"""Probe reward-bearing POPGym tasks for recurrent-memory headroom."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embryo.core.config import load_config
from embryo.eval.contamination import scan_actor_context
from embryo.eval.traces import write_jsonl
from embryo.models.popgym_recurrent import VectorObservationGRUPolicy, require_torch, save_popgym_recurrent_checkpoint
from embryo.run.train_popgym_recurrent import mapping, set_determinism


GO_REWARD_MEMORY_HEADROOM_FOUND = "GO_reward_memory_headroom_found"
NO_GO_BASELINE_NOT_COMPETENT = "NO_GO_baseline_not_competent"
NO_GO_NO_MEMORY_HEADROOM = "NO_GO_no_memory_headroom"
NO_GO_REWARD_NOT_RECALL_DEPENDENT = "NO_GO_reward_not_recall_dependent"
NO_GO_RUNTIME_UNAVAILABLE_OR_TOO_HEAVY = "NO_GO_runtime_unavailable_or_too_heavy"
NO_GO_TRAINING_UNSTABLE = "NO_GO_training_unstable"
NO_GO_CONTAMINATION_FAILURE = "NO_GO_contamination_failure"


POPGYM_CONCENTRATION_TASKS = {
    "concentration_easy": ("popgym.envs.concentration", "ConcentrationEasy"),
    "concentration_medium": ("popgym.envs.concentration", "ConcentrationMedium"),
    "concentration_hard": ("popgym.envs.concentration", "ConcentrationHard"),
}


@dataclass
class ObservationMemoryTeacher:
    """Reference player that uses only observed card reveals and its own memory."""

    facedown_value: int
    known_values: dict[int, int] = field(default_factory=dict)
    pending_second: int | None = None
    first_action: int | None = None
    waiting_second: bool = False

    def reset(self) -> None:
        self.known_values.clear()
        self.pending_second = None
        self.first_action = None
        self.waiting_second = False

    def act(self, observation: Sequence[int]) -> int:
        visible = [idx for idx, value in enumerate(observation) if int(value) != int(self.facedown_value)]
        for idx in visible:
            self.known_values[int(idx)] = int(observation[idx])
        facedown = [idx for idx, value in enumerate(observation) if int(value) == int(self.facedown_value)]
        if self.waiting_second:
            action = self._second_action(observation, facedown)
            self.waiting_second = False
            self.first_action = None
            self.pending_second = None
            return action
        pair = self._known_facedown_pair(facedown)
        if pair is not None:
            first, second = pair
            self.pending_second = int(second)
            return self._first_action(first)
        return self._first_action(self._unknown_or_first_facedown(facedown))

    def _first_action(self, action: int) -> int:
        self.first_action = int(action)
        self.waiting_second = True
        return int(action)

    def _second_action(self, observation: Sequence[int], facedown: Sequence[int]) -> int:
        if self.pending_second in facedown:
            return int(self.pending_second)
        if self.first_action is not None and int(self.first_action) in self.known_values:
            value = int(self.known_values[int(self.first_action)])
            for idx in facedown:
                if int(idx) != int(self.first_action) and self.known_values.get(int(idx)) == value:
                    return int(idx)
        return self._unknown_or_first_facedown(facedown)

    def _unknown_or_first_facedown(self, facedown: Sequence[int]) -> int:
        for idx in facedown:
            if int(idx) not in self.known_values:
                return int(idx)
        return int(facedown[0]) if facedown else 0

    def _known_facedown_pair(self, facedown: Sequence[int]) -> tuple[int, int] | None:
        by_value: dict[int, list[int]] = {}
        for idx in facedown:
            if int(idx) in self.known_values:
                by_value.setdefault(int(self.known_values[int(idx)]), []).append(int(idx))
        for indices in by_value.values():
            if len(indices) >= 2:
                return indices[0], indices[1]
        return None


def run_reward_memory_headroom_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    torch, _ = require_torch()
    run_cfg = mapping(config.get("run"))
    model_cfg = mapping(config.get("model"))
    training_cfg = mapping(config.get("training"))
    evaluation_cfg = mapping(config.get("evaluation"))
    set_determinism(int(training_cfg.get("seed", 20260608)))
    audit = candidate_audit()

    task_names = tuple(str(task) for task in run_cfg.get("tasks", ("concentration_easy", "concentration_medium", "concentration_hard")))
    task_results: dict[str, Any] = {}
    checkpoint_manifests: dict[str, Any] = {}
    training_logs: dict[str, list[dict[str, Any]]] = {}
    models: dict[str, Any] = {}
    contamination_failures: list[dict[str, Any]] = []
    for task in task_names:
        env = make_popgym_concentration_env(task)
        spec = concentration_spec(env)
        env.close()
        train_data = collect_teacher_dataset(
            task=task,
            seed_start=int(training_cfg.get("seed_start", 10000)),
            seed_count=int(training_cfg.get("seed_count", 128)),
        )
        contamination_failures.extend(train_data["contamination"]["failures"])
        model = VectorObservationGRUPolicy.build(
            action_count=spec["action_count"],
            obs_dim=spec["obs_dim"],
            obs_value_count=spec["obs_value_count"],
            obs_embedding_dim=int(model_cfg.get("obs_embedding_dim", 8)),
            prev_action_embedding_dim=int(model_cfg.get("prev_action_embedding_dim", 8)),
            hidden_size=int(model_cfg.get("hidden_size", 64)),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 0.003)))
        training_log = train_headroom_model(
            model,
            optimizer,
            train_data,
            epochs=int(training_cfg.get("epochs", 20)),
            batch_size=int(training_cfg.get("batch_size", 32)),
        )
        seed_start = int(evaluation_cfg.get("seed_start", 12000))
        seed_count = int(evaluation_cfg.get("seed_count", 32))
        recurrent_rows = evaluate_policy(task=task, policy="recurrent_baseline", model=model, seed_start=seed_start, seed_count=seed_count)
        random_rows = evaluate_policy(task=task, policy="random_valid_action", model=None, seed_start=seed_start, seed_count=seed_count)
        teacher_rows = evaluate_policy(task=task, policy="reference_observation_memory_teacher", model=None, seed_start=seed_start, seed_count=seed_count)
        task_results[task] = {
            "task": task,
            "demand": task_demand(task),
            "spec": spec,
            "recurrent_baseline": episode_summary(recurrent_rows),
            "random_valid_action": episode_summary(random_rows),
            "reference_observation_memory_teacher_eval_only": episode_summary(teacher_rows),
            "episodes": recurrent_rows + random_rows + teacher_rows,
        }
        checkpoint_manifests[task] = {
            "model_type": "popgym_reward_headroom_vector_gru_v0",
            "task": task,
            "action_count": spec["action_count"],
            "obs_dim": spec["obs_dim"],
            "obs_value_count": spec["obs_value_count"],
            "obs_embedding_dim": int(model_cfg.get("obs_embedding_dim", 8)),
            "prev_action_embedding_dim": int(model_cfg.get("prev_action_embedding_dim", 8)),
            "hidden_size": int(model_cfg.get("hidden_size", 64)),
            "weights": "model.pt",
            "training_boundary": "supervised_reference_policy_recurrent_baseline_probe",
        }
        training_logs[task] = training_log
        models[task] = model

    contamination = {"passed": not contamination_failures, "failure_count": len(contamination_failures), "failures": contamination_failures}
    metrics_by_demand = [task_results[task] for task in task_names]
    decision, reasons = decide_headroom(metrics_by_demand, contamination=contamination, evaluation_cfg=evaluation_cfg, audit=audit)
    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "objective": "reward_memory_headroom_probe",
        "candidate_audit": audit,
        "task_family": "popgym_concentration",
        "probe_boundary": "recurrent_baseline_only_no_memory_module",
        "metrics_by_demand": metrics_by_demand,
        "contamination": contamination,
    }
    return {
        "summary": summary,
        "metrics_by_demand": metrics_by_demand,
        "metrics_by_seed": flatten_seed_metrics(metrics_by_demand),
        "training_logs": training_logs,
        "checkpoint_manifests": checkpoint_manifests,
        "models": models,
        "config_manifest": json_safe(config),
        "contamination": contamination,
    }


def train_headroom_model(model: Any, optimizer: Any, data: Mapping[str, Any], *, epochs: int, batch_size: int) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    observations = torch.tensor(data["observations"], dtype=torch.long)
    previous_actions = torch.tensor(data["previous_actions"], dtype=torch.long)
    labels = torch.tensor(data["actions"], dtype=torch.long)
    mask = torch.tensor(data["mask"], dtype=torch.bool)
    row_count = int(observations.shape[0])
    log: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(row_count)
        losses: list[float] = []
        for start in range(0, row_count, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(observations[idx], previous_actions[idx])
            loss = torch.nn.functional.cross_entropy(logits[mask[idx]], labels[idx][mask[idx]])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            accuracy = teacher_action_accuracy(model, data)
            log.append({"epoch": epoch + 1, "loss": round(sum(losses) / len(losses), 6), "teacher_action_accuracy": accuracy})
    return log


def teacher_action_accuracy(model: Any, data: Mapping[str, Any]) -> float:
    torch, _ = require_torch()
    model.eval()
    observations = torch.tensor(data["observations"], dtype=torch.long)
    previous_actions = torch.tensor(data["previous_actions"], dtype=torch.long)
    labels = torch.tensor(data["actions"], dtype=torch.long)
    mask = torch.tensor(data["mask"], dtype=torch.bool)
    with torch.no_grad():
        logits, _ = model(observations, previous_actions)
    predicted = logits.argmax(dim=-1)
    correct = (predicted[mask] == labels[mask]).float()
    return round(float(correct.mean().item()), 6) if int(correct.numel()) else 0.0


def collect_teacher_dataset(*, task: str, seed_start: int, seed_count: int) -> dict[str, Any]:
    episodes = [collect_teacher_episode(task=task, seed=seed) for seed in range(int(seed_start), int(seed_start) + int(seed_count))]
    max_steps = max(len(ep["actions"]) for ep in episodes)
    obs_dim = len(episodes[0]["observations"][0])
    observations = np.zeros((len(episodes), max_steps, obs_dim), dtype=np.int64)
    previous_actions = np.zeros((len(episodes), max_steps), dtype=np.int64)
    actions = np.zeros((len(episodes), max_steps), dtype=np.int64)
    mask = np.zeros((len(episodes), max_steps), dtype=bool)
    contexts: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        length = len(episode["actions"])
        observations[idx, :length] = np.asarray(episode["observations"], dtype=np.int64)
        previous_actions[idx, :length] = np.asarray(episode["previous_actions"], dtype=np.int64)
        actions[idx, :length] = np.asarray(episode["actions"], dtype=np.int64)
        mask[idx, :length] = True
        contexts.extend(episode["contexts"])
    return {
        "observations": observations,
        "previous_actions": previous_actions,
        "actions": actions,
        "mask": mask,
        "contamination": contamination_summary(contexts),
    }


def collect_teacher_episode(*, task: str, seed: int) -> dict[str, Any]:
    env = make_popgym_concentration_env(task)
    try:
        spec = concentration_spec(env)
        teacher = ObservationMemoryTeacher(facedown_value=spec["facedown_value"])
        obs, _ = env.reset(seed=int(seed))
        previous_action = spec["action_count"]
        observations: list[list[int]] = []
        previous_actions: list[int] = []
        actions: list[int] = []
        contexts: list[dict[str, Any]] = []
        done = False
        while not done:
            safe_obs = observation_vector(obs)
            action = teacher.act(safe_obs)
            observations.append(safe_obs)
            previous_actions.append(int(previous_action))
            actions.append(int(action))
            contexts.append(actor_context(observation=safe_obs, previous_action=previous_action, action_count=spec["action_count"]))
            obs, _, terminated, truncated, _ = env.step(int(action))
            previous_action = int(action)
            done = bool(terminated or truncated)
        return {"observations": observations, "previous_actions": previous_actions, "actions": actions, "contexts": contexts}
    finally:
        env.close()


def evaluate_policy(*, task: str, policy: str, model: Any | None, seed_start: int, seed_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(int(seed_start), int(seed_start) + int(seed_count)):
        rows.append(evaluate_episode(task=task, policy=policy, model=model, seed=seed))
    return rows


def evaluate_episode(*, task: str, policy: str, model: Any | None, seed: int) -> dict[str, Any]:
    torch, _ = require_torch()
    env = make_popgym_concentration_env(task)
    rng = random.Random(int(seed) + 991)
    try:
        spec = concentration_spec(env)
        teacher = ObservationMemoryTeacher(facedown_value=spec["facedown_value"])
        obs, _ = env.reset(seed=int(seed))
        previous_action = spec["action_count"]
        hidden = None
        done = False
        total_reward = 0.0
        steps = 0
        terminated_flag = False
        while not done:
            safe_obs = observation_vector(obs)
            if policy == "recurrent_baseline":
                assert model is not None
                model.eval()
                obs_tensor = torch.tensor([[safe_obs]], dtype=torch.long)
                prev_tensor = torch.tensor([[int(previous_action)]], dtype=torch.long)
                with torch.no_grad():
                    logits, hidden = model(obs_tensor, prev_tensor, hidden)
                action = int(logits[0, -1].argmax().item())
            elif policy == "random_valid_action":
                action = int(rng.randrange(spec["action_count"]))
            elif policy == "reference_observation_memory_teacher":
                action = int(teacher.act(safe_obs))
            else:
                raise ValueError(f"Unknown policy: {policy}")
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            previous_action = int(action)
            steps += 1
            terminated_flag = bool(terminated)
            done = bool(terminated or truncated)
        return {
            "task": task,
            "demand": task_demand(task),
            "policy": policy,
            "seed": int(seed),
            "return": round(total_reward, 6),
            "episode_length": int(steps),
            "completed": bool(terminated_flag),
            "positive_return": bool(total_reward > 0.0),
        }
    finally:
        env.close()


def decide_headroom(
    metrics_by_demand: Sequence[Mapping[str, Any]],
    *,
    contamination: Mapping[str, Any],
    evaluation_cfg: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if int(contamination.get("failure_count", 0)) > 0:
        return NO_GO_CONTAMINATION_FAILURE, ["contamination_failure_count_nonzero"]
    if not metrics_by_demand:
        return NO_GO_RUNTIME_UNAVAILABLE_OR_TOO_HEAVY, ["no_candidate_metrics"]
    easy = metrics_by_demand[0]
    hard = metrics_by_demand[-1]
    easy_recurrent = mapping(easy.get("recurrent_baseline"))
    easy_random = mapping(easy.get("random_valid_action"))
    hard_recurrent = mapping(hard.get("recurrent_baseline"))
    short_min = float(evaluation_cfg.get("short_completion_min", 0.80))
    long_max = float(evaluation_cfg.get("long_completion_max", 0.50))
    drop_min = float(evaluation_cfg.get("completion_drop_min", 0.25))
    random_margin_min = float(evaluation_cfg.get("random_margin_min", 0.20))
    easy_completion = float(easy_recurrent.get("completion_rate", 0.0))
    hard_completion = float(hard_recurrent.get("completion_rate", 0.0))
    random_completion = float(easy_random.get("completion_rate", 0.0))
    if easy_completion < short_min:
        return NO_GO_BASELINE_NOT_COMPETENT, [f"short_completion_rate<{short_min}"]
    if easy_completion - random_completion < random_margin_min:
        return NO_GO_REWARD_NOT_RECALL_DEPENDENT, [f"short_recurrent_random_margin<{random_margin_min}"]
    if hard_completion > long_max or easy_completion - hard_completion < drop_min:
        return NO_GO_NO_MEMORY_HEADROOM, [f"completion_drop<{drop_min}_or_long_completion>{long_max}"]
    _ = audit
    return GO_REWARD_MEMORY_HEADROOM_FOUND, ["recurrent_baseline_competent_short_and_degrades_with_demand"]


def episode_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episode_count": 0,
            "mean_return": 0.0,
            "completion_rate": 0.0,
            "positive_return_rate": 0.0,
            "mean_episode_length": 0.0,
        }
    return {
        "episode_count": len(rows),
        "mean_return": round(float(np.mean([float(row["return"]) for row in rows])), 6),
        "completion_rate": round(float(np.mean([bool(row["completed"]) for row in rows])), 6),
        "positive_return_rate": round(float(np.mean([bool(row["positive_return"]) for row in rows])), 6),
        "mean_episode_length": round(float(np.mean([int(row["episode_length"]) for row in rows])), 3),
    }


def flatten_seed_metrics(metrics_by_demand: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in metrics_by_demand:
        rows.extend(task.get("episodes", ()))
    return rows


def candidate_audit() -> dict[str, Any]:
    return {
        "memory_maze": import_status("memory_maze"),
        "popgym_labyrinth": import_status("popgym.envs.labyrinth_escape"),
        "popgym_concentration": import_status("popgym.envs.concentration"),
    }


def import_status(module_name: str) -> dict[str, Any]:
    try:
        __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {"available": True}


def make_popgym_concentration_env(task: str) -> Any:
    module_name, class_name = POPGYM_CONCENTRATION_TASKS.get(str(task), ("", ""))
    if not module_name:
        raise ValueError(f"Unknown POPGym Concentration task: {task}")
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install popgym to run the reward headroom probe.") from exc
    return getattr(module, class_name)()


def concentration_spec(env: Any) -> dict[str, int]:
    nvec = getattr(env.observation_space, "nvec", None)
    if nvec is None:
        raise TypeError("Concentration observation space must be MultiDiscrete.")
    return {
        "obs_dim": int(len(nvec)),
        "obs_value_count": int(max(nvec)),
        "facedown_value": int(max(nvec) - 1),
        "action_count": int(env.action_space.n),
    }


def observation_vector(observation: Any) -> list[int]:
    return [int(value) for value in np.asarray(observation, dtype=np.int64).reshape(-1).tolist()]


def actor_context(*, observation: Sequence[int], previous_action: int, action_count: int) -> dict[str, Any]:
    return {
        "popgym_observation": list(int(value) for value in observation),
        "previous_action": "none" if int(previous_action) == int(action_count) else f"card_{int(previous_action)}",
    }


def contamination_summary(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for idx, context in enumerate(contexts):
        scan = scan_actor_context(context)
        for failure in scan.get("failures", []):
            failures.append({"row": idx, **dict(failure)})
    return {"passed": not failures, "failure_count": len(failures), "failures": failures}


def task_demand(task: str) -> int:
    return {"concentration_easy": 1, "concentration_medium": 2, "concentration_hard": 3}.get(str(task), 0)


def write_reward_memory_headroom_artifacts(result: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    root = Path(out)
    checkpoint_root = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline_headroom_summary_json": root / "baseline_headroom_summary.json",
        "baseline_headroom_summary_md": root / "baseline_headroom_summary.md",
        "metrics_by_demand": root / "metrics_by_demand.json",
        "metrics_by_seed": root / "metrics_by_seed.json",
        "training_log": root / "training_log.jsonl",
        "config_manifest": root / "config_manifest.json",
        "contamination_scan": root / "contamination_scan.json",
    }
    paths["baseline_headroom_summary_json"].write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["baseline_headroom_summary_md"].write_text(format_summary_markdown(result["summary"]), encoding="utf-8")
    paths["metrics_by_demand"].write_text(json.dumps(result["metrics_by_demand"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics_by_seed"].write_text(json.dumps(result["metrics_by_seed"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["config_manifest"].write_text(json.dumps(result["config_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["contamination_scan"].write_text(json.dumps(result["contamination"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training_rows = [{"task": task, **row} for task, rows in result["training_logs"].items() for row in rows]
    write_jsonl(paths["training_log"], training_rows)
    for task, model in result["models"].items():
        task_checkpoint = checkpoint_root / task
        saved = save_popgym_recurrent_checkpoint(task_checkpoint, model=model, manifest=result["checkpoint_manifests"][task])
        paths[f"checkpoint_{task}_weights"] = Path(saved["weights"])
        paths[f"checkpoint_{task}_manifest"] = Path(saved["manifest"])
    return {key: str(value) for key, value in paths.items()}


def format_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Reward Memory Headroom Probe",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Boundary: `{summary.get('probe_boundary')}`",
        f"- Task family: `{summary.get('task_family')}`",
        f"- Contamination failures: `{mapping(summary.get('contamination')).get('failure_count')}`",
        "",
        "## Candidate Audit",
        "",
    ]
    for name, row in mapping(summary.get("candidate_audit")).items():
        lines.append(f"- {name}: `{row}`")
    lines.extend(
        [
            "",
            "## Demand Sweep",
            "",
            "| Task | Recurrent return | Recurrent completion | Random completion | Teacher completion |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("metrics_by_demand", ()):
        item = mapping(row)
        recurrent = mapping(item.get("recurrent_baseline"))
        random_row = mapping(item.get("random_valid_action"))
        teacher = mapping(item.get("reference_observation_memory_teacher_eval_only"))
        lines.append(
            f"| {item.get('task')} | {recurrent.get('mean_return')} | {recurrent.get('completion_rate')} | "
            f"{random_row.get('completion_rate')} | {teacher.get('completion_rate')} |"
        )
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/popgym_reward_headroom.yaml")
    parser.add_argument("--out", default="runs/popgym_reward_headroom")
    args = parser.parse_args(argv)
    result = run_reward_memory_headroom_probe(load_config(args.config))
    artifacts = write_reward_memory_headroom_artifacts(result, args.out)
    print(json.dumps({"decision": result["summary"]["decision"], "artifacts": artifacts, "out": str(Path(args.out).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
