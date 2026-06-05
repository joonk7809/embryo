"""Offline fact-writer dataset construction from deployable observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from embryo.eval.contamination import scan_actor_context
from embryo.runtimes import make_runtime
from embryo.runtimes.crafter.features import (
    center_patch,
    extract_deployable_rgb_features,
    patch_edge_score,
    rgb_array,
)


ALLOWED_SOURCE_FIELDS = ("raw_rgb_frame", "previous_rgb_frame", "previous_action")
LABEL_SOURCE = "reference_rgb_scaffold_v1"
DEFAULT_ACTION_VOCAB = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "move_forward",
    "turn_left",
    "do",
    "sleep",
)
BASE_FEATURE_SCHEMA = (
    "center_mean_r",
    "center_mean_g",
    "center_mean_b",
    "center_std_r",
    "center_std_g",
    "center_std_b",
    "global_mean_r",
    "global_mean_g",
    "global_mean_b",
    "global_std",
    "center_global_abs_contrast",
    "center_saturation_mean",
    "center_variation",
    "center_edge_score",
    "rgb_delta_mean",
    "rgb_delta_max",
    "center_delta_mean",
    "center_delta_max",
)


@dataclass(frozen=True)
class FactWriterDataset:
    rows_by_split: dict[str, list[dict[str, Any]]]
    diagnostics_by_split: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any]
    contamination: dict[str, Any]


@dataclass(frozen=True)
class LabelProfile:
    salience_threshold: float = 0.5
    min_rgb_delta: float = 0.0
    center_vs_global_contrast_threshold: float = 0.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LabelProfile":
        raw = mapping(config.get("labels"))
        return cls(
            salience_threshold=float(raw.get("salience_threshold", 0.5)),
            min_rgb_delta=float(raw.get("min_rgb_delta", 0.0)),
            center_vs_global_contrast_threshold=float(raw.get("center_vs_global_contrast_threshold", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "salience_threshold": self.salience_threshold,
            "min_rgb_delta": self.min_rgb_delta,
            "center_vs_global_contrast_threshold": self.center_vs_global_contrast_threshold,
        }


def build_fact_writer_dataset(config: Mapping[str, Any]) -> FactWriterDataset:
    runtime_cfg = mapping(config.get("runtime"))
    split_cfg = mapping(config.get("splits"))
    collection_cfg = mapping(config.get("collection"))
    action_vocab = tuple(str(item) for item in collection_cfg.get("action_vocabulary", DEFAULT_ACTION_VOCAB))
    max_steps = int(collection_cfg.get("steps_per_seed", 128))
    label_profile = LabelProfile.from_config(config)
    runtime_name = str(runtime_cfg.get("name", "fixture_memory"))
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    diagnostics_by_split: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    split_manifest: dict[str, Any] = {}
    feature_schema = feature_schema_for_actions(action_vocab)

    for split_name, raw_split in split_cfg.items():
        split = mapping(raw_split)
        seeds = configured_seeds(split)
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            rows.extend(
                collect_seed_examples(
                    runtime_name,
                    seed=seed,
                    split=split_name,
                    max_steps=max_steps,
                    action_vocab=action_vocab,
                    runtime_cfg=runtime_cfg,
                    label_profile=label_profile,
                )
            )
        rows_by_split[split_name] = rows
        diagnostics_by_split[split_name] = [dict(row["scaffold_diagnostics"]) for row in rows]
        failures.extend(failure for row in rows for failure in row["contamination"]["failures"])
        split_manifest[split_name] = {
            "row_count": len(rows),
            "seed_count": len(seeds),
            "seed_start": seeds[0] if seeds else None,
            "seed_end": seeds[-1] if seeds else None,
            "label_balance": label_balance(rows),
        }

    contamination = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
    manifest = {
        "builder": "fact_writer_offline_dataset_v0",
        "runtime": {"name": runtime_name, "split_names": sorted(rows_by_split)},
        "splits": split_manifest,
        "source_input_fields": list(ALLOWED_SOURCE_FIELDS),
        "feature_schema": list(feature_schema),
        "label_schema": [
            "label_visual_anchor_visible",
            "label_center_salience_score",
            "label_visual_change_event",
        ],
        "label_source": LABEL_SOURCE,
        "label_profile": label_profile.to_dict(),
        "contamination": contamination,
    }
    return FactWriterDataset(
        rows_by_split=rows_by_split,
        diagnostics_by_split=diagnostics_by_split,
        manifest=manifest,
        contamination=contamination,
    )


def collect_seed_examples(
    runtime_name: str,
    *,
    seed: int,
    split: str,
    max_steps: int,
    action_vocab: Sequence[str],
    runtime_cfg: Mapping[str, Any],
    label_profile: LabelProfile,
) -> list[dict[str, Any]]:
    runtime = make_fact_writer_runtime(runtime_name, seed=seed, max_steps=max_steps, runtime_cfg=runtime_cfg)
    rows: list[dict[str, Any]] = []
    try:
        current = runtime.reset(seed=seed)
        cycle = action_cycle(runtime.spec.action_names, runtime.spec.resource_action, runtime.spec.fallback_action, runtime.spec.noop_action)
        for tick in range(max_steps):
            rows.append(
                example_from_observation(
                    current.observation,
                    split=split,
                    seed=seed,
                    tick=tick,
                    action_vocab=action_vocab,
                    label_profile=label_profile,
                )
            )
            if current.done:
                break
            action = cycle[tick % len(cycle)]
            current = runtime.step(action)
    finally:
        runtime.close()
    return rows


def example_from_observation(
    observation: Mapping[str, Any],
    *,
    split: str = "fixture",
    seed: int = 0,
    tick: int = 0,
    action_vocab: Sequence[str] = DEFAULT_ACTION_VOCAB,
    label_profile: LabelProfile | None = None,
) -> dict[str, Any]:
    profile = label_profile or LabelProfile()
    source_inputs = {field: observation.get(field) for field in ALLOWED_SOURCE_FIELDS}
    contamination = scan_actor_context(observation)
    features = compact_rgb_features(source_inputs, action_vocab=action_vocab)
    diagnostics = scaffold_diagnostics_from_observation(observation, features=features, split=split, seed=seed, tick=tick)
    labels = labels_from_reference_surface(diagnostics, profile)
    return {
        "split": str(split),
        "source_inputs": source_inputs,
        "features": features,
        "labels": labels,
        "label_source": LABEL_SOURCE,
        "scaffold_diagnostics": diagnostics,
        "metadata": {"seed": int(seed), "tick": int(tick)},
        "contamination": contamination,
    }


def compact_rgb_features(source_inputs: Mapping[str, Any], *, action_vocab: Sequence[str] = DEFAULT_ACTION_VOCAB) -> dict[str, float]:
    image = rgb_array(source_inputs.get("raw_rgb_frame"))
    previous = rgb_array(source_inputs.get("previous_rgb_frame"))
    if image is None:
        features = {name: 0.0 for name in BASE_FEATURE_SCHEMA}
    else:
        patch = center_patch(image)
        global_mean = image.mean(axis=(0, 1))
        center_mean = patch.mean(axis=(0, 1))
        center_std = patch.std(axis=(0, 1))
        delta = abs_delta(image, previous)
        patch_delta = abs_delta(patch, center_patch(previous) if previous is not None and previous.shape == image.shape else None)
        features = {
            "center_mean_r": float(center_mean[0]),
            "center_mean_g": float(center_mean[1]),
            "center_mean_b": float(center_mean[2]),
            "center_std_r": float(center_std[0]),
            "center_std_g": float(center_std[1]),
            "center_std_b": float(center_std[2]),
            "global_mean_r": float(global_mean[0]),
            "global_mean_g": float(global_mean[1]),
            "global_mean_b": float(global_mean[2]),
            "global_std": float(image.std()),
            "center_global_abs_contrast": abs(float(patch.mean()) - float(image.mean())),
            "center_saturation_mean": float((patch.max(axis=2) - patch.min(axis=2)).mean()),
            "center_variation": float(patch.std()),
            "center_edge_score": float(patch_edge_score(patch)),
            "rgb_delta_mean": float(delta.mean()) if delta is not None else 0.0,
            "rgb_delta_max": float(delta.max()) if delta is not None else 0.0,
            "center_delta_mean": float(patch_delta.mean()) if patch_delta is not None else 0.0,
            "center_delta_max": float(patch_delta.max()) if patch_delta is not None else 0.0,
        }
    previous_action = str(source_inputs.get("previous_action", "noop"))
    for action in action_vocab:
        features[f"previous_action_is_{safe_feature_name(action)}"] = 1.0 if previous_action == action else 0.0
    features["previous_action_is_other"] = 0.0 if previous_action in set(action_vocab) else 1.0
    return {key: round(float(value), 6) for key, value in features.items()}


def scaffold_diagnostics_from_observation(
    observation: Mapping[str, Any],
    *,
    features: Mapping[str, Any],
    split: str,
    seed: int,
    tick: int,
) -> dict[str, Any]:
    if all(key in observation for key in ("visual_anchor_visible", "center_salience_score", "visual_change_event")):
        scaffold = {
            "visual_anchor_visible": bool(observation.get("visual_anchor_visible", False)),
            "center_salience_score": round(float(observation.get("center_salience_score", 0.0)), 6),
            "rgb_delta_score": round(float(observation.get("rgb_delta_score", features.get("rgb_delta_mean", 0.0))), 6),
            "visual_change_event": bool(observation.get("visual_change_event", False)),
            "center_patch_hash": str(observation.get("center_patch_hash", "")),
        }
    else:
        scaffold = extract_deployable_rgb_features(
            observation.get("raw_rgb_frame"),
            previous_observation=observation.get("previous_rgb_frame"),
            previous_action=str(observation.get("previous_action", "noop")),
        )
    return {
        "split": str(split),
        "seed": int(seed),
        "tick": int(tick),
        "previous_action": str(observation.get("previous_action", "noop")),
        "center_salience_score": round(float(scaffold.get("center_salience_score", 0.0)), 6),
        "rgb_delta_score": round(float(scaffold.get("rgb_delta_score", features.get("rgb_delta_mean", 0.0))), 6),
        "center_vs_global_contrast": round(float(features.get("center_global_abs_contrast", 0.0)), 6),
        "visual_anchor_visible": bool(scaffold.get("visual_anchor_visible", False)),
        "visual_change_event": bool(scaffold.get("visual_change_event", False)),
        "center_patch_hash": str(scaffold.get("center_patch_hash", "")),
    }


def labels_from_reference_surface(diagnostics: Mapping[str, Any], profile: LabelProfile) -> dict[str, Any]:
    visible = bool(diagnostics.get("visual_anchor_visible", False))
    salience = float(diagnostics.get("center_salience_score", 0.0))
    rgb_delta = float(diagnostics.get("rgb_delta_score", 0.0))
    contrast = float(diagnostics.get("center_vs_global_contrast", 0.0))
    profile_visible = (
        visible
        and salience >= profile.salience_threshold
        and rgb_delta >= profile.min_rgb_delta
        and contrast >= profile.center_vs_global_contrast_threshold
    )
    return {
        "label_visual_anchor_visible": bool(profile_visible),
        "label_center_salience_score": round(salience, 6),
        "label_visual_change_event": bool(diagnostics.get("visual_change_event", False)),
    }


def label_balance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = sum(1 for row in rows if mapping(row.get("labels")).get("label_visual_anchor_visible", False))
    total = len(rows)
    negatives = total - positives
    return {
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": round(positives / total, 6) if total else 0.0,
    }


def feature_schema_for_actions(action_vocab: Sequence[str]) -> tuple[str, ...]:
    return (
        *BASE_FEATURE_SCHEMA,
        *(f"previous_action_is_{safe_feature_name(action)}" for action in action_vocab),
        "previous_action_is_other",
    )


def feature_matrix(rows: Sequence[Mapping[str, Any]], feature_schema: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(mapping(row.get("features")).get(name, 0.0)) for name in feature_schema] for row in rows], dtype=np.float64)


def visible_labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([1.0 if mapping(row.get("labels")).get("label_visual_anchor_visible", False) else 0.0 for row in rows], dtype=np.float64)


def change_labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([1.0 if mapping(row.get("labels")).get("label_visual_change_event", False) else 0.0 for row in rows], dtype=np.float64)


def salience_labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(mapping(row.get("labels")).get("label_center_salience_score", 0.0)) for row in rows], dtype=np.float64)


def make_fact_writer_runtime(runtime_name: str, *, seed: int, max_steps: int, runtime_cfg: Mapping[str, Any]):
    if runtime_name == "fixture_memory":
        return make_runtime(runtime_name, max_steps=max_steps)
    if runtime_name == "crafter_memory":
        return make_runtime(
            runtime_name,
            seed=seed,
            deterministic_backend_patch=bool(runtime_cfg.get("deterministic_backend_patch", True)),
        )
    try:
        return make_runtime(runtime_name, seed=seed)
    except TypeError:
        return make_runtime(runtime_name)


def configured_seeds(split: Mapping[str, Any]) -> list[int]:
    if split.get("seeds"):
        return [int(seed) for seed in split["seeds"]]
    seed_start = int(split.get("seed_start", 0))
    seed_count = int(split.get("seed_count", 1))
    return list(range(seed_start, seed_start + seed_count))


def action_cycle(action_names: Sequence[str], resource_action: str, fallback_action: str, noop_action: str) -> tuple[str, ...]:
    preferred = (resource_action, "move_up", "move_right", "move_down", "move_left", "do", fallback_action, noop_action)
    actions: list[str] = []
    valid = set(action_names)
    for action in preferred:
        if action in valid and action not in actions:
            actions.append(action)
    return tuple(actions or (noop_action,))


def abs_delta(image: np.ndarray, previous: np.ndarray | None) -> np.ndarray | None:
    if previous is None or previous.shape != image.shape:
        return None
    return np.abs(image - previous)


def safe_feature_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value).lower()).strip("_") or "unknown"


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
