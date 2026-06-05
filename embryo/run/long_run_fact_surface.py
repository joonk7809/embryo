"""Selectable visual fact surfaces for long-run memory scans."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from embryo.datasets.fact_writer import compact_rgb_features
from embryo.models import load_fact_writer_checkpoint
from embryo.runtimes.crafter.features import (
    extract_deployable_rgb_features,
    center_patch,
    patch_content_hash,
    rgb_array,
)


REFERENCE_RGB_SCAFFOLD = "reference_rgb_scaffold"
LEARNED_FACT_WRITER_V0 = "learned_fact_writer_v0"
LEARNED_VISUAL_ANCHOR_FAMILY = "learned_visual_anchor_v0"
REQUIRED_FACT_FIELDS = (
    "visual_anchor_visible",
    "visual_anchor_family",
    "center_salience_score",
    "center_patch_hash",
    "visual_change_event",
    "failed_action_event",
    "rgb_delta_score",
)
ALLOWED_LEARNED_INPUTS = ("raw_rgb_frame", "previous_rgb_frame", "previous_action")


class FactSurface:
    name = REFERENCE_RGB_SCAFFOLD

    def apply(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ReferenceRGBScaffoldSurface(FactSurface):
    name = REFERENCE_RGB_SCAFFOLD

    def apply(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(observation)
        if all(field in row for field in REQUIRED_FACT_FIELDS):
            return row
        row.update(reference_scaffold_fields(row))
        return row


class LearnedFactWriterV0Surface(FactSurface):
    name = LEARNED_FACT_WRITER_V0

    def __init__(self, *, checkpoint: str | Path, threshold: float = 0.5, root: str | Path | None = None) -> None:
        self.checkpoint = resolve_checkpoint_path(checkpoint, root=root)
        self.threshold = float(threshold)
        self.model = load_fact_writer_checkpoint(self.checkpoint)

    def apply(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(observation)
        source_inputs = learned_model_inputs(row)
        features = compact_rgb_features(source_inputs)
        scores = self.model.predict_scores(features)
        probability = float(scores["visual_anchor_probability"])
        reference = reference_scaffold_fields(row)
        row.update(
            {
                "visual_anchor_visible": probability >= self.threshold,
                "visual_anchor_family": LEARNED_VISUAL_ANCHOR_FAMILY,
                "center_salience_score": round(probability, 6),
                "center_patch_hash": deterministic_center_patch_hash(source_inputs.get("raw_rgb_frame")),
                "visual_change_event": bool(reference["visual_change_event"]),
                "failed_action_event": bool(reference["failed_action_event"]),
                "rgb_delta_score": float(reference["rgb_delta_score"]),
                "candidate_score": round(probability, 6),
            }
        )
        return row


def load_fact_surface(config: Mapping[str, Any], *, root: str | Path | None = None) -> FactSurface:
    name = str(config.get("name", REFERENCE_RGB_SCAFFOLD))
    if name == REFERENCE_RGB_SCAFFOLD:
        return ReferenceRGBScaffoldSurface()
    if name == LEARNED_FACT_WRITER_V0:
        checkpoint = config.get("checkpoint")
        if not checkpoint:
            raise ValueError("learned_fact_writer_v0 fact surface requires a checkpoint")
        return LearnedFactWriterV0Surface(
            checkpoint=str(checkpoint),
            threshold=float(config.get("threshold", 0.5)),
            root=root,
        )
    raise ValueError(f"Unsupported long-run fact surface: {name}")


def learned_model_inputs(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {field: observation.get(field) for field in ALLOWED_LEARNED_INPUTS}


def reference_scaffold_fields(observation: Mapping[str, Any]) -> dict[str, Any]:
    scaffold = extract_deployable_rgb_features(
        observation.get("raw_rgb_frame"),
        previous_observation=observation.get("previous_rgb_frame"),
        previous_action=str(observation.get("previous_action", "noop")),
    )
    return {
        "visual_anchor_visible": bool(observation.get("visual_anchor_visible", scaffold["visual_anchor_visible"])),
        "visual_anchor_family": str(observation.get("visual_anchor_family", scaffold["visual_anchor_family"])),
        "center_salience_score": float(observation.get("center_salience_score", scaffold["center_salience_score"])),
        "center_patch_hash": str(observation.get("center_patch_hash", scaffold["center_patch_hash"])),
        "visual_change_event": bool(observation.get("visual_change_event", scaffold["visual_change_event"])),
        "failed_action_event": bool(observation.get("failed_action_event", scaffold["failed_action_event"])),
        "rgb_delta_score": float(observation.get("rgb_delta_score", scaffold["rgb_delta_score"])),
    }


def deterministic_center_patch_hash(observation: Any) -> str:
    image = rgb_array(observation)
    if image is None:
        return ""
    return patch_content_hash(center_patch(image))


def resolve_checkpoint_path(checkpoint: str | Path, *, root: str | Path | None = None) -> Path:
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    base = Path(root) if root is not None else Path.cwd()
    return base / path
