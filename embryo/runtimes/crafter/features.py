"""Deployable RGB feature scaffold for the Crafter runtime."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


VISUAL_ANCHOR_FAMILY = "visual_salience_v1"
DEFAULT_SALIENCE_THRESHOLD = 0.5
DEFAULT_DELTA_THRESHOLD = 0.01
DEFAULT_LOW_DELTA_THRESHOLD = 0.003


def extract_deployable_rgb_features(
    observation: Any,
    *,
    previous_observation: Any = None,
    previous_action: str = "noop",
    salience_threshold: float = DEFAULT_SALIENCE_THRESHOLD,
) -> dict[str, Any]:
    """Extract a small deployable visual scaffold from RGB observations.

    This is a reference scaffold, not object recognition. It uses only pixels
    and agent-owned action history.
    """
    image = rgb_array(observation)
    previous = rgb_array(previous_observation)
    if image is None:
        return empty_features(previous_action=previous_action)

    patch = center_patch(image)
    salience = center_salience_score(patch, image)
    delta = rgb_delta_score(image, previous)
    patch_hash = patch_content_hash(patch)
    visual_change = delta >= DEFAULT_DELTA_THRESHOLD
    failed_action = bool(previous_action and previous_action != "noop" and delta <= DEFAULT_LOW_DELTA_THRESHOLD)
    visible = salience >= salience_threshold
    return {
        "rgb_delta_score": round(delta, 6),
        "center_salience_score": round(salience, 6),
        "center_patch_hash": patch_hash,
        "visual_change_event": visual_change,
        "failed_action_event": failed_action,
        "visual_anchor_visible": visible,
        "visual_anchor_family": VISUAL_ANCHOR_FAMILY,
        "candidate_score": round(salience, 6),
    }


def empty_features(*, previous_action: str) -> dict[str, Any]:
    return {
        "rgb_delta_score": 0.0,
        "center_salience_score": 0.0,
        "center_patch_hash": "",
        "visual_change_event": False,
        "failed_action_event": False if previous_action == "noop" else False,
        "visual_anchor_visible": False,
        "visual_anchor_family": VISUAL_ANCHOR_FAMILY,
        "candidate_score": 0.0,
    }


def rgb_array(observation: Any) -> np.ndarray | None:
    if observation is None:
        return None
    try:
        image = np.asarray(observation)
    except Exception:  # noqa: BLE001
        return None
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.shape[-1] != 3:
        return None
    image = image.astype(np.float32, copy=False)
    if float(np.nanmax(image)) > 1.0:
        image = image / 255.0
    return np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def center_patch(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = max(4, min(height, width) // 5)
    y0 = max(0, height // 2 - size // 2)
    x0 = max(0, width // 2 - size // 2)
    return image[y0 : y0 + size, x0 : x0 + size]


def center_salience_score(patch: np.ndarray, image: np.ndarray) -> float:
    ring = center_ring(image, patch.shape[0])
    saturation = float((patch.max(axis=2) - patch.min(axis=2)).mean())
    contrast = abs(float(patch.mean()) - float(ring.mean()))
    variation = float(patch.std())
    edge = patch_edge_score(patch)
    score = 1.5 * saturation + 1.5 * variation + 2.0 * contrast + edge
    return clamp01(score)


def center_ring(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    size = max(patch_size * 2, patch_size)
    y0 = max(0, height // 2 - size // 2)
    x0 = max(0, width // 2 - size // 2)
    return image[y0 : min(height, y0 + size), x0 : min(width, x0 + size)]


def patch_edge_score(patch: np.ndarray) -> float:
    vertical = np.abs(np.diff(patch, axis=0)).mean() if patch.shape[0] > 1 else 0.0
    horizontal = np.abs(np.diff(patch, axis=1)).mean() if patch.shape[1] > 1 else 0.0
    return float((vertical + horizontal) / 2.0)


def rgb_delta_score(image: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None or previous.shape != image.shape:
        return 0.0
    return float(np.abs(image - previous).mean())


def patch_content_hash(patch: np.ndarray) -> str:
    quantized = np.clip(np.rint(patch * 255.0), 0, 255).astype(np.uint8)
    return hashlib.sha1(quantized.tobytes()).hexdigest()[:16]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
