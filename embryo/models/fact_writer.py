"""Visual fact writer interfaces.

The forward package keeps model contracts separate from training code.  Learned
models can implement `VisualFactWriter`; the small threshold implementation is a
deterministic reference used by fixtures, tests, and smoke replays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from embryo.memory.facts import facing_candidate_fact


@dataclass(frozen=True)
class FactWriterOutput:
    predicted: bool
    confidence: float


class VisualFactWriter:
    """Interface for RGB-derived fact writers."""

    provenance = "learned_visual_fact_writer"

    def predict(self, features: Mapping[str, Any]) -> FactWriterOutput:
        raise NotImplementedError

    def fact(self, features: Mapping[str, Any]):
        output = self.predict(features)
        direction_bin = features.get("direction_bin")
        return facing_candidate_fact(
            output.predicted,
            confidence=output.confidence,
            provenance=self.provenance,
            direction_bin=str(direction_bin) if direction_bin is not None else None,
        )


@dataclass(frozen=True)
class ThresholdFactWriter(VisualFactWriter):
    """Reference writer over a scalar deployable visual feature."""

    feature_name: str = "candidate_score"
    threshold: float = 0.5
    provenance: str = "threshold_visual_fact_writer"

    def predict(self, features: Mapping[str, Any]) -> FactWriterOutput:
        raw = features.get(self.feature_name, 0.0)
        score = float(raw)
        confidence = max(0.0, min(1.0, score))
        return FactWriterOutput(predicted=score >= self.threshold, confidence=confidence)


@dataclass(frozen=True)
class LearnedFactWriterV0(VisualFactWriter):
    """Tiny offline fact writer over compact RGB-derived numeric features."""

    feature_schema: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    visible_weights: tuple[float, ...]
    visible_bias: float
    salience_weights: tuple[float, ...]
    salience_bias: float
    change_weights: tuple[float, ...]
    change_bias: float
    visible_threshold: float = 0.5
    change_threshold: float = 0.5
    provenance: str = "learned_fact_writer_v0"

    def predict(self, features: Mapping[str, Any]) -> FactWriterOutput:
        scores = self.predict_scores(features)
        return FactWriterOutput(
            predicted=bool(scores["visual_anchor_visible"]),
            confidence=float(scores["visual_anchor_probability"]),
        )

    def predict_scores(self, features: Mapping[str, Any]) -> dict[str, Any]:
        vector = self.normalized_vector(features)
        visible_probability = sigmoid(float(vector_dot(vector, np.asarray(self.visible_weights, dtype=np.float64)) + self.visible_bias))
        salience = float(vector_dot(vector, np.asarray(self.salience_weights, dtype=np.float64)) + self.salience_bias)
        change_probability = sigmoid(float(vector_dot(vector, np.asarray(self.change_weights, dtype=np.float64)) + self.change_bias))
        return {
            "visual_anchor_probability": round(visible_probability, 6),
            "visual_anchor_visible": visible_probability >= self.visible_threshold,
            "center_salience_score": round(clamp01(salience), 6),
            "visual_change_probability": round(change_probability, 6),
            "visual_change_event": change_probability >= self.change_threshold,
        }

    def predict_matrix(self, x: np.ndarray) -> dict[str, np.ndarray]:
        z = normalize_matrix(x, np.asarray(self.feature_mean), np.asarray(self.feature_scale))
        visible = sigmoid_array(matrix_vector(z, np.asarray(self.visible_weights, dtype=np.float64)) + self.visible_bias)
        salience = np.clip(matrix_vector(z, np.asarray(self.salience_weights, dtype=np.float64)) + self.salience_bias, 0.0, 1.0)
        change = sigmoid_array(matrix_vector(z, np.asarray(self.change_weights, dtype=np.float64)) + self.change_bias)
        return {"visible_probability": visible, "salience": salience, "change_probability": change}

    def normalized_vector(self, features: Mapping[str, Any]) -> np.ndarray:
        x = np.asarray([float(features.get(name, 0.0)) for name in self.feature_schema], dtype=np.float64)
        return (x - np.asarray(self.feature_mean, dtype=np.float64)) / np.asarray(self.feature_scale, dtype=np.float64)

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "model_type": "learned_fact_writer_v0",
            "feature_schema": list(self.feature_schema),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "visible_weights": list(self.visible_weights),
            "visible_bias": self.visible_bias,
            "salience_weights": list(self.salience_weights),
            "salience_bias": self.salience_bias,
            "change_weights": list(self.change_weights),
            "change_bias": self.change_bias,
            "visible_threshold": self.visible_threshold,
            "change_threshold": self.change_threshold,
            "provenance": self.provenance,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "LearnedFactWriterV0":
        return cls(
            feature_schema=tuple(str(item) for item in payload["feature_schema"]),
            feature_mean=tuple(float(item) for item in payload["feature_mean"]),
            feature_scale=tuple(float(item) for item in payload["feature_scale"]),
            visible_weights=tuple(float(item) for item in payload["visible_weights"]),
            visible_bias=float(payload["visible_bias"]),
            salience_weights=tuple(float(item) for item in payload["salience_weights"]),
            salience_bias=float(payload["salience_bias"]),
            change_weights=tuple(float(item) for item in payload["change_weights"]),
            change_bias=float(payload["change_bias"]),
            visible_threshold=float(payload.get("visible_threshold", 0.5)),
            change_threshold=float(payload.get("change_threshold", 0.5)),
            provenance=str(payload.get("provenance", "learned_fact_writer_v0")),
        )


def train_learned_fact_writer_v0(
    x: np.ndarray,
    *,
    visible: np.ndarray,
    salience: np.ndarray,
    change: np.ndarray,
    feature_schema: Sequence[str],
    epochs: int = 500,
    learning_rate: float = 0.2,
    l2: float = 1e-4,
    ridge: float = 1e-4,
) -> LearnedFactWriterV0:
    mean = x.mean(axis=0) if len(x) else np.zeros(len(feature_schema), dtype=np.float64)
    scale = x.std(axis=0) if len(x) else np.ones(len(feature_schema), dtype=np.float64)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z = normalize_matrix(x, mean, scale)
    visible_weights, visible_bias = fit_logistic(z, visible, epochs=epochs, learning_rate=learning_rate, l2=l2)
    change_weights, change_bias = fit_logistic(z, change, epochs=epochs, learning_rate=learning_rate, l2=l2)
    salience_weights, salience_bias = fit_ridge(z, salience, ridge=ridge)
    return LearnedFactWriterV0(
        feature_schema=tuple(str(item) for item in feature_schema),
        feature_mean=tuple(float(item) for item in mean),
        feature_scale=tuple(float(item) for item in scale),
        visible_weights=tuple(float(item) for item in visible_weights),
        visible_bias=float(visible_bias),
        salience_weights=tuple(float(item) for item in salience_weights),
        salience_bias=float(salience_bias),
        change_weights=tuple(float(item) for item in change_weights),
        change_bias=float(change_bias),
    )


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    positives = float((y == 1.0).sum())
    negatives = float((y == 0.0).sum())
    if positives == 0.0 or negatives == 0.0:
        prior = (positives + 0.5) / (float(len(y)) + 1.0) if len(y) else 0.5
        return weights, logit(prior)
    bias = 0.0
    sample_weight = balanced_sample_weight(y)
    denom = max(1.0, float(sample_weight.sum()))
    for _ in range(int(epochs)):
        probabilities = sigmoid_array(matrix_vector(x, weights) + bias)
        error = (probabilities - y) * sample_weight
        grad_w = np.clip(vector_matrix(error, x) / denom + l2 * weights, -5.0, 5.0)
        grad_b = float(np.clip(error.sum() / denom, -5.0, 5.0))
        weights = np.nan_to_num(weights - learning_rate * grad_w, nan=0.0, posinf=10.0, neginf=-10.0)
        bias = float(np.nan_to_num(bias - learning_rate * grad_b, nan=0.0, posinf=10.0, neginf=-10.0))
    return weights, bias


def fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge: float) -> tuple[np.ndarray, float]:
    if len(x) == 0:
        return np.zeros(x.shape[1], dtype=np.float64), 0.0
    design = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
    penalty = np.sqrt(float(ridge)) * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    augmented_x = np.vstack([design, penalty])
    augmented_y = np.concatenate([y, np.zeros(design.shape[1], dtype=np.float64)])
    params = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    return params[:-1], float(params[-1])


def save_fact_writer_checkpoint(model: LearnedFactWriterV0, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_checkpoint(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fact_writer_checkpoint(path: str | Path) -> LearnedFactWriterV0:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected fact-writer checkpoint mapping at {path}")
    return LearnedFactWriterV0.from_checkpoint(payload)


def normalize_matrix(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((x - mean) / scale, -10.0, 10.0)


def matrix_vector(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.einsum("ij,j->i", x, weights, optimize=True), nan=0.0, posinf=50.0, neginf=-50.0)


def vector_matrix(vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.einsum("i,ij->j", vector, x, optimize=True), nan=0.0, posinf=50.0, neginf=-50.0)


def vector_dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.nan_to_num(np.einsum("i,i->", left, right, optimize=True), nan=0.0, posinf=50.0, neginf=-50.0))


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    positives = float((y == 1.0).sum())
    negatives = float((y == 0.0).sum())
    weights = np.ones_like(y, dtype=np.float64)
    if positives > 0 and negatives > 0:
        weights = np.where(y == 1.0, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    return weights


def sigmoid(value: float) -> float:
    return float(sigmoid_array(np.asarray([value], dtype=np.float64))[0])


def sigmoid_array(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(value: float) -> float:
    clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
    return float(np.log(clipped / (1.0 - clipped)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
