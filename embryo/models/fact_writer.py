"""Visual fact writer interfaces.

The forward package keeps model contracts separate from training code.  Learned
models can implement `VisualFactWriter`; the small threshold implementation is a
deterministic reference used by fixtures, tests, and smoke replays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
