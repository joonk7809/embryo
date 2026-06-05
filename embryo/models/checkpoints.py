"""Checkpoint manifest helpers.

The clean package treats checkpoint manifests as public contracts.  Actual
weight formats can evolve, but the manifest remains the stable boundary between
training jobs, replay code, and packaging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


REQUIRED_MANIFEST_FIELDS = ("model_type", "model_name", "version", "feature_schema")


@dataclass(frozen=True)
class CheckpointManifest:
    model_type: str
    model_name: str
    version: str
    feature_schema: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CheckpointManifest":
        missing = [field_name for field_name in REQUIRED_MANIFEST_FIELDS if field_name not in data]
        if missing:
            raise ValueError(f"Checkpoint manifest missing required field(s): {', '.join(missing)}")
        feature_schema = data.get("feature_schema", ())
        if isinstance(feature_schema, str):
            raise ValueError("Checkpoint manifest feature_schema must be a sequence, not a string")
        return cls(
            model_type=str(data["model_type"]),
            model_name=str(data["model_name"]),
            version=str(data["version"]),
            feature_schema=tuple(str(item) for item in feature_schema),
            artifact_paths=dict(data.get("artifact_paths", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_schema"] = list(self.feature_schema)
        return payload


def load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected checkpoint manifest mapping at {target}")
    return data


def save_manifest(manifest: CheckpointManifest | Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    payload = manifest.to_dict() if isinstance(manifest, CheckpointManifest) else dict(manifest)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    return CheckpointManifest.from_mapping(load_manifest(path))


def validate_manifest(
    manifest: CheckpointManifest | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    require_artifacts: bool = False,
    allowed_model_types: set[str] | None = None,
) -> dict[str, Any]:
    parsed = manifest if isinstance(manifest, CheckpointManifest) else CheckpointManifest.from_mapping(manifest)
    failures: list[str] = []
    if allowed_model_types is not None and parsed.model_type not in allowed_model_types:
        failures.append(f"unsupported_model_type:{parsed.model_type}")
    if require_artifacts:
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        for label, relative_path in parsed.artifact_paths.items():
            if Path(relative_path).is_absolute():
                artifact_path = Path(relative_path)
            else:
                artifact_path = root / relative_path
            if not artifact_path.exists():
                failures.append(f"missing_artifact:{label}:{relative_path}")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "manifest": parsed.to_dict()}


def build_model_from_manifest(manifest: CheckpointManifest | Mapping[str, Any]):
    """Build a lightweight reference model from a manifest.

    This factory is intentionally small and dependency-free.  Torch checkpoints
    can be loaded by future training modules without changing the manifest API.
    """
    parsed = manifest if isinstance(manifest, CheckpointManifest) else CheckpointManifest.from_mapping(manifest)
    params = dict(parsed.metadata.get("parameters", {}))
    if parsed.model_type == "threshold_fact_writer":
        from embryo.models.fact_writer import ThresholdFactWriter

        return ThresholdFactWriter(**params)
    if parsed.model_type == "learned_fact_writer_v0":
        from embryo.models.fact_writer import load_fact_writer_checkpoint

        checkpoint = parsed.artifact_paths.get("weights", "")
        if not checkpoint:
            raise ValueError("learned_fact_writer_v0 manifest requires artifact_paths.weights")
        return load_fact_writer_checkpoint(checkpoint)
    if parsed.model_type == "rule_router":
        from embryo.models.router import RuleRouterModel

        return RuleRouterModel(**params)
    if parsed.model_type == "route_bc_policy":
        from embryo.models.bc_policy import RouteBCPolicy

        return RouteBCPolicy(**params)
    raise ValueError(f"Unsupported lightweight model_type: {parsed.model_type}")
