"""Learned stack manifest helpers."""

from __future__ import annotations

import json
import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from embryo.models.checkpoints import build_model_from_manifest, load_checkpoint_manifest, validate_manifest


STACK_COMPONENTS = ("fact_writer", "router", "bc_policy")


@dataclass(frozen=True)
class StackManifest:
    name: str
    version: str
    components: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StackManifest":
        if "name" not in data or "version" not in data or "components" not in data:
            raise ValueError("Stack manifest requires name, version, and components")
        components = dict(data["components"])
        missing = [name for name in STACK_COMPONENTS if name not in components]
        if missing:
            raise ValueError(f"Stack manifest missing component(s): {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            components={str(key): str(value) for key, value in components.items()},
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stack_manifest(path: str | Path) -> StackManifest:
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected stack manifest mapping at {target}")
    return StackManifest.from_mapping(data)


def save_stack_manifest(manifest: StackManifest | Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    payload = manifest.to_dict() if isinstance(manifest, StackManifest) else dict(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assemble_stack_manifest(
    *,
    fact_writer: str,
    router: str,
    bc_policy: str,
    name: str = "reference_memory_stack",
    version: str = "1",
    metadata: Mapping[str, Any] | None = None,
) -> StackManifest:
    return StackManifest(
        name=name,
        version=version,
        components={"fact_writer": fact_writer, "router": router, "bc_policy": bc_policy},
        metadata=dict(metadata or {}),
    )


def validate_stack_manifest(manifest: StackManifest | Mapping[str, Any], *, base_dir: str | Path | None = None) -> dict[str, Any]:
    parsed = manifest if isinstance(manifest, StackManifest) else StackManifest.from_mapping(manifest)
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    failures: list[str] = []
    component_validations: dict[str, Any] = {}
    expected_types = {
        "fact_writer": {"threshold_fact_writer"},
        "router": {"rule_router"},
        "bc_policy": {"route_bc_policy"},
    }
    for component, relative_path in parsed.components.items():
        path = resolve_component_path(root, relative_path)
        if not path.exists():
            failures.append(f"missing_component_manifest:{component}:{relative_path}")
            continue
        component_manifest = load_checkpoint_manifest(path)
        validation = validate_manifest(component_manifest, base_dir=path.parent, allowed_model_types=expected_types.get(component))
        component_validations[component] = validation
        if not validation["passed"]:
            failures.append(f"invalid_component_manifest:{component}")
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "stack": parsed.to_dict(),
        "components": component_validations,
    }


def build_stack_from_manifest(path: str | Path):
    from embryo.run.replay import StackModules

    manifest_path = Path(path)
    manifest = load_stack_manifest(manifest_path)
    root = manifest_path.parent
    validation = validate_stack_manifest(manifest, base_dir=root)
    if not validation["passed"]:
        raise ValueError(f"Invalid stack manifest: {validation['failures']}")
    components = {}
    for name, relative_path in manifest.components.items():
        component_manifest = load_checkpoint_manifest(resolve_component_path(root, relative_path))
        components[name] = build_model_from_manifest(component_manifest)
    return StackModules(fact_writer=components["fact_writer"], router=components["router"], policy=components["bc_policy"])


def resolve_component_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble a stack manifest from component manifests.")
    parser.add_argument("--fact-writer", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--bc-policy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="reference_memory_stack")
    parser.add_argument("--version", default="1")
    args = parser.parse_args(argv)
    out = Path(args.out)
    manifest = assemble_stack_manifest(
        fact_writer=relative_to_out(args.fact_writer, out),
        router=relative_to_out(args.router, out),
        bc_policy=relative_to_out(args.bc_policy, out),
        name=args.name,
        version=args.version,
        metadata={"assembly_boundary": "component_checkpoint_manifests"},
    )
    save_stack_manifest(manifest, out)
    validation = validate_stack_manifest(manifest, base_dir=out.parent)
    (out.parent / "stack_manifest_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "validation": {"passed": validation["passed"], "failure_count": validation["failure_count"]}}, sort_keys=True))
    return 0


def relative_to_out(path_value: str, out: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return str(path.relative_to(out.parent))
        except ValueError:
            return str(path)
    return str(path)
