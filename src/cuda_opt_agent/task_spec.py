"""
Task specification loading for structured CUDA optimization requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models.data import OperatorSpec


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 only
        import tomli as tomllib

    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Spec file must contain a table/object: {path}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as e:  # pragma: no cover - dependency guard
        raise RuntimeError("YAML specs require PyYAML to be installed") from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Spec file must contain a mapping/object: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Spec file must contain an object: {path}")
    return data


def resolve_existing_cuda_path(path: str | Path, *, base_dir: Path | None = None) -> str:
    """Resolve and validate an existing .cu file path."""
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    candidate = candidate.expanduser().resolve()

    if candidate.suffix.lower() != ".cu":
        raise ValueError(f"Seed code must be a .cu file: {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"Seed code file not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Seed code path is not a file: {candidate}")
    return str(candidate)


def load_operator_spec(path: str | Path) -> OperatorSpec:
    """Load an OperatorSpec from YAML, TOML, or JSON."""
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    if not spec_path.is_file():
        raise ValueError(f"Spec path is not a file: {spec_path}")

    suffix = spec_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _load_yaml(spec_path)
    elif suffix == ".toml":
        data = _load_toml(spec_path)
    elif suffix == ".json":
        data = _load_json(spec_path)
    else:
        raise ValueError(f"Unsupported spec format '{suffix}'; use .yaml, .yml, .toml, or .json")

    spec = OperatorSpec.model_validate(data)
    if spec.seed_code_path:
        spec.seed_code_path = resolve_existing_cuda_path(spec.seed_code_path, base_dir=spec_path.parent)
    return spec
