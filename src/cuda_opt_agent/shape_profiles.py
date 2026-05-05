"""Shape profile parsing and executable argument helpers."""

from __future__ import annotations

from typing import Any


ShapeProfile = dict[str, Any]


DEFAULT_PROFILES: dict[str, dict[str, list[ShapeProfile]]] = {
    "gemm": {
        "small": [{"M": 1024, "N": 1024, "K": 1024}],
        "medium": [{"M": 2048, "N": 2048, "K": 2048}],
        "large": [{"M": 4096, "N": 4096, "K": 4096}],
        "sweep": [
            {"M": 1024, "N": 1024, "K": 1024},
            {"M": 2048, "N": 2048, "K": 2048},
            {"M": 4096, "N": 4096, "K": 4096},
        ],
    },
    "softmax": {
        "small": [{"B": 1024, "N": 1024}],
        "medium": [{"B": 2048, "N": 2048}],
        "large": [{"B": 4096, "N": 4096}],
        "sweep": [
            {"B": 1024, "N": 1024},
            {"B": 4096, "N": 4096},
        ],
    },
}

OPERATOR_DIM_KEYS = {
    "gemm": ["M", "N", "K"],
    "softmax": ["B", "N"],
}


def dim_keys_for_operator(operator: str, count: int) -> list[str]:
    keys = OPERATOR_DIM_KEYS.get(operator.lower())
    if keys and len(keys) == count:
        return keys
    return [f"D{i}" for i in range(count)]


def dims_to_profile(operator: str, dims: list[int]) -> ShapeProfile:
    return dict(zip(dim_keys_for_operator(operator, len(dims)), dims))


def _parse_dim_token(token: str) -> list[int]:
    token = token.strip()
    if "^" in token:
        base, power = token.split("^", 1)
        return [int(base.strip())] * int(power.strip())
    return [int(part.strip()) for part in token.split(",") if part.strip()]


def _parse_scalar_or_list(value: str) -> Any:
    value = value.strip().strip("[]")
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_shape_profiles(operator: str, shapes: str) -> list[ShapeProfile]:
    """Parse --shapes strings like 1024^3;2048^3 or M=1024,N=1024,K=1024."""
    profiles = []
    for token in shapes.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            profiles.append(dims_to_profile(operator, _parse_dim_token(token)))
            continue
        profile: ShapeProfile = {}
        for part in token.replace(" ", ",").split(","):
            part = part.strip()
            if not part:
                continue
            key, value = part.split("=", 1)
            profile[key.strip()] = _parse_scalar_or_list(value)
        profiles.append(profile)
    return profiles


def default_shape_profiles(operator: str, profile_name: str) -> list[ShapeProfile]:
    operator_profiles = DEFAULT_PROFILES.get(operator.lower())
    if operator_profiles is None:
        raise ValueError(f"No default shape profiles for operator '{operator}'")
    profiles = operator_profiles.get(profile_name.lower())
    if profiles is None:
        choices = ", ".join(sorted(operator_profiles))
        raise ValueError(f"Unknown shape profile '{profile_name}' for {operator}; choose one of: {choices}")
    return [dict(profile) for profile in profiles]


def shape_profile_to_args(profile: ShapeProfile) -> list[str]:
    shape_items = [
        (key, value)
        for key, value in profile.items()
        if key not in {"weight", "_weight"} and not key.startswith("_")
    ]
    if not shape_items:
        return []
    args = ["--shape"]
    for key, value in shape_items:
        if isinstance(value, (list, tuple)):
            value_str = ",".join(str(v) for v in value)
        else:
            value_str = str(value)
        args.append(f"{key}={value_str}")
    return args


def shape_profile_label(profile: ShapeProfile) -> str:
    parts = []
    for key, value in profile.items():
        if key in {"weight", "_weight"} or key.startswith("_"):
            continue
        if isinstance(value, (list, tuple)):
            value_str = "x".join(str(v) for v in value)
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return ",".join(parts) or "default"


def profile_weight(profile: ShapeProfile) -> float:
    value = profile.get("_weight", profile.get("weight", 1.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0
