"""Safe numeric coercion and formatting helpers (shared by services, reporting, notifications)."""

import math
from typing import Any, Iterable, Mapping


def num(mapping: Mapping[str, Any], key: str) -> float:
    return float(mapping.get(key, 0.0) or 0.0)


def to_float(value: Any, default: float = 0.0) -> float:
    """float(value), or float(default) when unconvertible or NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if math.isnan(number) else number


def float_or(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_number(mapping: Mapping[str, Any], keys: Iterable[str]) -> float:
    """First present key that converts (falsy -> 0.0); 0.0 when none does."""
    for key in keys:
        if key not in mapping:
            continue
        try:
            return float(mapping.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return ""


def fmt_pct(value: Any, fallback: str = "") -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return fallback
