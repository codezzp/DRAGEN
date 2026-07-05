"""Small numeric helpers for Evidence-v2 construction."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def log1p_pos(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def gini(values: Iterable[float]) -> float:
    xs = sorted(max(float(v), 0.0) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(xs))
    return (2 * weighted) / (n * total) - (n + 1) / n


def entropy_from_counts(counts: Iterable[int]) -> float:
    vals = [c for c in counts if c > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = -sum((c / total) * math.log(c / total + 1e-12) for c in vals)
    return ent / math.log(max(len(vals), 2))


def row_value(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    return to_float(row.get(key), default)
