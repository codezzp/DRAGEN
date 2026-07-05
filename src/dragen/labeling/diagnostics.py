"""Diagnostics for weak label versions."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def build_label_diagnostics(rows: List[Mapping[str, Any]], *, extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    n = len(rows)
    counts = {str(label): labels.count(label) for label in [-1, 0, 1]}
    observed = [float(row.get("observed_retweet_count", 0.0)) for row in rows]
    weak_scores = [float(row.get("weak_score", 0.0)) for row in rows]
    warnings: List[str] = []
    corr = pearson(observed, [float(v) for v in labels]) if rows else 0.0
    if abs(corr) >= 0.5:
        warnings.append(f"observed_retweet_count_label_correlation_high:{corr:.4f}")
    by_bucket = label_by(rows, "size_bucket")
    for bucket, bucket_counts in by_bucket.items():
        if bucket_counts.get("1", 0) < 10:
            warnings.append(f"few_positive_in_bucket:{bucket}:{bucket_counts.get('1', 0)}")
        if bucket_counts.get("0", 0) < 10:
            warnings.append(f"few_negative_in_bucket:{bucket}:{bucket_counts.get('0', 0)}")
    diag: Dict[str, Any] = {
        "num_cascades": n,
        "positive": counts.get("1", 0),
        "negative": counts.get("0", 0),
        "ignore": counts.get("-1", 0),
        "label_ratio": {key: value / max(n, 1) for key, value in counts.items()},
        "label_by_size_bucket": by_bucket,
        "observed_retweet_count_by_label": describe_by_label(rows, "observed_retweet_count"),
        "score_by_label": describe_by_label(rows, "weak_score"),
        "feature_correlation_with_label": {
            "observed_retweet_count": corr,
            "weak_score": pearson(weak_scores, [float(v) for v in labels]) if rows else 0.0,
        },
        "split_distribution": split_distribution(rows),
        "warnings": warnings,
    }
    if extra:
        diag.update(extra)
    return diag


def label_by(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        out[str(row.get(field, ""))][str(row["label"])] += 1
    return {key: dict(value) for key, value in out.items()}


def split_distribution(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        out[str(row.get("split", ""))][str(row["label"])] += 1
    return {key: dict(value) for key, value in out.items()}


def describe_by_label(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        values[str(row["label"])].append(float(row.get(field, 0.0)))
    return {label: describe(vals) for label, vals in values.items()}


def describe(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"count": 0, "mean": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "p50": vals[len(vals) // 2],
        "max": vals[-1],
    }


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return num / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
