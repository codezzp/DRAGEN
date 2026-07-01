"""Build event-level weak labels for fast DRAGEN experiment closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"

LABEL_FIELDS = ["cascade_idx", "weak_score", "weak_label", "label_confidence", "split"]
COMPONENT_FIELDS = [
    "burst_score",
    "text_repetition_score",
    "coordination_score",
    "structure_concentration_score",
    "persistence_score",
]


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    feature_dir = args.feature_dir or run_dir / "features" / "obs_1800_step300_multiscale_hybrid_tree"
    out_dir = args.out_dir or run_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = aggregate_components(feature_dir)
    normalized = normalize_components(raw)
    labels = assign_labels(normalized)
    write_labels(out_dir / "weak_event_labels.csv", labels)
    diagnostics = build_diagnostics(labels, feature_dir)
    write_json(out_dir / "label_diagnostics.json", diagnostics)
    print(
        f"Wrote weak labels to {out_dir / 'weak_event_labels.csv'} "
        f"cascades={diagnostics['num_cascades']} pos={diagnostics['label_counts'].get('1', 0)} "
        f"neg={diagnostics['label_counts'].get('0', 0)} ignore={diagnostics['label_counts'].get('-1', 0)}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-level weak labels.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def aggregate_components(feature_dir: Path) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with (feature_dir / "window_features.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            bucket = agg[cascade_idx]
            retweets = to_float(row.get("num_retweets_cur"))
            heat = to_float(row.get("heat_cur"))
            delta = to_float(row.get("delta_heat_cur"))
            bucket["max_delta_heat"] = max(bucket["max_delta_heat"], delta)
            bucket["max_heat"] = max(bucket["max_heat"], heat)
            bucket["sum_retweets"] += retweets
            bucket["active_windows"] += 1 if retweets > 0 else 0
            bucket["num_windows"] += 1
    with (feature_dir / "node_window_features.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            bucket = agg[cascade_idx]
            posts_cum = to_float(row.get("num_posts_cum"))
            texts_visible = to_float(row.get("num_texts_visible"))
            out_cum = to_float(row.get("out_degree_cum"))
            in_cum = to_float(row.get("in_degree_cum"))
            bucket["max_user_posts_cum"] = max(bucket["max_user_posts_cum"], posts_cum)
            bucket["sum_visible_texts"] += texts_visible
            bucket["max_degree_cum"] = max(bucket["max_degree_cum"], out_cum + in_cum)
            bucket["sum_degree_cum"] += out_cum + in_cum
            bucket["active_user_window_rows"] += 1
    for bucket in agg.values():
        total_retweets = max(bucket.get("sum_retweets", 0.0), 1.0)
        total_degree = max(bucket.get("sum_degree_cum", 0.0), 1.0)
        total_user_rows = max(bucket.get("active_user_window_rows", 0.0), 1.0)
        bucket["burst_raw"] = bucket.get("max_delta_heat", 0.0) + bucket.get("max_heat", 0.0)
        bucket["text_repetition_raw"] = bucket.get("max_user_posts_cum", 0.0) / total_retweets
        bucket["coordination_raw"] = bucket.get("sum_visible_texts", 0.0) / total_user_rows
        bucket["structure_concentration_raw"] = bucket.get("max_degree_cum", 0.0) / total_degree
        bucket["persistence_raw"] = bucket.get("active_windows", 0.0) / max(bucket.get("num_windows", 0.0), 1.0)
    return dict(agg)


def normalize_components(raw: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    raw_names = {
        "burst_score": "burst_raw",
        "text_repetition_score": "text_repetition_raw",
        "coordination_score": "coordination_raw",
        "structure_concentration_score": "structure_concentration_raw",
        "persistence_score": "persistence_raw",
    }
    ranges: Dict[str, tuple[float, float]] = {}
    for out_name, raw_name in raw_names.items():
        values = [float(bucket.get(raw_name, 0.0)) for bucket in raw.values()]
        ranges[out_name] = (min(values), max(values)) if values else (0.0, 0.0)
    normalized: Dict[str, Dict[str, float]] = {}
    for cascade_idx, bucket in raw.items():
        item: Dict[str, float] = {}
        for out_name, raw_name in raw_names.items():
            lo, hi = ranges[out_name]
            value = float(bucket.get(raw_name, 0.0))
            item[out_name] = (value - lo) / (hi - lo) if hi > lo else 0.0
        item["weak_score"] = (
            0.25 * item["burst_score"]
            + 0.20 * item["text_repetition_score"]
            + 0.20 * item["coordination_score"]
            + 0.20 * item["structure_concentration_score"]
            + 0.15 * item["persistence_score"]
        )
        normalized[str(cascade_idx)] = item
    return normalized


def assign_labels(components: Mapping[str, Mapping[str, float]]) -> List[Dict[str, Any]]:
    items = sorted(components.items(), key=lambda kv: (kv[1]["weak_score"], int(kv[0])))
    n = len(items)
    neg_cut = int(n * 0.50)
    pos_cut = int(n * 0.80)
    rows: List[Dict[str, Any]] = []
    for rank, (cascade_idx, comp) in enumerate(items):
        if rank < neg_cut:
            label = 0
            confidence = 1.0 - rank / max(neg_cut, 1)
        elif rank >= pos_cut:
            label = 1
            confidence = (rank - pos_cut + 1) / max(n - pos_cut, 1)
        else:
            label = -1
            confidence = 0.0
        rows.append(
            {
                "cascade_idx": cascade_idx,
                "weak_score": round(float(comp["weak_score"]), 8),
                "weak_label": label,
                "label_confidence": round(confidence, 8),
                "split": split_for_cascade(cascade_idx),
                **{name: round(float(comp[name]), 8) for name in COMPONENT_FIELDS},
            }
        )
    return sorted(rows, key=lambda row: int(row["cascade_idx"]))


def split_for_cascade(cascade_idx: str) -> str:
    digest = hashlib.md5(str(cascade_idx).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def write_labels(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS + COMPONENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_diagnostics(rows: List[Mapping[str, Any]], feature_dir: Path) -> Dict[str, Any]:
    label_counts: Dict[str, int] = defaultdict(int)
    split_counts: Dict[str, int] = defaultdict(int)
    split_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        label = str(row["weak_label"])
        split = str(row["split"])
        label_counts[label] += 1
        split_counts[split] += 1
        split_label_counts[split][label] += 1
    scores = [float(row["weak_score"]) for row in rows]
    return {
        "feature_dir": str(feature_dir),
        "num_cascades": len(rows),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "split_label_counts": {split: dict(counts) for split, counts in split_label_counts.items()},
        "weak_score_min": min(scores) if scores else 0.0,
        "weak_score_max": max(scores) if scores else 0.0,
    }


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
