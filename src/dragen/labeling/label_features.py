"""Shared event-level features for weak-label builders."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"
DEFAULT_WINDOW_NAME = "obs_1800_step300_multiscale_hybrid_tree"
SIZE_BUCKETS = [
    (8, 20, "8-19"),
    (20, 50, "20-49"),
    (50, 100, "50-99"),
    (100, 300, "100-299"),
    (300, None, "300+"),
]
LABEL_FIELDS = [
    "cascade_idx",
    "label",
    "split",
    "label_confidence",
    "weak_score",
    "label_method",
    "size_bucket",
    "observed_retweet_count",
    "final_retweet_count",
]


def default_run_dir(run_id: str) -> Path:
    return PROJECT_ROOT / "work" / "runs" / run_id


def default_feature_dir(run_id: str) -> Path:
    return default_run_dir(run_id) / "features" / DEFAULT_WINDOW_NAME


def default_cascade_table(run_id: str) -> Path:
    return default_run_dir(run_id) / "org_task" / "cascade_table.csv"


def default_global_candidate_edges(run_id: str) -> Path:
    return default_run_dir(run_id) / "global_graph" / DEFAULT_WINDOW_NAME / "global_candidate_edge_table.csv"


def split_for_cascade(cascade_idx: str) -> str:
    digest = hashlib.md5(str(cascade_idx).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def size_bucket(observed_retweet_count: float) -> str:
    for left, right, label in SIZE_BUCKETS:
        if observed_retweet_count >= left and (right is None or observed_retweet_count < right):
            return label
    return "<8"


def load_event_features(
    feature_dir: Path,
    cascade_table: Path,
    global_candidate_edges: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    features = read_cascade_base(cascade_table)
    aggregate_window_features(feature_dir / "window_features.csv", features)
    aggregate_node_features(feature_dir / "node_window_features.csv", features)
    if global_candidate_edges is not None and global_candidate_edges.exists():
        aggregate_follow_candidates(global_candidate_edges, features)
    finalize_features(features)
    return features


def read_cascade_base(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            observed = to_float(row.get("observed_retweet_count"))
            final = to_float(row.get("final_retweet_count"))
            out[cascade_idx] = {
                "cascade_idx": cascade_idx,
                "observed_retweet_count": observed,
                "final_retweet_count": final,
                "size_bucket": size_bucket(observed),
                "split": split_for_cascade(cascade_idx),
                "duration": to_float(row.get("duration")),
                "observed_duration": to_float(row.get("observed_duration")),
                "root_text_len": float(len(row.get("root_text", "") or "")),
            }
    return out


def aggregate_window_features(path: Path, features: Dict[str, Dict[str, Any]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item = features.get(str(row["cascade_idx"]))
            if item is None:
                continue
            retweets = to_float(row.get("num_retweets_cur"))
            heat = to_float(row.get("heat_cur"))
            delta = to_float(row.get("delta_heat_cur"))
            active = to_float(row.get("num_active_users_cur"))
            item["sum_retweets_cur"] = item.get("sum_retweets_cur", 0.0) + retweets
            item["sum_active_users_cur"] = item.get("sum_active_users_cur", 0.0) + active
            item["max_heat"] = max(item.get("max_heat", 0.0), heat)
            item["max_delta_heat"] = max(item.get("max_delta_heat", 0.0), delta)
            item["max_window_retweets"] = max(item.get("max_window_retweets", 0.0), retweets)
            item["active_windows"] = item.get("active_windows", 0.0) + (1.0 if retweets > 0 else 0.0)
            item["num_windows"] = item.get("num_windows", 0.0) + 1.0


def aggregate_node_features(path: Path, features: Dict[str, Dict[str, Any]]) -> None:
    users_by_cascade: Dict[str, set[str]] = defaultdict(set)
    first_seen_values: Dict[str, List[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            item = features.get(cascade_idx)
            if item is None:
                continue
            user_idx = str(row.get("user_idx", ""))
            if user_idx:
                users_by_cascade[cascade_idx].add(user_idx)
            posts_cur = to_float(row.get("num_posts_cur"))
            posts_cum = to_float(row.get("num_posts_cum"))
            texts_cur = to_float(row.get("num_texts_cur"))
            texts_visible = to_float(row.get("num_texts_visible"))
            avg_text_len = to_float(row.get("avg_text_len_visible"))
            degree_cum = to_float(row.get("in_degree_cum")) + to_float(row.get("out_degree_cum"))
            degree_ctx = to_float(row.get("in_degree_ctx")) + to_float(row.get("out_degree_ctx"))
            depth = to_float(row.get("depth"))
            parent_score = to_float(row.get("parent_score"))
            text_score = to_float(row.get("text_score"))
            first_seen = to_float(row.get("first_seen_time"))
            if posts_cum > 0 or texts_visible > 0:
                first_seen_values[cascade_idx].append(first_seen)
            item["max_user_posts_cum"] = max(item.get("max_user_posts_cum", 0.0), posts_cum)
            item["sum_posts_cur"] = item.get("sum_posts_cur", 0.0) + posts_cur
            item["sum_texts_cur"] = item.get("sum_texts_cur", 0.0) + texts_cur
            item["sum_texts_visible"] = item.get("sum_texts_visible", 0.0) + texts_visible
            item["sum_avg_text_len_visible"] = item.get("sum_avg_text_len_visible", 0.0) + avg_text_len
            item["max_degree_cum"] = max(item.get("max_degree_cum", 0.0), degree_cum)
            item["sum_degree_cum"] = item.get("sum_degree_cum", 0.0) + degree_cum
            item["sum_degree_ctx"] = item.get("sum_degree_ctx", 0.0) + degree_ctx
            item["max_depth"] = max(item.get("max_depth", 0.0), depth)
            item["sum_depth"] = item.get("sum_depth", 0.0) + depth
            item["sum_parent_score"] = item.get("sum_parent_score", 0.0) + parent_score
            item["sum_text_score"] = item.get("sum_text_score", 0.0) + text_score
            item["node_rows"] = item.get("node_rows", 0.0) + 1.0
    for cascade_idx, users in users_by_cascade.items():
        features[cascade_idx]["num_visible_users"] = float(len(users))
    for cascade_idx, values in first_seen_values.items():
        features[cascade_idx]["first_seen_std"] = stddev(values)
        features[cascade_idx]["first_seen_span"] = max(values) - min(values) if values else 0.0


def aggregate_follow_candidates(path: Path, features: Dict[str, Dict[str, Any]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item = features.get(str(row["cascade_idx"]))
            if item is None:
                continue
            item["global_candidate_edges"] = item.get("global_candidate_edges", 0.0) + 1.0


def finalize_features(features: Dict[str, Dict[str, Any]]) -> None:
    for item in features.values():
        observed = max(float(item.get("observed_retweet_count", 0.0)), 0.0)
        final = max(float(item.get("final_retweet_count", 0.0)), 0.0)
        users = max(float(item.get("num_visible_users", 0.0)), 1.0)
        rows = max(float(item.get("node_rows", 0.0)), 1.0)
        total_degree = max(float(item.get("sum_degree_cum", 0.0)), 1.0)
        duration = max(float(item.get("observed_duration", 0.0)), 1.0)
        item["burst_raw"] = item.get("max_delta_heat", 0.0) + item.get("max_heat", 0.0) + safe_ratio(item.get("max_window_retweets", 0.0), observed + 1.0)
        item["coordination_raw"] = safe_ratio(item.get("sum_active_users_cur", 0.0), item.get("active_windows", 0.0) + 1.0) + safe_ratio(users, math.log1p(duration))
        item["structure_raw"] = safe_ratio(item.get("max_degree_cum", 0.0), total_degree) + safe_ratio(item.get("sum_degree_ctx", 0.0), rows) + safe_ratio(item.get("global_candidate_edges", 0.0), users * users)
        item["text_raw"] = safe_ratio(item.get("sum_text_score", 0.0), rows) + safe_ratio(item.get("sum_texts_visible", 0.0), rows) + safe_ratio(item.get("max_user_posts_cum", 0.0), observed + 1.0)
        item["temporal_sync_raw"] = safe_ratio(1.0, item.get("first_seen_std", 0.0) + 1.0) + safe_ratio(item.get("max_window_retweets", 0.0), observed + 1.0)
        item["follow_density_raw"] = safe_ratio(item.get("global_candidate_edges", 0.0), users * max(users - 1.0, 1.0))
        item["natural_spread_raw"] = math.log1p(observed) + item.get("first_seen_span", 0.0) / 1800.0 - item["coordination_raw"] - item["structure_raw"]
        item["observed_retweet_count"] = observed
        item["final_retweet_count"] = final
        item["size_bucket"] = size_bucket(observed)
        item["split"] = split_for_cascade(str(item["cascade_idx"]))


def add_percentile_scores(features: Dict[str, Dict[str, Any]], raw_to_score: Mapping[str, str], *, by_bucket: bool = False) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in features.values():
        groups[str(item.get("size_bucket", "all")) if by_bucket else "all"].append(item)
    for rows in groups.values():
        for raw_name, score_name in raw_to_score.items():
            ranked = percentile_rank([float(row.get(raw_name, 0.0)) for row in rows])
            for row, score in zip(rows, ranked):
                row[score_name] = score


def percentile_rank(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (values[i], i))
    out = [0.0] * n
    if n == 1:
        out[order[0]] = 1.0
        return out
    for rank, idx in enumerate(order):
        out[idx] = rank / (n - 1)
    return out


def bucket_quantiles(rows: Iterable[Mapping[str, Any]], field: str, qs: Sequence[float]) -> Dict[str, Dict[float, float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("size_bucket", ""))].append(float(row.get(field, 0.0)))
    return {bucket: {q: quantile(vals, q) for q in qs} for bucket, vals in groups.items()}


def quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def make_label_row(item: Mapping[str, Any], label: int, confidence: float, weak_score: float, method: str) -> Dict[str, Any]:
    return {
        "cascade_idx": str(item["cascade_idx"]),
        "label": int(label),
        "split": str(item["split"]),
        "label_confidence": round(float(confidence), 8),
        "weak_score": round(float(weak_score), 8),
        "label_method": method,
        "size_bucket": str(item["size_bucket"]),
        "observed_retweet_count": int(float(item.get("observed_retweet_count", 0.0))),
        "final_retweet_count": int(float(item.get("final_retweet_count", 0.0))),
    }


def write_label_csv(path: Path, rows: Iterable[Mapping[str, Any]], extra_fields: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = LABEL_FIELDS + [field for field in extra_fields if field not in LABEL_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_label_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
