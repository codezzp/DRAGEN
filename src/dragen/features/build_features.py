"""Build lightweight statistical features from window tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"

WINDOW_FEATURE_FIELDS = [
    "cascade_idx",
    "window_idx",
    "num_retweets_cur",
    "num_retweets_ctx",
    "num_retweets_cum",
    "num_active_users_cur",
    "num_active_users_ctx",
    "num_active_users_cum",
    "num_edges_cur",
    "num_edges_ctx",
    "heat_cur",
    "heat_ctx",
    "heat_cum",
    "delta_heat_cur",
]

NODE_FEATURE_FIELDS = [
    "cascade_idx",
    "window_idx",
    "user_idx",
    "is_root",
    "first_seen_time",
    "time_since_first_seen",
    "num_posts_cur",
    "num_posts_ctx",
    "num_posts_cum",
    "in_degree_cur",
    "out_degree_cur",
    "in_degree_ctx",
    "out_degree_ctx",
    "in_degree_cum",
    "out_degree_cum",
    "active_window_count",
    "num_texts_cur",
    "num_texts_visible",
    "avg_text_len_cur",
    "avg_text_len_visible",
    "has_tree_feature",
    "depth",
    "parent_time_gap",
    "parent_score",
    "time_score",
    "text_score",
    "activity_score",
    "depth_penalty",
    "load_penalty",
    "root_fallback_flag",
]


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    if args.window_dir:
        jobs = [(args.window_dir, args.out_dir or default_feature_dir(run_dir, args.window_dir.name))]
    else:
        jobs = default_jobs(run_dir)

    tree_features = read_tree_features(args.tree_edges) if args.tree_edges else {}
    diagnostics: Dict[str, Any] = {"run_id": args.run_id, "jobs": []}
    for window_dir, out_dir in jobs:
        if not window_dir.exists():
            raise FileNotFoundError(f"window directory not found: {window_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        job_diag = build_feature_tables(window_dir, out_dir, tree_features)
        diagnostics["jobs"].append(job_diag)
        print(
            f"Wrote features to {out_dir} "
            f"window_rows={job_diag['num_window_features']} node_rows={job_diag['num_node_window_features']}"
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DRAGEN v1 statistical features.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--window-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--tree-edges",
        type=Path,
        default=PROJECT_ROOT / "work" / "runs" / DEFAULT_RUN_ID / "edges" / "hybrid_tree_light" / "inferred_tree_edge_table.csv",
    )
    return parser.parse_args()


def default_jobs(run_dir: Path) -> List[Tuple[Path, Path]]:
    windows = run_dir / "windows"
    features = run_dir / "features"
    star_input = windows / "obs_1800_win300_step300_star"
    if not star_input.exists():
        star_input = windows / "obs_1800_win300_step300"
    return [
        (star_input, features / "obs_1800_win300_step300_star"),
        (windows / "obs_1800_win300_step300_hybrid_tree", features / "obs_1800_win300_step300_hybrid_tree"),
        (
            windows / "obs_1800_step300_multiscale_hybrid_tree",
            features / "obs_1800_step300_multiscale_hybrid_tree",
        ),
    ]


def default_feature_dir(run_dir: Path, window_name: str) -> Path:
    return run_dir / "features" / window_name


def build_feature_tables(
    window_dir: Path,
    out_dir: Path,
    tree_features: Mapping[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    window_diag = build_window_features(window_dir / "window_table.csv", out_dir / "window_features.csv")
    node_diag = build_node_features(
        window_dir / "node_window_table.csv",
        out_dir / "node_window_features.csv",
        text_path=window_dir / "text_window_table.csv",
        tree_features=tree_features,
    )
    edge_scope_counts = count_edge_scopes(window_dir / "edge_window_table.csv")
    diagnostics = {
        "window_dir": str(window_dir),
        "out_dir": str(out_dir),
        **window_diag,
        **node_diag,
        "edge_scope_counts": edge_scope_counts,
    }
    write_json(out_dir / "feature_diagnostics.json", diagnostics)
    return diagnostics


def build_window_features(path: Path, out_path: Path) -> Dict[str, Any]:
    row_count = 0
    nan_count = 0
    inf_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=WINDOW_FEATURE_FIELDS)
        writer.writeheader()
        for row in reader:
            feat = normalize_window_row(row)
            n_nan, n_inf = count_bad_numbers(feat)
            nan_count += n_nan
            inf_count += n_inf
            writer.writerow(feat)
            row_count += 1
    return {"num_window_features": row_count, "window_nan_count": nan_count, "window_inf_count": inf_count}


def build_node_features(
    path: Path,
    out_path: Path,
    *,
    text_path: Path,
    tree_features: Mapping[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    row_count = 0
    nan_count = 0
    inf_count = 0
    text_reader = TextStatsByWindow(text_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=NODE_FEATURE_FIELDS)
        writer.writeheader()
        for row in reader:
            feat = normalize_node_row(row)
            texts = text_reader.get(str(row["cascade_idx"]), str(row["window_idx"]), str(row["user_idx"]))
            feat.update(
                {
                    "num_texts_cur": int(texts.get("num_texts_cur", 0)),
                    "num_texts_visible": int(texts.get("num_texts_visible", 0)),
                    "avg_text_len_cur": round_float(texts.get("avg_text_len_cur", 0.0)),
                    "avg_text_len_visible": round_float(texts.get("avg_text_len_visible", 0.0)),
                }
            )
            tree = tree_features.get((str(row["cascade_idx"]), str(row["user_idx"])))
            feat.update(tree_feature_values(tree))
            n_nan, n_inf = count_bad_numbers(feat)
            nan_count += n_nan
            inf_count += n_inf
            writer.writerow(feat)
            row_count += 1
    text_reader.close()
    return {"num_node_window_features": row_count, "node_nan_count": nan_count, "node_inf_count": inf_count}


def normalize_window_row(row: Mapping[str, str]) -> Dict[str, Any]:
    if "num_retweets_cur" in row:
        return {
            "cascade_idx": row["cascade_idx"],
            "window_idx": row["window_idx"],
            "num_retweets_cur": to_int(row.get("num_retweets_cur")),
            "num_retweets_ctx": to_int(row.get("num_retweets_ctx")),
            "num_retweets_cum": to_int(row.get("num_retweets_cum")),
            "num_active_users_cur": to_int(row.get("num_active_users_cur")),
            "num_active_users_ctx": to_int(row.get("num_active_users_ctx")),
            "num_active_users_cum": to_int(row.get("num_active_users_cum")),
            "num_edges_cur": to_int(row.get("num_edges_cur")),
            "num_edges_ctx": to_int(row.get("num_edges_ctx")),
            "heat_cur": round_float(row.get("heat_cur")),
            "heat_ctx": round_float(row.get("heat_ctx")),
            "heat_cum": round_float(row.get("heat_cum")),
            "delta_heat_cur": round_float(row.get("delta_heat_cur")),
        }
    cur_retweets = to_int(row.get("num_retweets"))
    cur_users = to_int(row.get("num_active_users"))
    cur_edges = to_int(row.get("num_edges"))
    cur_heat = round_float(row.get("window_heat"))
    return {
        "cascade_idx": row["cascade_idx"],
        "window_idx": row["window_idx"],
        "num_retweets_cur": cur_retweets,
        "num_retweets_ctx": cur_retweets,
        "num_retweets_cum": to_int(row.get("cum_retweets")),
        "num_active_users_cur": cur_users,
        "num_active_users_ctx": cur_users,
        "num_active_users_cum": cur_users,
        "num_edges_cur": cur_edges,
        "num_edges_ctx": cur_edges,
        "heat_cur": cur_heat,
        "heat_ctx": cur_heat,
        "heat_cum": cur_heat,
        "delta_heat_cur": round_float(row.get("delta_heat")),
    }


def normalize_node_row(row: Mapping[str, str]) -> Dict[str, Any]:
    if "num_posts_cur" in row:
        return {
            "cascade_idx": row["cascade_idx"],
            "window_idx": row["window_idx"],
            "user_idx": row["user_idx"],
            "is_root": to_int(row.get("is_root")),
            "first_seen_time": to_int(row.get("first_seen_time")),
            "time_since_first_seen": to_int(row.get("time_since_first_seen")),
            "num_posts_cur": to_int(row.get("num_posts_cur")),
            "num_posts_ctx": to_int(row.get("num_posts_ctx")),
            "num_posts_cum": to_int(row.get("num_posts_cum")),
            "in_degree_cur": to_int(row.get("in_degree_cur")),
            "out_degree_cur": to_int(row.get("out_degree_cur")),
            "in_degree_ctx": to_int(row.get("in_degree_ctx")),
            "out_degree_ctx": to_int(row.get("out_degree_ctx")),
            "in_degree_cum": to_int(row.get("in_degree_cum")),
            "out_degree_cum": to_int(row.get("out_degree_cum")),
            "active_window_count": to_int(row.get("active_window_count")),
        }
    return {
        "cascade_idx": row["cascade_idx"],
        "window_idx": row["window_idx"],
        "user_idx": row["user_idx"],
        "is_root": to_int(row.get("is_root")),
        "first_seen_time": to_int(row.get("first_seen_time")),
        "time_since_first_seen": to_int(row.get("time_since_first_seen")),
        "num_posts_cur": to_int(row.get("num_posts_in_window")),
        "num_posts_ctx": to_int(row.get("num_posts_in_window")),
        "num_posts_cum": to_int(row.get("cum_posts")),
        "in_degree_cur": to_int(row.get("in_degree_window")),
        "out_degree_cur": to_int(row.get("out_degree_window")),
        "in_degree_ctx": to_int(row.get("in_degree_window")),
        "out_degree_ctx": to_int(row.get("out_degree_window")),
        "in_degree_cum": to_int(row.get("cum_in_degree")),
        "out_degree_cum": to_int(row.get("cum_out_degree")),
        "active_window_count": 1 if to_int(row.get("num_posts_in_window")) > 0 else 0,
    }


class TextStatsByWindow:
    """Stream text statistics one cascade/window block at a time."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("r", encoding="utf-8-sig", newline="")
        self._reader = csv.DictReader(self._file)
        self._pending: Optional[Dict[str, str]] = None
        self._current_key: Optional[Tuple[int, int]] = None
        self._current_stats: Dict[str, Dict[str, float]] = {}
        self._done = False

    def get(self, cascade_idx: str, window_idx: str, user_idx: str) -> Dict[str, float]:
        target = (int(cascade_idx), int(window_idx))
        while not self._done and (self._current_key is None or self._current_key < target):
            self._load_next_window()
        if self._current_key == target:
            return self._current_stats.get(str(user_idx), {})
        return {}

    def close(self) -> None:
        self._file.close()

    def _load_next_window(self) -> None:
        first = self._pending
        if first is None:
            try:
                first = next(self._reader)
            except StopIteration:
                self._done = True
                self._current_key = None
                self._current_stats = {}
                return
        self._pending = None
        key = (int(first["cascade_idx"]), int(first["window_idx"]))
        stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._add_text_row(stats, first)
        for row in self._reader:
            row_key = (int(row["cascade_idx"]), int(row["window_idx"]))
            if row_key != key:
                self._pending = row
                break
            self._add_text_row(stats, row)
        self._current_key = key
        self._current_stats = {user_idx: self._finalize_text_bucket(bucket) for user_idx, bucket in stats.items()}

    @staticmethod
    def _add_text_row(stats: Dict[str, Dict[str, float]], row: Mapping[str, str]) -> None:
        user_idx = str(row["user_idx"])
        text_len = len(row.get("text") or "")
        bucket = stats[user_idx]
        bucket["num_texts_visible"] += 1
        bucket["sum_text_len_visible"] += text_len
        if row.get("post_type") != "root":
            bucket["num_texts_cur"] += 1
            bucket["sum_text_len_cur"] += text_len

    @staticmethod
    def _finalize_text_bucket(bucket: Mapping[str, float]) -> Dict[str, float]:
        visible = bucket.get("num_texts_visible", 0.0)
        cur = bucket.get("num_texts_cur", 0.0)
        return {
            "num_texts_visible": visible,
            "num_texts_cur": cur,
            "avg_text_len_visible": bucket.get("sum_text_len_visible", 0.0) / visible if visible else 0.0,
            "avg_text_len_cur": bucket.get("sum_text_len_cur", 0.0) / cur if cur else 0.0,
        }


def read_tree_features(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    accum: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row["cascade_idx"]), str(row["child_user_idx"]))
            bucket = accum[key]
            bucket["count"] += 1
            bucket["depth"] = max(bucket["depth"], to_float(row.get("child_depth")))
            for src, dst in [
                ("time_gap", "parent_time_gap"),
                ("parent_score", "parent_score"),
                ("time_score", "time_score"),
                ("text_score", "text_score"),
                ("activity_score", "activity_score"),
                ("depth_penalty", "depth_penalty"),
                ("load_penalty", "load_penalty"),
                ("root_fallback_flag", "root_fallback_flag"),
            ]:
                bucket[dst] += to_float(row.get(src))
    features: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, bucket in accum.items():
        count = max(bucket.pop("count", 0.0), 1.0)
        features[key] = {
            "depth": round_float(bucket.get("depth", 0.0)),
            "parent_time_gap": round_float(bucket.get("parent_time_gap", 0.0) / count),
            "parent_score": round_float(bucket.get("parent_score", 0.0) / count),
            "time_score": round_float(bucket.get("time_score", 0.0) / count),
            "text_score": round_float(bucket.get("text_score", 0.0) / count),
            "activity_score": round_float(bucket.get("activity_score", 0.0) / count),
            "depth_penalty": round_float(bucket.get("depth_penalty", 0.0) / count),
            "load_penalty": round_float(bucket.get("load_penalty", 0.0) / count),
            "root_fallback_flag": 1 if bucket.get("root_fallback_flag", 0.0) > 0 else 0,
        }
    return features


def tree_feature_values(tree: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not tree:
        return {
            "has_tree_feature": 0,
            "depth": 0,
            "parent_time_gap": 0,
            "parent_score": 0,
            "time_score": 0,
            "text_score": 0,
            "activity_score": 0,
            "depth_penalty": 0,
            "load_penalty": 0,
            "root_fallback_flag": 0,
        }
    return {"has_tree_feature": 1, **dict(tree)}


def count_edge_scopes(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counts[row.get("window_scope") or "current"] += 1
    return dict(counts)


def count_bad_numbers(row: Mapping[str, Any]) -> Tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for value in row.values():
        if isinstance(value, (int, float)):
            if math.isnan(float(value)):
                nan_count += 1
            if math.isinf(float(value)):
                inf_count += 1
    return nan_count, inf_count


def to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def round_float(value: Any, digits: int = 8) -> float:
    return round(to_float(value), digits)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
