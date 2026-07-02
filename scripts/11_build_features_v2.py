from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dragen.features.build_features import (
    TextStatsByWindow,
    count_bad_numbers,
    normalize_node_row,
    normalize_window_row,
    read_tree_features,
    round_float,
    to_float,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "run_0002"
DEFAULT_WINDOW_NAME = "obs_1800_step300_multiscale_hybrid_tree"

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
    "window_position_norm",
    "retweet_growth_rate",
    "retweet_acceleration",
    "active_user_growth_rate",
    "active_user_ratio",
    "edge_density_cur",
    "edge_density_ctx",
    "edge_density_gap",
    "heat_per_user_cur",
    "heat_per_edge_cur",
    "burstiness_ratio",
    "active_span_ratio",
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
    "num_posts_delta_cur",
    "num_posts_growth_rate",
    "in_degree_cur",
    "out_degree_cur",
    "in_degree_ctx",
    "out_degree_ctx",
    "in_degree_cum",
    "out_degree_cum",
    "degree_balance_cur",
    "degree_balance_ctx",
    "degree_balance_cum",
    "degree_growth_rate",
    "active_window_count",
    "active_window_ratio",
    "num_texts_cur",
    "num_texts_visible",
    "text_visibility_ratio",
    "avg_text_len_cur",
    "avg_text_len_visible",
    "text_length_gap",
    "activity_intensity",
    "first_seen_norm",
    "time_since_first_seen_norm",
    "window_position_norm",
    "same_window_active_share",
    "same_window_post_share",
    "node_contribution_share_cur",
    "node_contribution_share_cum",
    "node_degree_share_cur",
    "node_degree_share_cum",
    "reactivation_flag",
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
    out_dir = args.out_dir or run_dir / "features_v2" / DEFAULT_WINDOW_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_features = read_tree_features(args.tree_edges) if args.tree_edges else {}
    edge_stats = read_edge_stats(args.window_dir / "edge_window_table.csv")
    window_summary = collect_window_summary(args.window_dir / "window_table.csv")
    node_summary, cascade_node_totals = collect_node_summary(args.window_dir / "node_window_table.csv")

    diagnostics: Dict[str, Any] = {
        "run_id": args.run_id,
        "window_dir": str(args.window_dir),
        "out_dir": str(out_dir),
        "feature_variant": "v2",
        "jobs": [],
    }
    job_diag = build_feature_tables(
        args.window_dir,
        out_dir,
        tree_features=tree_features,
        edge_stats=edge_stats,
        window_summary=window_summary,
        node_summary=node_summary,
        cascade_node_totals=cascade_node_totals,
    )
    diagnostics["jobs"].append(job_diag)
    write_json(out_dir / "feature_diagnostics.json", diagnostics)
    print(
        f"Wrote v2 features to {out_dir} window_rows={job_diag['num_window_features']} node_rows={job_diag['num_node_window_features']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DRAGEN v2 statistical features.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--window-dir", type=Path, default=PROJECT_ROOT / "work" / "runs" / DEFAULT_RUN_ID / "windows" / DEFAULT_WINDOW_NAME)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--tree-edges",
        type=Path,
        default=PROJECT_ROOT / "work" / "runs" / DEFAULT_RUN_ID / "edges" / "hybrid_tree_light" / "inferred_tree_edge_table.csv",
    )
    return parser.parse_args()


def build_feature_tables(
    window_dir: Path,
    out_dir: Path,
    *,
    tree_features: Mapping[Tuple[str, str], Dict[str, Any]],
    edge_stats: Mapping[Tuple[str, str], Dict[str, float]],
    window_summary: Mapping[str, Dict[str, float]],
    node_summary: Mapping[Tuple[str, str], Dict[str, float]],
    cascade_node_totals: Mapping[str, Dict[str, float]],
) -> Dict[str, Any]:
    window_diag = build_window_features(
        window_dir / "window_table.csv",
        out_dir / "window_features.csv",
        edge_stats=edge_stats,
        window_summary=window_summary,
    )
    node_diag = build_node_features(
        window_dir / "node_window_table.csv",
        out_dir / "node_window_features.csv",
        text_path=window_dir / "text_window_table.csv",
        tree_features=tree_features,
        node_summary=node_summary,
        cascade_node_totals=cascade_node_totals,
        window_summary=window_summary,
    )
    diagnostics = {
        "window_dir": str(window_dir),
        "out_dir": str(out_dir),
        **window_diag,
        **node_diag,
        "edge_scope_counts": count_edge_scopes(window_dir / "edge_window_table.csv"),
        "window_feature_columns": WINDOW_FEATURE_FIELDS,
        "node_feature_columns": NODE_FEATURE_FIELDS,
    }
    write_json(out_dir / "feature_diagnostics.json", diagnostics)
    return diagnostics


def collect_window_summary(path: Path) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            c = str(row["cascade_idx"])
            bucket = summary[c]
            bucket["num_windows"] += 1
            bucket["sum_retweets_cur"] += to_float(row.get("num_retweets_cur"))
            bucket["sum_retweets_cum"] = max(bucket.get("sum_retweets_cum", 0.0), to_float(row.get("num_retweets_cum")))
            bucket["sum_active_users_cur"] += to_float(row.get("num_active_users_cur"))
            bucket["sum_active_users_cum"] = max(bucket.get("sum_active_users_cum", 0.0), to_float(row.get("num_active_users_cum")))
            bucket["sum_edges_cur"] += to_float(row.get("num_edges_cur"))
            bucket["sum_edges_ctx"] += to_float(row.get("num_edges_ctx"))
            bucket["max_heat"] = max(bucket.get("max_heat", 0.0), to_float(row.get("heat_cur")))
            bucket["max_delta_heat"] = max(bucket.get("max_delta_heat", 0.0), to_float(row.get("delta_heat_cur")))
    return {k: dict(v) for k, v in summary.items()}


def collect_node_summary(path: Path) -> tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, Dict[str, float]]]:
    window_summary: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cascade_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            c = str(row["cascade_idx"])
            w = str(row["window_idx"])
            key = (c, w)
            window_bucket = window_summary[key]
            cascade_bucket = cascade_totals[c]

            posts_cur = to_float(row.get("num_posts_cur"))
            posts_cum = to_float(row.get("num_posts_cum"))
            in_cur = to_float(row.get("in_degree_cur"))
            out_cur = to_float(row.get("out_degree_cur"))
            in_ctx = to_float(row.get("in_degree_ctx"))
            out_ctx = to_float(row.get("out_degree_ctx"))
            in_cum = to_float(row.get("in_degree_cum"))
            out_cum = to_float(row.get("out_degree_cum"))
            texts_cur = to_float(row.get("num_texts_cur"))
            texts_visible = to_float(row.get("num_texts_visible"))
            avg_len_cur = to_float(row.get("avg_text_len_cur"))
            avg_len_visible = to_float(row.get("avg_text_len_visible"))
            depth = to_float(row.get("depth"))

            degree_cur = in_cur + out_cur
            degree_ctx = in_ctx + out_ctx
            degree_cum = in_cum + out_cum

            window_bucket["active_count"] += 1.0 if posts_cur > 0 else 0.0
            window_bucket["total_posts_cur"] += posts_cur
            window_bucket["total_posts_cum"] += posts_cum
            window_bucket["total_degree_cur"] += degree_cur
            window_bucket["total_degree_ctx"] += degree_ctx
            window_bucket["total_degree_cum"] += degree_cum
            window_bucket["total_texts_cur"] += texts_cur
            window_bucket["total_texts_visible"] += texts_visible
            window_bucket["sum_avg_text_len_cur"] += avg_len_cur
            window_bucket["sum_avg_text_len_visible"] += avg_len_visible
            window_bucket["max_depth"] = max(window_bucket.get("max_depth", 0.0), depth)

            cascade_bucket["num_rows"] += 1.0
            cascade_bucket["sum_posts_cur"] += posts_cur
            cascade_bucket["sum_posts_cum"] += posts_cum
            cascade_bucket["sum_degree_cur"] += degree_cur
            cascade_bucket["sum_degree_cum"] += degree_cum
            cascade_bucket["sum_texts_cur"] += texts_cur
            cascade_bucket["sum_texts_visible"] += texts_visible
            cascade_bucket["sum_avg_text_len_cur"] += avg_len_cur
            cascade_bucket["sum_avg_text_len_visible"] += avg_len_visible
            cascade_bucket["max_posts_cum"] = max(cascade_bucket.get("max_posts_cum", 0.0), posts_cum)
            cascade_bucket["max_depth"] = max(cascade_bucket.get("max_depth", 0.0), depth)
    return {k: dict(v) for k, v in window_summary.items()}, {k: dict(v) for k, v in cascade_totals.items()}


def build_window_features(
    path: Path,
    out_path: Path,
    *,
    edge_stats: Mapping[Tuple[str, str], Dict[str, float]],
    window_summary: Mapping[str, Dict[str, float]],
) -> Dict[str, Any]:
    row_count = 0
    nan_count = 0
    inf_count = 0
    prev_state: Dict[str, Dict[str, float]] = defaultdict(lambda: {"heat": 0.0, "delta": 0.0, "active": 0.0})
    with path.open("r", encoding="utf-8-sig", newline="") as f, out_path.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=WINDOW_FEATURE_FIELDS)
        writer.writeheader()
        for row in reader:
            feat = normalize_window_row(row)
            c = str(row["cascade_idx"])
            w = int(row["window_idx"])
            summary = window_summary.get(c, {})
            prev = prev_state[c]
            cur_heat = to_float(row.get("heat_cur"))
            delta_heat = to_float(row.get("delta_heat_cur"))
            cur_active = to_float(row.get("num_active_users_cur"))
            ctx_active = to_float(row.get("num_active_users_ctx"))
            cur_edges = to_float(row.get("num_edges_cur"))
            ctx_edges = to_float(row.get("num_edges_ctx"))
            cur_cum = to_float(row.get("num_retweets_cum"))

            feat.update(
                {
                    "window_position_norm": round_float(safe_div(w - 1, max(summary.get("num_windows", 1.0) - 1.0, 1.0))),
                    "retweet_growth_rate": round_float(safe_ratio(delta_heat, max(prev["heat"], 1.0))),
                    "retweet_acceleration": round_float(delta_heat - prev["delta"]),
                    "active_user_growth_rate": round_float(safe_ratio(cur_active - prev["active"], max(prev["active"], 1.0))),
                    "active_user_ratio": round_float(safe_ratio(cur_active, max(to_float(row.get("num_active_users_cum")), 1.0))),
                    "edge_density_cur": round_float(safe_ratio(cur_edges, max(cur_active * max(cur_active - 1.0, 1.0), 1.0))),
                    "edge_density_ctx": round_float(safe_ratio(ctx_edges, max(ctx_active * max(ctx_active - 1.0, 1.0), 1.0))),
                    "edge_density_gap": round_float(safe_ratio(cur_edges, max(cur_active * max(cur_active - 1.0, 1.0), 1.0)) - safe_ratio(ctx_edges, max(ctx_active * max(ctx_active - 1.0, 1.0), 1.0))),
                    "heat_per_user_cur": round_float(safe_ratio(cur_heat, max(cur_active, 1.0))),
                    "heat_per_edge_cur": round_float(safe_ratio(cur_heat, max(cur_edges, 1.0))),
                    "burstiness_ratio": round_float(safe_ratio(delta_heat, max(cur_heat, 1.0))),
                    "active_span_ratio": round_float(safe_ratio(cur_active, max(cur_cum, 1.0))),
                }
            )
            edge_key = (c, str(w))
            if edge_key in edge_stats:
                feat["edge_density_gap"] = round_float(feat["edge_density_gap"])
                feat["heat_per_edge_cur"] = round_float(feat["heat_per_edge_cur"])
            n_nan, n_inf = count_bad_numbers(feat)
            nan_count += n_nan
            inf_count += n_inf
            writer.writerow(feat)
            row_count += 1
            prev_state[c] = {"heat": cur_heat, "delta": delta_heat, "active": cur_active}
    return {"num_window_features": row_count, "window_nan_count": nan_count, "window_inf_count": inf_count}


def build_node_features(
    path: Path,
    out_path: Path,
    *,
    text_path: Path,
    tree_features: Mapping[Tuple[str, str], Dict[str, Any]],
    node_summary: Mapping[Tuple[str, str], Dict[str, float]],
    cascade_node_totals: Mapping[str, Dict[str, float]],
    window_summary: Mapping[str, Dict[str, float]],
) -> Dict[str, Any]:
    row_count = 0
    nan_count = 0
    inf_count = 0
    text_reader = TextStatsByWindow(text_path)
    prev_posts: Dict[Tuple[str, str], float] = defaultdict(float)
    prev_delta: Dict[Tuple[str, str], float] = defaultdict(float)
    current_cascade: Optional[str] = None
    with path.open("r", encoding="utf-8-sig", newline="") as f, out_path.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=NODE_FEATURE_FIELDS)
        writer.writeheader()
        for row in reader:
            c = str(row["cascade_idx"])
            if c != current_cascade:
                current_cascade = c
                for key in [key for key in list(prev_posts.keys()) if key[0] != c]:
                    prev_posts.pop(key, None)
                    prev_delta.pop(key, None)
            w = str(row["window_idx"])
            u = str(row["user_idx"])
            key = (c, u)
            feat = normalize_node_row(row)
            texts = text_reader.get(c, w, u)
            base_summary = node_summary.get((c, w), {})
            cascade_totals = cascade_node_totals.get(c, {})
            cascade_windows = max(window_summary.get(c, {}).get("num_windows", 1.0), 1.0)
            total_posts_cur = max(base_summary.get("total_posts_cur", 0.0), 1.0)
            total_posts_cum = max(cascade_totals.get("sum_posts_cum", 0.0), 1.0)
            total_degree_cur = max(base_summary.get("total_degree_cur", 0.0), 1.0)
            total_degree_cum = max(cascade_totals.get("sum_degree_cum", 0.0), 1.0)
            active_count = max(base_summary.get("active_count", 0.0), 1.0)
            posts_cur = to_float(row.get("num_posts_cur"))
            posts_ctx = to_float(row.get("num_posts_ctx"))
            posts_cum = to_float(row.get("num_posts_cum"))
            in_cur = to_float(row.get("in_degree_cur"))
            out_cur = to_float(row.get("out_degree_cur"))
            in_ctx = to_float(row.get("in_degree_ctx"))
            out_ctx = to_float(row.get("out_degree_ctx"))
            in_cum = to_float(row.get("in_degree_cum"))
            out_cum = to_float(row.get("out_degree_cum"))
            deg_cur = in_cur + out_cur
            deg_ctx = in_ctx + out_ctx
            deg_cum = in_cum + out_cum
            delta_posts = posts_cur - posts_ctx
            prev_posts_value = prev_posts[key]
            prev_delta_value = prev_delta[key]
            active_window_count = to_float(row.get("active_window_count"))
            first_seen = to_float(row.get("first_seen_time"))
            time_since_seen = to_float(row.get("time_since_first_seen"))
            time_denom = max(first_seen + time_since_seen, 1.0)
            window_idx = int(float(w))
            summary_row = node_summary.get((c, w), {})
            same_window_active_share = safe_ratio(1.0, max(summary_row.get("active_count", 0.0), 1.0)) if posts_cur > 0 else 0.0
            same_window_post_share = safe_ratio(posts_cur, total_posts_cur)
            feat.update(
                {
                    "num_posts_delta_cur": round_float(delta_posts),
                    "num_posts_growth_rate": round_float(safe_ratio(delta_posts, max(posts_ctx, 1.0))),
                    "degree_balance_cur": round_float(out_cur - in_cur),
                    "degree_balance_ctx": round_float(out_ctx - in_ctx),
                    "degree_balance_cum": round_float(out_cum - in_cum),
                    "degree_growth_rate": round_float(safe_ratio(deg_cur - deg_ctx, max(deg_ctx, 1.0))),
                    "active_window_ratio": round_float(safe_ratio(active_window_count, cascade_windows)),
                    "text_visibility_ratio": round_float(safe_ratio(to_float(row.get("num_texts_visible")), max(posts_cum, 1.0))),
                    "text_length_gap": round_float(to_float(row.get("avg_text_len_visible")) - to_float(row.get("avg_text_len_cur"))),
                    "activity_intensity": round_float(safe_ratio(posts_cur, max(time_since_seen + 1.0, 1.0))),
                    "first_seen_norm": round_float(safe_ratio(first_seen, time_denom)),
                    "time_since_first_seen_norm": round_float(safe_ratio(time_since_seen, time_denom)),
                    "window_position_norm": round_float(safe_ratio(window_idx - 1, max(cascade_windows - 1.0, 1.0))),
                    "same_window_active_share": round_float(same_window_active_share),
                    "same_window_post_share": round_float(same_window_post_share),
                    "node_contribution_share_cur": round_float(safe_ratio(posts_cur, total_posts_cur)),
                    "node_contribution_share_cum": round_float(safe_ratio(posts_cum, total_posts_cum)),
                    "node_degree_share_cur": round_float(safe_ratio(deg_cur, total_degree_cur)),
                    "node_degree_share_cum": round_float(safe_ratio(deg_cum, total_degree_cum)),
                    "reactivation_flag": int(posts_cur > 0 and prev_posts_value <= 0 and posts_cum > posts_cur),
                }
            )
            tree = tree_features.get((c, u))
            feat.update(tree_feature_values(tree))
            n_nan, n_inf = count_bad_numbers(feat)
            nan_count += n_nan
            inf_count += n_inf
            writer.writerow(feat)
            row_count += 1
            prev_posts[key] = posts_cur
            prev_delta[key] = delta_posts
    text_reader.close()
    return {"num_node_window_features": row_count, "node_nan_count": nan_count, "node_inf_count": inf_count}


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


def read_edge_stats(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row["cascade_idx"]), str(row["window_idx"]))
            bucket = stats[key]
            bucket["edge_count"] += 1.0
            if "follow" in str(row.get("edge_type", "")).lower():
                bucket["follow_edge_count"] += 1.0
            scope = str(row.get("window_scope") or "current")
            if scope == "current":
                bucket["current_edge_count"] += 1.0
            elif scope == "context":
                bucket["context_edge_count"] += 1.0
    return {k: dict(v) for k, v in stats.items()}


def count_edge_scopes(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counts[row.get("window_scope") or "current"] += 1
    return dict(counts)


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def safe_div(num: float, den: float) -> float:
    return safe_ratio(num, den)


if __name__ == "__main__":
    raise SystemExit(main())
