"""Top-level window construction entry point."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dragen.windowing.edge_window_builder import build_edge_window_rows
from dragen.windowing.node_window_builder import build_node_window_rows
from dragen.windowing.text_window_builder import build_text_window_rows


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"

WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "start_offset",
    "end_offset",
    "num_retweets",
    "cum_retweets",
    "num_active_users",
    "num_edges",
    "window_heat",
    "delta_heat",
]

NODE_WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "user_idx",
    "first_seen_time",
    "is_root",
    "num_posts_in_window",
    "cum_posts",
    "in_degree_window",
    "out_degree_window",
    "cum_in_degree",
    "cum_out_degree",
    "time_since_root",
    "time_since_first_seen",
]

EDGE_WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "src_user_idx",
    "dst_user_idx",
    "src_tweet_idx",
    "dst_tweet_idx",
    "edge_time",
    "edge_offset",
    "edge_type",
]

TEXT_WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "user_idx",
    "tweet_idx",
    "post_type",
    "text",
    "text_visible_type",
    "post_offset",
]

MULTISCALE_WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "endpoint",
    "cur_start_offset",
    "ctx_start_offset",
    "cum_start_offset",
    "end_offset",
    "num_retweets_cur",
    "num_retweets_ctx",
    "num_retweets_cum",
    "num_active_users_cur",
    "num_active_users_ctx",
    "num_active_users_cum",
    "num_edges_cur",
    "num_edges_ctx",
    "num_edges_cum",
    "heat_cur",
    "heat_ctx",
    "heat_cum",
    "delta_heat_cur",
]

MULTISCALE_NODE_WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "user_idx",
    "first_seen_time",
    "is_root",
    "num_posts_cur",
    "num_posts_ctx",
    "num_posts_cum",
    "in_degree_cur",
    "out_degree_cur",
    "in_degree_ctx",
    "out_degree_ctx",
    "in_degree_cum",
    "out_degree_cum",
    "time_since_first_seen",
    "active_window_count",
]

MULTISCALE_EDGE_WINDOW_FIELDS = [
    "cascade_idx",
    "window_idx",
    "window_scope",
    "src_user_idx",
    "dst_user_idx",
    "src_tweet_idx",
    "dst_tweet_idx",
    "edge_time",
    "edge_offset",
    "edge_type",
]


def main() -> int:
    args = parse_args()
    config = load_simple_config(args.window_config) if args.window_config else {}

    run_id = args.run_id or str(config.get("run_id") or DEFAULT_RUN_ID)
    obs_seconds = int(args.obs_seconds or config.get("obs_seconds") or 1800)
    step_seconds = int(args.step_seconds or config.get("step_seconds") or 300)
    window_size_seconds = int(
        args.window_size_seconds
        or config.get("window_size_seconds")
        or config.get("window_seconds")
        or step_seconds
    )
    edge_mode = str(args.edge_mode or config.get("edge_mode") or "star")
    window_mode = str(args.window_mode or config.get("window_mode") or "fixed")
    context_window_seconds = int(config.get("context_window_seconds") or 600)

    run_dir = PROJECT_ROOT / "work" / "runs" / run_id
    org_task_dir = args.org_task_dir or run_dir / "org_task"
    edge_suffix = "tree" if edge_mode == "inferred_tree" else "star"
    default_name = (
        f"obs_{obs_seconds}_step{step_seconds}_multiscale_{edge_suffix}"
        if window_mode == "causal_multiscale"
        else f"obs_{obs_seconds}_win{window_size_seconds}_step{step_seconds}_{edge_suffix}"
    )
    out_dir = args.out_dir or run_dir / "windows" / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cascade_filter = read_cascade_filter(args.cascade_list)
    posts_by_cascade = read_grouped_csv(
        org_task_dir / "post_table.csv",
        cascade_filter=cascade_filter,
        max_cascades=args.max_cascades,
        stop_after_max=True,
    )
    keep = set(posts_by_cascade)
    edge_table = (
        args.inferred_tree_edge_table
        if args.inferred_tree_edge_table is not None
        else run_dir / "edges" / "inferred_tree_edge_table.csv"
        if edge_mode == "inferred_tree"
        else org_task_dir / "cascade_edge_table.csv"
    )
    edges_by_cascade = read_grouped_csv(
        edge_table,
        cascade_filter=keep,
        max_cascades=None,
        stop_after_max=bool(args.max_cascades),
    )
    if edge_mode == "inferred_tree":
        edges_by_cascade = normalize_inferred_tree_edges(edges_by_cascade)

    if window_mode == "causal_multiscale":
        windows_by_cascade = build_multiscale_windows_by_cascade(
            posts_by_cascade.keys(), obs_seconds, step_seconds, window_size_seconds, context_window_seconds
        )
        window_rows = build_multiscale_window_rows(posts_by_cascade, edges_by_cascade, windows_by_cascade)
        node_rows = build_multiscale_node_window_rows(posts_by_cascade, edges_by_cascade, windows_by_cascade)
        edge_rows = build_multiscale_edge_window_rows(edges_by_cascade, windows_by_cascade)
        text_rows = build_text_window_rows(posts_by_cascade, windows_by_cascade)
        write_csv(out_dir / "window_table.csv", MULTISCALE_WINDOW_FIELDS, window_rows)
        write_csv(out_dir / "node_window_table.csv", MULTISCALE_NODE_WINDOW_FIELDS, node_rows)
        write_csv(out_dir / "edge_window_table.csv", MULTISCALE_EDGE_WINDOW_FIELDS, edge_rows)
    else:
        windows_by_cascade = build_windows_by_cascade(
            posts_by_cascade.keys(), obs_seconds, window_size_seconds, step_seconds
        )
        window_rows = build_window_rows(posts_by_cascade, edges_by_cascade, windows_by_cascade)
        node_rows = build_node_window_rows(posts_by_cascade, edges_by_cascade, windows_by_cascade)
        edge_rows = build_edge_window_rows(edges_by_cascade, windows_by_cascade)
        text_rows = build_text_window_rows(posts_by_cascade, windows_by_cascade)
        write_csv(out_dir / "window_table.csv", WINDOW_FIELDS, window_rows)
        write_csv(out_dir / "node_window_table.csv", NODE_WINDOW_FIELDS, node_rows)
        write_csv(out_dir / "edge_window_table.csv", EDGE_WINDOW_FIELDS, edge_rows)
    write_csv(out_dir / "text_window_table.csv", TEXT_WINDOW_FIELDS, text_rows)

    cascade_window_index = {
        cascade_idx: [
            {
                "window_idx": int(window["window_idx"]),
                "start_offset": int(window["start_offset"]),
                "end_offset": int(window["end_offset"]),
            }
            for window in windows
        ]
        for cascade_idx, windows in windows_by_cascade.items()
    }
    write_json(out_dir / "cascade_window_index.json", cascade_window_index)
    write_json(
        out_dir / "window_diagnostics.json",
        build_diagnostics(
            run_id=run_id,
            edge_mode=edge_mode,
            edge_table=str(edge_table),
            window_mode=window_mode,
            obs_seconds=obs_seconds,
            window_size_seconds=window_size_seconds,
            context_window_seconds=context_window_seconds,
            step_seconds=step_seconds,
            posts_by_cascade=posts_by_cascade,
            edges_by_cascade=edges_by_cascade,
            windows_by_cascade=windows_by_cascade,
            window_rows=window_rows,
            node_rows=node_rows,
            edge_rows=edge_rows,
            text_rows=text_rows,
        ),
    )

    print(f"Wrote windows to {out_dir}")
    print(
        f"cascades={len(windows_by_cascade)} windows={len(window_rows)} "
        f"node_windows={len(node_rows)} edge_windows={len(edge_rows)} text_windows={len(text_rows)}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sliding-window tables for a DRAGEN run.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--window-config", type=Path, default=None)
    parser.add_argument("--org-task-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cascade-list", type=Path, default=None)
    parser.add_argument("--obs-seconds", type=int, default=None)
    parser.add_argument("--window-size-seconds", type=int, default=None)
    parser.add_argument("--step-seconds", type=int, default=None)
    parser.add_argument("--window-mode", choices=["fixed", "causal_multiscale"], default=None)
    parser.add_argument("--edge-mode", choices=["star", "inferred_tree"], default=None)
    parser.add_argument("--inferred-tree-edge-table", type=Path, default=None)
    parser.add_argument("--max-cascades", type=int, default=None)
    return parser.parse_args()


def load_simple_config(path: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip().strip("'\"")
        if value.isdigit():
            config[key.strip()] = int(value)
        else:
            config[key.strip()] = value
    return config


def read_cascade_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_grouped_csv(
    path: Path,
    *,
    cascade_filter: Optional[set[str]],
    max_cascades: Optional[int],
    stop_after_max: bool,
) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    selected: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            if cascade_filter is not None and cascade_idx not in cascade_filter:
                if stop_after_max and selected and _sort_key(cascade_idx) > max(_sort_key(key) for key in selected):
                    break
                continue
            if cascade_idx not in selected:
                if max_cascades is not None and len(selected) >= max_cascades:
                    if stop_after_max:
                        break
                    continue
                selected.add(cascade_idx)
            grouped[cascade_idx].append(row)
    return dict(grouped)


def normalize_inferred_tree_edges(
    rows_by_cascade: Mapping[str, Sequence[Dict[str, str]]],
) -> Dict[str, List[Dict[str, str]]]:
    normalized: Dict[str, List[Dict[str, str]]] = {}
    for cascade_idx, rows in rows_by_cascade.items():
        normalized[cascade_idx] = [
            {
                "cascade_idx": row["cascade_idx"],
                "src_user_idx": row["parent_user_idx"],
                "dst_user_idx": row["child_user_idx"],
                "src_tweet_idx": row["parent_tweet_idx"],
                "dst_tweet_idx": row["child_tweet_idx"],
                "edge_type": row.get("parent_source", "inferred_tree"),
                "time_epoch": "",
                "relative_time": row["child_time"],
            }
            for row in rows
        ]
    return normalized


def build_windows_by_cascade(
    cascade_ids: Iterable[str],
    obs_seconds: int,
    window_size_seconds: int,
    step_seconds: int,
) -> Dict[str, List[Dict[str, int]]]:
    starts = list(range(0, max(0, obs_seconds - window_size_seconds) + 1, step_seconds))
    return {
        cascade_idx: [
            {
                "cascade_idx": cascade_idx,
                "window_idx": idx,
                "start_offset": start,
                "end_offset": start + window_size_seconds,
            }
            for idx, start in enumerate(starts, start=1)
        ]
        for cascade_idx in sorted(cascade_ids, key=_sort_key)
    }


def build_multiscale_windows_by_cascade(
    cascade_ids: Iterable[str],
    obs_seconds: int,
    step_seconds: int,
    current_window_seconds: int,
    context_window_seconds: int,
) -> Dict[str, List[Dict[str, int]]]:
    endpoints = list(range(step_seconds, obs_seconds + 1, step_seconds))
    return {
        cascade_idx: [
            {
                "cascade_idx": cascade_idx,
                "window_idx": idx,
                "start_offset": max(0, endpoint - current_window_seconds),
                "end_offset": endpoint,
                "endpoint": endpoint,
                "cur_start_offset": max(0, endpoint - current_window_seconds),
                "ctx_start_offset": max(0, endpoint - context_window_seconds),
                "cum_start_offset": 0,
            }
            for idx, endpoint in enumerate(endpoints, start=1)
        ]
        for cascade_idx in sorted(cascade_ids, key=_sort_key)
    }


def build_multiscale_window_rows(
    posts_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        posts = [post for post in posts_by_cascade.get(cascade_idx, []) if not _truthy(post.get("is_root"))]
        edges = list(edges_by_cascade.get(cascade_idx, []))
        previous_heat = 0
        for window in windows:
            endpoint = int(window["endpoint"])
            cur_start = int(window["cur_start_offset"])
            ctx_start = int(window["ctx_start_offset"])
            cur_posts = [post for post in posts if cur_start <= int(post["relative_time"]) < endpoint]
            ctx_posts = [post for post in posts if ctx_start <= int(post["relative_time"]) < endpoint]
            cum_posts = [post for post in posts if int(post["relative_time"]) < endpoint]
            cur_edges = [edge for edge in edges if cur_start <= int(edge["relative_time"]) < endpoint]
            ctx_edges = [edge for edge in edges if ctx_start <= int(edge["relative_time"]) < endpoint]
            cum_edges = [edge for edge in edges if int(edge["relative_time"]) < endpoint]
            heat_cur = len(cur_posts)
            rows.append(
                {
                    "cascade_idx": cascade_idx,
                    "window_idx": window["window_idx"],
                    "endpoint": endpoint,
                    "cur_start_offset": cur_start,
                    "ctx_start_offset": ctx_start,
                    "cum_start_offset": 0,
                    "end_offset": endpoint,
                    "num_retweets_cur": len(cur_posts),
                    "num_retweets_ctx": len(ctx_posts),
                    "num_retweets_cum": len(cum_posts),
                    "num_active_users_cur": len({str(post["user_idx"]) for post in cur_posts}),
                    "num_active_users_ctx": len({str(post["user_idx"]) for post in ctx_posts}),
                    "num_active_users_cum": len({str(post["user_idx"]) for post in cum_posts}),
                    "num_edges_cur": len(cur_edges),
                    "num_edges_ctx": len(ctx_edges),
                    "num_edges_cum": len(cum_edges),
                    "heat_cur": heat_cur,
                    "heat_ctx": len(ctx_posts),
                    "heat_cum": len(cum_posts),
                    "delta_heat_cur": heat_cur - previous_heat,
                }
            )
            previous_heat = heat_cur
    return rows


def build_multiscale_node_window_rows(
    posts_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        posts = list(posts_by_cascade.get(cascade_idx, []))
        edges = list(edges_by_cascade.get(cascade_idx, []))
        first_seen: Dict[str, int] = {}
        root_users = set()
        active_window_counts: Dict[str, int] = defaultdict(int)
        for post in posts:
            user_idx = str(post["user_idx"])
            rel = int(post["relative_time"])
            first_seen[user_idx] = min(first_seen.get(user_idx, rel), rel)
            if _truthy(post.get("is_root")):
                root_users.add(user_idx)
        for edge in edges:
            rel = int(edge["relative_time"])
            for key in ("src_user_idx", "dst_user_idx"):
                user_idx = str(edge[key])
                first_seen[user_idx] = min(first_seen.get(user_idx, rel), rel)

        for window in windows:
            endpoint = int(window["endpoint"])
            cur_start = int(window["cur_start_offset"])
            ctx_start = int(window["ctx_start_offset"])
            posts_cur = Counter(str(post["user_idx"]) for post in posts if cur_start <= int(post["relative_time"]) < endpoint)
            posts_ctx = Counter(str(post["user_idx"]) for post in posts if ctx_start <= int(post["relative_time"]) < endpoint)
            posts_cum = Counter(str(post["user_idx"]) for post in posts if int(post["relative_time"]) < endpoint)
            in_cur, out_cur = _degree_counts(edge for edge in edges if cur_start <= int(edge["relative_time"]) < endpoint)
            in_ctx, out_ctx = _degree_counts(edge for edge in edges if ctx_start <= int(edge["relative_time"]) < endpoint)
            in_cum, out_cum = _degree_counts(edge for edge in edges if int(edge["relative_time"]) < endpoint)
            visible_users = sorted(
                (user_idx for user_idx, rel in first_seen.items() if rel < endpoint),
                key=lambda user_idx: (first_seen[user_idx], int(user_idx) if user_idx.isdigit() else user_idx),
            )
            for user_idx in visible_users:
                if posts_cur[user_idx] or in_cur[user_idx] or out_cur[user_idx]:
                    active_window_counts[user_idx] += 1
                seen = int(first_seen[user_idx])
                rows.append(
                    {
                        "cascade_idx": cascade_idx,
                        "window_idx": window["window_idx"],
                        "user_idx": user_idx,
                        "first_seen_time": seen,
                        "is_root": 1 if user_idx in root_users else 0,
                        "num_posts_cur": posts_cur[user_idx],
                        "num_posts_ctx": posts_ctx[user_idx],
                        "num_posts_cum": posts_cum[user_idx],
                        "in_degree_cur": in_cur[user_idx],
                        "out_degree_cur": out_cur[user_idx],
                        "in_degree_ctx": in_ctx[user_idx],
                        "out_degree_ctx": out_ctx[user_idx],
                        "in_degree_cum": in_cum[user_idx],
                        "out_degree_cum": out_cum[user_idx],
                        "time_since_first_seen": max(0, endpoint - seen),
                        "active_window_count": active_window_counts[user_idx],
                    }
                )
    return rows


def build_multiscale_edge_window_rows(
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        edges = list(edges_by_cascade.get(cascade_idx, []))
        for window in windows:
            endpoint = int(window["endpoint"])
            scopes = {
                "current": int(window["cur_start_offset"]),
                "context": int(window["ctx_start_offset"]),
            }
            for scope, start in scopes.items():
                for edge in edges:
                    rel = int(edge["relative_time"])
                    if start <= rel < endpoint:
                        rows.append(
                            {
                                "cascade_idx": cascade_idx,
                                "window_idx": window["window_idx"],
                                "window_scope": scope,
                                "src_user_idx": edge["src_user_idx"],
                                "dst_user_idx": edge["dst_user_idx"],
                                "src_tweet_idx": edge["src_tweet_idx"],
                                "dst_tweet_idx": edge["dst_tweet_idx"],
                                "edge_time": edge.get("time_epoch", ""),
                                "edge_offset": rel,
                                "edge_type": edge.get("edge_type", "repost"),
                            }
                        )
    return rows


def build_window_rows(
    posts_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        posts = [post for post in posts_by_cascade.get(cascade_idx, []) if not _truthy(post.get("is_root"))]
        edges = list(edges_by_cascade.get(cascade_idx, []))
        previous_heat = 0
        for window in windows:
            start = int(window["start_offset"])
            end = int(window["end_offset"])
            posts_in_window = [post for post in posts if start <= int(post["relative_time"]) < end]
            cum_posts = [post for post in posts if int(post["relative_time"]) < end]
            edges_in_window = [edge for edge in edges if start <= int(edge["relative_time"]) < end]
            active_users = {str(post["user_idx"]) for post in posts_in_window}
            window_heat = len(posts_in_window)
            rows.append(
                {
                    "cascade_idx": cascade_idx,
                    "window_idx": window["window_idx"],
                    "start_offset": start,
                    "end_offset": end,
                    "num_retweets": len(posts_in_window),
                    "cum_retweets": len(cum_posts),
                    "num_active_users": len(active_users),
                    "num_edges": len(edges_in_window),
                    "window_heat": window_heat,
                    "delta_heat": window_heat - previous_heat,
                }
            )
            previous_heat = window_heat
    return rows


def build_diagnostics(
    *,
    run_id: str,
    edge_mode: str,
    edge_table: str,
    window_mode: str,
    obs_seconds: int,
    window_size_seconds: int,
    context_window_seconds: int,
    step_seconds: int,
    posts_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
    window_rows: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    text_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_windows = len(next(iter(windows_by_cascade.values()), []))
    root_text_windows = sum(1 for row in text_rows if row.get("text_visible_type") == "root_always_visible")
    retweet_early_violations = sum(
        1
        for row in text_rows
        if row.get("post_type") == "retweet"
        and int(row.get("post_offset", 0)) >= _window_end(windows_by_cascade, row)
    )
    return {
        "run_id": run_id,
        "edge_mode": edge_mode,
        "edge_table": edge_table,
        "window_mode": window_mode,
        "obs_seconds": obs_seconds,
        "window_size_seconds": window_size_seconds,
        "context_window_seconds": context_window_seconds,
        "step_seconds": step_seconds,
        "num_cascades": len(windows_by_cascade),
        "windows_per_cascade": expected_windows,
        "num_posts": sum(len(rows) for rows in posts_by_cascade.values()),
        "num_edges": sum(len(rows) for rows in edges_by_cascade.values()),
        "num_window_rows": len(window_rows),
        "num_node_window_rows": len(node_rows),
        "num_edge_window_rows": len(edge_rows),
        "num_text_window_rows": len(text_rows),
        "root_text_window_rows": root_text_windows,
        "retweet_text_early_violations": retweet_early_violations,
    }


def _window_end(
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]], row: Mapping[str, Any]
) -> int:
    for window in windows_by_cascade[str(row["cascade_idx"])]:
        if int(window["window_idx"]) == int(row["window_idx"]):
            return int(window["end_offset"])
    return -1


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "root"}


def _degree_counts(edges: Iterable[Dict[str, Any]]) -> Tuple[defaultdict, defaultdict]:
    in_degree: defaultdict = defaultdict(int)
    out_degree: defaultdict = defaultdict(int)
    for edge in edges:
        src = str(edge["src_user_idx"])
        dst = str(edge["dst_user_idx"])
        out_degree[src] += 1
        in_degree[dst] += 1
    return in_degree, out_degree


def _sort_key(value: str) -> Tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)
