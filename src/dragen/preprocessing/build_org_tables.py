#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build organization-spread anomaly task tables from processed weibo2casicff data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CASCADE_FIELDS = [
    "cascade_idx",
    "root_tweet_idx",
    "root_user_idx",
    "root_time_epoch",
    "root_text",
    "final_retweet_count",
    "observed_retweet_count",
    "duration",
    "observed_duration",
    "valid_for_training",
    "drop_reason",
]

POST_FIELDS = [
    "cascade_idx",
    "tweet_idx",
    "user_idx",
    "parent_tweet_idx",
    "parent_user_idx",
    "time_epoch",
    "relative_time",
    "text",
    "is_root",
    "in_observation",
]

USER_FIELDS = [
    "user_idx",
    "profile_text",
    "is_active_in_events",
    "is_root_user",
    "is_retweet_user",
    "num_root_posts",
    "num_retweets",
]

EDGE_FIELDS = [
    "cascade_idx",
    "src_user_idx",
    "dst_user_idx",
    "src_tweet_idx",
    "dst_tweet_idx",
    "edge_type",
    "time_epoch",
    "relative_time",
    "in_observation",
]

NODE_WINDOW_FIELDS = [
    "cascade_idx",
    "window_id",
    "user_idx",
    "window_start",
    "window_end",
    "active_count",
    "retweet_count",
    "text_count",
    "first_active_time",
    "last_active_time",
    "in_degree_cur",
    "out_degree_cur",
    "in_degree_cum",
    "out_degree_cum",
    "is_root",
    "has_text",
]

FOLLOW_FIELDS = [
    "src_user_idx",
    "dst_user_idx",
    "weight",
    "src_active",
    "dst_active",
    "both_active",
]

ORG_TABLE_SCHEMAS = [
    (
        "cascade_table.csv",
        "级联级表，一行对应一个 cascade。",
        [
            ("cascade_idx", "int", "级联编号。"),
            ("root_tweet_idx", "int", "根帖编号。"),
            ("root_user_idx", "int", "根用户编号。"),
            ("root_time_epoch", "int", "根帖发布时间，Unix epoch 秒。"),
            ("root_text", "str", "根帖文本，缺失时为空字符串。"),
            ("final_retweet_count", "int", "该级联全量转发数，不限观测期。"),
            ("observed_retweet_count", "int", "观测期内转发数，默认统计 0 到 obs 秒。"),
            ("duration", "int", "全量级联持续时间，单位秒。"),
            ("observed_duration", "int", "观测期内持续时间，单位秒。"),
            ("valid_for_training", "int", "是否满足训练过滤条件，1 表示有效，0 表示无效。"),
            ("drop_reason", "str", "无效原因，多个原因用分号连接。"),
        ],
    ),
    (
        "post_table.csv",
        "帖子级表，root 和 retweet 统一作为 post。",
        [
            ("cascade_idx", "int", "级联编号。"),
            ("tweet_idx", "int", "帖子编号。"),
            ("user_idx", "int", "发帖或转发用户编号。"),
            ("parent_tweet_idx", "int", "父帖子编号；retweet 无真实父节点时填 root_tweet_idx。"),
            ("parent_user_idx", "int", "父用户编号；retweet 无真实父节点时填 root_user_idx。"),
            ("time_epoch", "int", "帖子发布时间，Unix epoch 秒。"),
            ("relative_time", "int", "相对根帖的时间差，单位秒。"),
            ("text", "str", "帖子文本，缺失时为空字符串。"),
            ("is_root", "int", "root 帖为 1，retweet 为 0。"),
            ("in_observation", "int", "是否在观测期内，1 表示是，0 表示否。"),
        ],
    ),
    (
        "user_table.csv",
        "用户表，只输出事件相关活跃用户。",
        [
            ("user_idx", "int", "用户编号。"),
            ("profile_text", "str", "用户画像文本，缺失时为空字符串。"),
            ("is_active_in_events", "int", "是否出现在当前事件数据中。"),
            ("is_root_user", "int", "是否作为 root 用户出现。"),
            ("is_retweet_user", "int", "是否作为转发用户出现。"),
            ("num_root_posts", "int", "作为 root 用户的次数。"),
            ("num_retweets", "int", "作为转发用户的次数。"),
        ],
    ),
    (
        "cascade_edge_table.csv",
        "传播边表。第一版使用星型结构 root_user_idx -> retweet_user_idx。",
        [
            ("cascade_idx", "int", "级联编号。"),
            ("src_user_idx", "int", "信息来源用户编号。"),
            ("dst_user_idx", "int", "当前转发用户编号。"),
            ("src_tweet_idx", "int", "来源帖子编号。"),
            ("dst_tweet_idx", "int", "当前转发帖子编号。"),
            ("edge_type", "str", "边类型，第一版统一为 repost。"),
            ("time_epoch", "int", "当前转发时间，Unix epoch 秒。"),
            ("relative_time", "int", "当前转发相对根帖时间，单位秒。"),
            ("in_observation", "int", "是否在观测期内。"),
        ],
    ),
    (
        "node_window_table.csv",
        "节点-窗口表，后续模型主输入。一行对应一个 cascade、窗口、用户的聚合结果。",
        [
            ("cascade_idx", "int", "级联编号。"),
            ("window_id", "int", "窗口编号，从 1 开始。"),
            ("user_idx", "int", "用户编号。"),
            ("window_start", "int", "窗口起点，相对时间秒。"),
            ("window_end", "int", "窗口终点，相对时间秒。"),
            ("active_count", "int", "用户在当前窗口内出现次数。"),
            ("retweet_count", "int", "用户在当前窗口内转发数。"),
            ("text_count", "int", "当前窗口内该用户有文本的帖子数。"),
            ("first_active_time", "int", "当前窗口内首次活跃相对时间。"),
            ("last_active_time", "int", "当前窗口内最后活跃相对时间。"),
            ("in_degree_cur", "int", "当前窗口传播图入度。"),
            ("out_degree_cur", "int", "当前窗口传播图出度。"),
            ("in_degree_cum", "int", "从 0 到当前窗口累计入度。"),
            ("out_degree_cum", "int", "从 0 到当前窗口累计出度。"),
            ("is_root", "int", "是否 root 用户。"),
            ("has_text", "int", "当前窗口是否有非空文本。"),
        ],
    ),
    (
        "follow_edge_table.csv",
        "关注边表，仅在开启 include_follow_subgraph 时生成，只保留至少一个端点为活跃用户的边。",
        [
            ("src_user_idx", "int", "关注边源用户编号。"),
            ("dst_user_idx", "int", "关注边目标用户编号。"),
            ("weight", "int", "边权重，缺失时为 1。"),
            ("src_active", "int", "源用户是否为当前事件活跃用户。"),
            ("dst_active", "int", "目标用户是否为当前事件活跃用户。"),
            ("both_active", "int", "两个端点是否都为当前事件活跃用户。"),
        ],
    ),
    (
        "org_task_diagnostics.json",
        "诊断报告，记录参数、计数、质量、覆盖率、分布和硬性检查结果。",
        [
            ("run_id", "str", "运行编号。"),
            ("processed_dir", "str", "输入 processed 目录。"),
            ("out_dir", "str", "输出 org_task 目录。"),
            ("params", "object", "脚本运行参数。"),
            ("counts", "object", "输出规模统计。"),
            ("quality", "object", "缺失、重复、异常事件等质量统计。"),
            ("coverage", "object", "文本和用户画像覆盖率。"),
            ("window_stats", "object", "窗口数量和窗口内节点/转发统计。"),
            ("distributions", "object", "观测转发数和持续时间分布。"),
            ("edge_construction", "str", "传播边构造方式，当前为 star_from_root。"),
            ("follow_edges", "object", "关注图路径、检测状态和是否纳入输出。"),
            ("checks", "object", "index-only、索引合法性、窗口/root 等验收检查。"),
        ],
    ),
]

FORBIDDEN_FIELD_NAMES = {
    "raw_user_id",
    "raw_tweet_id",
    "raw_cascade_id",
    "user_id",
    "tweet_id",
    "cascade_id",
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def iter_jsonl(path: Path, *, strict: bool = True) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
                continue
            if isinstance(obj, dict):
                yield obj
            elif strict:
                raise ValueError(f"{path}:{line_no}: expected JSON object")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} not found")
    if not path.is_file():
        fail(f"{label} is not a file")


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def quantile(sorted_values: Sequence[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def distribution(values: Sequence[int]) -> Dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {k: 0 for k in ["min", "p25", "median", "mean", "p75", "p90", "p95", "p99", "max"]}
    return {
        "min": int(ordered[0]),
        "p25": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.50),
        "mean": float(sum(ordered) / len(ordered)),
        "p75": quantile(ordered, 0.75),
        "p90": quantile(ordered, 0.90),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": int(ordered[-1]),
    }


def build_windows(obs: int, window_size: int, step: int) -> List[Tuple[int, int, bool]]:
    windows: List[Tuple[int, int, bool]] = []
    start = 0
    while start < obs:
        end = min(start + window_size, obs)
        windows.append((start, end, False))
        start += step
    if not windows:
        fail("obs/window_size/step produce no windows")
    return [(s, e, idx == len(windows) - 1) for idx, (s, e, _) in enumerate(windows)]


def window_ids_for_time(relative_time: int, windows: Sequence[Tuple[int, int, bool]]) -> List[int]:
    hits: List[int] = []
    for idx, (start, end, is_last) in enumerate(windows, start=1):
        if start <= relative_time < end or (is_last and relative_time == end):
            hits.append(idx)
    return hits


def load_roots(path: Path) -> Dict[int, Dict[str, int]]:
    roots: Dict[int, Dict[str, int]] = {}
    for row in iter_jsonl(path):
        cascade_idx = as_int(row.get("cascade_idx"))
        if cascade_idx in roots:
            fail("cascade_idx duplicated in cascade_root.jsonl")
        roots[cascade_idx] = {
            "cascade_idx": cascade_idx,
            "root_user_idx": as_int(row.get("root_user_idx")),
            "root_time_epoch": as_int(row.get("root_time_epoch", row.get("root_time"))),
            "root_tweet_idx": as_int(row.get("root_tweet_idx", row.get("tweet_idx"))),
        }
    return roots


def load_texts(processed_dir: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    root_text_by_cascade: Dict[int, str] = {}
    text_by_tweet: Dict[int, str] = {}

    for row in iter_jsonl(processed_dir / "text" / "root_text.jsonl", strict=False):
        tweet_idx = as_int(row.get("tweet_idx"))
        cascade_idx = as_int(row.get("cascade_idx"))
        text = as_text(row.get("text"))
        if tweet_idx > 0:
            text_by_tweet[tweet_idx] = text
        if cascade_idx > 0:
            root_text_by_cascade[cascade_idx] = text

    retweet_path = processed_dir / "text" / "retweet_text.jsonl"
    if retweet_path.exists():
        for row in iter_jsonl(retweet_path, strict=False):
            tweet_idx = as_int(row.get("tweet_idx"))
            if tweet_idx > 0:
                text_by_tweet[tweet_idx] = as_text(row.get("text"))

    post_path = processed_dir / "text" / "post_text.jsonl"
    if post_path.exists():
        for row in iter_jsonl(post_path, strict=False):
            tweet_idx = as_int(row.get("tweet_idx"))
            if tweet_idx > 0:
                text_by_tweet[tweet_idx] = as_text(row.get("text"))

    return root_text_by_cascade, text_by_tweet


def load_user_profiles(path: Path) -> Dict[int, str]:
    profiles: Dict[int, str] = {}
    for row in iter_jsonl(path, strict=False):
        user_idx = as_int(row.get("user_idx"))
        if user_idx > 0:
            profiles[user_idx] = as_text(row.get("profile_text"))
    return profiles


def make_window_row(
    cascade_idx: int,
    window_id: int,
    user_idx: int,
    windows: Sequence[Tuple[int, int, bool]],
    is_root: int,
) -> Dict[str, Any]:
    start, end, _ = windows[window_id - 1]
    return {
        "cascade_idx": cascade_idx,
        "window_id": window_id,
        "user_idx": user_idx,
        "window_start": start,
        "window_end": end,
        "active_count": 0,
        "retweet_count": 0,
        "text_count": 0,
        "first_active_time": "",
        "last_active_time": "",
        "in_degree_cur": 0,
        "out_degree_cur": 0,
        "in_degree_cum": 0,
        "out_degree_cum": 0,
        "is_root": is_root,
        "has_text": 0,
    }


def bump_window_activity(row: Dict[str, Any], relative_time: int, *, is_retweet: bool, has_text: bool) -> None:
    row["active_count"] += 1
    if is_retweet:
        row["retweet_count"] += 1
    if has_text:
        row["text_count"] += 1
        row["has_text"] = 1
    if row["first_active_time"] == "" or relative_time < row["first_active_time"]:
        row["first_active_time"] = relative_time
    if row["last_active_time"] == "" or relative_time > row["last_active_time"]:
        row["last_active_time"] = relative_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", required=True, type=Path)
    parser.add_argument("--follow_edges", type=Path, default=None)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--obs", type=int, default=1800)
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--step", type=int, default=300)
    parser.add_argument("--min_retweets", type=int, default=8)
    parser.add_argument("--min_duration", type=int, default=300)
    parser.add_argument("--include_follow_subgraph", action="store_true")
    parser.add_argument("--max_follow_edges", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.obs <= 0 or args.window_size <= 0 or args.step <= 0:
        fail("obs, window_size and step must be positive")

    processed_dir = args.processed_dir
    out_dir = args.out_dir

    required = {
        "cascade_root.jsonl": processed_dir / "mapping" / "cascade_root.jsonl",
        "events.jsonl": processed_dir / "events" / "events.jsonl",
        "root_text.jsonl": processed_dir / "text" / "root_text.jsonl",
        "retweet_text.jsonl": processed_dir / "text" / "retweet_text.jsonl",
        "post_text.jsonl": processed_dir / "text" / "post_text.jsonl",
        "user_profile.jsonl": processed_dir / "user" / "user_profile.jsonl",
    }
    for label, path in required.items():
        require_file(path, label)

    targets = [
        "cascade_table.csv",
        "post_table.csv",
        "user_table.csv",
        "cascade_edge_table.csv",
        "node_window_table.csv",
        "org_task_diagnostics.json",
        "README_CHECK.md",
        "org_task_result_report.md",
    ]
    if args.follow_edges and args.include_follow_subgraph:
        targets.append("follow_edge_table.csv")
    if out_dir.exists() and any((out_dir / name).exists() for name in targets):
        if not args.force:
            fail(f"{out_dir} already contains output files; pass --force to overwrite")
        for name in targets:
            path = out_dir / name
            if path.exists():
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = build_windows(args.obs, args.window_size, args.step)
    roots = load_roots(required["cascade_root.jsonl"])
    root_text_by_cascade, text_by_tweet = load_texts(processed_dir)

    cascade_stats: Dict[int, Dict[str, int]] = {
        cidx: {
            "final_retweet_count": 0,
            "observed_retweet_count": 0,
            "last_relative_time": 0,
            "last_observed_relative_time": 0,
        }
        for cidx in roots
    }
    post_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    node_windows: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    cur_in_degree: Counter[Tuple[int, int, int]] = Counter()
    cur_out_degree: Counter[Tuple[int, int, int]] = Counter()
    active_users: set[int] = set()
    root_users: set[int] = set()
    retweet_users: set[int] = set()
    root_post_counts: Counter[int] = Counter()
    retweet_counts_by_user: Counter[int] = Counter()
    seen_tweets: set[int] = set()
    quality = {
        "missing_root_text": 0,
        "missing_retweet_text": 0,
        "missing_user_profile": 0,
        "duplicate_tweet_idx": 0,
        "negative_relative_time": 0,
        "missing_cascade_root": 0,
        "missing_user_idx": 0,
        "bad_event_rows": 0,
    }

    for cascade_idx, root in sorted(roots.items()):
        root_user_idx = root["root_user_idx"]
        root_tweet_idx = root["root_tweet_idx"]
        root_time = root["root_time_epoch"]
        root_text = text_by_tweet.get(root_tweet_idx, root_text_by_cascade.get(cascade_idx, ""))
        if not root_text:
            quality["missing_root_text"] += 1
        if root_user_idx <= 0 or root_tweet_idx <= 0 or root_time <= 0:
            quality["bad_event_rows"] += 1
        active_users.add(root_user_idx)
        root_users.add(root_user_idx)
        root_post_counts[root_user_idx] += 1
        seen_tweets.add(root_tweet_idx)
        post_rows.append(
            {
                "cascade_idx": cascade_idx,
                "tweet_idx": root_tweet_idx,
                "user_idx": root_user_idx,
                "parent_tweet_idx": 0,
                "parent_user_idx": 0,
                "time_epoch": root_time,
                "relative_time": 0,
                "text": root_text,
                "is_root": 1,
                "in_observation": 1,
            }
        )
        key = (cascade_idx, 1, root_user_idx)
        node_windows[key] = make_window_row(cascade_idx, 1, root_user_idx, windows, 1)
        bump_window_activity(node_windows[key], 0, is_retweet=False, has_text=bool(root_text))

    for event in iter_jsonl(required["events.jsonl"], strict=False):
        cascade_idx = as_int(event.get("cascade_idx"))
        tweet_idx = as_int(event.get("tweet_idx"))
        retweet_user_idx = as_int(event.get("retweet_user_idx", event.get("user_idx")))
        retweet_time = as_int(event.get("retweet_time", event.get("time_epoch")))
        root = roots.get(cascade_idx)
        if root is None:
            quality["missing_cascade_root"] += 1
            continue
        if tweet_idx <= 0 or retweet_user_idx <= 0 or retweet_time <= 0:
            quality["bad_event_rows"] += 1
            if retweet_user_idx <= 0:
                quality["missing_user_idx"] += 1
            continue

        root_time = root["root_time_epoch"]
        root_user_idx = root["root_user_idx"]
        root_tweet_idx = root["root_tweet_idx"]
        relative_time = retweet_time - root_time
        stat = cascade_stats[cascade_idx]
        stat["final_retweet_count"] += 1
        stat["last_relative_time"] = max(stat["last_relative_time"], relative_time)
        if relative_time < 0:
            quality["negative_relative_time"] += 1
        in_obs = 1 if 0 <= relative_time <= args.obs else 0
        if in_obs:
            stat["observed_retweet_count"] += 1
            stat["last_observed_relative_time"] = max(stat["last_observed_relative_time"], relative_time)

        if tweet_idx in seen_tweets:
            quality["duplicate_tweet_idx"] += 1
            continue
        seen_tweets.add(tweet_idx)

        text = text_by_tweet.get(tweet_idx, "")
        if not text:
            quality["missing_retweet_text"] += 1
        active_users.add(retweet_user_idx)
        retweet_users.add(retweet_user_idx)
        retweet_counts_by_user[retweet_user_idx] += 1
        post_rows.append(
            {
                "cascade_idx": cascade_idx,
                "tweet_idx": tweet_idx,
                "user_idx": retweet_user_idx,
                "parent_tweet_idx": root_tweet_idx,
                "parent_user_idx": root_user_idx,
                "time_epoch": retweet_time,
                "relative_time": relative_time,
                "text": text,
                "is_root": 0,
                "in_observation": in_obs,
            }
        )
        edge_rows.append(
            {
                "cascade_idx": cascade_idx,
                "src_user_idx": root_user_idx,
                "dst_user_idx": retweet_user_idx,
                "src_tweet_idx": root_tweet_idx,
                "dst_tweet_idx": tweet_idx,
                "edge_type": "repost",
                "time_epoch": retweet_time,
                "relative_time": relative_time,
                "in_observation": in_obs,
            }
        )

        if in_obs:
            for window_id in window_ids_for_time(relative_time, windows):
                key = (cascade_idx, window_id, retweet_user_idx)
                row = node_windows.setdefault(
                    key, make_window_row(cascade_idx, window_id, retweet_user_idx, windows, 0)
                )
                bump_window_activity(row, relative_time, is_retweet=True, has_text=bool(text))
                cur_in_degree[key] += 1
                root_key = (cascade_idx, window_id, root_user_idx)
                root_row = node_windows.setdefault(
                    root_key, make_window_row(cascade_idx, window_id, root_user_idx, windows, 1)
                )
                root_row["is_root"] = 1
                cur_out_degree[root_key] += 1

    profiles = load_user_profiles(required["user_profile.jsonl"])
    for user_idx in active_users:
        if user_idx > 0 and user_idx not in profiles:
            quality["missing_user_profile"] += 1

    cascade_rows: List[Dict[str, Any]] = []
    for cascade_idx, root in sorted(roots.items()):
        stat = cascade_stats[cascade_idx]
        observed_retweets = stat["observed_retweet_count"]
        observed_duration = stat["last_observed_relative_time"] if observed_retweets > 0 else 0
        duration = max(0, stat["last_relative_time"]) if stat["final_retweet_count"] > 0 else 0
        reasons: List[str] = []
        if observed_retweets < args.min_retweets:
            reasons.append("min_retweets")
        if observed_duration < args.min_duration:
            reasons.append("min_duration")
        if root["root_user_idx"] <= 0:
            reasons.append("bad_root_user_idx")
        if root["root_time_epoch"] <= 0:
            reasons.append("bad_root_time_epoch")
        cascade_rows.append(
            {
                "cascade_idx": cascade_idx,
                "root_tweet_idx": root["root_tweet_idx"],
                "root_user_idx": root["root_user_idx"],
                "root_time_epoch": root["root_time_epoch"],
                "root_text": text_by_tweet.get(root["root_tweet_idx"], root_text_by_cascade.get(cascade_idx, "")),
                "final_retweet_count": stat["final_retweet_count"],
                "observed_retweet_count": observed_retweets,
                "duration": duration,
                "observed_duration": observed_duration,
                "valid_for_training": 0 if reasons else 1,
                "drop_reason": ";".join(reasons),
            }
        )

    cumulative_in: Counter[Tuple[int, int]] = Counter()
    cumulative_out: Counter[Tuple[int, int]] = Counter()
    node_window_rows: List[Dict[str, Any]] = []
    for key in sorted(node_windows):
        cascade_idx, _window_id, user_idx = key
        cumulative_in[(cascade_idx, user_idx)] += cur_in_degree[key]
        cumulative_out[(cascade_idx, user_idx)] += cur_out_degree[key]
        row = node_windows[key]
        row["in_degree_cur"] = cur_in_degree[key]
        row["out_degree_cur"] = cur_out_degree[key]
        row["in_degree_cum"] = cumulative_in[(cascade_idx, user_idx)]
        row["out_degree_cum"] = cumulative_out[(cascade_idx, user_idx)]
        node_window_rows.append(row)

    user_rows = [
        {
            "user_idx": user_idx,
            "profile_text": profiles.get(user_idx, ""),
            "is_active_in_events": 1,
            "is_root_user": 1 if user_idx in root_users else 0,
            "is_retweet_user": 1 if user_idx in retweet_users else 0,
            "num_root_posts": root_post_counts[user_idx],
            "num_retweets": retweet_counts_by_user[user_idx],
        }
        for user_idx in sorted(u for u in active_users if u > 0)
    ]

    post_rows.sort(key=lambda r: (r["cascade_idx"], r["relative_time"], r["tweet_idx"]))
    edge_rows.sort(key=lambda r: (r["cascade_idx"], r["relative_time"], r["dst_tweet_idx"]))
    cascade_rows.sort(key=lambda r: r["cascade_idx"])

    counts: Dict[str, Any] = {
        "num_cascades": len(cascade_rows),
        "num_valid_training_cascades": sum(1 for r in cascade_rows if r["valid_for_training"] == 1),
        "num_root_posts": len(roots),
        "num_retweet_posts_total": sum(r["final_retweet_count"] for r in cascade_rows),
        "num_retweet_posts_in_obs": sum(r["observed_retweet_count"] for r in cascade_rows),
        "num_posts": len(post_rows),
        "num_users_active": len(user_rows),
        "num_root_users": len(root_users),
        "num_retweet_users": len(retweet_users),
        "num_cascade_edges": len(edge_rows),
        "num_cascade_edges_in_obs": sum(1 for r in edge_rows if r["in_observation"] == 1),
        "num_node_window_rows": len(node_window_rows),
    }

    write_csv(out_dir / "cascade_table.csv", CASCADE_FIELDS, cascade_rows)
    write_csv(out_dir / "post_table.csv", POST_FIELDS, post_rows)
    write_csv(out_dir / "user_table.csv", USER_FIELDS, user_rows)
    write_csv(out_dir / "cascade_edge_table.csv", EDGE_FIELDS, edge_rows)
    write_csv(out_dir / "node_window_table.csv", NODE_WINDOW_FIELDS, node_window_rows)

    follow_rows_written = 0
    follow_detected = bool(args.follow_edges and args.follow_edges.exists())
    if args.follow_edges and args.include_follow_subgraph:
        require_file(args.follow_edges, "follow_edges.tsv")

        def follow_rows() -> Iterable[Dict[str, Any]]:
            nonlocal follow_rows_written
            with args.follow_edges.open("r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, start=1):
                    if args.max_follow_edges is not None and line_no > args.max_follow_edges:
                        break
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    src = as_int(parts[0])
                    dst = as_int(parts[1])
                    if src <= 0 or dst <= 0:
                        continue
                    src_active = 1 if src in active_users else 0
                    dst_active = 1 if dst in active_users else 0
                    if not src_active and not dst_active:
                        continue
                    weight = as_int(parts[2], 1) if len(parts) >= 3 else 1
                    follow_rows_written += 1
                    yield {
                        "src_user_idx": src,
                        "dst_user_idx": dst,
                        "weight": weight,
                        "src_active": src_active,
                        "dst_active": dst_active,
                        "both_active": 1 if src_active and dst_active else 0,
                    }

        write_csv(out_dir / "follow_edge_table.csv", FOLLOW_FIELDS, follow_rows())
        counts["num_follow_edges"] = follow_rows_written

    root_text_coverage = 1.0 - quality["missing_root_text"] / max(1, len(roots))
    retweet_text_coverage = 1.0 - quality["missing_retweet_text"] / max(1, len(edge_rows))
    user_profile_coverage = 1.0 - quality["missing_user_profile"] / max(1, len(user_rows))
    if root_text_coverage < 0.95:
        warn(f"root text coverage below 95%: {root_text_coverage:.4f}")
    if retweet_text_coverage < 0.95:
        warn(f"retweet text coverage below 95%: {retweet_text_coverage:.4f}")
    if user_profile_coverage < 0.95:
        warn(f"user profile coverage below 95%: {user_profile_coverage:.4f}")
    if quality["missing_retweet_text"]:
        warn(f"missing retweet text: {quality['missing_retweet_text']}")
    if quality["negative_relative_time"]:
        warn(f"negative relative time events: {quality['negative_relative_time']}")

    nodes_per_window = Counter((r["cascade_idx"], r["window_id"]) for r in node_window_rows)
    retweets_per_window: Counter[Tuple[int, int]] = Counter()
    for row in edge_rows:
        if row["in_observation"] != 1:
            continue
        for window_id in window_ids_for_time(row["relative_time"], windows):
            retweets_per_window[(row["cascade_idx"], window_id)] += 1
    node_counts = list(nodes_per_window.values())
    retweet_counts = list(retweets_per_window.values())

    all_fields = CASCADE_FIELDS + POST_FIELDS + USER_FIELDS + EDGE_FIELDS + NODE_WINDOW_FIELDS
    if args.follow_edges and args.include_follow_subgraph:
        all_fields += FOLLOW_FIELDS

    root_w1_count = len(
        {
            r["cascade_idx"]
            for r in node_window_rows
            if r["window_id"] == 1 and r["is_root"] == 1 and r["active_count"] >= 1
        }
    )

    diagnostics = {
        "run_id": processed_dir.parent.name,
        "processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "params": {
            "obs": args.obs,
            "window_size": args.window_size,
            "step": args.step,
            "min_retweets": args.min_retweets,
            "min_duration": args.min_duration,
            "include_follow_subgraph": bool(args.include_follow_subgraph),
            "max_follow_edges": args.max_follow_edges,
        },
        "counts": counts,
        "quality": quality,
        "coverage": {
            "root_text_coverage": root_text_coverage,
            "retweet_text_coverage": retweet_text_coverage,
            "user_profile_coverage": user_profile_coverage,
        },
        "window_stats": {
            "num_windows": len(windows),
            "window_size": args.window_size,
            "avg_nodes_per_window": float(sum(node_counts) / len(node_counts)) if node_counts else 0.0,
            "median_nodes_per_window": float(statistics.median(node_counts)) if node_counts else 0.0,
            "avg_retweets_per_window": float(sum(retweet_counts) / len(retweet_counts)) if retweet_counts else 0.0,
            "median_retweets_per_window": float(statistics.median(retweet_counts)) if retweet_counts else 0.0,
        },
        "distributions": {
            "observed_retweet_count": distribution([r["observed_retweet_count"] for r in cascade_rows]),
            "duration": distribution([r["duration"] for r in cascade_rows]),
        },
        "edge_construction": "star_from_root",
        "follow_edges": {
            "path": str(args.follow_edges) if args.follow_edges else "",
            "detected": follow_detected,
            "included": bool(args.follow_edges and args.include_follow_subgraph),
        },
        "checks": {
            "index_only": not any(field in FORBIDDEN_FIELD_NAMES for field in all_fields),
            "all_user_idx_positive": all(r["user_idx"] > 0 for r in user_rows)
            and all(r["root_user_idx"] > 0 for r in cascade_rows)
            and all(r["src_user_idx"] > 0 and r["dst_user_idx"] > 0 for r in edge_rows),
            "all_tweet_idx_positive": all(r["tweet_idx"] > 0 for r in post_rows)
            and all(r["root_tweet_idx"] > 0 for r in cascade_rows)
            and all(r["src_tweet_idx"] > 0 and r["dst_tweet_idx"] > 0 for r in edge_rows),
            "all_cascade_idx_positive": all(r["cascade_idx"] > 0 for r in cascade_rows)
            and all(r["cascade_idx"] > 0 for r in post_rows)
            and all(r["cascade_idx"] > 0 for r in edge_rows)
            and all(r["cascade_idx"] > 0 for r in node_window_rows),
            "all_window_id_valid": all(1 <= r["window_id"] <= len(windows) for r in node_window_rows),
            "relative_time_non_negative_in_window_table": True,
            "root_exists_for_each_cascade": len({r["cascade_idx"] for r in post_rows if r["is_root"] == 1}) == len(roots),
            "root_w1_count": root_w1_count,
            "root_w1_total": len(roots),
        },
    }

    diag_path = out_dir / "org_task_diagnostics.json"
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    readme_check_path = out_dir / "README_CHECK.md"
    with readme_check_path.open("w", encoding="utf-8") as f:
        f.write("# README_CHECK\n\n")
        f.write(f"run_id: {diagnostics['run_id']}\n\n")
        f.write(f"obs: {diagnostics['params']['obs']}\n\n")
        f.write(f"window_size: {diagnostics['params']['window_size']}\n\n")
        f.write(f"step: {diagnostics['params']['step']}\n\n")
        f.write(f"num_cascades: {diagnostics['counts']['num_cascades']}\n\n")
        f.write(f"num_windows: {diagnostics['window_stats']['num_windows']}\n\n")
        f.write(f"edge_construction: {diagnostics['edge_construction']}\n\n")
        f.write(f"missing_retweet_text: {diagnostics['quality']['missing_retweet_text']}\n\n")
        f.write(f"index_only: {str(diagnostics['checks']['index_only']).lower()}\n\n")
        f.write(
            "root_w1_check: "
            f"{diagnostics['checks']['root_w1_count']} / {diagnostics['checks']['root_w1_total']}\n"
        )

    report_path = out_dir / "org_task_result_report.md"
    output_files = [
        "cascade_table.csv",
        "post_table.csv",
        "user_table.csv",
        "cascade_edge_table.csv",
        "node_window_table.csv",
        "org_task_diagnostics.json",
        "README_CHECK.md",
    ]
    if args.follow_edges and args.include_follow_subgraph:
        output_files.append("follow_edge_table.csv")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# 组织化传播异常任务标准表构建结果报告\n\n")
        f.write("## 1. 执行信息\n\n")
        f.write(f"- 实现脚本：`scripts/build_org_tables.py`\n")
        f.write(f"- 输入目录：`{diagnostics['processed_dir']}`\n")
        f.write(f"- 输出目录：`{diagnostics['out_dir']}`\n")
        f.write(f"- run_id：`{diagnostics['run_id']}`\n\n")
        f.write("执行命令示例：\n\n")
        f.write("```powershell\n")
        f.write(
            "python scripts\\build_org_tables.py "
            f"--processed_dir {processed_dir} "
            f"--follow_edges {args.follow_edges if args.follow_edges else '<follow_edges.tsv>'} "
            f"--out_dir {out_dir} "
            f"--obs {args.obs} --window_size {args.window_size} --step {args.step} "
            f"--min_retweets {args.min_retweets} --min_duration {args.min_duration}\n"
        )
        f.write("```\n\n")

        f.write("## 2. 输出文件\n\n")
        f.write("| 文件 | 大小 |\n")
        f.write("|---|---:|\n")
        for name in output_files:
            path = out_dir / name
            size = path.stat().st_size if path.exists() else 0
            f.write(f"| `{name}` | {size:,} bytes |\n")
        f.write("\n")
        f.write(f"- 传播边构建方式：`{diagnostics['edge_construction']}`\n")
        f.write(
            f"- 关注图检测：`{str(diagnostics['follow_edges']['detected']).lower()}`；"
            f"是否生成关注子图：`{str(diagnostics['follow_edges']['included']).lower()}`\n\n"
        )

        f.write("## 3. 构建规模统计\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|---|---:|\n")
        count_labels = [
            ("num_cascades", "级联数"),
            ("num_valid_training_cascades", "有效训练级联数"),
            ("num_root_posts", "root 帖数"),
            ("num_retweet_posts_total", "总转发帖数"),
            ("num_retweet_posts_in_obs", "观测期内转发帖数"),
            ("num_posts", "总帖子数"),
            ("num_users_active", "活跃用户数"),
            ("num_root_users", "root 用户数"),
            ("num_retweet_users", "转发用户数"),
            ("num_cascade_edges", "传播边数"),
            ("num_cascade_edges_in_obs", "观测期内传播边数"),
            ("num_node_window_rows", "节点-窗口表行数"),
        ]
        for key, label in count_labels:
            f.write(f"| {label} | {diagnostics['counts'].get(key, 0):,} |\n")
        f.write("\n")

        f.write("## 4. 数据质量\n\n")
        f.write("| 质量项 | 数值 |\n")
        f.write("|---|---:|\n")
        quality_labels = [
            ("missing_root_text", "缺失 root 文本"),
            ("missing_retweet_text", "缺失 retweet 文本"),
            ("missing_user_profile", "缺失用户画像"),
            ("duplicate_tweet_idx", "重复 tweet_idx"),
            ("negative_relative_time", "负 relative_time 事件"),
            ("missing_cascade_root", "缺失 cascade root"),
            ("missing_user_idx", "缺失 user_idx"),
            ("bad_event_rows", "异常事件行"),
        ]
        for key, label in quality_labels:
            f.write(f"| {label} | {diagnostics['quality'].get(key, 0):,} |\n")
        f.write("\n")
        f.write("| 覆盖率 | 数值 |\n")
        f.write("|---|---:|\n")
        for key, value in diagnostics["coverage"].items():
            f.write(f"| {key} | {value:.6f} |\n")
        f.write("\n")

        f.write("## 5. 窗口统计\n\n")
        f.write(f"- `obs = {args.obs}`\n")
        f.write(f"- `window_size = {args.window_size}`\n")
        f.write(f"- `step = {args.step}`\n")
        f.write(f"- `num_windows = {len(windows)}`\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|---|---:|\n")
        for key in [
            "avg_nodes_per_window",
            "median_nodes_per_window",
            "avg_retweets_per_window",
            "median_retweets_per_window",
        ]:
            f.write(f"| {key} | {diagnostics['window_stats'][key]:.6f} |\n")
        f.write("\n")

        f.write("## 6. 验收检查\n\n")
        f.write("| 检查项 | 结果 |\n")
        f.write("|---|---|\n")
        check_labels = [
            ("index_only", "index-only 字段要求"),
            ("all_user_idx_positive", "user_idx 全部为正"),
            ("all_tweet_idx_positive", "tweet_idx 全部为正"),
            ("all_cascade_idx_positive", "cascade_idx 全部为正"),
            ("all_window_id_valid", "window_id 合法"),
            ("relative_time_non_negative_in_window_table", "node_window_table 无负 relative_time 来源"),
            ("root_exists_for_each_cascade", "每个 cascade 都有 root post"),
        ]
        for key, label in check_labels:
            f.write(f"| {label} | {'通过' if diagnostics['checks'].get(key) else '未通过'} |\n")
        f.write(
            "| 每个 cascade 的 root 用户都出现在第 1 个窗口 | "
            f"{diagnostics['checks']['root_w1_count']} / {diagnostics['checks']['root_w1_total']} |\n\n"
        )

        f.write("## 7. 输出字段说明\n\n")
        for table_name, table_desc, fields in ORG_TABLE_SCHEMAS:
            if table_name == "follow_edge_table.csv" and not diagnostics["follow_edges"]["included"]:
                f.write(f"### {table_name}\n\n")
                f.write(f"{table_desc}\n\n")
                f.write("本次未开启 `--include_follow_subgraph`，因此未生成该文件。\n\n")
                continue
            f.write(f"### {table_name}\n\n")
            f.write(f"{table_desc}\n\n")
            f.write("| 字段 | 类型 | 说明 |\n")
            f.write("|---|---|---|\n")
            for field_name, field_type, field_desc in fields:
                f.write(f"| `{field_name}` | {field_type} | {field_desc} |\n")
            f.write("\n")

        f.write("## 8. 结论\n\n")
        f.write("本次已完成组织化传播异常任务标准表构建。输出保持 index-only，")
        f.write("训练相关表不包含真实微博 ID、真实用户 ID 或真实 cascade ID。")
        f.write("第一版传播边采用 `root_user_idx -> retweet_user_idx` 星型结构。\n")

    for name in targets:
        path = out_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[OK] {name}")
        else:
            fail(f"{name} missing or empty")

    if not diagnostics["checks"]["index_only"]:
        fail("forbidden raw ID field found in output schema")
    if not diagnostics["checks"]["all_user_idx_positive"]:
        fail("user_idx <= 0 found in output")
    if not diagnostics["checks"]["all_tweet_idx_positive"]:
        fail("tweet_idx <= 0 found in output")
    if not diagnostics["checks"]["all_cascade_idx_positive"]:
        fail("cascade_idx <= 0 found in output")
    if not diagnostics["checks"]["all_window_id_valid"]:
        fail("window_id out of range")
    if not diagnostics["checks"]["root_exists_for_each_cascade"]:
        fail("root post missing for one or more cascades")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
