"""Build node-window tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def build_node_window_rows(
    posts_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    """Build one row per visible user per cascade window."""

    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        posts = list(posts_by_cascade.get(cascade_idx, []))
        edges = list(edges_by_cascade.get(cascade_idx, []))
        first_seen: Dict[str, int] = {}
        root_users = set()

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
            start = int(window["start_offset"])
            end = int(window["end_offset"])
            posts_in_window = Counter(
                str(post["user_idx"])
                for post in posts
                if start <= int(post["relative_time"]) < end
            )
            cum_posts = Counter(
                str(post["user_idx"]) for post in posts if int(post["relative_time"]) < end
            )
            in_degree_window, out_degree_window = _degree_counts(
                edge for edge in edges if start <= int(edge["relative_time"]) < end
            )
            cum_in_degree, cum_out_degree = _degree_counts(
                edge for edge in edges if int(edge["relative_time"]) < end
            )

            visible_users = sorted(
                (user_idx for user_idx, rel in first_seen.items() if rel < end),
                key=lambda user_idx: (first_seen[user_idx], int(user_idx) if user_idx.isdigit() else user_idx),
            )
            for user_idx in visible_users:
                seen = int(first_seen[user_idx])
                rows.append(
                    {
                        "cascade_idx": cascade_idx,
                        "window_idx": window["window_idx"],
                        "user_idx": user_idx,
                        "first_seen_time": seen,
                        "is_root": 1 if user_idx in root_users else 0,
                        "num_posts_in_window": posts_in_window[user_idx],
                        "cum_posts": cum_posts[user_idx],
                        "in_degree_window": in_degree_window[user_idx],
                        "out_degree_window": out_degree_window[user_idx],
                        "cum_in_degree": cum_in_degree[user_idx],
                        "cum_out_degree": cum_out_degree[user_idx],
                        "time_since_root": seen,
                        "time_since_first_seen": max(0, end - seen),
                    }
                )
    return rows


def _degree_counts(edges: Iterable[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    in_degree: Counter = Counter()
    out_degree: Counter = Counter()
    for edge in edges:
        src = str(edge["src_user_idx"])
        dst = str(edge["dst_user_idx"])
        out_degree[src] += 1
        in_degree[dst] += 1
    return in_degree, out_degree


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "root"}
