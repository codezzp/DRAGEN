"""Align root and retweet text evidence with observation windows."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def build_text_window_rows(
    posts_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    """Build text visibility rows for root and retweet posts."""

    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        posts = sorted(
            posts_by_cascade.get(cascade_idx, []),
            key=lambda post: (int(post["relative_time"]), str(post["tweet_idx"])),
        )
        root_posts = [post for post in posts if _truthy(post.get("is_root"))]
        retweet_posts = [post for post in posts if not _truthy(post.get("is_root"))]

        for window in windows:
            start = int(window["start_offset"])
            end = int(window["end_offset"])
            for post in root_posts:
                rows.append(_text_row(cascade_idx, window["window_idx"], post, "root", "root_always_visible"))
            for post in retweet_posts:
                rel = int(post["relative_time"])
                if rel < end:
                    visible_type = "current_window" if start <= rel < end else "history_visible"
                    rows.append(_text_row(cascade_idx, window["window_idx"], post, "retweet", visible_type))
    return rows


def _text_row(
    cascade_idx: str,
    window_idx: int,
    post: Dict[str, Any],
    post_type: str,
    visible_type: str,
) -> Dict[str, Any]:
    return {
        "cascade_idx": cascade_idx,
        "window_idx": window_idx,
        "user_idx": post["user_idx"],
        "tweet_idx": post["tweet_idx"],
        "post_type": post_type,
        "text": post.get("text", ""),
        "text_visible_type": visible_type,
        "post_offset": post.get("relative_time", 0),
    }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "root"}
