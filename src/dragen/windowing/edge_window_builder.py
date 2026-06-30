"""Build propagation edges inside each observation window."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def build_edge_window_rows(
    edges_by_cascade: Mapping[str, Sequence[Dict[str, Any]]],
    windows_by_cascade: Mapping[str, Sequence[Dict[str, int]]],
) -> List[Dict[str, Any]]:
    """Build current-window propagation edge rows."""

    rows: List[Dict[str, Any]] = []
    for cascade_idx, windows in windows_by_cascade.items():
        edges = list(edges_by_cascade.get(cascade_idx, []))
        for window in windows:
            start = int(window["start_offset"])
            end = int(window["end_offset"])
            for edge in edges:
                rel = int(edge["relative_time"])
                if start <= rel < end:
                    rows.append(
                        {
                            "cascade_idx": cascade_idx,
                            "window_idx": window["window_idx"],
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
