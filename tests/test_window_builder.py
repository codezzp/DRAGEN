"""Tests for window construction."""

from __future__ import annotations

from dragen.windowing.edge_window_builder import build_edge_window_rows
from dragen.windowing.node_window_builder import build_node_window_rows
from dragen.windowing.text_window_builder import build_text_window_rows
from dragen.windowing.window_builder import build_window_rows, build_windows_by_cascade


def test_build_window_tables_align_posts_edges_and_text_visibility() -> None:
    posts_by_cascade = {
        "1": [
            {
                "cascade_idx": "1",
                "tweet_idx": "10",
                "user_idx": "100",
                "relative_time": "0",
                "text": "root",
                "is_root": "1",
            },
            {
                "cascade_idx": "1",
                "tweet_idx": "11",
                "user_idx": "101",
                "relative_time": "120",
                "text": "early",
                "is_root": "0",
            },
            {
                "cascade_idx": "1",
                "tweet_idx": "12",
                "user_idx": "102",
                "relative_time": "420",
                "text": "late",
                "is_root": "0",
            },
        ]
    }
    edges_by_cascade = {
        "1": [
            {
                "cascade_idx": "1",
                "src_user_idx": "100",
                "dst_user_idx": "101",
                "src_tweet_idx": "10",
                "dst_tweet_idx": "11",
                "edge_type": "repost",
                "time_epoch": "1120",
                "relative_time": "120",
            },
            {
                "cascade_idx": "1",
                "src_user_idx": "101",
                "dst_user_idx": "102",
                "src_tweet_idx": "11",
                "dst_tweet_idx": "12",
                "edge_type": "repost",
                "time_epoch": "1420",
                "relative_time": "420",
            },
        ]
    }
    windows = build_windows_by_cascade(["1"], obs_seconds=600, window_size_seconds=300, step_seconds=300)

    window_rows = build_window_rows(posts_by_cascade, edges_by_cascade, windows)
    node_rows = build_node_window_rows(posts_by_cascade, edges_by_cascade, windows)
    edge_rows = build_edge_window_rows(edges_by_cascade, windows)
    text_rows = build_text_window_rows(posts_by_cascade, windows)

    assert [row["num_retweets"] for row in window_rows] == [1, 1]
    assert [row["cum_retweets"] for row in window_rows] == [1, 2]
    assert len(edge_rows) == 2
    assert {row["user_idx"] for row in node_rows if row["window_idx"] == 1} == {"100", "101"}
    assert {row["user_idx"] for row in node_rows if row["window_idx"] == 2} == {"100", "101", "102"}

    root_rows = [row for row in text_rows if row["text_visible_type"] == "root_always_visible"]
    assert len(root_rows) == 2
    first_window_texts = [row["tweet_idx"] for row in text_rows if row["window_idx"] == 1]
    assert first_window_texts == ["10", "11"]
    second_window_retweets = [
        (row["tweet_idx"], row["text_visible_type"])
        for row in text_rows
        if row["window_idx"] == 2 and row["post_type"] == "retweet"
    ]
    assert second_window_retweets == [("11", "history_visible"), ("12", "current_window")]
