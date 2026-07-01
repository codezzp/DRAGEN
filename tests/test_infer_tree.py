"""Tests for proxy propagation tree inference."""

from __future__ import annotations

from dragen.graph.infer_tree import infer_trees
from dragen.windowing.window_builder import normalize_inferred_tree_edges


def test_infer_time_only_tree_is_time_consistent_and_normalizable() -> None:
    cascades = {
        "1": [
            {
                "cascade_idx": "1",
                "tweet_idx": "10",
                "user_idx": "100",
                "relative_time": "0",
                "is_root": "1",
            },
            {
                "cascade_idx": "1",
                "tweet_idx": "11",
                "user_idx": "101",
                "relative_time": "120",
                "is_root": "0",
            },
            {
                "cascade_idx": "1",
                "tweet_idx": "12",
                "user_idx": "102",
                "relative_time": "420",
                "is_root": "0",
            },
        ]
    }

    rows, diagnostics = infer_trees(
        cascades,
        follow_pairs=set(),
        method="time_only",
        tau_seconds=300.0,
        max_candidate_lookback=10,
        max_parent_gap=3600,
        depth_penalty_weight=0.35,
        activity_weight=0.25,
        root_bias=0.15,
        child_penalty=0.03,
        follow_weight=0.20,
        text_weight=0.20,
        exposure_weight=0.0,
        root_threshold=-0.25,
        window_seconds=300,
    )

    assert len(rows) == 2
    assert diagnostics["tree_valid_ratio"] == 1.0
    assert diagnostics["invalid_time_edges"] == 0
    assert rows[0]["parent_tweet_idx"] == "10"
    assert rows[1]["parent_tweet_idx"] == "11"
    assert {row["child_tweet_idx"] for row in rows} == {"11", "12"}
    assert all(int(row["parent_time"]) < int(row["child_time"]) for row in rows)

    normalized = normalize_inferred_tree_edges({"1": rows})
    assert normalized["1"][0]["src_user_idx"] == "100"
    assert normalized["1"][0]["dst_user_idx"] == "101"
    assert normalized["1"][0]["relative_time"] == 120
