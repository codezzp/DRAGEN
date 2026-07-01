"""Feature schema used by DRAGEN-Full."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


ROLE_NAMES = ["producer", "amplifier", "suppressor", "reframer", "ordinary"]
SOURCE_NAMES = ["text", "emotion", "behavior", "structure"]

WINDOW_FEATURE_COLUMNS = [
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

NODE_FEATURE_COLUMNS = [
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

FEATURE_GROUPS: Dict[str, List[str]] = {
    "text": [
        "num_texts_cur",
        "num_texts_visible",
        "avg_text_len_cur",
        "avg_text_len_visible",
    ],
    "emotion": [],
    "behavior": [
        "num_posts_cur",
        "num_posts_ctx",
        "num_posts_cum",
        "active_window_count",
        "time_since_first_seen",
    ],
    "structure": [
        "in_degree_cur",
        "out_degree_cur",
        "in_degree_ctx",
        "out_degree_ctx",
        "in_degree_cum",
        "out_degree_cum",
        "depth",
        "parent_time_gap",
        "parent_score",
        "time_score",
        "text_score",
        "activity_score",
        "depth_penalty",
        "load_penalty",
        "root_fallback_flag",
    ],
}


@dataclass(frozen=True)
class FeatureSchema:
    node_columns: List[str]
    window_columns: List[str]
    groups: Mapping[str, List[str]]

    @property
    def node_input_dim(self) -> int:
        return len(self.node_columns)

    @property
    def window_input_dim(self) -> int:
        return len(self.window_columns)

    def group_indices(self, group_name: str) -> List[int]:
        names = self.groups.get(group_name, [])
        missing = [name for name in names if name not in self.node_columns]
        if missing:
            raise ValueError(f"feature group {group_name!r} references missing columns: {missing}")
        return [self.node_columns.index(name) for name in names]


DEFAULT_SCHEMA = FeatureSchema(
    node_columns=NODE_FEATURE_COLUMNS,
    window_columns=WINDOW_FEATURE_COLUMNS,
    groups=FEATURE_GROUPS,
)


def schema_from_meta(meta: Mapping[str, object] | None) -> FeatureSchema:
    if not meta:
        return DEFAULT_SCHEMA
    node_columns = list(meta.get("node_feature_columns") or NODE_FEATURE_COLUMNS)  # type: ignore[arg-type]
    window_columns = list(meta.get("window_feature_columns") or WINDOW_FEATURE_COLUMNS)  # type: ignore[arg-type]
    if node_columns != NODE_FEATURE_COLUMNS:
        raise ValueError(
            "pack node_feature_columns do not match DRAGEN-Full fixed schema; "
            f"expected {NODE_FEATURE_COLUMNS}, got {node_columns}"
        )
    if window_columns != WINDOW_FEATURE_COLUMNS:
        raise ValueError(
            "pack window_feature_columns do not match DRAGEN-Full fixed schema; "
            f"expected {WINDOW_FEATURE_COLUMNS}, got {window_columns}"
        )
    return FeatureSchema(node_columns=node_columns, window_columns=window_columns, groups=FEATURE_GROUPS)
