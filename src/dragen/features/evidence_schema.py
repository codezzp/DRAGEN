"""Evidence-v2 feature schema definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

NODE_EVIDENCE_GROUPS: Dict[str, List[str]] = {
    "behavior": [
        "beh_active_cur",
        "beh_posts_cur_log",
        "beh_posts_cum_log",
        "beh_visible_texts_cur_log",
        "beh_visible_texts_cum_log",
        "beh_contribution_share_cur",
        "beh_contribution_share_cum",
        "beh_active_window_count_log",
        "beh_active_window_ratio",
        "beh_first_seen_norm",
        "beh_time_since_first_seen_norm",
        "beh_window_reactivation",
    ],
    "temporal": [
        "tmp_window_pos",
        "tmp_first_seen_window_norm",
        "tmp_time_since_first_seen_norm",
        "tmp_is_early_participant",
        "tmp_is_late_participant",
        "tmp_activity_delta",
        "tmp_activity_acceleration",
        "tmp_same_bin_user_count_log",
        "tmp_same_bin_user_share",
        "tmp_nearest_event_gap_norm",
        "tmp_burst_bin_rank",
        "tmp_temporal_sync_score",
    ],
    "structure": [
        "str_in_degree_cur_log",
        "str_out_degree_cur_log",
        "str_total_degree_cur_log",
        "str_in_degree_ctx_log",
        "str_out_degree_ctx_log",
        "str_total_degree_ctx_log",
        "str_degree_cum_log",
        "str_degree_delta",
        "str_degree_share_cur",
        "str_degree_share_cum",
        "str_proxy_depth_norm",
        "str_parent_score",
        "str_child_count_log",
        "str_local_clustering",
    ],
    "coordination": [
        "coord_sync_degree_cur_log",
        "coord_sync_weight_sum_cur",
        "coord_sync_degree_cum_log",
        "coord_component_size_ratio",
        "coord_component_rank_norm",
        "coord_follow_supported_sync_ratio",
        "coord_same_parent_sync_count_log",
        "coord_same_time_bin_sync_count_log",
        "coord_coordination_clustering",
        "coord_coordination_density_local",
    ],
    "global_relation": [
        "glob_follow_in_cand_log",
        "glob_follow_out_cand_log",
        "glob_follow_total_cand_log",
        "glob_follow_candidate_share",
        "glob_follow_current_overlap",
        "glob_follow_context_overlap",
        "glob_follow_sync_overlap",
        "glob_follow_neighbor_active_mean",
        "glob_follow_neighbor_degree_mean",
        "glob_follow_reciprocal_count_log",
        "glob_follow_supported_edge_ratio",
    ],
}

WINDOW_EVIDENCE_GROUPS: Dict[str, List[str]] = {
    "behavior": [
        "win_heat_cur_log",
        "win_heat_cum_log",
        "win_delta_heat",
        "win_growth_rate",
        "win_acceleration",
        "win_active_users_cur_log",
        "win_active_users_cum_log",
        "win_new_user_ratio",
        "win_repeat_user_ratio",
    ],
    "temporal": [
        "win_temporal_entropy",
        "win_max_bin_share",
        "win_gini_time_bin",
        "win_burstiness",
        "win_active_bin_ratio",
    ],
    "structure": [
        "win_edge_count_cur_log",
        "win_edge_count_ctx_log",
        "win_density_cur",
        "win_density_ctx",
        "win_degree_gini",
        "win_largest_component_ratio",
        "win_depth_mean",
        "win_depth_max",
        "win_branch_entropy",
    ],
    "coordination": [
        "win_coord_edge_count_log",
        "win_coord_density",
        "win_coord_largest_component_ratio",
        "win_coord_avg_component_size",
        "win_coord_follow_supported_ratio",
    ],
    "global_relation": [
        "win_global_candidate_edge_count_log",
        "win_global_candidate_density",
        "win_global_follow_overlap_cur",
        "win_global_follow_overlap_ctx",
    ],
}

FORBIDDEN_INPUT_COLUMNS = {
    "weak_score",
    "label_confidence",
    "positive_votes",
    "negative_votes",
    "abstain_votes",
    "coordination_score",
    "ensemble_score",
    "final_label",
    "label",
    "weak_label",
}


@dataclass(frozen=True)
class EvidenceSchema:
    node_groups: Mapping[str, List[str]]
    window_groups: Mapping[str, List[str]]

    @property
    def node_columns(self) -> List[str]:
        return [name for cols in self.node_groups.values() for name in cols]

    @property
    def window_columns(self) -> List[str]:
        return [name for cols in self.window_groups.values() for name in cols]

    def node_slices(self) -> Dict[str, List[int]]:
        cols = self.node_columns
        return {group: [cols.index(col) for col in names] for group, names in self.node_groups.items()}

    def to_json(self) -> Dict[str, object]:
        return {
            "version": "evidence_v2",
            "node_evidence": dict(self.node_groups),
            "window_evidence": dict(self.window_groups),
            "node_evidence_columns": self.node_columns,
            "window_evidence_columns": self.window_columns,
            "node_evidence_slices": self.node_slices(),
            "forbidden_input_columns": sorted(FORBIDDEN_INPUT_COLUMNS),
        }


DEFAULT_EVIDENCE_SCHEMA = EvidenceSchema(NODE_EVIDENCE_GROUPS, WINDOW_EVIDENCE_GROUPS)


def write_schema(path: Path, schema: EvidenceSchema = DEFAULT_EVIDENCE_SCHEMA) -> None:
    path.write_text(json.dumps(schema.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_schema(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
