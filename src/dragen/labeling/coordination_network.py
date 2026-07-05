"""Lightweight coordination-network label features."""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

from dragen.labeling.label_features import add_percentile_scores, safe_ratio


def build_coordination_network_scores(
    features: Dict[str, Dict[str, Any]],
    out_dir: Path,
    *,
    write_edges: bool = True,
) -> Dict[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edges_to_write: List[Dict[str, Any]] = []
    for cascade_idx, item in features.items():
        n = int(max(float(item.get("num_visible_users", 0.0)), 0.0))
        follow_edges = int(max(float(item.get("global_candidate_edges", 0.0)), 0.0))
        # Feature-derived proxy edge counts. These are event-level approximations; training never reads them.
        temporal_edges = int(max(float(item.get("max_window_retweets", 0.0)) - 1.0, 0.0))
        text_edges = int(max(float(item.get("sum_text_score", 0.0)), 0.0))
        structure_edges = int(max(float(item.get("sum_degree_ctx", 0.0)), 0.0))
        total_edges = follow_edges + temporal_edges + text_edges + structure_edges
        possible = max(n * max(n - 1, 1), 1)
        density = safe_ratio(total_edges, possible)
        largest_ratio = min(1.0, safe_ratio(math.sqrt(max(total_edges, 0.0)) + 1.0, max(n, 1))) if n else 0.0
        clustering = min(1.0, safe_ratio(structure_edges + text_edges, total_edges + 1.0))
        temporal_burst = float(item.get("temporal_sync_score", 0.0))
        text_sim = float(item.get("text_score", 0.0))
        follow_ratio = safe_ratio(follow_edges, total_edges + 1.0)
        row = {
            "cascade_idx": cascade_idx,
            "size_bucket": item.get("size_bucket", ""),
            "coord_edge_density_raw": density,
            "largest_coord_component_ratio_raw": largest_ratio,
            "coord_clustering_raw": clustering,
            "coord_temporal_burst_raw": temporal_burst,
            "coord_text_similarity_mean_raw": text_sim,
            "follow_coord_ratio_raw": follow_ratio,
            "coordination_edge_count": total_edges,
        }
        rows.append(row)
        if write_edges and total_edges > 0:
            # Keep the edge artifact compact and deterministic: one summary row per source type.
            for source, count in [
                ("follow", follow_edges),
                ("temporal", temporal_edges),
                ("text", text_edges),
                ("structure", structure_edges),
            ]:
                if count > 0:
                    edges_to_write.append({"cascade_idx": cascade_idx, "edge_source": source, "edge_count": count})
    score_map = {row["cascade_idx"]: row for row in rows}
    add_percentile_scores(score_map, {
        "coord_edge_density_raw": "coord_edge_density",
        "largest_coord_component_ratio_raw": "largest_coord_component_ratio",
        "coord_clustering_raw": "coord_clustering",
        "coord_temporal_burst_raw": "coord_temporal_burst",
        "coord_text_similarity_mean_raw": "coord_text_similarity_mean",
        "follow_coord_ratio_raw": "follow_coord_ratio",
    }, by_bucket=True)
    for row in rows:
        row["coordination_score"] = (
            0.25 * row["coord_edge_density"]
            + 0.25 * row["largest_coord_component_ratio"]
            + 0.15 * row["coord_clustering"]
            + 0.15 * row["coord_temporal_burst"]
            + 0.10 * row["coord_text_similarity_mean"]
            + 0.10 * row["follow_coord_ratio"]
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "coordination_event_scores.csv", rows)
    write_csv(out_dir / "coordination_edges.csv", edges_to_write)
    return score_map


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
