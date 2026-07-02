from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.labeling.coordination_network import build_coordination_network_scores
from dragen.labeling.diagnostics import build_label_diagnostics, write_json
from dragen.labeling.label_features import (
    DEFAULT_RUN_ID,
    bucket_quantiles,
    default_cascade_table,
    default_feature_dir,
    default_global_candidate_edges,
    default_run_dir,
    load_event_features,
    make_label_row,
    write_label_csv,
)


def main() -> int:
    args = parse_args()
    run_dir = default_run_dir(args.run_id)
    out_dir = args.out_dir or run_dir / "labels_v4_coordination_network"
    features = load_event_features(args.feature_dir or default_feature_dir(args.run_id), args.cascade_table or default_cascade_table(args.run_id), args.global_candidate_edges or default_global_candidate_edges(args.run_id))
    coord = build_coordination_network_scores(features, out_dir)
    for cascade_idx, item in features.items():
        item.update(coord[cascade_idx])
        item["weak_score"] = item["coordination_score"]
    qs_score = bucket_quantiles(features.values(), "coordination_score", [0.4, 0.8])
    qs_comp = bucket_quantiles(features.values(), "largest_coord_component_ratio", [0.4, 0.7])
    qs_density = bucket_quantiles(features.values(), "coord_edge_density", [0.4, 0.7])
    rows = []
    edge_count = 0
    for item in features.values():
        bucket = item["size_bucket"]
        edge_count += int(item.get("coordination_edge_count", 0))
        if bucket != "<8" and item["coordination_score"] >= qs_score[bucket][0.8] and item["largest_coord_component_ratio"] >= qs_comp[bucket][0.7] and item["coord_edge_density"] >= qs_density[bucket][0.7]:
            label = 1
        elif bucket != "<8" and item["coordination_score"] <= qs_score[bucket][0.4] and item["largest_coord_component_ratio"] <= qs_comp[bucket][0.4] and item["coord_edge_density"] <= qs_density[bucket][0.4]:
            label = 0
        else:
            label = -1
        confidence = abs(float(item["coordination_score"]) - 0.5) * 2.0
        row = make_label_row(item, label, min(confidence, 1.0), item["coordination_score"], "coordination_network")
        row.update({
            "coord_edge_density": round(float(item["coord_edge_density"]), 8),
            "largest_coord_component_ratio": round(float(item["largest_coord_component_ratio"]), 8),
            "coord_clustering": round(float(item["coord_clustering"]), 8),
            "coordination_edge_count": int(item.get("coordination_edge_count", 0)),
        })
        rows.append(row)
    rows = sorted(rows, key=lambda row: int(row["cascade_idx"]))
    write_label_csv(out_dir / "weak_event_labels.csv", rows, ["coord_edge_density", "largest_coord_component_ratio", "coord_clustering", "coordination_edge_count"])
    extra = {
        "coordination_edge_count": edge_count,
        "coordination_component_stats": build_component_stats(rows),
    }
    write_json(out_dir / "label_diagnostics.json", build_label_diagnostics(rows, extra=extra))
    print(f"Wrote Label-v4 coordination_network to {out_dir} rows={len(rows)}")
    return 0


def build_component_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    vals = [float(row.get("largest_coord_component_ratio", 0.0)) for row in rows]
    return {"mean_largest_component_ratio": sum(vals) / max(len(vals), 1), "max_largest_component_ratio": max(vals) if vals else 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Label-v4 coordination network weak labels.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--cascade-table", type=Path, default=None)
    parser.add_argument("--global-candidate-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
