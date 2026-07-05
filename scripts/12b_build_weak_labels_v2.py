from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from dragen.labeling.diagnostics import build_label_diagnostics, write_json
from dragen.labeling.label_features import (
    DEFAULT_RUN_ID,
    add_percentile_scores,
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
    feature_dir = args.feature_dir or default_feature_dir(args.run_id)
    cascade_table = args.cascade_table or default_cascade_table(args.run_id)
    global_edges = args.global_candidate_edges or default_global_candidate_edges(args.run_id)
    out_dir = args.out_dir or run_dir / "labels_v2_stratified_score"
    features = load_event_features(feature_dir, cascade_table, global_edges)
    add_percentile_scores(features, {
        "burst_raw": "burst_score",
        "coordination_raw": "coordination_score",
        "structure_raw": "structure_score",
        "text_raw": "text_score",
    }, by_bucket=True)
    for item in features.values():
        item["weak_score"] = 0.25 * item["burst_score"] + 0.30 * item["coordination_score"] + 0.25 * item["structure_score"] + 0.20 * item["text_score"]
        item["evidence_hit_count"] = sum(1 for name in ["burst_score", "coordination_score", "structure_score", "text_score"] if item[name] >= 0.8)
    qs = bucket_quantiles(features.values(), "weak_score", [0.5, 0.8])
    rows = []
    for item in features.values():
        bucket = item["size_bucket"]
        if bucket == "<8":
            label = -1
        elif item["weak_score"] >= qs[bucket][0.8] and item["evidence_hit_count"] >= 2:
            label = 1
        elif item["weak_score"] <= qs[bucket][0.5] and item["evidence_hit_count"] <= 1:
            label = 0
        else:
            label = -1
        confidence = abs(item["weak_score"] - 0.5) * (1.0 + min(item["evidence_hit_count"], 4) / 4.0)
        row = make_label_row(item, label, min(confidence, 1.0), item["weak_score"], "stratified_score")
        row.update({k: round(float(item[k]), 8) for k in ["burst_score", "coordination_score", "structure_score", "text_score"]})
        row["evidence_hit_count"] = int(item["evidence_hit_count"])
        rows.append(row)
    rows = sorted(rows, key=lambda row: int(row["cascade_idx"]))
    write_label_csv(out_dir / "weak_event_labels.csv", rows, ["burst_score", "coordination_score", "structure_score", "text_score", "evidence_hit_count"])
    write_json(out_dir / "label_diagnostics.json", build_label_diagnostics(rows))
    print(f"Wrote Label-v2 stratified_score to {out_dir} rows={len(rows)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Label-v2 stratified score weak labels.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--cascade-table", type=str, default=None)
    parser.add_argument("--global-candidate-edges", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    from pathlib import Path
    for name in ["feature_dir", "cascade_table", "global_candidate_edges", "out_dir"]:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    return args


if __name__ == "__main__":
    raise SystemExit(main())
