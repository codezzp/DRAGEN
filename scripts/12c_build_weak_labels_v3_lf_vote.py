from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
from dragen.labeling.label_functions import LABEL_FUNCTIONS, apply_label_functions, confidence_from_votes, vote_counts


def main() -> int:
    args = parse_args()
    run_dir = default_run_dir(args.run_id)
    out_dir = args.out_dir or run_dir / "labels_v3_lf_vote"
    features = load_event_features(args.feature_dir or default_feature_dir(args.run_id), args.cascade_table or default_cascade_table(args.run_id), args.global_candidate_edges or default_global_candidate_edges(args.run_id))
    add_percentile_scores(features, {
        "burst_raw": "burst_score",
        "coordination_raw": "coordination_score",
        "structure_raw": "structure_score",
        "text_raw": "text_score",
        "temporal_sync_raw": "temporal_sync_score",
        "follow_density_raw": "follow_density_score",
        "natural_spread_raw": "natural_spread_score",
    }, by_bucket=True)
    for item in features.values():
        item["weak_score"] = 0.25 * item["burst_score"] + 0.30 * item["coordination_score"] + 0.25 * item["structure_score"] + 0.20 * item["text_score"]
    qs = bucket_quantiles(features.values(), "weak_score", [0.5, 0.8])
    rows = []
    vote_rows = []
    lf_active = {name: 0 for name in LABEL_FUNCTIONS}
    conflicts = 0
    for item in features.values():
        votes = apply_label_functions(item)
        pos, neg, abstain = vote_counts(votes)
        confidence = confidence_from_votes(pos, neg)
        bucket = item["size_bucket"]
        if bucket != "<8" and pos >= 2 and confidence >= 0.6 and item["weak_score"] >= qs[bucket][0.8]:
            label = 1
        elif bucket != "<8" and neg >= 2 and pos == 0 and item["weak_score"] <= qs[bucket][0.5]:
            label = 0
        else:
            label = -1
        if pos > 0 and neg > 0:
            conflicts += 1
        for name, value in votes.items():
            if value != -1:
                lf_active[name] += 1
        row = make_label_row(item, label, confidence, item["weak_score"], "lf_vote")
        row.update({"positive_votes": pos, "negative_votes": neg, "abstain_votes": abstain})
        rows.append(row)
        vote_rows.append({"cascade_idx": item["cascade_idx"], **votes})
    rows = sorted(rows, key=lambda row: int(row["cascade_idx"]))
    vote_rows = sorted(vote_rows, key=lambda row: int(row["cascade_idx"]))
    write_label_csv(out_dir / "weak_event_labels.csv", rows, ["positive_votes", "negative_votes", "abstain_votes"])
    write_votes(out_dir / "label_function_votes.csv", vote_rows)
    extra = {
        "lf_coverage": {name: count / max(len(rows), 1) for name, count in lf_active.items()},
        "lf_conflict_rate": conflicts / max(len(rows), 1),
        "lf_agreement_matrix": {},
    }
    write_json(out_dir / "label_diagnostics.json", build_label_diagnostics(rows, extra=extra))
    print(f"Wrote Label-v3 lf_vote to {out_dir} rows={len(rows)}")
    return 0


def write_votes(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cascade_idx", *LABEL_FUNCTIONS.keys()])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Label-v3 LF vote weak labels.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--cascade-table", type=Path, default=None)
    parser.add_argument("--global-candidate-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
