from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.labeling.diagnostics import pearson
from dragen.labeling.label_features import DEFAULT_RUN_ID, default_run_dir, read_label_csv

VERSIONS = {
    "label_v1_score_rank": "labels_v1_score_rank",
    "label_v2_stratified_score": "labels_v2_stratified_score",
    "label_v3_lf_vote": "labels_v3_lf_vote",
    "label_v4_coordination_network": "labels_v4_coordination_network",
    "label_v5_ensemble_consensus": "labels_v5_ensemble_consensus",
}


def main() -> int:
    args = parse_args()
    run_dir = default_run_dir(args.run_id)
    out_dir = args.out_dir or run_dir / "label_comparison"
    rows = []
    for version, dirname in VERSIONS.items():
        path = run_dir / dirname / "weak_event_labels.csv"
        if not path.exists():
            continue
        labels = read_label_csv(path)
        n = len(labels)
        y = [int(row["label"]) for row in labels]
        obs = [float(row.get("observed_retweet_count", 0.0)) for row in labels]
        conf = [float(row.get("label_confidence", 0.0)) for row in labels]
        pos = y.count(1)
        neg = y.count(0)
        ign = y.count(-1)
        rows.append({
            "label_version": version,
            "positive": pos,
            "negative": neg,
            "ignore": ign,
            "pos_ratio": pos / max(n, 1),
            "neg_ratio": neg / max(n, 1),
            "ignore_ratio": ign / max(n, 1),
            "corr_with_observed_retweet_count": pearson(obs, [float(v) for v in y]) if labels else 0.0,
            "mean_confidence": sum(conf) / max(len(conf), 1),
            "size_bucket_balance_score": size_balance(labels),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "label_version_comparison.csv"
    if rows:
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("", encoding="utf-8")
    print(f"Wrote label comparison to {out} versions={len(rows)}")
    return 0


def size_balance(rows: list[dict[str, str]]) -> float:
    buckets: dict[str, list[int]] = {}
    for row in rows:
        label = int(row["label"])
        if label < 0:
            continue
        buckets.setdefault(row.get("size_bucket", ""), []).append(label)
    if not buckets:
        return 0.0
    scores = []
    for labels in buckets.values():
        pos = labels.count(1)
        neg = labels.count(0)
        scores.append(min(pos, neg) / max(pos + neg, 1))
    return sum(scores) / len(scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare weak label versions.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
