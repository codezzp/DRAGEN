from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.labeling.diagnostics import build_label_diagnostics, write_json
from dragen.labeling.label_features import DEFAULT_RUN_ID, default_cascade_table, default_run_dir, read_label_csv, split_for_cascade, size_bucket, write_label_csv


def main() -> int:
    args = parse_args()
    run_dir = default_run_dir(args.run_id)
    source = args.source_labels or run_dir / "labels" / "weak_event_labels.csv"
    cascade_table = args.cascade_table or default_cascade_table(args.run_id)
    out_dir = args.out_dir or run_dir / "labels_v1_score_rank"
    cascade_meta = load_cascade_meta(cascade_table)
    rows = []
    for row in read_label_csv(source):
        cascade_idx = str(row["cascade_idx"])
        meta = cascade_meta.get(cascade_idx, {})
        observed = int(float(meta.get("observed_retweet_count", 0.0)))
        final = int(float(meta.get("final_retweet_count", 0.0)))
        label = int(row.get("weak_label", row.get("label", -1)))
        rows.append({
            "cascade_idx": cascade_idx,
            "label": label,
            "split": row.get("split") or split_for_cascade(cascade_idx),
            "label_confidence": row.get("label_confidence", 0.0),
            "weak_score": row.get("weak_score", 0.0),
            "label_method": "score_rank",
            "size_bucket": size_bucket(observed),
            "observed_retweet_count": observed,
            "final_retweet_count": final,
        })
    rows = sorted(rows, key=lambda item: int(item["cascade_idx"]))
    write_label_csv(out_dir / "weak_event_labels.csv", rows)
    write_json(out_dir / "label_diagnostics.json", build_label_diagnostics(rows))
    print(f"Wrote Label-v1 score_rank unified copy to {out_dir} rows={len(rows)}")
    return 0


def load_cascade_meta(path: Path) -> dict[str, dict[str, str]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row["cascade_idx"]): row for row in csv.DictReader(f)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export existing Label-v1 score_rank labels to the unified label schema.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-labels", type=Path, default=None)
    parser.add_argument("--cascade-table", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
