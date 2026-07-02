from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.labeling.diagnostics import build_label_diagnostics, write_json
from dragen.labeling.label_features import DEFAULT_RUN_ID, default_run_dir, write_label_csv
from dragen.labeling.label_model import ensemble_consensus


def main() -> int:
    args = parse_args()
    run_dir = default_run_dir(args.run_id)
    out_dir = args.out_dir or run_dir / "labels_v5_ensemble_consensus"
    label_paths = {
        "v2_stratified_score": args.v2_labels or run_dir / "labels_v2_stratified_score" / "weak_event_labels.csv",
        "v3_lf_vote": args.v3_labels or run_dir / "labels_v3_lf_vote" / "weak_event_labels.csv",
        "v4_coordination_network": args.v4_labels or run_dir / "labels_v4_coordination_network" / "weak_event_labels.csv",
    }
    rows, extra = ensemble_consensus(label_paths)
    rows = sorted(rows, key=lambda row: int(row["cascade_idx"]))
    write_label_csv(out_dir / "weak_event_labels.csv", rows, ["v2_stratified_score_label", "v3_lf_vote_label", "v4_coordination_network_label"])
    write_json(out_dir / "label_diagnostics.json", build_label_diagnostics(rows, extra=extra))
    print(f"Wrote Label-v5 ensemble_consensus to {out_dir} rows={len(rows)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Label-v5 ensemble consensus weak labels.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--v2-labels", type=Path, default=None)
    parser.add_argument("--v3-labels", type=Path, default=None)
    parser.add_argument("--v4-labels", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
