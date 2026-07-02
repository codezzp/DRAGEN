from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.evaluation.training_curves import plot_training_curves


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    reports = artifact_dir / "reports"
    out = Path(args.out) if args.out else reports / "training_curves.png"
    ok = plot_training_curves(reports / "epoch_metrics.csv", reports / "loss_breakdown.json", out)
    if ok:
        actual_out = out if out.exists() else out.with_suffix(".html")
        print(f"wrote {actual_out}")
        return 0
    print("no training curves written")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DRAGEN training curves from report files.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
