from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.config import apply_config
from dragen.evaluation.result_tables import export_ablation_results, export_main_results, export_risk_retrieval_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fair event-level DRAGEN result tables.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dirs", nargs="+", default=None, help="Artifact directories for fair main/risk tables.")
    parser.add_argument("--out-dir", default="work/artifacts/reports")
    parser.add_argument("--main-out", default=None)
    parser.add_argument("--risk-out", default=None)
    parser.add_argument("--ablation-run-dirs", nargs="*", default=None)
    parser.add_argument("--full-run-dir", default=None)
    parser.add_argument("--ablation-out", default=None)
    args = apply_config(parser, parser.parse_args(), sys.argv[1:])
    if not args.run_dirs:
        raise SystemExit("Missing required argument --run-dirs or config field tables.run_dirs")

    out_dir = Path(args.out_dir)
    run_dirs = [Path(p) for p in args.run_dirs]
    main_out = Path(args.main_out) if args.main_out else out_dir / "main_results.csv"
    risk_out = Path(args.risk_out) if args.risk_out else out_dir / "risk_retrieval_results.csv"
    export_main_results(run_dirs, main_out)
    export_risk_retrieval_results(run_dirs, risk_out)

    if args.ablation_run_dirs:
        ablation_out = Path(args.ablation_out) if args.ablation_out else out_dir / "ablation_results.csv"
        full_run = Path(args.full_run_dir) if args.full_run_dir else None
        export_ablation_results([Path(p) for p in args.ablation_run_dirs], ablation_out, full_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
