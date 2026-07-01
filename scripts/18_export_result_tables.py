from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.evaluation.result_tables import export_main_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compact DRAGEN result tables.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out", default="work/artifacts/reports/main_results.csv")
    args = parser.parse_args()
    export_main_results([Path(p) for p in args.run_dirs], Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
