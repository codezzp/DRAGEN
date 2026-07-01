from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from dragen.baselines.campaign_gnn import main as campaign_main
from dragen.baselines.temporal_gnn import main as temporal_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GNN baselines.")
    parser.add_argument("--baseline", choices=["campaign_gnn", "temporal_gnn"], required=True)
    args = parser.parse_args()
    return campaign_main() if args.baseline == "campaign_gnn" else temporal_main()


if __name__ == "__main__":
    raise SystemExit(main())
