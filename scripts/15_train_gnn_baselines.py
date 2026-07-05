from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from dragen.config import apply_config
from dragen.baselines.campaign_gnn import main as campaign_main
from dragen.baselines.temporal_gnn import main as temporal_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GNN baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline", choices=["campaign_gnn", "temporal_gnn"], default=None)
    args = apply_config(parser, parser.parse_args(), sys.argv[1:])
    if not args.baseline:
        raise SystemExit("Missing required argument --baseline or config field baseline")
    return campaign_main() if args.baseline == "campaign_gnn" else temporal_main()


if __name__ == "__main__":
    raise SystemExit(main())
