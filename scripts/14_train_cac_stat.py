from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from dragen.config import apply_config
from dragen.baselines.cac_stat import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAC-Stat baseline.")
    parser.add_argument("--config", default=None)
    apply_config(parser, parser.parse_args(), sys.argv[1:])
    raise SystemExit(main())
