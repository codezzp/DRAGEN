from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401
from dragen.config import apply_config


ABLATION_FLAGS = {
    "no_tree": [],
    "no_multiscale": [],
    "no_role": ["--no-use-role", "--lambda-role", "0.0"],
    "no_memory": ["--no-use-memory"],
    "no_global_prior": ["--no-use-global-prior"],
    "no_adaptive_sampling": ["--no-use-adaptive-sampler"],
    "no_gate": ["--no-use-gate"],
    "no_uncertainty": ["--no-use-uncertainty", "--lambda-uncertainty", "0.0"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one DRAGEN-Full ablation by forwarding to script 16.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation", default=None, choices=sorted(ABLATION_FLAGS))
    args, rest = parser.parse_known_args()
    args = apply_config(parser, args, sys.argv[1:])
    if not args.ablation:
        raise SystemExit("Missing required argument --ablation or config field ablation")
    config_args = ["--config", args.config] if args.config else []
    cmd = [sys.executable, "scripts/16_train_dragen_full.py", *config_args, *rest, *ABLATION_FLAGS[args.ablation]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
