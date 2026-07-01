from __future__ import annotations

import argparse
import subprocess
import sys


ABLATION_FLAGS = {
    "no_role": ["--no-use-role", "--lambda-role", "0.0"],
    "no_memory": ["--no-use-memory"],
    "no_global_prior": ["--no-use-global-prior"],
    "no_adaptive_sampling": ["--no-use-adaptive-sampler"],
    "no_gate": ["--no-use-gate"],
    "no_uncertainty": ["--no-use-uncertainty", "--lambda-uncertainty", "0.0"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one DRAGEN-Full ablation by forwarding to script 16.")
    parser.add_argument("--ablation", required=True, choices=sorted(ABLATION_FLAGS))
    args, rest = parser.parse_known_args()
    cmd = [sys.executable, "scripts/16_train_dragen_full.py", *rest, *ABLATION_FLAGS[args.ablation]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
