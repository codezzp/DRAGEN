from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from dragen.config import apply_config
from dragen.training.trainer import train_dragen_full


def main() -> int:
    args = parse_args()
    train_dragen_full(args)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train thesis-aligned DRAGEN-Full.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--role-num", type=int, default=5)
    parser.add_argument("--top-k-global", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--lambda-jump", type=float, default=0.01)
    parser.add_argument("--lambda-struct", type=float, default=0.005)
    parser.add_argument("--lambda-align", type=float, default=0.001)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.001)
    parser.add_argument("--lambda-role", type=float, default=0.0)
    parser.add_argument("--lambda-sampler-edge", type=float, default=0.005)
    parser.add_argument("--lambda-sampler-hub", type=float, default=0.001)
    parser.add_argument("--lambda-sampler-temp", type=float, default=0.005)
    parser.add_argument("--use-global-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-adaptive-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-uncertainty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-role", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tb-log-dir", default=None)
    args = apply_config(parser, parser.parse_args(), sys.argv[1:])
    require_arg(args, "pack_dir")
    require_arg(args, "out_dir")
    return args


def require_arg(args: argparse.Namespace, name: str) -> None:
    if getattr(args, name, None) in (None, ""):
        raise SystemExit(f"Missing required argument --{name.replace('_', '-')} or config field")


if __name__ == "__main__":
    raise SystemExit(main())
