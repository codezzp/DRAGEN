from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.features.non_text_evidence_v2 import build_non_text_evidence_v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    feature_dir = args.feature_dir or run_dir / "features" / "obs_1800_step300_multiscale_hybrid_tree"
    window_dir = args.window_dir or run_dir / "windows" / "obs_1800_step300_multiscale_hybrid_tree"
    global_candidate_edges = args.global_candidate_edges
    if global_candidate_edges is None:
        candidate = run_dir / "global_graph" / "obs_1800_step300_multiscale_hybrid_tree" / "global_candidate_edge_table.csv"
        global_candidate_edges = candidate if candidate.exists() else None
    out_dir = args.out_dir or run_dir / "evidence" / "obs_1800_step300_multiscale_hybrid_tree_global_follow"
    diagnostics = build_non_text_evidence_v2(
        feature_dir=feature_dir,
        window_dir=window_dir,
        global_candidate_edges=global_candidate_edges,
        out_dir=out_dir,
    )
    print(
        f"Wrote Evidence-v2 to {out_dir} "
        f"node_rows={diagnostics['num_node_rows']} window_rows={diagnostics['num_window_rows']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-text Evidence-v2 observable evidence features.")
    parser.add_argument("--run-id", default="run_0002")
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--window-dir", type=Path, default=None)
    parser.add_argument("--global-candidate-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
