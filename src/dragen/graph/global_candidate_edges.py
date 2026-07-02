"""Build per-cascade global follow candidate edges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"
DEFAULT_WINDOW_NAME = "obs_1800_step300_multiscale_hybrid_tree"


FIELDS = [
    "cascade_idx",
    "src_user_idx",
    "dst_user_idx",
    "src_local_idx",
    "dst_local_idx",
    "edge_weight",
    "edge_source",
]


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    feature_dir = args.feature_dir or run_dir / "features" / DEFAULT_WINDOW_NAME
    follow_edges = args.follow_edges or PROJECT_ROOT / "graph" / "follow_edges.tsv"
    out_dir = args.out_dir or run_dir / "global_graph" / DEFAULT_WINDOW_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    cascade_users, user_to_cascades = read_visible_users(feature_dir / "node_window_features.csv", args.max_cascades)
    local_pos = {
        cascade_idx: {user_idx: i for i, user_idx in enumerate(sorted(users, key=lambda x: int(x)))}
        for cascade_idx, users in cascade_users.items()
    }

    out_table = out_dir / "global_candidate_edge_table.csv"
    diagnostics = build_candidate_table(
        follow_edges=follow_edges,
        out_table=out_table,
        user_to_cascades=user_to_cascades,
        local_pos=local_pos,
        edge_weight=args.edge_weight,
        max_follow_edges=args.max_follow_edges,
    )
    diagnostics.update(
        {
            "run_id": args.run_id,
            "feature_dir": str(feature_dir),
            "follow_edges": str(follow_edges),
            "output": str(out_table),
            "num_cascades": len(cascade_users),
            "num_visible_user_memberships": sum(len(users) for users in cascade_users.values()),
            "num_unique_visible_users": len(user_to_cascades),
        }
    )
    write_json(out_dir / "global_candidate_diagnostics.json", diagnostics)
    print(
        f"Wrote global candidate edges to {out_table} "
        f"scanned={diagnostics['follow_edges_scanned']} kept={diagnostics['candidate_edges_written']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-cascade candidate follow edges for DRAGEN-Full.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--follow-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--max-cascades", type=int, default=None)
    parser.add_argument("--max-follow-edges", type=int, default=None)
    return parser.parse_args()


def read_visible_users(path: Path, max_cascades: int | None = None) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    cascade_users: Dict[str, Set[str]] = defaultdict(set)
    selected: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            if cascade_idx not in selected:
                if max_cascades is not None and len(selected) >= max_cascades:
                    continue
                selected.add(cascade_idx)
            if cascade_idx not in selected:
                continue
            cascade_users[cascade_idx].add(str(row["user_idx"]))

    user_to_cascades: Dict[str, Set[str]] = defaultdict(set)
    for cascade_idx, users in cascade_users.items():
        for user_idx in users:
            user_to_cascades[user_idx].add(cascade_idx)
    return dict(cascade_users), dict(user_to_cascades)


def build_candidate_table(
    *,
    follow_edges: Path,
    out_table: Path,
    user_to_cascades: Mapping[str, Set[str]],
    local_pos: Mapping[str, Mapping[str, int]],
    edge_weight: float,
    max_follow_edges: int | None,
) -> Dict[str, Any]:
    scanned = 0
    skipped_malformed = 0
    candidate_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with follow_edges.open("r", encoding="utf-8", errors="ignore", newline="") as src:
        reader = csv.reader(src, delimiter="\t")
        for row in reader:
            if max_follow_edges is not None and scanned >= max_follow_edges:
                break
            if len(row) < 2:
                skipped_malformed += 1
                continue
            scanned += 1
            src_user = row[0].strip()
            dst_user = row[1].strip()
            if not src_user or not dst_user:
                skipped_malformed += 1
                continue
            src_cascades = user_to_cascades.get(src_user)
            dst_cascades = user_to_cascades.get(dst_user)
            if not src_cascades or not dst_cascades:
                continue
            for cascade_idx in src_cascades & dst_cascades:
                positions = local_pos[cascade_idx]
                candidate_rows[cascade_idx].append(
                    {
                        "cascade_idx": cascade_idx,
                        "src_user_idx": src_user,
                        "dst_user_idx": dst_user,
                        "src_local_idx": positions[src_user],
                        "dst_local_idx": positions[dst_user],
                        "edge_weight": edge_weight,
                        "edge_source": "follow_graph",
                    }
                )

    written = 0
    with out_table.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for cascade_idx in sorted(candidate_rows, key=lambda x: int(x)):
            rows = sorted(
                candidate_rows[cascade_idx],
                key=lambda row: (int(row["src_local_idx"]), int(row["dst_local_idx"])),
            )
            writer.writerows(rows)
            written += len(rows)

    counts = [len(rows) for rows in candidate_rows.values()]
    return {
        "follow_edges_scanned": scanned,
        "candidate_edges_written": written,
        "skipped_malformed_edges": skipped_malformed,
        "num_cascades_with_candidates": len(candidate_rows),
        "min_candidate_edges_per_cascade": min(counts) if counts else 0,
        "max_candidate_edges_per_cascade": max(counts) if counts else 0,
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
