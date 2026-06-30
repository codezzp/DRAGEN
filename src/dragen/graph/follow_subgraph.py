"""Build a run-level follow subgraph from a large follow edge TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Set


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    post_table = args.post_table or run_dir / "org_task" / "post_table.csv"
    follow_edges = args.follow_edges or PROJECT_ROOT / "graph" / "follow_edges.tsv"
    out_dir = args.out_dir or run_dir / "graphs" / "follow_subgraph"
    out_dir.mkdir(parents=True, exist_ok=True)

    users = read_run_users(post_table, args.max_cascades)
    out_edges = out_dir / "follow_edges_run.tsv"
    kept = 0
    scanned = 0
    with follow_edges.open("r", encoding="utf-8", errors="ignore", newline="") as src, out_edges.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        for row in reader:
            if len(row) < 2:
                continue
            scanned += 1
            u = row[0].strip()
            v = row[1].strip()
            if u in users and v in users:
                writer.writerow([u, v])
                kept += 1

    diagnostics: Dict[str, object] = {
        "run_id": args.run_id,
        "num_run_users": len(users),
        "follow_edges_scanned": scanned,
        "follow_edges_kept": kept,
        "output": str(out_edges),
    }
    (out_dir / "follow_subgraph_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote run follow subgraph to {out_edges}")
    print(f"users={len(users)} scanned={scanned} kept={kept}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a run-level follow subgraph.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--post-table", type=Path, default=None)
    parser.add_argument("--follow-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-cascades", type=int, default=None)
    return parser.parse_args()


def read_run_users(path: Path, max_cascades: int | None) -> Set[str]:
    users: Set[str] = set()
    selected: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            if cascade_idx not in selected:
                if max_cascades is not None and len(selected) >= max_cascades:
                    break
                selected.add(cascade_idx)
            users.add(str(row["user_idx"]))
    return users
