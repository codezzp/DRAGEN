#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 pack.sqlite 的结构和内容，用于验证打包结果。

用法:
  # 基本检查
  python scripts/inspect_pack.py --pack work/deliverables/run_0001/pack.sqlite --cascades 3 --nodes 10

  # 显示文本和用户信息（完整输出，不截断）
  python scripts/inspect_pack.py --pack work/deliverables/run_0001/pack.sqlite --cascades 3 --nodes 10 --show_text --show_users

  # 限制文本输出长度
  python scripts/inspect_pack.py --pack work/deliverables/run_0001/pack.sqlite --show_text --show_users --sample_max_chars 200
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _print_table_info(cur: sqlite3.Cursor, table: str) -> None:
    rows = cur.execute(f"PRAGMA table_info({table});").fetchall()
    if not rows:
        print(f"- {table}: <no table_info>")
        return
    cols = ", ".join([f"{r[1]}:{r[2]}" for r in rows])  # (cid,name,type,notnull,dflt,pk)
    print(f"- {table}: {cols}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="Path to pack.sqlite")
    ap.add_argument("--cascades", type=int, default=3, help="How many cascades to show")
    ap.add_argument("--nodes", type=int, default=10, help="How many nodes per cascade to show")
    ap.add_argument("--show_text", action="store_true", help="Show text samples for first few post_idx")
    ap.add_argument("--show_users", action="store_true", help="Show user profile samples for first few user_idx")
    ap.add_argument(
        "--sample_max_chars",
        type=int,
        default=0,
        help="Max chars to print for text/profile samples (0 means no truncation)",
    )
    args = ap.parse_args()

    pack = Path(args.pack)
    if not pack.exists():
        print(f"ERROR: pack not found: {pack}")
        return 2

    conn = sqlite3.connect(str(pack))
    try:
        cur = conn.cursor()

        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()]
        print("## Tables")
        for t in tables:
            print(f"- {t}")

        print("\n## Schema")
        for t in tables:
            _print_table_info(cur, t)

        if "meta" in tables:
            print("\n## meta")
            for k, v in cur.execute("SELECT k,v FROM meta ORDER BY k;").fetchall():
                print(f"- {k} = {v}")

        # Cascades samples
        n_cascades = max(0, int(args.cascades))
        n_nodes = max(0, int(args.nodes))
        if "cascades" in tables and n_cascades > 0:
            crows = cur.execute(
                "SELECT cascade_idx, root_post_idx, root_user_idx, root_time_epoch "
                "FROM cascades ORDER BY cascade_idx LIMIT ?;",
                (n_cascades,),
            ).fetchall()
            print("\n## Cascades (first {})".format(n_cascades))
            for r in crows:
                print(f"- cascade_idx={r[0]} root_post_idx={r[1]} root_user_idx={r[2]} root_time_epoch={r[3]}")

            if "nodes" in tables and n_nodes > 0:
                print("\n## Nodes (per cascade, first {} nodes)".format(n_nodes))
                for cidx, *_ in crows:
                    rows = cur.execute(
                        "SELECT node_idx, post_idx, user_idx, time_epoch, is_root "
                        "FROM nodes WHERE cascade_idx=? ORDER BY node_idx LIMIT ?;",
                        (int(cidx), n_nodes),
                    ).fetchall()
                    print(f"- cascade_idx={cidx} nodes={len(rows)}")
                    for rr in rows:
                        print(f"  node_idx={rr[0]} post_idx={rr[1]} user_idx={rr[2]} time_epoch={rr[3]} is_root={rr[4]}")

        if args.show_text and "texts" in tables:
            print("\n## Text samples (first 5)")
            for post_idx, text in cur.execute("SELECT post_idx, text FROM texts ORDER BY post_idx LIMIT 5;").fetchall():
                t = (text or "").replace("\n", "\\n")
                if int(args.sample_max_chars) > 0 and len(t) > int(args.sample_max_chars):
                    t = t[: int(args.sample_max_chars)] + "..."
                print(f"- post_idx={post_idx} text={t!r}")

        if args.show_users and "users" in tables:
            print("\n## User profile samples (first 5)")
            for user_idx, text in cur.execute("SELECT user_idx, profile_text FROM users ORDER BY user_idx LIMIT 5;").fetchall():
                t = (text or "").replace("\n", "\\n")
                if int(args.sample_max_chars) > 0 and len(t) > int(args.sample_max_chars):
                    t = t[: int(args.sample_max_chars)] + "..."
                print(f"- user_idx={user_idx} profile_text={t!r}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

