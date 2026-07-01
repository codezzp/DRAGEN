#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check that processed/mapping/user_id_map.tsv matches uidlist.txt order.

user_id_map.tsv format:
  user_idx \t raw_uid

uidlist.txt format:
  raw_uid (one per line)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uidlist", required=True, help="Path to uidlist.txt")
    ap.add_argument("--user_id_map", required=True, help="Path to user_id_map.tsv")
    ap.add_argument("--start_from", type=int, default=1, help="Expected start index (0 or 1)")
    ap.add_argument("--limit", type=int, default=0, help="Only check first N lines (0 means full)")
    args = ap.parse_args()

    uidlist = Path(args.uidlist)
    user_map = Path(args.user_id_map)
    start_from = int(args.start_from)
    limit = int(args.limit)

    if start_from not in (0, 1):
        raise SystemExit("--start_from must be 0 or 1")
    if not uidlist.exists():
        raise SystemExit(f"uidlist not found: {uidlist}")
    if not user_map.exists():
        raise SystemExit(f"user_id_map not found: {user_map}")

    checked = 0
    with uidlist.open("r", encoding="utf-8-sig", errors="replace") as fu, user_map.open(
        "r", encoding="utf-8", errors="replace"
    ) as fm:
        while True:
            if limit and checked >= limit:
                break
            u = fu.readline()
            m = fm.readline()
            if not u or not m:
                # length mismatch or both ended
                if bool(u) != bool(m):
                    print(f"FAIL length mismatch at line {checked+1}: uidlist_has_more={bool(u)} map_has_more={bool(m)}")
                    return 2
                break

            raw_uid = u.strip().lstrip("\ufeff")
            parts = m.rstrip("\n").split("\t")
            if len(parts) < 2:
                print(f"FAIL bad map line at line {checked+1}: {m[:120]!r}")
                return 3
            try:
                idx = int(parts[0])
            except ValueError:
                print(f"FAIL bad index at line {checked+1}: {parts[0]!r}")
                return 4
            uid2 = parts[1].lstrip("\ufeff")
            exp = checked + start_from
            if idx != exp or uid2 != raw_uid:
                print(
                    "FAIL mismatch at line {}: expected idx={},uid={!r} got idx={},uid={!r}".format(
                        checked + 1, exp, raw_uid, idx, uid2
                    )
                )
                return 5
            checked += 1

    print(f"OK user_id_map matches uidlist (start_from={start_from}), checked_lines={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

