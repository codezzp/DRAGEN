#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deduplicate Weibo cascade raw files IN PLACE (optional).

Cascade file format (as used by this repo):
  line1: root_tweet_id <tab> root_user_id <tab> root_time_str <tab> actual_count
  line2: N (dataset/stat retweet count, may be inconsistent in raw data)
  then repeated pairs:
    - event line: uid time retweet_id
    - text line

This script deduplicates retweet pairs by (event_line + text_line) by default,
and updates line2 to the number of kept pairs.

It can operate on:
  - a single .txt file
  - a directory containing many cascade .txt files

"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _iter_targets(path: Path, glob_pattern: str, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    if not path.is_dir():
        return
    if recursive:
        yield from path.rglob(glob_pattern)
    else:
        yield from path.glob(glob_pattern)


def _ensure_newline(s: str) -> str:
    return s if s.endswith("\n") else (s + "\n")


def dedup_one_file(
    fp: Path,
    *,
    inplace: bool,
    key_mode: str,
    update_count_line2: bool,
    encoding: str,
) -> Tuple[int, int, int]:
    """
    Returns (kept_pairs, dropped_pairs, total_pairs_seen).
    """
    # Read header
    with open(fp, "r", encoding=encoding, errors="replace") as f:
        header1 = f.readline()
        header2 = f.readline()
        if not header1 or not header2:
            return 0, 0, 0

        # Temp file that stores deduped pairs only (no headers)
        tmp_pairs = fp.with_suffix(fp.suffix + ".dedup_pairs.tmp")
        kept = 0
        dropped = 0
        total = 0
        seen = set()

        with open(tmp_pairs, "w", encoding=encoding, errors="replace") as out_pairs:
            while True:
                event_line = f.readline()
                if not event_line:
                    break
                text_line = f.readline()
                if not text_line:
                    break
                total += 1

                # Keep blank event lines as-is to avoid breaking alignment.
                if not event_line.strip():
                    out_pairs.write(_ensure_newline(event_line.rstrip("\n").rstrip("\r")))
                    out_pairs.write(_ensure_newline(text_line.rstrip("\n").rstrip("\r")))
                    kept += 1
                    continue

                ev = event_line.rstrip("\n").rstrip("\r")
                tx = text_line.rstrip("\n").rstrip("\r")
                if key_mode == "event":
                    key = ev
                else:
                    # default: event_text
                    key = (ev, tx)

                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)

                out_pairs.write(_ensure_newline(ev))
                out_pairs.write(_ensure_newline(tx))
                kept += 1

    # Build final tmp file: header + (updated) count + pairs
    tmp_final = fp.with_suffix(fp.suffix + ".dedup.tmp")
    with open(tmp_final, "w", encoding=encoding, errors="replace") as out:
        out.write(_ensure_newline(header1.rstrip("\n").rstrip("\r")))
        if update_count_line2:
            out.write(f"{kept}\n")
        else:
            out.write(_ensure_newline(header2.rstrip("\n").rstrip("\r")))
        with open(fp.with_suffix(fp.suffix + ".dedup_pairs.tmp"), "r", encoding=encoding, errors="replace") as in_pairs:
            shutil.copyfileobj(in_pairs, out)

    # Cleanup pairs tmp
    try:
        fp.with_suffix(fp.suffix + ".dedup_pairs.tmp").unlink()
    except Exception:
        pass

    if not inplace:
        # Keep output next to input
        out_path = fp.with_suffix(fp.suffix + ".deduped.txt")
        shutil.move(str(tmp_final), str(out_path))
        return kept, dropped, total

    os.replace(str(tmp_final), str(fp))
    return kept, dropped, total


def _worker(args) -> Tuple[str, int, int, int, str | None]:
    """
    Worker entry (picklable) for multiprocessing on Windows.
    Returns: (path, kept, dropped, total, error)
    """
    fp_s, inplace, key_mode, update_count_line2, encoding = args
    fp = Path(fp_s)
    try:
        kept, dropped, total = dedup_one_file(
            fp,
            inplace=bool(inplace),
            key_mode=str(key_mode),
            update_count_line2=bool(update_count_line2),
            encoding=str(encoding),
        )
        return str(fp), kept, dropped, total, None
    except Exception as e:
        return str(fp), 0, 0, 0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Deduplicate raw cascade files (event+text pairs).")
    ap.add_argument("--path", required=True, help="A cascade .txt file, or a directory containing cascades.")
    ap.add_argument("--glob", default="*.txt", help="Glob for directory mode (default: *.txt)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--inplace", action="store_true", help="Modify files in place (default: output *.deduped.txt)")
    ap.add_argument(
        "--key_mode",
        choices=["event_text", "event"],
        default="event_text",
        help="Dedup key: event_text=(event line + text line) or event=(event line only)",
    )
    ap.add_argument("--no_update_count_line2", action="store_true", help="Do not update line2 retweet count")
    ap.add_argument("--max_files", type=int, default=None, help="Process at most N files (for quick test)")
    ap.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    ap.add_argument("--verbose", action="store_true", help="Print per-file results (default: only a progress bar + summary)")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = ap.parse_args()

    path = Path(args.path)

    files = [p for p in _iter_targets(path, args.glob, args.recursive) if p.is_file()]
    files.sort()
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]

    if not files:
        print("No files matched.")
        return 2

    total_kept = 0
    total_dropped = 0
    total_pairs = 0
    changed = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    workers = args.workers
    if workers is None:
        workers = os.cpu_count() or 4
    workers = max(1, int(workers))

    tasks = [
        (
            str(fp),
            bool(args.inplace),
            str(args.key_mode),
            (not bool(args.no_update_count_line2)),
            str(args.encoding),
        )
        for fp in files
    ]

    iterator = None
    if tqdm is not None:
        iterator = tqdm(total=len(tasks), desc=f"Dedup cascades (workers={workers})", unit="file")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            fp_s, kept, dropped, total, err = fut.result()
            if err:
                failed += 1
                errors.append((fp_s, err))
                if args.verbose:
                    if iterator is not None:
                        iterator.write(f"[FAIL] {fp_s}: {err}")
                    else:
                        print(f"[FAIL] {fp_s}: {err}")
            else:
                total_kept += kept
                total_dropped += dropped
                total_pairs += total
                if dropped > 0:
                    changed += 1
                if args.verbose:
                    msg = f"{fp_s}: pairs={total} kept={kept} dropped={dropped}"
                    if iterator is not None:
                        iterator.write(msg)
                    else:
                        print(msg)
            if iterator is not None:
                iterator.update(1)

    if iterator is not None:
        iterator.close()

    if failed and not args.verbose:
        # Keep output short: show only a few failures.
        show_n = 10
        head = errors[:show_n]
        print(f"WARN: failed={failed} (showing first {len(head)} errors)")
        for fp_s, err in head:
            print(f"- {fp_s}: {err}")

    print(
        "DONE: "
        f"files={len(files)}, changed={changed}, "
        f"pairs={total_pairs}, kept={total_kept}, dropped={total_dropped}, failed={failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

