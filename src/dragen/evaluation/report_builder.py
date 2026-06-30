#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a data readiness statistics report for one processed run.

This script does not train anything. It inspects the files under
work/runs/<run_id>/processed and writes:

  work/runs/<run_id>/org_task/stats_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


DEFAULT_SIZE_BINS = [
    (0, 0),
    (1, 10),
    (11, 50),
    (51, 100),
    (101, 500),
    (501, 1000),
    (1001, 5000),
    (5001, None),
]

DEFAULT_DURATION_BINS_SECONDS = [
    (0, 0, "0s"),
    (1, 3600, "<=1h"),
    (3601, 6 * 3600, "1-6h"),
    (6 * 3600 + 1, 24 * 3600, "6-24h"),
    (24 * 3600 + 1, 7 * 24 * 3600, "1-7d"),
    (7 * 24 * 3600 + 1, 30 * 24 * 3600, "7-30d"),
    (30 * 24 * 3600 + 1, None, ">30d"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line") from exc


def _count_tsv_rows(path: Path, has_header: bool = False) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if has_header and line_no == 1:
                continue
            if line.strip():
                count += 1
    return count


def _quantile(sorted_values: List[int], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _numeric_summary(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    vals = sorted(values)
    return {
        "count": len(vals),
        "min": vals[0],
        "p25": _quantile(vals, 0.25),
        "median": _quantile(vals, 0.50),
        "p75": _quantile(vals, 0.75),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def _bin_label(left: int, right: Optional[int]) -> str:
    if right is None:
        return f"{left}+"
    if left == right:
        return str(left)
    return f"{left}-{right}"


def _bin_counts(values: List[int], bins: List[tuple]) -> Dict[str, Dict[str, Any]]:
    total = len(values)
    out: Dict[str, Dict[str, Any]] = {}
    for item in bins:
        if len(item) == 3:
            left, right, label = item
        else:
            left, right = item
            label = _bin_label(left, right)
        cnt = 0
        for val in values:
            if val >= left and (right is None or val <= right):
                cnt += 1
        out[label] = {
            "count": cnt,
            "ratio": (cnt / total) if total else None,
        }
    return out


def _coverage(covered: int, total: int, status: str = "ok", path: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "covered": int(covered),
        "total": int(total),
        "ratio": (covered / total) if total else None,
        "status": status,
        "path": str(path) if path is not None else None,
    }


def _load_nonempty_text_ids(path: Path, id_key: str) -> Set[int]:
    ids: Set[int] = set()
    if not path.exists():
        return ids
    for obj in _iter_jsonl(path):
        if id_key not in obj:
            continue
        text = obj.get("text", "")
        if text is not None and str(text).strip():
            ids.add(int(obj[id_key]))
    return ids


def _load_nonempty_profile_users(path: Path) -> Set[int]:
    users: Set[int] = set()
    if not path.exists():
        return users
    for obj in _iter_jsonl(path):
        if "user_idx" not in obj:
            continue
        text = obj.get("profile_text", "")
        if text is not None and str(text).strip():
            users.add(int(obj["user_idx"]))
    return users


def _scan_follow_coverage(follow_edges: Path, retweet_users: Set[int]) -> Dict[str, Any]:
    if not follow_edges.exists():
        return {
            **_coverage(0, len(retweet_users), status="missing", path=follow_edges),
            "num_edges": 0,
            "checked_as": "endpoint",
        }

    missing = set(retweet_users)
    num_edges = 0
    bad_rows = 0
    with open(follow_edges, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                bad_rows += 1
                continue
            try:
                src = int(parts[0])
                dst = int(parts[1])
            except ValueError:
                bad_rows += 1
                continue
            num_edges += 1
            if src in missing:
                missing.remove(src)
            if dst in missing:
                missing.remove(dst)
            if not missing:
                # Count coverage exactly; edge count is partial after early stop.
                break

    covered = len(retweet_users) - len(missing)
    return {
        **_coverage(covered, len(retweet_users), status="ok", path=follow_edges),
        "num_edges_scanned": num_edges,
        "bad_rows": bad_rows,
        "checked_as": "endpoint",
        "note": "A retweet user is covered if it appears as src_user_idx or dst_user_idx in follow_edges.tsv.",
    }


def _resolve_run_id(config_path: Path, explicit_run_id: Optional[str]) -> str:
    if explicit_run_id:
        return explicit_run_id
    if config_path.exists():
        cfg = _load_json(config_path)
        run_id = cfg.get("build", {}).get("run", {}).get("run_id")
        if run_id:
            return str(run_id)
    return "run_0001"


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    run_id = _resolve_run_id(repo_root / args.config, args.run_id)
    processed = Path(args.processed_dir) if args.processed_dir else repo_root / "work" / "runs" / run_id / "processed"
    processed = processed.resolve()

    mapping_dir = processed / "mapping"
    events_dir = processed / "events"
    text_dir = processed / "text"
    user_dir = processed / "user"
    graph_dir = processed / "graph"

    cascade_root = mapping_dir / "cascade_root.jsonl"
    events = events_dir / "events.jsonl"
    user_id_map = mapping_dir / "user_id_map.tsv"
    root_text = text_dir / "root_text.jsonl"
    retweet_text = text_dir / "retweet_text.jsonl"
    post_text = text_dir / "post_text.jsonl"
    user_profile = user_dir / "user_profile.jsonl"
    common_follow_edges = repo_root / "work" / "graph" / "follow_edges.tsv"
    legacy_follow_edges = graph_dir / "follow_edges.tsv"
    follow_edges = common_follow_edges if common_follow_edges.exists() else legacy_follow_edges
    events_stats = processed / "events_stats.json"

    if not processed.exists():
        raise FileNotFoundError(f"processed dir not found: {processed}")
    if not cascade_root.exists():
        raise FileNotFoundError(f"missing required file: {cascade_root}")
    if not events.exists():
        raise FileNotFoundError(f"missing required file: {events}")

    root_by_cascade: Dict[int, Dict[str, int]] = {}
    root_tweet_ids: Set[int] = set()
    root_users: Set[int] = set()
    for obj in _iter_jsonl(cascade_root):
        cidx = int(obj["cascade_idx"])
        rec = {
            "root_user_idx": int(obj["root_user_idx"]),
            "root_time_epoch": int(obj["root_time_epoch"]),
            "root_tweet_idx": int(obj["root_tweet_idx"]),
        }
        root_by_cascade[cidx] = rec
        root_tweet_ids.add(rec["root_tweet_idx"])
        root_users.add(rec["root_user_idx"])

    retweet_counts: Counter[int] = Counter()
    max_retweet_time: Dict[int, int] = {}
    retweet_tweet_ids: List[int] = []
    retweet_users: Set[int] = set()
    all_event_users: Set[int] = set(root_users)
    missing_root_for_event = 0

    for obj in _iter_jsonl(events):
        cidx = int(obj["cascade_idx"])
        retweet_counts[cidx] += 1
        tweet_idx = int(obj["tweet_idx"])
        user_idx = int(obj["retweet_user_idx"])
        retweet_tweet_ids.append(tweet_idx)
        retweet_users.add(user_idx)
        all_event_users.add(user_idx)
        retweet_time = int(obj["retweet_time"])
        old = max_retweet_time.get(cidx)
        if old is None or retweet_time > old:
            max_retweet_time[cidx] = retweet_time
        if cidx not in root_by_cascade:
            missing_root_for_event += 1

    num_cascades = len(root_by_cascade)
    num_retweets = len(retweet_tweet_ids)
    num_users = _count_tsv_rows(user_id_map)

    cascade_sizes = [int(retweet_counts.get(cidx, 0)) for cidx in root_by_cascade.keys()]
    durations = []
    negative_durations = 0
    for cidx, root in root_by_cascade.items():
        last_time = max_retweet_time.get(cidx, root["root_time_epoch"])
        duration = int(last_time) - int(root["root_time_epoch"])
        if duration < 0:
            negative_durations += 1
            duration = 0
        durations.append(duration)

    root_text_status = "ok" if root_text.exists() else "missing"
    root_text_ids = _load_nonempty_text_ids(root_text, "tweet_idx") if root_text.exists() else set()

    retweet_text_status = "ok" if retweet_text.exists() else "missing"
    retweet_text_ids = _load_nonempty_text_ids(retweet_text, "tweet_idx") if retweet_text.exists() else set()
    if not retweet_text.exists() and post_text.exists():
        # Keep the requested metric strict, but report a useful fallback signal.
        retweet_text_status = "missing_retweet_text_jsonl_post_text_available"

    profile_status = "ok" if user_profile.exists() else "missing"
    profile_users = _load_nonempty_profile_users(user_profile) if user_profile.exists() else set()

    if args.skip_follow_scan:
        follow_coverage = {
            **_coverage(0, len(retweet_users), status="skipped", path=follow_edges),
            "checked_as": "endpoint",
        }
    else:
        follow_coverage = _scan_follow_coverage(follow_edges, retweet_users)

    report = {
        "schema_version": 1,
        "created_at_epoch": int(time.time()),
        "run_id": run_id,
        "processed_dir": str(processed),
        "inputs": {
            "cascade_root": str(cascade_root),
            "events": str(events),
            "user_id_map": str(user_id_map),
            "root_text": str(root_text),
            "retweet_text": str(retweet_text),
            "post_text": str(post_text),
            "user_profile": str(user_profile),
            "follow_edges": str(follow_edges),
            "common_follow_edges": str(common_follow_edges),
            "legacy_run_follow_edges": str(legacy_follow_edges),
            "events_stats": str(events_stats),
        },
        "num_cascades": num_cascades,
        "num_retweets": num_retweets,
        "num_users": num_users,
        "root_text_coverage": _coverage(
            len(root_tweet_ids & root_text_ids),
            len(root_tweet_ids),
            status=root_text_status,
            path=root_text,
        ),
        "retweet_text_coverage": _coverage(
            len(set(retweet_tweet_ids) & retweet_text_ids),
            len(set(retweet_tweet_ids)),
            status=retweet_text_status,
            path=retweet_text,
        ),
        "profile_coverage": _coverage(
            len(all_event_users & profile_users),
            len(all_event_users),
            status=profile_status,
            path=user_profile,
        ),
        "follow_coverage": follow_coverage,
        "retweet_count_distribution": {
            "unit": "retweets_per_cascade",
            "summary": _numeric_summary(cascade_sizes),
            "bins": _bin_counts(cascade_sizes, DEFAULT_SIZE_BINS),
        },
        "duration_distribution": {
            "unit": "seconds",
            "summary": _numeric_summary(durations),
            "bins": _bin_counts(durations, DEFAULT_DURATION_BINS_SECONDS),
            "negative_duration_cascades_clamped_to_zero": negative_durations,
        },
        "diagnostics": {
            "num_root_users": len(root_users),
            "num_retweet_users": len(retweet_users),
            "num_active_users_in_events": len(all_event_users),
            "num_unique_retweet_posts": len(set(retweet_tweet_ids)),
            "events_missing_cascade_root": missing_root_for_event,
            "events_stats": _load_json(events_stats) if events_stats.exists() else None,
        },
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="Repository root. Defaults to current directory.")
    ap.add_argument("--config", default="config.json", help="Config path relative to repo_root.")
    ap.add_argument("--run_id", default=None, help="Run id. Defaults to build.run.run_id in config.json.")
    ap.add_argument("--processed_dir", default=None, help="Override processed directory.")
    ap.add_argument("--out", default=None, help="Output JSON path.")
    ap.add_argument(
        "--skip_follow_scan",
        action="store_true",
        help="Skip scanning graph/follow_edges.tsv. Useful for quick checks on very large graph files.",
    )
    args = ap.parse_args()

    report = build_report(args)
    run_id = report["run_id"]
    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out) if args.out else repo_root / "work" / "runs" / run_id / "org_task" / "stats_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(out_path)
    print(f"OK: stats report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
