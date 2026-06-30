#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze retweet relative-time distributions for one processed run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "work" / "runs" / "run_0003" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "work" / "runs" / "run_0003" / "time_distribution"

TIME_BINS = [
    ("0_5m", 0, 300),
    ("5_10m", 300, 600),
    ("10_15m", 600, 900),
    ("15_20m", 900, 1200),
    ("20_25m", 1200, 1500),
    ("25_30m", 1500, 1800),
    ("30_60m", 1800, 3600),
    ("1_2h", 3600, 7200),
    ("2_6h", 7200, 21600),
    ("6_24h", 21600, 86400),
    ("1_7d", 86400, 604800),
    ("7_30d", 604800, 2592000),
    ("gt_30d", 2592000, None),
]

CUM_THRESHOLDS = [
    ("30m", 1800),
    ("1h", 3600),
    ("2h", 7200),
    ("6h", 21600),
    ("24h", 86400),
]

PASS_FIELDS = [
    ("pass_30m_min5", 1800, 5),
    ("pass_30m_min8", 1800, 8),
    ("pass_1h_min5", 3600, 5),
    ("pass_1h_min8", 3600, 8),
    ("pass_2h_min8", 7200, 8),
    ("pass_2h_min10", 7200, 10),
]

OBS_THRESHOLD_COMBOS = [
    (1800, 5),
    (1800, 8),
    (3600, 5),
    (3600, 8),
    (7200, 8),
    (7200, 10),
    (21600, 10),
    (86400, 10),
]

SIZE_BUCKETS = [
    ("0-10", 0, 10),
    ("11-50", 11, 50),
    ("51-100", 51, 100),
    ("101-500", 101, 500),
    ("501-1000", 501, 1000),
    ("1001-5000", 1001, 5000),
    ("5001+", 5001, None),
]

CASCADE_FIELDS = (
    ["cascade_idx", "total_retweets", "duration"]
    + [f"bin_{name}" for name, _start, _end in TIME_BINS]
    + [f"cum_{name}" for name, _threshold in CUM_THRESHOLDS]
    + [f"ratio_{name}" for name, _threshold in CUM_THRESHOLDS]
    + [name for name, _threshold, _min_retweets in PASS_FIELDS]
)

BIN_SUMMARY_FIELDS = [
    "time_bin",
    "retweet_count",
    "retweet_ratio",
    "cascade_count_with_retweet",
    "cascade_ratio_with_retweet",
]

SIZE_BUCKET_FIELDS = [
    "size_bucket",
    "num_cascades",
    "total_retweets",
    "time_bin",
    "retweet_count",
    "retweet_ratio",
    "cascade_count_with_retweet",
    "cascade_ratio_with_retweet",
]

OBS_SUMMARY_FIELDS = [
    "obs_seconds",
    "min_retweets",
    "num_pass_cascades",
    "pass_ratio",
    "avg_observed_retweets",
    "median_observed_retweets",
    "avg_observed_duration",
    "median_observed_duration",
]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield obj


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def load_roots(path: Path) -> Dict[int, int]:
    roots: Dict[int, int] = {}
    for row in iter_jsonl(path):
        cascade_idx = as_int(row.get("cascade_idx"))
        root_time = as_int(row.get("root_time_epoch", row.get("root_time")))
        if cascade_idx <= 0:
            continue
        if cascade_idx in roots:
            raise ValueError(f"duplicated cascade_idx in {path}: {cascade_idx}")
        roots[cascade_idx] = root_time
    return roots


def bucket_for_count(count: int) -> str:
    for label, left, right in SIZE_BUCKETS:
        if count >= left and (right is None or count <= right):
            return label
    return "unknown"


def bin_for_time(relative_time: int) -> Optional[str]:
    for idx, (name, start, end) in enumerate(TIME_BINS):
        if end is None:
            if relative_time > start:
                return name
            continue
        left_ok = relative_time >= start if idx == 0 else relative_time > start
        if left_ok and relative_time <= end:
            return name
    return None


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--index", type=Path, default=Path("work/index.csv"))
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_dir = args.processed_dir
    events_path = processed_dir / "events" / "events.jsonl"
    roots_path = processed_dir / "mapping" / "cascade_root.jsonl"
    out_dir = args.out_dir

    if not events_path.exists():
        raise SystemExit(f"[FAIL] events.jsonl not found: {events_path}")
    if not roots_path.exists():
        raise SystemExit(f"[FAIL] cascade_root.jsonl not found: {roots_path}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"[FAIL] {out_dir} already exists and is not empty; pass --force")
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = load_roots(roots_path)
    rel_times_by_cascade: Dict[int, List[int]] = {cascade_idx: [] for cascade_idx in roots}
    bad_event_rows = 0
    missing_cascade_root = 0
    negative_relative_time = 0
    total_event_rows = 0

    for row in iter_jsonl(events_path):
        total_event_rows += 1
        cascade_idx = as_int(row.get("cascade_idx"))
        root_time = as_int(row.get("root_time"))
        retweet_time = as_int(row.get("retweet_time"))
        if cascade_idx <= 0 or retweet_time <= 0:
            bad_event_rows += 1
            continue
        if cascade_idx not in roots:
            missing_cascade_root += 1
            continue
        if root_time <= 0:
            root_time = roots[cascade_idx]
        if root_time <= 0:
            bad_event_rows += 1
            continue
        relative_time = retweet_time - root_time
        if relative_time < 0:
            negative_relative_time += 1
            continue
        rel_times_by_cascade[cascade_idx].append(relative_time)

    cascade_rows: List[Dict[str, Any]] = []
    global_bin_counts: Counter[str] = Counter()
    global_bin_cascades: Dict[str, set[int]] = defaultdict(set)
    observed_counts_by_threshold: Dict[int, List[int]] = {threshold: [] for _name, threshold in CUM_THRESHOLDS}
    observed_durations_by_threshold: Dict[int, List[int]] = {threshold: [] for _name, threshold in CUM_THRESHOLDS}

    for cascade_idx in sorted(roots):
        rel_times = sorted(rel_times_by_cascade.get(cascade_idx, []))
        total_retweets = len(rel_times)
        duration = rel_times[-1] if rel_times else 0
        bin_counts: Counter[str] = Counter()
        for relative_time in rel_times:
            bin_name = bin_for_time(relative_time)
            if bin_name is None:
                continue
            bin_counts[bin_name] += 1
            global_bin_counts[bin_name] += 1
            global_bin_cascades[bin_name].add(cascade_idx)

        row: Dict[str, Any] = {
            "cascade_idx": cascade_idx,
            "total_retweets": total_retweets,
            "duration": duration,
        }
        for bin_name, _start, _end in TIME_BINS:
            row[f"bin_{bin_name}"] = bin_counts[bin_name]
        for name, threshold in CUM_THRESHOLDS:
            count = sum(1 for value in rel_times if value <= threshold)
            observed_duration = max((value for value in rel_times if value <= threshold), default=0)
            row[f"cum_{name}"] = count
            row[f"ratio_{name}"] = f"{(count / total_retweets) if total_retweets else 0.0:.6f}"
            observed_counts_by_threshold[threshold].append(count)
            observed_durations_by_threshold[threshold].append(observed_duration)
        for field_name, threshold, min_retweets in PASS_FIELDS:
            row[field_name] = 1 if row[f"cum_{threshold_label(threshold)}"] >= min_retweets else 0
        cascade_rows.append(row)

    write_csv(out_dir / "cascade_retweet_time_bins.csv", CASCADE_FIELDS, cascade_rows)

    num_cascades = len(roots)
    total_nonnegative_retweets = sum(row["total_retweets"] for row in cascade_rows)
    bin_summary_rows = []
    for bin_name, _start, _end in TIME_BINS:
        retweet_count = global_bin_counts[bin_name]
        cascade_count = len(global_bin_cascades[bin_name])
        bin_summary_rows.append(
            {
                "time_bin": bin_name,
                "retweet_count": retweet_count,
                "retweet_ratio": f"{(retweet_count / total_nonnegative_retweets) if total_nonnegative_retweets else 0.0:.6f}",
                "cascade_count_with_retweet": cascade_count,
                "cascade_ratio_with_retweet": f"{(cascade_count / num_cascades) if num_cascades else 0.0:.6f}",
            }
        )
    write_csv(out_dir / "retweet_time_bin_summary.csv", BIN_SUMMARY_FIELDS, bin_summary_rows)

    by_bucket_rel_times: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
    for cascade_idx, rel_times in rel_times_by_cascade.items():
        bucket = bucket_for_count(len(rel_times))
        by_bucket_rel_times[bucket][cascade_idx] = rel_times

    size_bucket_rows = []
    for bucket_label, _left, _right in SIZE_BUCKETS:
        cascades = by_bucket_rel_times.get(bucket_label, {})
        bucket_total = sum(len(values) for values in cascades.values())
        bucket_bin_counts: Counter[str] = Counter()
        bucket_bin_cascades: Dict[str, set[int]] = defaultdict(set)
        for cascade_idx, rel_times in cascades.items():
            for relative_time in rel_times:
                bin_name = bin_for_time(relative_time)
                if bin_name is None:
                    continue
                bucket_bin_counts[bin_name] += 1
                bucket_bin_cascades[bin_name].add(cascade_idx)
        for bin_name, _start, _end in TIME_BINS:
            retweet_count = bucket_bin_counts[bin_name]
            cascade_count = len(bucket_bin_cascades[bin_name])
            size_bucket_rows.append(
                {
                    "size_bucket": bucket_label,
                    "num_cascades": len(cascades),
                    "total_retweets": bucket_total,
                    "time_bin": bin_name,
                    "retweet_count": retweet_count,
                    "retweet_ratio": f"{(retweet_count / bucket_total) if bucket_total else 0.0:.6f}",
                    "cascade_count_with_retweet": cascade_count,
                    "cascade_ratio_with_retweet": f"{(cascade_count / len(cascades)) if cascades else 0.0:.6f}",
                }
            )
    write_csv(out_dir / "retweet_time_bin_by_size_bucket.csv", SIZE_BUCKET_FIELDS, size_bucket_rows)

    obs_summary_rows = []
    for obs_seconds, min_retweets in OBS_THRESHOLD_COMBOS:
        observed_counts = [sum(1 for value in rel_times if value <= obs_seconds) for rel_times in rel_times_by_cascade.values()]
        observed_durations = [
            max((value for value in rel_times if value <= obs_seconds), default=0)
            for rel_times in rel_times_by_cascade.values()
        ]
        num_pass = sum(1 for count in observed_counts if count >= min_retweets)
        obs_summary_rows.append(
            {
                "obs_seconds": obs_seconds,
                "min_retweets": min_retweets,
                "num_pass_cascades": num_pass,
                "pass_ratio": f"{(num_pass / num_cascades) if num_cascades else 0.0:.6f}",
                "avg_observed_retweets": f"{mean(observed_counts):.6f}",
                "median_observed_retweets": f"{median(observed_counts):.6f}",
                "avg_observed_duration": f"{mean(observed_durations):.6f}",
                "median_observed_duration": f"{median(observed_durations):.6f}",
            }
        )
    write_csv(out_dir / "observation_threshold_summary.csv", OBS_SUMMARY_FIELDS, obs_summary_rows)

    diagnostics = {
        "run_id": processed_dir.parent.name,
        "processed_dir": str(processed_dir),
        "events_path": str(events_path),
        "cascade_root_path": str(roots_path),
        "index_path": str(args.index) if args.index.exists() else "",
        "out_dir": str(out_dir),
        "counts": {
            "num_cascades": num_cascades,
            "total_event_rows": total_event_rows,
            "total_nonnegative_retweets": total_nonnegative_retweets,
            "bad_event_rows": bad_event_rows,
            "missing_cascade_root": missing_cascade_root,
            "negative_relative_time": negative_relative_time,
        },
        "time_bins": [
            {"name": name, "start": start, "end": end}
            for name, start, end in TIME_BINS
        ],
        "cum_thresholds": [
            {"name": name, "seconds": threshold}
            for name, threshold in CUM_THRESHOLDS
        ],
        "checks": {
            "index_only": True,
            "all_cascade_idx_positive": all(cascade_idx > 0 for cascade_idx in roots),
            "negative_relative_time_excluded": True,
        },
    }
    with (out_dir / "retweet_time_distribution_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    max_cascade_idx = 0
    max_cascade_retweets = 0
    nonzero_cascades = 0
    for row in cascade_rows:
        total_retweets = int(row["total_retweets"])
        if total_retweets > 0:
            nonzero_cascades += 1
        if total_retweets > max_cascade_retweets:
            max_cascade_idx = int(row["cascade_idx"])
            max_cascade_retweets = total_retweets

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Retweet Time Distribution\n\n")
        f.write(f"本目录统计 `{diagnostics['run_id']}` 中每个级联的转发时间分布。\n\n")
        f.write("核心明细文件：\n\n")
        f.write("```text\ncascade_retweet_time_bins.csv\n```\n\n")
        f.write("该文件是“每个级联一行”的明细表：\n\n")
        f.write(f"- 行数：{num_cascades}\n")
        f.write(f"- 覆盖级联数：{num_cascades}\n")
        f.write(f"- 有转发的级联数：{nonzero_cascades}\n")
        f.write(f"- 总转发数：{total_nonnegative_retweets}\n")
        f.write(
            f"- 转发数最多的级联：cascade_idx = {max_cascade_idx}，"
            f"total_retweets = {max_cascade_retweets}\n\n"
        )

        f.write("## cascade_retweet_time_bins.csv 字段\n\n")
        f.write("| 字段 | 说明 |\n")
        f.write("|---|---|\n")
        field_docs = [
            ("cascade_idx", "级联编号，index-only，不是真实 cascade ID。"),
            ("total_retweets", "该级联总转发数，只统计 relative_time >= 0 的转发。"),
            ("duration", "该级联最后一次转发相对根帖的时间差，单位秒。"),
            ("bin_0_5m", "0 到 5 分钟内的转发数。"),
            ("bin_5_10m", "5 到 10 分钟内的转发数。"),
            ("bin_10_15m", "10 到 15 分钟内的转发数。"),
            ("bin_15_20m", "15 到 20 分钟内的转发数。"),
            ("bin_20_25m", "20 到 25 分钟内的转发数。"),
            ("bin_25_30m", "25 到 30 分钟内的转发数。"),
            ("bin_30_60m", "30 到 60 分钟内的转发数。"),
            ("bin_1_2h", "1 到 2 小时内的转发数。"),
            ("bin_2_6h", "2 到 6 小时内的转发数。"),
            ("bin_6_24h", "6 到 24 小时内的转发数。"),
            ("bin_1_7d", "1 到 7 天内的转发数。"),
            ("bin_7_30d", "7 到 30 天内的转发数。"),
            ("bin_gt_30d", "30 天之后的转发数。"),
            ("cum_30m", "30 分钟内累计转发数。"),
            ("cum_1h", "1 小时内累计转发数。"),
            ("cum_2h", "2 小时内累计转发数。"),
            ("cum_6h", "6 小时内累计转发数。"),
            ("cum_24h", "24 小时内累计转发数。"),
            ("ratio_30m", "30 分钟内累计转发数占该级联总转发数的比例。"),
            ("ratio_1h", "1 小时内累计转发数占该级联总转发数的比例。"),
            ("ratio_2h", "2 小时内累计转发数占该级联总转发数的比例。"),
            ("ratio_6h", "6 小时内累计转发数占该级联总转发数的比例。"),
            ("ratio_24h", "24 小时内累计转发数占该级联总转发数的比例。"),
            ("pass_30m_min5", "30 分钟内累计转发数是否达到 5。"),
            ("pass_30m_min8", "30 分钟内累计转发数是否达到 8。"),
            ("pass_1h_min5", "1 小时内累计转发数是否达到 5。"),
            ("pass_1h_min8", "1 小时内累计转发数是否达到 8。"),
            ("pass_2h_min8", "2 小时内累计转发数是否达到 8。"),
            ("pass_2h_min10", "2 小时内累计转发数是否达到 10。"),
        ]
        for field_name, desc in field_docs:
            f.write(f"| `{field_name}` | {desc} |\n")
        f.write("\n")

        f.write("## 其他汇总文件\n\n")
        f.write("| 文件 | 说明 |\n")
        f.write("|---|---|\n")
        f.write("| `retweet_time_bin_summary.csv` | 全局转发时间区间分布。 |\n")
        f.write("| `retweet_time_bin_by_size_bucket.csv` | 按级联规模桶统计的时间区间分布。 |\n")
        f.write("| `observation_threshold_summary.csv` | 不同 obs 和 min_retweets 组合下的通过级联数。 |\n")
        f.write("| `retweet_time_distribution_diagnostics.json` | 诊断信息，包括事件总数、负时间事件数和 index-only 检查。 |\n\n")

        f.write("## 读取示例\n\n")
        f.write("查看某个级联，例如 `cascade_idx = 28`：\n\n")
        f.write("```powershell\n")
        f.write(
            "python -c \"import csv; p='work/runs/run_0001/time_distribution/"
            "cascade_retweet_time_bins.csv'; [print(r) for r in "
            "csv.DictReader(open(p, encoding='utf-8-sig')) if r['cascade_idx']=='28']\"\n"
        )
        f.write("```\n\n")
        f.write("筛选 30 分钟内达到 8 次转发的级联：\n\n")
        f.write("```powershell\n")
        f.write(
            "python -c \"import csv; p='work/runs/run_0001/time_distribution/"
            "cascade_retweet_time_bins.csv'; print([r['cascade_idx'] for r in "
            "csv.DictReader(open(p, encoding='utf-8-sig')) if r['pass_30m_min8']=='1'][:20])\"\n"
        )
        f.write("```\n")

    for name in [
        "cascade_retweet_time_bins.csv",
        "retweet_time_bin_summary.csv",
        "retweet_time_bin_by_size_bucket.csv",
        "observation_threshold_summary.csv",
        "retweet_time_distribution_diagnostics.json",
        "README.md",
    ]:
        path = out_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[OK] {name}")
        else:
            raise SystemExit(f"[FAIL] {name} missing or empty")
    return 0


def threshold_label(threshold: int) -> str:
    labels = {
        1800: "30m",
        3600: "1h",
        7200: "2h",
        21600: "6h",
        86400: "24h",
    }
    return labels[threshold]


if __name__ == "__main__":
    raise SystemExit(main())
