#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build dev cascade lists from per-cascade retweet time bins."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "work" / "runs" / "run_0001" / "time_distribution" / "cascade_retweet_time_bins.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "work" / "runs" / "run_0001" / "time_distribution" / "dev_cascade_lists"

LIST_CONFIGS = [
    {
        "list_name": "dev_A_30m_min8_cascades",
        "file_name": "dev_A_30m_min8_cascades.txt",
        "obs_seconds": 1800,
        "min_retweets": 8,
        "pass_field": "pass_30m_min8",
        "cum_field": "cum_30m",
        "ratio_field": "ratio_30m",
    },
    {
        "list_name": "dev_B_1h_min8_cascades",
        "file_name": "dev_B_1h_min8_cascades.txt",
        "obs_seconds": 3600,
        "min_retweets": 8,
        "pass_field": "pass_1h_min8",
        "cum_field": "cum_1h",
        "ratio_field": "ratio_1h",
    },
    {
        "list_name": "dev_C_2h_min10_cascades",
        "file_name": "dev_C_2h_min10_cascades.txt",
        "obs_seconds": 7200,
        "min_retweets": 10,
        "pass_field": "pass_2h_min10",
        "cum_field": "cum_2h",
        "ratio_field": "ratio_2h",
    },
]

SUMMARY_FIELDS = [
    "list_name",
    "obs_seconds",
    "min_retweets",
    "num_cascades",
    "avg_total_retweets",
    "median_total_retweets",
    "avg_cum_obs",
    "median_cum_obs",
    "avg_ratio_obs",
    "median_ratio_obs",
]


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_list(path: Path, cascade_indices: Iterable[int]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for cascade_idx in cascade_indices:
            f.write(f"{cascade_idx}\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    out_dir = args.out_dir

    if not input_path.exists():
        raise SystemExit(f"[FAIL] input not found: {input_path}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"[FAIL] {out_dir} already exists and is not empty; pass --force")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    summary_rows: List[Dict[str, Any]] = []

    for config in LIST_CONFIGS:
        selected = [
            row for row in rows
            if row.get(config["pass_field"], "0") == "1"
        ]
        selected.sort(key=lambda row: as_int(row.get("cascade_idx")))
        cascade_indices = [as_int(row["cascade_idx"]) for row in selected]
        list_count = write_list(out_dir / config["file_name"], cascade_indices)

        total_retweets = [float(as_int(row.get("total_retweets"))) for row in selected]
        cum_obs = [float(as_int(row.get(config["cum_field"]))) for row in selected]
        ratio_obs = [as_float(row.get(config["ratio_field"])) for row in selected]
        summary_rows.append(
            {
                "list_name": config["list_name"],
                "obs_seconds": config["obs_seconds"],
                "min_retweets": config["min_retweets"],
                "num_cascades": list_count,
                "avg_total_retweets": f"{mean(total_retweets):.6f}",
                "median_total_retweets": f"{median(total_retweets):.6f}",
                "avg_cum_obs": f"{mean(cum_obs):.6f}",
                "median_cum_obs": f"{median(cum_obs):.6f}",
                "avg_ratio_obs": f"{mean(ratio_obs):.6f}",
                "median_ratio_obs": f"{median(ratio_obs):.6f}",
            }
        )
        print(f"[OK] {config['file_name']}: {list_count}")

    summary_path = out_dir / "dev_cascade_list_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] {summary_path}")

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Dev Cascade Lists\n\n")
        f.write("这些列表仅用于代码验证，不作为最终论文实验采样分布。\n\n")
        f.write(f"输入文件：`{input_path}`\n\n")
        f.write("输出列表只包含 `cascade_idx`，一行一个，保持 index-only。\n\n")
        f.write("| 文件 | 条件 |\n")
        f.write("|---|---|\n")
        f.write("| `dev_A_30m_min8_cascades.txt` | `pass_30m_min8 = 1` |\n")
        f.write("| `dev_B_1h_min8_cascades.txt` | `pass_1h_min8 = 1` |\n")
        f.write("| `dev_C_2h_min10_cascades.txt` | `pass_2h_min10 = 1` |\n")
        f.write("\n汇总文件：`dev_cascade_list_summary.csv`\n")
    print(f"[OK] {out_dir / 'README.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
