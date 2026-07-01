#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 index.csv 中 retweet_count_in_dataset 的区间分布

用法:
  python scripts/stats_retweet_intervals.py --index work/index.csv
  python scripts/stats_retweet_intervals.py --index work/index.csv --bins "0:10,11:50,51:100,101:500,501:1000,1001:inf" --out work/stats/retweet_intervals.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def parse_bins(bins_str: str) -> List[Tuple[int, int | None]]:
    """
    解析区间字符串，例如 "0:10,11:50,51:100,101:inf"
    返回: [(0, 10), (11, 50), (51, 100), (101, None)]
    """
    bins = []
    for part in bins_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"区间格式错误，应为 'min:max'，但得到: {part}")
        left, right = part.split(":", 1)
        left = int(left.strip())
        if right.strip().lower() == "inf":
            bins.append((left, None))
        else:
            bins.append((left, int(right.strip())))
    # 检查区间是否连续且不重叠
    for i in range(len(bins) - 1):
        if bins[i][1] is None:
            raise ValueError(f"只有最后一个区间可以是 inf，但第 {i+1} 个区间是 {bins[i]}")
        if bins[i][1] >= bins[i + 1][0]:
            raise ValueError(f"区间重叠或顺序错误: {bins[i]} 和 {bins[i+1]}")
    return bins


def get_default_bins() -> List[Tuple[int, int | None]]:
    """默认区间：0-10, 11-50, 51-100, 101-500, 501-1000, 1001-5000, 5001+"""
    return [
        (0, 10),
        (11, 50),
        (51, 100),
        (101, 500),
        (501, 1000),
        (1001, 5000),
        (5001, None),  # 5001+
    ]


def get_bin_label(bin_range: Tuple[int, int | None]) -> str:
    """生成区间标签"""
    left, right = bin_range
    if right is None:
        return f"{left}+"
    if left == right:
        return str(left)
    return f"{left}-{right}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index.csv 路径")
    ap.add_argument(
        "--bins",
        default=None,
        help='区间定义，例如 "0:10,11:50,51:100,101:inf"（默认使用预设区间）',
    )
    ap.add_argument("--out", default=None, help="输出文件路径（可选，默认打印到控制台）")
    ap.add_argument("--show_examples", type=int, default=0, help="每个区间显示 N 个示例 cascade_idx（0=不显示）")
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"文件不存在: {index_path}")

    # 解析区间
    if args.bins:
        bins = parse_bins(args.bins)
    else:
        bins = get_default_bins()

    # 读取 CSV 并统计
    counts = {i: 0 for i in range(len(bins))}
    examples = {i: [] for i in range(len(bins))} if args.show_examples > 0 else {}
    total = 0
    invalid = 0

    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "retweet_count_in_dataset" not in reader.fieldnames:
            raise SystemExit(f"CSV 缺少列 'retweet_count_in_dataset'，现有列: {reader.fieldnames}")

        for row in reader:
            total += 1
            try:
                count = int(row["retweet_count_in_dataset"])
            except (ValueError, KeyError):
                invalid += 1
                continue

            # 找到所属区间
            bin_idx = None
            for i, (left, right) in enumerate(bins):
                if count >= left:
                    if right is None or count <= right:
                        bin_idx = i
                        break
                else:
                    break

            if bin_idx is not None:
                counts[bin_idx] += 1
                if args.show_examples > 0 and len(examples[bin_idx]) < args.show_examples:
                    cascade_idx = row.get("cascade_idx", "?")
                    examples[bin_idx].append((cascade_idx, count))

    # 输出结果
    lines = []
    lines.append(f"## retweet_count_in_dataset 区间统计")
    lines.append(f"总级联数: {total}")
    if invalid > 0:
        lines.append(f"无效/缺失值: {invalid}")
    lines.append("")
    lines.append("区间\t数量\t占比")
    lines.append("-" * 40)

    for i, bin_range in enumerate(bins):
        count = counts[i]
        pct = (count / total * 100) if total > 0 else 0.0
        label = get_bin_label(bin_range)
        lines.append(f"{label}\t{count}\t{pct:.2f}%")

        if args.show_examples > 0 and i in examples and examples[i]:
            lines.append(f"  示例: {', '.join([f'cascade_idx={cidx}(count={c})' for cidx, c in examples[i]])}")

    output = "\n".join(lines)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"OK: 统计结果已保存到 {out_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
