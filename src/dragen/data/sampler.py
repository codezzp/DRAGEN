#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 index.csv 中随机采样级联，支持两种模式：
1. 完全随机采样（random）：从所有级联中随机选择 N 个
2. 分层随机采样（stratified）：按 retweet_count_in_dataset 区间比例随机采样，确保分布均匀

用法:
  # 完全随机采样 110000 个级联
  python scripts/sample_stratified_by_retweets.py --index work/index.csv --n 110000 --mode random --out work/sample_random_110k.txt

  # 分层随机采样（按区间比例）
  python scripts/sample_stratified_by_retweets.py --index work/index.csv --n 11000 --mode stratified --out work/sample_stratified_11k.txt

  # 自定义区间进行分层采样
  python scripts/sample_stratified_by_retweets.py --index work/index.csv --n 11000 --mode stratified --bins "0:10,11:50,51:100,101:inf" --out work/sample_stratified_11k.txt

  # 输出 cascade_id 而不是 cascade_idx（需要 cascade_id_map）
  python scripts/sample_stratified_by_retweets.py --index work/index.csv --n 11000 --cascade_id_map work/mapping/cascade_id_map.tsv --output_mode id --out work/sample_stratified_11k_ids.txt
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def parse_bins(bins_str: str) -> List[Tuple[int, int | None]]:
    """解析区间字符串"""
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
    for i in range(len(bins) - 1):
        if bins[i][1] is None:
            raise ValueError(f"只有最后一个区间可以是 inf，但第 {i+1} 个区间是 {bins[i]}")
        if bins[i][1] >= bins[i + 1][0]:
            raise ValueError(f"区间重叠或顺序错误: {bins[i]} 和 {bins[i+1]}")
    return bins


def get_default_bins() -> List[Tuple[int, int | None]]:
    """默认区间"""
    return [
        (0, 10),
        (11, 50),
        (51, 100),
        (101, 500),
        (501, 1000),
        (1001, 5000),
        (5001, None),
    ]


def get_bin_label(bin_range: Tuple[int, int | None]) -> str:
    """生成区间标签"""
    left, right = bin_range
    if right is None:
        return f"{left}+"
    if left == right:
        return str(left)
    return f"{left}-{right}"


def load_cascade_id_map(cascade_id_map_path: Path) -> dict[int, str]:
    """加载 cascade_id_map.tsv: cascade_idx -> cascade_id"""
    idx2id = {}
    with open(cascade_id_map_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0].lstrip("\ufeff"))
                cid = parts[1].lstrip("\ufeff")
                idx2id[idx] = cid
            except ValueError:
                continue
    return idx2id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index.csv 路径")
    ap.add_argument("--n", type=int, required=True, help="要采样的级联总数")
    ap.add_argument(
        "--mode",
        type=str,
        choices=["random", "stratified"],
        default="stratified",
        help="采样模式：random=完全随机；stratified=按 retweet_count_in_dataset 区间分层随机（默认，确保分布与源数据一致）",
    )
    ap.add_argument(
        "--bins",
        default=None,
        help='区间定义（仅用于 stratified 模式），例如 "0:10,11:50,51:100,101:inf"（默认使用预设区间）',
    )
    ap.add_argument(
        "--output_mode",
        type=str,
        choices=["idx", "id"],
        default="idx",
        help="输出格式：idx=输出 cascade_idx（用于 build run --cascade_list_mode=idx）；id=输出 cascade_id（用于 build run --cascade_list_mode=id）",
    )
    ap.add_argument(
        "--cascade_id_map",
        default=None,
        help="cascade_id_map.tsv 路径（当 output_mode=id 时必需）",
    )
    ap.add_argument("--out", required=True, help="输出级联列表文件路径（每行一个 cascade_idx 或 cascade_id）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现）")
    ap.add_argument("--show_stats", action="store_true", help="显示采样统计信息")
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"文件不存在: {index_path}")

    n_total = int(args.n)
    if n_total <= 0:
        raise SystemExit(f"--n 必须为正整数，但得到: {n_total}")

    # 设置随机种子
    random.seed(int(args.seed))

    # 如果需要输出 cascade_id，加载映射
    idx2id = {}
    if args.output_mode == "id":
        if not args.cascade_id_map:
            raise SystemExit("--output_mode=id 时需要提供 --cascade_id_map")
        cascade_id_map_path = Path(args.cascade_id_map)
        if not cascade_id_map_path.exists():
            raise SystemExit(f"文件不存在: {cascade_id_map_path}")
        idx2id = load_cascade_id_map(cascade_id_map_path)

    # 读取 CSV
    all_cascades: List[str] = []  # 所有有效的 cascade_idx
    cascades_by_bin: dict[int, List[Tuple[str, int]]] = defaultdict(list)  # bin_idx -> [(cascade_idx, retweet_count), ...]
    total_available = 0
    invalid = 0

    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "cascade_idx" not in reader.fieldnames:
            raise SystemExit(f"CSV 缺少列 'cascade_idx'，现有列: {reader.fieldnames}")

        # 如果是分层模式，需要 retweet_count_in_dataset
        if args.mode == "stratified":
            if "retweet_count_in_dataset" not in reader.fieldnames:
                raise SystemExit(f"CSV 缺少列 'retweet_count_in_dataset'（stratified 模式需要），现有列: {reader.fieldnames}")

        for row in reader:
            total_available += 1
            try:
                cascade_idx = str(row["cascade_idx"]).strip()
                if not cascade_idx:
                    invalid += 1
                    continue
            except (ValueError, KeyError):
                invalid += 1
                continue

            all_cascades.append(cascade_idx)

            # 如果是分层模式，按区间分组
            if args.mode == "stratified":
                try:
                    count = int(row["retweet_count_in_dataset"])
                except (ValueError, KeyError):
                    invalid += 1
                    continue

                # 解析区间（如果还没解析）
                if not hasattr(args, "_bins_parsed"):
                    if args.bins:
                        bins = parse_bins(args.bins)
                    else:
                        bins = get_default_bins()
                    args._bins_parsed = bins
                else:
                    bins = args._bins_parsed

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
                    cascades_by_bin[bin_idx].append((cascade_idx, count))

    # 根据模式进行采样
    sampled: List[str] = []
    stats = {}

    if args.mode == "random":
        # 完全随机采样
        total_valid = len(all_cascades)
        if total_valid < n_total:
            print(f"警告: 可用级联数({total_valid}) < 目标采样数({n_total})，将采样所有可用级联", file=__import__("sys").stderr)
            sampled = all_cascades.copy()
        else:
            sampled = random.sample(all_cascades, n_total)
        stats = {"total_available": total_valid, "sampled": len(sampled)}

    else:  # stratified
        # 解析区间（如果还没解析）
        if not hasattr(args, "_bins_parsed"):
            if args.bins:
                bins = parse_bins(args.bins)
            else:
                bins = get_default_bins()
            args._bins_parsed = bins
        else:
            bins = args._bins_parsed

        # 计算每个区间应该采样多少（按比例）
        bin_sizes = {i: len(cascades_by_bin[i]) for i in range(len(bins))}
        total_in_bins = sum(bin_sizes.values())

        if total_in_bins < n_total:
            print(f"警告: 可用级联数({total_in_bins}) < 目标采样数({n_total})，将采样所有可用级联", file=__import__("sys").stderr)
            n_total = total_in_bins

        # 按比例分配（先按比例，再处理余数）
        bin_targets: dict[int, int] = {}
        allocated = 0
        for i in range(len(bins)):
            if total_in_bins == 0:
                bin_targets[i] = 0
            else:
                target = int(bin_sizes[i] / total_in_bins * n_total)
                # 不能超过该区间的实际数量
                target = min(target, bin_sizes[i])
                bin_targets[i] = target
                allocated += target

        # 处理余数：按剩余可用数量从大到小分配
        remainder = n_total - allocated
        if remainder > 0:
            # 按剩余可用数量排序（降序）
            candidates = sorted(
                [(i, bin_sizes[i] - bin_targets[i]) for i in range(len(bins))],
                key=lambda x: x[1],
                reverse=True,
            )
            for i, available in candidates:
                if remainder <= 0:
                    break
                if available > 0:
                    add = min(remainder, available)
                    bin_targets[i] += add
                    remainder -= add

        # 从每个区间采样
        for i in range(len(bins)):
            candidates = cascades_by_bin[i]
            target = bin_targets[i]
            if target == 0:
                stats[i] = {"sampled": 0, "available": len(candidates)}
                continue

            # 如果目标数 >= 可用数，全部取
            if target >= len(candidates):
                selected = [cidx for cidx, _ in candidates]
            else:
                # 随机采样
                selected_items = random.sample(candidates, target)
                selected = [cidx for cidx, _ in selected_items]

            sampled.extend(selected)
            stats[i] = {"sampled": len(selected), "available": len(candidates)}

    # 打乱最终顺序（避免按区间连续）
    random.shuffle(sampled)

    # 输出
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        if args.output_mode == "id":
            # 转换为 cascade_id
            missing = 0
            for cidx_str in sampled:
                try:
                    cidx = int(cidx_str)
                    if cidx in idx2id:
                        f.write(f"{idx2id[cidx]}\n")
                    else:
                        missing += 1
                except ValueError:
                    missing += 1
            if missing > 0:
                print(f"警告: {missing} 个 cascade_idx 在 cascade_id_map 中未找到，已跳过", file=__import__("sys").stderr)
        else:
            # 直接输出 cascade_idx
            for cidx in sampled:
                f.write(f"{cidx}\n")

    print(f"OK: 已采样 {len(sampled)} 个级联，保存到 {out_path}")

    # 显示统计信息
    if args.show_stats:
        if args.mode == "random":
            print("\n## 采样统计（完全随机）")
            print(f"可用级联数: {stats['total_available']}")
            print(f"采样级联数: {stats['sampled']}")
            print(f"采样率: {stats['sampled'] / stats['total_available'] * 100:.2f}%")
            print(f"无效/缺失: {invalid}")
        else:  # stratified
            print("\n## 采样统计（分层随机）")
            print("区间\t可用数\t采样数\t采样率")
            print("-" * 40)
            bins = args._bins_parsed
            total_in_bins = sum(s.get("available", 0) for s in stats.values() if isinstance(s, dict))
            for i, bin_range in enumerate(bins):
                label = get_bin_label(bin_range)
                s = stats.get(i, {"sampled": 0, "available": 0})
                available = s["available"]
                sampled_count = s["sampled"]
                rate = (sampled_count / available * 100) if available > 0 else 0.0
                print(f"{label}\t{available}\t{sampled_count}\t{rate:.2f}%")
            print(f"\n总计: 可用={total_in_bins}, 采样={len(sampled)}, 无效/缺失={invalid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
