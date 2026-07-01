#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep org-task table construction parameters for run_0001."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROCESSED_DIR = Path("work/runs/run_0001/processed")
SWEEP_DIR = Path("work/runs/run_0001/org_task_sweep")
BUILD_SCRIPT = Path("scripts/build_org_tables.py")

CONFIGS = [
    {
        "config_name": "A",
        "obs": 1800,
        "window_size": 300,
        "step": 300,
        "min_retweets": 8,
        "min_duration": 300,
    },
    {
        "config_name": "B",
        "obs": 3600,
        "window_size": 600,
        "step": 600,
        "min_retweets": 8,
        "min_duration": 600,
    },
    {
        "config_name": "C",
        "obs": 7200,
        "window_size": 600,
        "step": 600,
        "min_retweets": 10,
        "min_duration": 600,
    },
    {
        "config_name": "D",
        "obs": 1800,
        "window_size": 300,
        "step": 300,
        "min_retweets": 5,
        "min_duration": 300,
    },
    {
        "config_name": "E",
        "obs": 3600,
        "window_size": 600,
        "step": 600,
        "min_retweets": 5,
        "min_duration": 600,
    },
]

SUMMARY_FIELDS = [
    "config_name",
    "obs",
    "window_size",
    "step",
    "min_retweets",
    "min_duration",
    "num_cascades",
    "num_valid_training_cascades",
    "valid_ratio",
    "num_retweets_in_obs",
    "num_node_window_rows",
    "avg_nodes_per_window",
    "median_nodes_per_window",
    "avg_retweets_per_window",
    "median_retweets_per_window",
    "drop_min_retweets",
    "drop_min_retweets_min_duration",
    "drop_min_duration",
    "root_w1_check",
    "index_only",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        fail(f"{path} is not a JSON object")
    return obj


def count_drop_reasons(cascade_table: Path) -> Dict[str, int]:
    counts = {
        "drop_min_retweets": 0,
        "drop_min_retweets_min_duration": 0,
        "drop_min_duration": 0,
    }
    with cascade_table.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            reason = row.get("drop_reason", "")
            parts = set(part for part in reason.split(";") if part)
            has_min_retweets = "min_retweets" in parts
            has_min_duration = "min_duration" in parts
            if has_min_retweets and has_min_duration:
                counts["drop_min_retweets_min_duration"] += 1
            elif has_min_retweets:
                counts["drop_min_retweets"] += 1
            elif has_min_duration:
                counts["drop_min_duration"] += 1
    return counts


def run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config_name = str(config["config_name"])
    out_dir = SWEEP_DIR / config_name
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--processed_dir",
        str(PROCESSED_DIR),
        "--out_dir",
        str(out_dir),
        "--obs",
        str(config["obs"]),
        "--window_size",
        str(config["window_size"]),
        "--step",
        str(config["step"]),
        "--min_retweets",
        str(config["min_retweets"]),
        "--min_duration",
        str(config["min_duration"]),
        "--force",
    ]
    print(
        f"[RUN] config {config_name}: obs={config['obs']} "
        f"window_size={config['window_size']} step={config['step']}",
        flush=True,
    )
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        fail(f"config {config_name} failed with exit code {completed.returncode}")

    diagnostics = load_json(out_dir / "org_task_diagnostics.json")
    drops = count_drop_reasons(out_dir / "cascade_table.csv")
    counts = diagnostics["counts"]
    window_stats = diagnostics["window_stats"]
    checks = diagnostics["checks"]
    num_cascades = int(counts["num_cascades"])
    num_valid = int(counts["num_valid_training_cascades"])

    row: Dict[str, Any] = {
        "config_name": config_name,
        "obs": config["obs"],
        "window_size": config["window_size"],
        "step": config["step"],
        "min_retweets": config["min_retweets"],
        "min_duration": config["min_duration"],
        "num_cascades": num_cascades,
        "num_valid_training_cascades": num_valid,
        "valid_ratio": f"{(num_valid / num_cascades) if num_cascades else 0.0:.6f}",
        "num_retweets_in_obs": counts["num_retweet_posts_in_obs"],
        "num_node_window_rows": counts["num_node_window_rows"],
        "avg_nodes_per_window": f"{float(window_stats['avg_nodes_per_window']):.6f}",
        "median_nodes_per_window": f"{float(window_stats['median_nodes_per_window']):.6f}",
        "avg_retweets_per_window": f"{float(window_stats['avg_retweets_per_window']):.6f}",
        "median_retweets_per_window": f"{float(window_stats['median_retweets_per_window']):.6f}",
        "drop_min_retweets": drops["drop_min_retweets"],
        "drop_min_retweets_min_duration": drops["drop_min_retweets_min_duration"],
        "drop_min_duration": drops["drop_min_duration"],
        "root_w1_check": f"{checks['root_w1_count']} / {checks['root_w1_total']}",
        "index_only": str(bool(checks["index_only"])).lower(),
    }
    print(
        f"[OK] config {config_name}: valid={row['num_valid_training_cascades']} "
        f"ratio={row['valid_ratio']} rows={row['num_node_window_rows']}",
        flush=True,
    )
    return row


def write_summary(rows: Iterable[Dict[str, Any]]) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SWEEP_DIR / "sweep_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {summary_path}", flush=True)


def main() -> int:
    if not BUILD_SCRIPT.exists():
        fail(f"{BUILD_SCRIPT} not found")
    if not PROCESSED_DIR.exists():
        fail(f"{PROCESSED_DIR} not found")

    rows: List[Dict[str, Any]] = []
    for config in CONFIGS:
        rows.append(run_config(config))
    write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
