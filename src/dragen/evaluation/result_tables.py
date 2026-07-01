"""Export compact result tables from DRAGEN artifact directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping


def export_main_results(run_dirs: Iterable[Path], out_path: Path) -> None:
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "reports" / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        test = metrics.get("test", {})
        rows.append({"run": run_dir.name, **{f"test_{k}": v for k, v in test.items()}})
    write_csv(out_path, rows)


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
