"""Export fair event-level result tables from artifact directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from dragen.config import load_config
from dragen.evaluation.metrics import binary_metrics


MAIN_FIELDS = [
    "model",
    "input_variant",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "macro_f1",
    "auc",
    "ap",
    "mcc",
    "brier",
    "ece",
]

RISK_FIELDS = [
    "model",
    "input_variant",
    "precision_at_100",
    "precision_at_500",
    "recall_at_500",
    "precision_at_1pct",
    "recall_at_1pct",
    "precision_at_5pct",
    "recall_at_5pct",
]

ABLATION_FIELDS = [
    "model",
    "input_variant",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "ap",
    "mcc",
    "delta_auc",
    "delta_ap",
    "delta_f1",
    "delta_mcc",
]


def export_main_results(run_dirs: Iterable[Path], out_path: Path) -> None:
    rows = [load_event_metrics(run_dir) for run_dir in run_dirs]
    rows = [row for row in rows if row]
    write_csv(out_path, [select_fields(row, MAIN_FIELDS) for row in rows], MAIN_FIELDS)


def export_risk_retrieval_results(run_dirs: Iterable[Path], out_path: Path) -> None:
    rows = [load_event_metrics(run_dir) for run_dir in run_dirs]
    rows = [row for row in rows if row]
    write_csv(out_path, [select_fields(row, RISK_FIELDS) for row in rows], RISK_FIELDS)


def export_ablation_results(run_dirs: Iterable[Path], out_path: Path, full_run: Path | None = None) -> None:
    rows = [load_event_metrics(run_dir) for run_dir in run_dirs]
    rows = [row for row in rows if row]
    baseline = load_event_metrics(full_run) if full_run else find_full_row(rows)
    for row in rows:
        for key in ["auc", "ap", "f1", "mcc"]:
            row[f"delta_{key}"] = float(baseline.get(key, 0.0)) - float(row.get(key, 0.0)) if baseline else 0.0
    write_csv(out_path, [select_fields(row, ABLATION_FIELDS) for row in rows], ABLATION_FIELDS)


def load_event_metrics(run_dir: Path) -> Dict[str, object]:
    metrics = load_metrics_json(run_dir)
    if not metrics:
        metrics = compute_metrics_from_event_predictions(run_dir)
    if not metrics:
        return {}
    metadata = load_run_metadata(run_dir)
    return {
        "model": metadata.get("model") or infer_model_name(run_dir),
        "input_variant": metadata.get("input_variant") or infer_input_variant(run_dir),
        **metrics,
    }


def load_run_metadata(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "reports" / "resolved_config.yaml"
    if not path.exists():
        return {}
    data = load_config(str(path))
    source = data.get("source_config", {}) if isinstance(data, dict) else {}
    resolved = data.get("resolved_args", {}) if isinstance(data, dict) else {}
    model = ""
    input_variant = ""
    if isinstance(source, dict):
        model_section = source.get("model", {})
        data_section = source.get("data", {})
        if isinstance(model_section, dict):
            model = str(model_section.get("name") or "")
        if isinstance(data_section, dict):
            input_variant = str(data_section.get("input_variant") or "")
    if isinstance(resolved, dict):
        input_variant = input_variant or str(resolved.get("input_variant") or "")
    return {"model": model, "input_variant": input_variant}


def load_metrics_json(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "reports" / "metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "test" in data and isinstance(data["test"], dict):
        return {k: float(v) for k, v in data["test"].items() if is_number(v)}
    return {k: float(v) for k, v in data.items() if is_number(v)}


def compute_metrics_from_event_predictions(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "predictions" / "event_predictions.csv"
    if not path.exists():
        return {}
    rows = read_csv(path)
    if not rows:
        return {}
    return binary_metrics([int(float(row["y_true"])) for row in rows], [float(row["y_prob"]) for row in rows])


def infer_model_name(run_dir: Path) -> str:
    name = run_dir.name
    mapping = {
        "dragen_full": "DRAGEN-Full",
        "cac_stat": "CAC-Stat",
        "campaign_gnn": "Campaign-GNN",
        "temporal_gnn": "Temporal-GNN",
        "ablation_no_tree": "w/o Tree",
        "ablation_no_multiscale": "w/o MultiScale",
        "ablation_no_role": "w/o Role",
        "ablation_no_memory": "w/o Memory",
        "ablation_no_global_prior": "w/o Global Prior",
        "ablation_no_adaptive_sampling": "w/o Adaptive Sampling",
        "ablation_no_gate": "w/o Gate",
        "ablation_no_uncertainty": "w/o Uncertainty",
    }
    for key, value in mapping.items():
        if key in name:
            return value
    return name


def infer_input_variant(run_dir: Path) -> str:
    name = run_dir.name
    if "no_tree" in name or "star" in name:
        return "Fixed-5m-Star"
    if "no_multiscale" in name:
        return "Fixed-5m-HybridTree"
    if "hybrid_tree" in name or "dragen_full" in name or "ablation" in name:
        return "MultiScale-HybridTree"
    return ""


def find_full_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    for row in rows:
        if row.get("model") == "DRAGEN-Full":
            return row
    return rows[0] if rows else {}


def select_fields(row: Mapping[str, object], fields: Sequence[str]) -> Dict[str, object]:
    return {field: row.get(field, "") for field in fields}


def write_csv(path: Path, rows: list[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True
