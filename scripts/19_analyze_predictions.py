from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401
from dragen.evaluation.diagnostics import (
    compute_attention_statistics,
    compute_gate_statistics,
    compute_role_statistics,
    compute_temporal_stability,
    compute_uncertainty_statistics,
)
from dragen.evaluation.metrics import binary_metrics, risk_retrieval_metrics


EVENT_FIELDS = [
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
    "precision_at_100",
    "precision_at_500",
    "recall_at_500",
    "precision_at_1pct",
    "recall_at_1pct",
    "precision_at_5pct",
    "recall_at_5pct",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze event-level and DRAGEN-Full diagnostic predictions.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir) if args.out_dir else artifact_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = artifact_dir / "predictions"
    event_path = pred_dir / "event_predictions.csv"
    event_metrics = compute_event_metrics(event_path)
    risk_metrics = {key: event_metrics[key] for key in RISK_FIELDS if key in event_metrics}
    if event_metrics:
        write_json(out_dir / "event_metrics_extended.json", {key: event_metrics[key] for key in EVENT_FIELDS if key in event_metrics})
        write_json(out_dir / "risk_retrieval_metrics.json", risk_metrics)

    temporal = compute_temporal_stability(pred_dir / "node_window_predictions.csv")
    role = compute_role_statistics(pred_dir / "role_distribution.csv")
    gate = compute_gate_statistics(pred_dir / "gate_weights.csv")
    uncertainty = compute_uncertainty_statistics(pred_dir / "uncertainty.csv", event_path)
    attention = compute_attention_statistics(pred_dir / "event_attention.csv")
    interpretability = {**role, **gate, **uncertainty, **attention}

    write_json(out_dir / "temporal_stability_metrics.json", temporal)
    write_json(out_dir / "interpretability_metrics.json", interpretability)
    write_diagnostic_summary(out_dir / "diagnostic_summary.csv", artifact_dir.name, temporal, interpretability)
    return 0


def compute_event_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        warnings.warn(f"Missing event prediction file, skip event metrics: {path}")
        return {}
    rows = read_csv(path)
    if not rows:
        warnings.warn(f"Empty event prediction file, skip event metrics: {path}")
        return {}
    y_true = [int(float(row["y_true"])) for row in rows]
    y_prob = [float(row["y_prob"]) for row in rows]
    metrics = binary_metrics(y_true, y_prob)
    metrics.update(risk_retrieval_metrics(y_true, y_prob))
    return metrics


def write_diagnostic_summary(path: Path, model: str, temporal: dict[str, object], interpretability: dict[str, object]) -> None:
    row = {"model": model, **temporal}
    for key, value in interpretability.items():
        if isinstance(value, dict):
            continue
        row[key] = value
    if len(row) == 1:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
