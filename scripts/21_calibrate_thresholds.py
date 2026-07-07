from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

import _bootstrap  # noqa: F401
from dragen.evaluation.metrics import binary_metrics


DEFAULT_RUNS = [
    "dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0",
    "dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1",
    "dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2",
]

METRIC_FIELDS = [
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
    "precision_at_100",
    "precision_at_500",
    "recall_at_500",
    "precision_at_1pct",
    "recall_at_1pct",
    "precision_at_5pct",
    "recall_at_5pct",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds on valid predictions and apply them to test predictions.")
    parser.add_argument("--artifact-root", type=Path, default=Path("work/artifacts/_artifacts"))
    parser.add_argument("--run", action="append", dest="runs", help="Run directory name under artifact-root. Repeatable.")
    parser.add_argument("--out-dir", type=Path, default=Path("work/artifacts/_analysis/run_0002_threshold_calibration"))
    parser.add_argument("--steps", type=int, default=1000, help="Number of grid thresholds between 0 and 1.")
    args = parser.parse_args()

    runs = args.runs or DEFAULT_RUNS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    threshold_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for run_name in runs:
        run_dir = args.artifact_root / run_name
        valid_y, valid_p = read_predictions(run_dir / "predictions" / "valid_event_predictions.csv")
        test_y, test_p = read_predictions(run_dir / "predictions" / "test_event_predictions.csv")
        strategies = {
            "default_0.5": 0.5,
            "valid_best_f1": best_threshold(valid_y, valid_p, "f1", args.steps),
            "valid_best_mcc": best_threshold(valid_y, valid_p, "mcc", args.steps),
        }
        for strategy, threshold in strategies.items():
            valid_metrics = binary_metrics(valid_y, valid_p, threshold=threshold)
            test_metrics = binary_metrics(test_y, test_p, threshold=threshold)
            threshold_rows.append(
                {
                    "run": run_name,
                    "strategy": strategy,
                    "threshold": f"{threshold:.6f}",
                    "valid_f1": f"{valid_metrics['f1']:.6f}",
                    "valid_mcc": f"{valid_metrics['mcc']:.6f}",
                    "valid_precision": f"{valid_metrics['precision']:.6f}",
                    "valid_recall": f"{valid_metrics['recall']:.6f}",
                }
            )
            row = {"run": run_name, "strategy": strategy, "threshold": f"{threshold:.6f}"}
            row.update({field: f"{test_metrics[field]:.6f}" for field in METRIC_FIELDS if field in test_metrics})
            metric_rows.append(row)

    summary_rows = summarize(metric_rows)
    write_csv(args.out_dir / "threshold_by_seed.csv", threshold_rows)
    write_csv(args.out_dir / "threshold_test_metrics.csv", metric_rows)
    write_csv(args.out_dir / "threshold_summary_mean_std.csv", summary_rows)
    write_json(args.out_dir / "threshold_summary_mean_std.json", summary_json(summary_rows))
    write_markdown(args.out_dir / "threshold_calibration_summary.md", summary_rows, threshold_rows)
    print(f"wrote threshold calibration outputs to {args.out_dir}")
    return 0


def read_predictions(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    y: list[int] = []
    p: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            y.append(int(float(row["y_true"])))
            p.append(min(max(float(row["y_prob"]), 0.0), 1.0))
    if not y:
        raise ValueError(f"empty prediction file: {path}")
    return y, p


def best_threshold(y: list[int], p: list[float], metric: str, steps: int) -> float:
    candidates = sorted(set(p))
    if len(candidates) > steps + 1:
        candidates = [i / steps for i in range(steps + 1)]
    best_t = 0.5
    best_value = -math.inf
    best_precision = -math.inf
    for threshold in candidates:
        metrics = binary_metrics(y, p, threshold=threshold)
        value = metrics[metric]
        precision = metrics["precision"]
        if value > best_value or (value == best_value and precision > best_precision):
            best_t = float(threshold)
            best_value = value
            best_precision = precision
    return best_t


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    strategies = sorted({str(row["strategy"]) for row in rows})
    for strategy in strategies:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        summary: dict[str, object] = {"strategy": strategy, "n": len(strategy_rows)}
        for field in ["threshold", *METRIC_FIELDS]:
            vals = [float(row[field]) for row in strategy_rows if field in row]
            if not vals:
                continue
            summary[f"{field}_mean"] = f"{statistics.mean(vals):.6f}"
            summary[f"{field}_std"] = f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000"
        out.append(summary)
    return out


def summary_json(rows: list[dict[str, object]]) -> dict[str, object]:
    return {str(row["strategy"]): {k: v for k, v in row.items() if k != "strategy"} for row in rows}


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_markdown(path: Path, summary_rows: list[dict[str, object]], threshold_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Threshold Calibration Summary",
        "",
        "Thresholds are selected on validation predictions and applied to test predictions.",
        "",
        "## Test Mean +/- Std",
        "",
        "| Strategy | Threshold | Acc | Precision | Recall | F1 | AUC | AP | MCC | P@100 | P@500 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {strategy} | {threshold_mean} +/- {threshold_std} | {accuracy_mean} +/- {accuracy_std} | "
            "{precision_mean} +/- {precision_std} | {recall_mean} +/- {recall_std} | {f1_mean} +/- {f1_std} | "
            "{auc_mean} +/- {auc_std} | {ap_mean} +/- {ap_std} | {mcc_mean} +/- {mcc_std} | "
            "{precision_at_100_mean} +/- {precision_at_100_std} | {precision_at_500_mean} +/- {precision_at_500_std} |".format(**row)
        )
    lines.extend(["", "## Selected Thresholds", "", "| Run | Strategy | Threshold | Valid F1 | Valid MCC | Valid Precision | Valid Recall |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in threshold_rows:
        lines.append(
            f"| {row['run']} | {row['strategy']} | {row['threshold']} | {row['valid_f1']} | {row['valid_mcc']} | {row['valid_precision']} | {row['valid_recall']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
