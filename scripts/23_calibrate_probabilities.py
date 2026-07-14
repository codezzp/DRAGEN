from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Callable, Iterable, Sequence

import _bootstrap  # noqa: F401
from dragen.evaluation.metrics import binary_metrics, negative_log_likelihood


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
    "nll",
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
    parser = argparse.ArgumentParser(description="Fit probability calibrators on valid predictions and apply them to test predictions.")
    parser.add_argument("--artifact-root", type=Path, default=Path("work/artifacts/_artifacts"))
    parser.add_argument("--run", action="append", dest="runs", help="Run directory name under artifact-root. Repeatable.")
    parser.add_argument("--out-dir", type=Path, default=Path("work/artifacts/_analysis/run_0002_probability_calibration"))
    parser.add_argument("--steps", type=int, default=1000, help="Threshold grid size for calibrated valid best-F1/MCC.")
    parser.add_argument("--write-predictions", action="store_true", help="Also export calibrated per-sample prediction CSV files.")
    args = parser.parse_args()

    runs = args.runs or DEFAULT_RUNS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    for run_name in runs:
        run_dir = args.artifact_root / run_name
        valid_rows = read_prediction_rows(run_dir / "predictions" / "valid_event_predictions.csv")
        test_rows = read_prediction_rows(run_dir / "predictions" / "test_event_predictions.csv")
        valid_y = [row.y for row in valid_rows]
        valid_p = [row.p for row in valid_rows]
        test_y = [row.y for row in test_rows]
        test_p = [row.p for row in test_rows]

        calibrators = fit_calibrators(valid_y, valid_p)
        for method, fitted in calibrators.items():
            valid_cal = [fitted.apply(p) for p in valid_p]
            test_cal = [fitted.apply(p) for p in test_p]
            if args.write_predictions:
                write_calibrated_predictions(args.out_dir / "predictions" / run_name / f"valid_{method}.csv", valid_rows, valid_cal, method)
                write_calibrated_predictions(args.out_dir / "predictions" / run_name / f"test_{method}.csv", test_rows, test_cal, method)
            param_rows.append({"run": run_name, "method": method, **fitted.params})

            thresholds = {
                "default_0.5": 0.5,
                "valid_best_f1": best_threshold(valid_y, valid_cal, "f1", args.steps),
                "valid_best_mcc": best_threshold(valid_y, valid_cal, "mcc", args.steps),
            }
            for threshold_strategy, threshold in thresholds.items():
                valid_metrics = binary_metrics(valid_y, valid_cal, threshold=threshold)
                test_metrics = binary_metrics(test_y, test_cal, threshold=threshold)
                row: dict[str, object] = {
                    "run": run_name,
                    "method": method,
                    "threshold_strategy": threshold_strategy,
                    "threshold": f"{threshold:.6f}",
                    "valid_f1": f"{valid_metrics['f1']:.6f}",
                    "valid_mcc": f"{valid_metrics['mcc']:.6f}",
                    "valid_brier": f"{valid_metrics['brier']:.6f}",
                    "valid_nll": f"{valid_metrics['nll']:.6f}",
                    "valid_ece": f"{valid_metrics['ece']:.6f}",
                }
                row.update({field: f"{test_metrics[field]:.6f}" for field in METRIC_FIELDS if field in test_metrics})
                metric_rows.append(row)

    summary_rows = summarize(metric_rows)
    write_csv(args.out_dir / "calibration_params.csv", param_rows)
    write_csv(args.out_dir / "probability_calibration_test_metrics.csv", metric_rows)
    write_csv(args.out_dir / "probability_calibration_summary_mean_std.csv", summary_rows)
    write_json(args.out_dir / "probability_calibration_summary_mean_std.json", summary_json(summary_rows))
    write_markdown(args.out_dir / "probability_calibration_summary.md", summary_rows, param_rows, bool(args.write_predictions))
    print(f"wrote probability calibration outputs to {args.out_dir}")
    return 0


class PredictionRow:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.y = int(float(values["y_true"]))
        self.p = clamp01(float(values["y_prob"]))


class FittedCalibrator:
    def __init__(self, name: str, apply: Callable[[float], float], params: dict[str, object]) -> None:
        self.name = name
        self.apply = apply
        self.params = params


def fit_calibrators(y: Sequence[int], p: Sequence[float]) -> dict[str, FittedCalibrator]:
    return {
        "none": FittedCalibrator("none", lambda value: clamp01(value), {}),
        "temperature": fit_temperature_scaling(y, p),
        "platt": fit_platt_scaling(y, p),
        "isotonic": fit_isotonic_regression(y, p),
    }


def fit_temperature_scaling(y: Sequence[int], p: Sequence[float]) -> FittedCalibrator:
    logits = [prob_to_logit(value) for value in p]

    def objective(log_temp: float) -> float:
        temp = math.exp(log_temp)
        cal = [sigmoid(logit / temp) for logit in logits]
        return negative_log_likelihood(y, cal)

    lo, hi = -4.0, 4.0
    for _ in range(100):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            hi = m2
        else:
            lo = m1
    temp = math.exp((lo + hi) / 2.0)
    return FittedCalibrator(
        "temperature",
        lambda value: sigmoid(prob_to_logit(value) / temp),
        {"temperature": f"{temp:.8f}", "valid_nll": f"{objective(math.log(temp)):.8f}"},
    )


def fit_platt_scaling(y: Sequence[int], p: Sequence[float]) -> FittedCalibrator:
    logits = [prob_to_logit(value) for value in p]
    mean = statistics.mean(logits) if logits else 0.0
    std = statistics.pstdev(logits) if len(logits) > 1 else 1.0
    std = std if std > 1e-12 else 1.0
    x = [(value - mean) / std for value in logits]
    a, b = 1.0, 0.0
    lr = 0.05
    l2 = 1e-4
    n = max(len(x), 1)
    for step in range(3000):
        grad_a = 0.0
        grad_b = 0.0
        for xi, yi in zip(x, y):
            pred = sigmoid(a * xi + b)
            grad_a += (pred - yi) * xi
            grad_b += pred - yi
        grad_a = grad_a / n + l2 * a
        grad_b /= n
        rate = lr / math.sqrt(1.0 + step / 500.0)
        a -= rate * grad_a
        b -= rate * grad_b
    return FittedCalibrator(
        "platt",
        lambda value: sigmoid(a * ((prob_to_logit(value) - mean) / std) + b),
        {"a": f"{a:.8f}", "b": f"{b:.8f}", "logit_mean": f"{mean:.8f}", "logit_std": f"{std:.8f}"},
    )


def fit_isotonic_regression(y: Sequence[int], p: Sequence[float]) -> FittedCalibrator:
    pairs = sorted((float(pi), int(yi)) for yi, pi in zip(y, p))
    blocks: list[dict[str, float]] = []
    for xi, yi in pairs:
        blocks.append({"sum_y": float(yi), "weight": 1.0, "max_x": xi})
        while len(blocks) >= 2 and block_value(blocks[-2]) > block_value(blocks[-1]):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "sum_y": left["sum_y"] + right["sum_y"],
                    "weight": left["weight"] + right["weight"],
                    "max_x": right["max_x"],
                }
            )
    thresholds = [block["max_x"] for block in blocks]
    values = [clamp01(block_value(block)) for block in blocks]

    def apply(value: float) -> float:
        if not values:
            return clamp01(value)
        idx = bisect.bisect_left(thresholds, float(value))
        idx = min(max(idx, 0), len(values) - 1)
        return values[idx]

    return FittedCalibrator("isotonic", apply, {"blocks": len(blocks)})


def block_value(block: dict[str, float]) -> float:
    return block["sum_y"] / max(block["weight"], 1e-12)


def best_threshold(y: Sequence[int], p: Sequence[float], metric: str, steps: int) -> float:
    candidates = sorted(set(float(value) for value in p))
    if len(candidates) > steps + 1:
        candidates = [i / steps for i in range(steps + 1)]
    best_t = 0.5
    best_value = -math.inf
    best_precision = -math.inf
    for threshold in candidates:
        value, precision = threshold_metric(y, p, threshold, metric)
        if value > best_value or (value == best_value and precision > best_precision):
            best_t = float(threshold)
            best_value = value
            best_precision = precision
    return best_t


def threshold_metric(y: Sequence[int], p: Sequence[float], threshold: float, metric: str) -> tuple[float, float]:
    tp = tn = fp = fn = 0
    for yi, pi in zip(y, p):
        pred = 1 if pi >= threshold else 0
        if yi == 1 and pred == 1:
            tp += 1
        elif yi == 0 and pred == 0:
            tn += 1
        elif yi == 0 and pred == 1:
            fp += 1
        elif yi == 1 and pred == 0:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if metric == "f1":
        value = 2.0 * precision * recall / max(precision + recall, 1e-12)
    elif metric == "mcc":
        denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
        value = ((tp * tn) - (fp * fn)) / denom
    else:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    return value, precision


def read_prediction_rows(path: Path) -> list[PredictionRow]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [PredictionRow(row) for row in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    return rows


def write_calibrated_predictions(path: Path, rows: Sequence[PredictionRow], probs: Sequence[float], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row, prob in zip(rows, probs):
        values = dict(row.values)
        values["y_prob_raw"] = values["y_prob"]
        values["y_prob"] = f"{clamp01(prob):.10f}"
        values["y_pred"] = 1 if clamp01(prob) >= 0.5 else 0
        values["calibration_method"] = method
        out_rows.append(values)
    write_csv(path, out_rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(row["method"]), str(row["threshold_strategy"])) for row in rows})
    for method, threshold_strategy in keys:
        subset = [row for row in rows if row["method"] == method and row["threshold_strategy"] == threshold_strategy]
        summary: dict[str, object] = {"method": method, "threshold_strategy": threshold_strategy, "n": len(subset)}
        for field in ["threshold", "valid_f1", "valid_mcc", "valid_brier", "valid_nll", "valid_ece", *METRIC_FIELDS]:
            vals = [float(row[field]) for row in subset if field in row]
            if not vals:
                continue
            summary[f"{field}_mean"] = f"{statistics.mean(vals):.6f}"
            summary[f"{field}_std"] = f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000"
        out.append(summary)
    return out


def summary_json(rows: list[dict[str, object]]) -> dict[str, object]:
    return {f"{row['method']}:{row['threshold_strategy']}": row for row in rows}


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_markdown(path: Path, summary_rows: list[dict[str, object]], param_rows: list[dict[str, object]], wrote_predictions: bool) -> None:
    lines = [
        "# Probability Calibration Summary",
        "",
        "Calibrators are fitted on validation predictions and then frozen for test predictions.",
        "",
        "## Test Mean +/- Std",
        "",
        "| Method | Threshold | F1 | Precision | Recall | AUC | AP | MCC | Brier | NLL | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {threshold_strategy} | {f1_mean} +/- {f1_std} | {precision_mean} +/- {precision_std} | "
            "{recall_mean} +/- {recall_std} | {auc_mean} +/- {auc_std} | {ap_mean} +/- {ap_std} | "
            "{mcc_mean} +/- {mcc_std} | {brier_mean} +/- {brier_std} | {nll_mean} +/- {nll_std} | "
            "{ece_mean} +/- {ece_std} |".format(**row)
        )
    if wrote_predictions:
        lines.extend(["", "Calibrated per-sample prediction CSV files were written under `predictions/`.", ""])
    lines.extend(["", "## Fitted Parameters", "", "See `calibration_params.csv` for complete values.", ""])
    for row in param_rows[:12]:
        params = ", ".join(f"{key}={value}" for key, value in row.items() if key not in {"run", "method"}) or "identity"
        lines.append(f"- {row['run']} / {row['method']}: {params}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prob_to_logit(value: float) -> float:
    p = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
