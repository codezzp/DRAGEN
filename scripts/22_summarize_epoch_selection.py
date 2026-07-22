from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Iterable

import _bootstrap  # noqa: F401


DEFAULT_RUNS = [
    "dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0",
    "dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1",
    "dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2",
]

METRICS = ["valid_accuracy", "valid_precision", "valid_recall", "valid_f1", "valid_auc", "valid_ap", "valid_mcc"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize final and best validation epoch metrics.")
    parser.add_argument("--artifact-root", type=Path, default=Path("work/artifacts/_artifacts"))
    parser.add_argument("--run", action="append", dest="runs", help="Run directory name under artifact-root. Repeatable.")
    parser.add_argument("--out-dir", type=Path, default=Path("work/artifacts/_analysis/run_0002_epoch_selection"))
    args = parser.parse_args()

    runs = args.runs or DEFAULT_RUNS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for run_name in runs:
        epochs = read_epoch_metrics(args.artifact_root / run_name / "reports" / "epoch_metrics.csv")
        rows.append(make_row(run_name, "final_epoch", epochs[-1]))
        rows.append(make_row(run_name, "best_valid_f1", max(epochs, key=lambda r: r["valid_f1"])))
        rows.append(make_row(run_name, "best_valid_auc", max(epochs, key=lambda r: r["valid_auc"])))
        rows.append(make_row(run_name, "best_valid_mcc", max(epochs, key=lambda r: r["valid_mcc"])))

    summary_rows = summarize(rows)
    write_csv(args.out_dir / "epoch_selection_by_seed.csv", rows)
    write_csv(args.out_dir / "epoch_selection_summary_mean_std.csv", summary_rows)
    write_json(args.out_dir / "epoch_selection_summary_mean_std.json", {str(r["strategy"]): r for r in summary_rows})
    write_markdown(args.out_dir / "epoch_selection_summary.md", rows, summary_rows)
    print(f"wrote epoch selection outputs to {args.out_dir}")
    return 0


def read_epoch_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"empty epoch metrics file: {path}")
    return rows


def make_row(run: str, strategy: str, row: dict[str, float]) -> dict[str, object]:
    out: dict[str, object] = {"run": run, "strategy": strategy, "epoch": int(row["epoch"])}
    for metric in METRICS:
        out[metric] = f"{row[metric]:.6f}"
    out["valid_loss"] = f"{row['valid_loss']:.6f}"
    out["train_loss"] = f"{row['train_loss']:.6f}"
    return out


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for strategy in sorted({str(row["strategy"]) for row in rows}):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        summary: dict[str, object] = {"strategy": strategy, "n": len(strategy_rows)}
        for field in ["epoch", *METRICS, "valid_loss", "train_loss"]:
            vals = [float(row[field]) for row in strategy_rows]
            summary[f"{field}_mean"] = f"{statistics.mean(vals):.6f}"
            summary[f"{field}_std"] = f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else "0.000000"
        out.append(summary)
    return out


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


def write_markdown(path: Path, rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Epoch Selection Summary",
        "",
        "This summarizes validation metrics under different model-selection rules.",
        "",
        "## Mean +/- Std",
        "",
        "| Strategy | Epoch | Acc | Precision | Recall | F1 | AUC | AP | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {strategy} | {epoch_mean} +/- {epoch_std} | {valid_accuracy_mean} +/- {valid_accuracy_std} | "
            "{valid_precision_mean} +/- {valid_precision_std} | {valid_recall_mean} +/- {valid_recall_std} | "
            "{valid_f1_mean} +/- {valid_f1_std} | {valid_auc_mean} +/- {valid_auc_std} | "
            "{valid_ap_mean} +/- {valid_ap_std} | {valid_mcc_mean} +/- {valid_mcc_std} |".format(**row)
        )
    lines.extend(["", "## By Seed", "", "| Run | Strategy | Epoch | F1 | AUC | AP | MCC |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['run']} | {row['strategy']} | {row['epoch']} | {row['valid_f1']} | {row['valid_auc']} | {row['valid_ap']} | {row['valid_mcc']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
