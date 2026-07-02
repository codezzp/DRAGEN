"""Plot training metrics from DRAGEN report files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


def plot_training_curves(epoch_metrics: Path, loss_breakdown: Path, out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency.
        rows = read_epoch_metrics(epoch_metrics)
        if not rows:
            return False
        html = out_png.with_suffix(".html")
        write_training_curves_html(rows, html)
        print(f"training curve HTML written to {html}; PNG skipped because matplotlib is unavailable ({exc})", flush=True)
        return True

    rows = read_epoch_metrics(epoch_metrics)
    if not rows:
        return False
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_line(axes[0][0], epochs, rows, "train_loss", "Train Loss")
    plot_line(axes[0][0], epochs, rows, "valid_loss", "Valid Loss")
    axes[0][0].set_title("Loss")
    axes[0][0].legend()

    for key in ["valid_auc", "valid_ap", "valid_f1", "valid_mcc"]:
        plot_line(axes[0][1], epochs, rows, key, key)
    axes[0][1].set_title("Validation Metrics")
    axes[0][1].legend()

    for key in ["valid_accuracy", "valid_precision", "valid_recall"]:
        plot_line(axes[1][0], epochs, rows, key, key)
    axes[1][0].set_title("Classification Metrics")
    axes[1][0].legend()

    plot_loss_breakdown(axes[1][1], loss_breakdown)
    axes[1][1].set_title("Loss Breakdown")
    axes[1][1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def read_epoch_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_line(ax: Any, epochs: List[int], rows: List[Mapping[str, str]], key: str, label: str) -> None:
    values = [to_float(row.get(key, "")) for row in rows]
    if all(value is None for value in values):
        return
    xs = [epoch for epoch, value in zip(epochs, values) if value is not None]
    ys = [value for value in values if value is not None]
    if xs and ys:
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)


def plot_loss_breakdown(ax: Any, path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history", [])
    if not history:
        return
    epochs = [int(row.get("epoch", i + 1)) for i, row in enumerate(history)]
    keys = ["loss_event", "loss_struct", "loss_sampler_edge", "loss_sampler_hub", "loss_sampler_temp"]
    for key in keys:
        values = []
        for row in history:
            valid_loss = row.get("valid_loss", {}) if isinstance(row, Mapping) else {}
            value = valid_loss.get(key) if isinstance(valid_loss, Mapping) else None
            values.append(float(value) if isinstance(value, (int, float)) else None)
        if any(value is not None for value in values):
            ax.plot([e for e, v in zip(epochs, values) if v is not None], [v for v in values if v is not None], marker="o", linewidth=1.2, label=key)


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None



def write_training_curves_html(rows: List[Mapping[str, str]], out_html: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    panels = [
        ("Loss", ["train_loss", "valid_loss"]),
        ("Validation Metrics", ["valid_auc", "valid_ap", "valid_f1", "valid_mcc"]),
        ("Classification Metrics", ["valid_accuracy", "valid_precision", "valid_recall"]),
        ("Runtime", ["epoch_time_sec"]),
    ]
    body = "\n".join(render_svg_panel(title, epochs, rows, keys) for title, keys in panels)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'><title>DRAGEN Training Curves</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#1f2933}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}"
        ".panel{border:1px solid #d7dde5;border-radius:8px;padding:12px}"
        "svg{width:100%;height:auto}.legend{font-size:12px}</style></head>"
        f"<body><h1>DRAGEN Training Curves</h1><div class='grid'>{body}</div></body></html>\n",
        encoding="utf-8",
    )


def render_svg_panel(title: str, epochs: List[int], rows: List[Mapping[str, str]], keys: List[str]) -> str:
    width, height = 560, 320
    left, top, right, bottom = 52, 28, 18, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    series = []
    for key in keys:
        points = [(epoch, to_float(row.get(key, ""))) for epoch, row in zip(epochs, rows)]
        points = [(x, y) for x, y in points if y is not None]
        if points:
            series.append((key, points))
    if not series:
        return f"<div class='panel'><h3>{title}</h3><p>No data.</p></div>"
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    polylines = []
    legends = []
    for idx, (key, pts) in enumerate(series):
        color = colors[idx % len(colors)]
        coords = []
        for x, y in pts:
            sx = left + (x - x_min) / (x_max - x_min) * plot_w
            sy = top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h
            coords.append(f"{sx:.1f},{sy:.1f}")
        polylines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(coords)}'/>")
        legends.append(f"<span style='color:{color};margin-right:12px'>{key}</span>")
    axes = (
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#64748b'/>"
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#64748b'/>"
        f"<text x='{left}' y='{height - 14}' font-size='11'>epoch {x_min}-{x_max}</text>"
        f"<text x='{left}' y='18' font-size='11'>{y_min:.4g} to {y_max:.4g}</text>"
    )
    return f"<div class='panel'><h3>{title}</h3><svg viewBox='0 0 {width} {height}'>{axes}{''.join(polylines)}</svg><div class='legend'>{''.join(legends)}</div></div>"
