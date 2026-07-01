"""Export DRAGEN-Full event and node-window explanation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch

from dragen.data.feature_schema import ROLE_NAMES
from dragen.evaluation.metrics import binary_metrics
from dragen.utils.progress import progress_iter


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    desc: str = "export",
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    model.eval()
    event_rows: List[Dict[str, Any]] = []
    detail = {
        "node_window": [],
        "role_distribution": [],
        "gate_weights": [],
        "uncertainty": [],
        "event_attention": [],
        "sampled_neighbors": [],
    }
    total = len(loader) if hasattr(loader, "__len__") else None
    every = max((total or 10) // 10, 1)
    for batch in progress_iter(loader, total=total, desc=desc, every=every):
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        B, T, N = out["node_prob"].shape
        for b in range(B):
            cascade_idx = int(batch["cascade_idx"][b].cpu())
            y_true = int(batch["y"][b].cpu())
            y_prob = float(out["event_prob"][b].cpu())
            event_rows.append(
                {
                    "cascade_idx": cascade_idx,
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "y_pred": 1 if y_prob >= 0.5 else 0,
                    "event_strength": float(out["event_strength"][b].cpu()),
                }
            )
            for t in range(T):
                for n in range(N):
                    if not bool(out["node_mask"][b, t, n].cpu()):
                        continue
                    role_id = int(out["dominant_role"][b, t, n].cpu())
                    dominant = ROLE_NAMES[role_id]
                    base = {"cascade_idx": cascade_idx, "window_idx": t + 1, "local_node_idx": n}
                    detail["node_window"].append(
                        {
                            **base,
                            "node_prob": float(out["node_prob"][b, t, n].cpu()),
                            "node_strength": float(out["node_strength"][b, t, n].cpu()),
                            "shock": float(out["shock"][b, t, n].cpu()),
                            "dominant_role": dominant,
                            "gate_obs_weight": float(out["gate_obs_weight"][b, t, n].cpu()),
                            "gate_prior_weight": float(out["gate_prior_weight"][b, t, n].cpu()),
                            "uncertainty": float(out["uncertainty_log_var"][b, t, n].exp().cpu()),
                            "event_attention": float(out["event_attention"][b, t, n].cpu()),
                        }
                    )
                    role_probs = out["role_prob"][b, t, n].detach().cpu().tolist()
                    detail["role_distribution"].append(
                        {
                            **base,
                            "role_producer": role_probs[0],
                            "role_amplifier": role_probs[1],
                            "role_suppressor": role_probs[2],
                            "role_reframer": role_probs[3],
                            "role_ordinary": role_probs[4],
                            "dominant_role": dominant,
                        }
                    )
                    detail["gate_weights"].append(
                        {
                            **base,
                            "gate_obs_weight": float(out["gate_obs_weight"][b, t, n].cpu()),
                            "gate_prior_weight": float(out["gate_prior_weight"][b, t, n].cpu()),
                        }
                    )
                    log_var = float(out["uncertainty_log_var"][b, t, n].cpu())
                    detail["uncertainty"].append({**base, "uncertainty_log_var": log_var, "uncertainty_score": float(torch.exp(torch.tensor(log_var)))})
                    detail["event_attention"].append({**base, "event_attention": float(out["event_attention"][b, t, n].cpu())})
            for t, neigh_by_batch in enumerate(out.get("sampled_global_neighbors", []), start=1):
                if b >= len(neigh_by_batch):
                    continue
                for row in neigh_by_batch[b]:
                    detail["sampled_neighbors"].append(
                        {
                            "cascade_idx": cascade_idx,
                            "window_idx": t,
                            "local_node_idx": row["local_node_idx"],
                            "neighbor_local_node_idx": row["neighbor_local_node_idx"],
                            "sample_weight": row["sample_weight"],
                            "source_type": row["source_type"],
                        }
                    )
    return event_rows, detail


def export_prediction_files(out_dir: Path, split: str, event_rows: List[Mapping[str, Any]], detail: Mapping[str, List[Mapping[str, Any]]]) -> Dict[str, float]:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = [{**row, "split": split} for row in event_rows]
    write_csv(pred_dir / f"{split}_event_predictions.csv", rows)
    if split == "test":
        write_csv(pred_dir / "event_predictions.csv", rows)
        write_csv(pred_dir / "node_window_predictions.csv", detail["node_window"])
        write_csv(pred_dir / "role_distribution.csv", detail["role_distribution"])
        write_csv(pred_dir / "gate_weights.csv", detail["gate_weights"])
        write_csv(pred_dir / "uncertainty.csv", detail["uncertainty"])
        write_csv(pred_dir / "event_attention.csv", detail["event_attention"])
        write_csv(pred_dir / "sampled_global_neighbors.csv", detail["sampled_neighbors"])
    return binary_metrics([int(r["y_true"]) for r in event_rows], [float(r["y_prob"]) for r in event_rows])


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif key.startswith("edge_index"):
            moved[key] = [[edge.to(device) for edge in per_sample] for per_sample in value]
        else:
            moved[key] = value
    return moved
