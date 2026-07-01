"""Training loop for DRAGEN-Full."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader

from dragen.data.pack_reader import collate_fn, make_datasets
from dragen.evaluation.export_predictions import collect_predictions, export_prediction_files, move_batch_to_device, write_json
from dragen.evaluation.metrics import binary_metrics
from dragen.models.dragen_full import DRAGENFull
from dragen.training.losses import dragen_full_loss


def train_dragen_full(args: Any) -> Dict[str, Any]:
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    datasets = make_datasets(args.pack_dir, args.max_train_samples, args.max_valid_samples, args.max_test_samples)
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"), collate_fn=collate_fn)
        for split, ds in datasets.items()
    }
    model = DRAGENFull(
        hidden_dim=args.hidden_dim,
        role_num=args.role_num,
        top_k_global=args.top_k_global,
        dropout=args.dropout,
        use_global_prior=args.use_global_prior,
        use_adaptive_sampler=args.use_adaptive_sampler,
        use_memory=args.use_memory,
        use_gate=args.use_gate,
        use_uncertainty=args.use_uncertainty,
        use_role=args.use_role,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {
        "lambda_jump": args.lambda_jump,
        "lambda_struct": args.lambda_struct,
        "lambda_align": args.lambda_align,
        "lambda_uncertainty": args.lambda_uncertainty,
        "lambda_role": args.lambda_role,
    }
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optim, weights, device)
        valid_metrics, valid_breakdown = evaluate_loss_and_metrics(model, loaders["valid"], weights, device)
        score = valid_metrics.get("auc", 0.0) or valid_metrics.get("f1", 0.0)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid": valid_metrics, "valid_loss": valid_breakdown})
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_auc={valid_metrics['auc']:.4f} valid_f1={valid_metrics['f1']:.4f}")
        if score >= best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "checkpoints" / "best.pt")
    metrics = {}
    loss_breakdown = {"history": history}
    for split in ["valid", "test"]:
        events, detail = collect_predictions(model, loaders[split], device)
        metrics[split] = export_prediction_files(out_dir, split, events, detail)
    write_json(out_dir / "reports" / "metrics.json", metrics)
    write_json(out_dir / "reports" / "loss_breakdown.json", loss_breakdown)
    return {"metrics": metrics, "history": history}


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, weights: Dict[str, float], device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optim.zero_grad(set_to_none=True)
        out = model(batch)
        loss, _ = dragen_full_loss(out, batch, weights)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loss_and_metrics(model: torch.nn.Module, loader: DataLoader, weights: Dict[str, float], device: torch.device) -> tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    y_true = []
    y_prob = []
    breakdown_sum: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        _, breakdown = dragen_full_loss(out, batch, weights)
        for k, v in breakdown.items():
            breakdown_sum[k] = breakdown_sum.get(k, 0.0) + float(v)
        y_true.extend([int(v) for v in batch["y"].detach().cpu().tolist()])
        y_prob.extend([float(v) for v in out["event_prob"].detach().cpu().tolist()])
        count += 1
    return binary_metrics(y_true, y_prob), {k: v / max(count, 1) for k, v in breakdown_sum.items()}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
