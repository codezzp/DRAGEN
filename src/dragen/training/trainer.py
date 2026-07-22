"""Training loop for DRAGEN-Full."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only when tensorboard is absent.
    SummaryWriter = None  # type: ignore[assignment]

from dragen.config import write_run_metadata
from dragen.data.pack_reader import collate_fn, make_datasets
from dragen.evaluation.export_predictions import collect_predictions, export_prediction_files, move_batch_to_device, write_json
from dragen.evaluation.metrics import binary_metrics
from dragen.models.dragen_full import DRAGENFull
from dragen.training.losses import dragen_full_loss
from dragen.evaluation.training_curves import plot_training_curves
from dragen.utils.progress import progress_iter


def train_dragen_full(args: Any) -> Dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, args)
    datasets = make_datasets(args.pack_dir, args.max_train_samples, args.max_valid_samples, args.max_test_samples)
    print(
        f"dataset sizes: train={len(datasets['train'])} valid={len(datasets['valid'])} test={len(datasets['test'])}",
        flush=True,
    )
    loader_kwargs = build_dataloader_kwargs(args, device)
    print(f"dataloader kwargs: {loader_kwargs}", flush=True)
    loaders = build_loaders(datasets, args, loader_kwargs)
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
        text_semantic_dim=getattr(args, "text_semantic_dim", 64),
        global_sampling_mode=getattr(args, "global_sampling_mode", "edge_list"),
        key_user_max_hops=getattr(args, "key_user_max_hops", 4),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = create_tensorboard_writer(args, out_dir)
    weights = {
        "lambda_jump": args.lambda_jump,
        "lambda_struct": args.lambda_struct,
        "lambda_align": args.lambda_align,
        "lambda_uncertainty": args.lambda_uncertainty,
        "lambda_role": args.lambda_role,
        "lambda_sampler_edge": args.lambda_sampler_edge,
        "lambda_sampler_hub": args.lambda_sampler_hub,
        "lambda_sampler_temp": args.lambda_sampler_temp,
        "event_loss": getattr(args, "event_loss", "bce"),
        "pos_weight": resolve_pos_weight_arg(getattr(args, "pos_weight", "auto"), datasets["train"]),
        "focal_alpha": getattr(args, "focal_alpha", 0.75),
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
    }
    best_score = -1.0
    history = []
    start_epoch = 1
    if args.resume:
        resume_state = load_checkpoint(Path(args.resume), model, optim, device)
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_score = float(resume_state.get("best_score", best_score))
        history = list(resume_state.get("history", []))
        print(f"resumed from {args.resume}: start_epoch={start_epoch} best_score={best_score:.4f}", flush=True)
    initialize_epoch_metrics(out_dir / "reports" / "epoch_metrics.csv", append=bool(args.resume))
    loss_breakdown = {"history": history}
    write_json(out_dir / "reports" / "loss_breakdown.json", loss_breakdown)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        print(f"epoch {epoch}/{args.epochs} start", flush=True)
        train_loss = train_one_epoch(model, loaders["train"], optim, weights, device, epoch=epoch, epochs=args.epochs)
        should_eval = epoch % max(args.eval_every, 1) == 0 or epoch == args.epochs
        valid_metrics: Dict[str, float] = {}
        valid_breakdown: Dict[str, float] = {}
        valid_loss = None
        score = best_score
        if should_eval:
            valid_metrics, valid_breakdown = evaluate_loss_and_metrics(
                model, loaders["valid"], weights, device, desc=f"valid epoch {epoch}/{args.epochs}"
            )
            valid_loss = float(valid_breakdown.get("loss_total", 0.0))
            score = valid_metrics.get("auc", 0.0) or valid_metrics.get("f1", 0.0)
        epoch_time_sec = time.time() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid": valid_metrics,
                "valid_loss": valid_breakdown,
                "epoch_time_sec": epoch_time_sec,
            }
        )
        loss_breakdown = {"history": history}
        write_json(out_dir / "reports" / "loss_breakdown.json", loss_breakdown)
        append_epoch_metrics(
            out_dir / "reports" / "epoch_metrics.csv",
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_metrics=valid_metrics,
            lr=current_lr(optim),
            epoch_time_sec=epoch_time_sec,
        )
        write_tensorboard_epoch(
            writer,
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_metrics=valid_metrics,
            valid_breakdown=valid_breakdown,
            lr=current_lr(optim),
            epoch_time_sec=epoch_time_sec,
        )
        if getattr(args, "plot_every_epoch", True):
            plot_training_curves(out_dir / "reports" / "epoch_metrics.csv", out_dir / "reports" / "loss_breakdown.json", out_dir / "reports" / "training_curves.png")
        checkpoint = build_checkpoint(model, optim, args, epoch, best_score, history)
        torch.save(checkpoint, out_dir / "checkpoints" / "last.pt")
        if args.save_every_epoch:
            torch.save(checkpoint, out_dir / "checkpoints" / f"epoch_{epoch}.pt")
        if should_eval:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} valid_auc={valid_metrics['auc']:.4f} "
                f"valid_f1={valid_metrics['f1']:.4f} epoch_time_sec={epoch_time_sec:.1f}",
                flush=True,
            )
        else:
            print(f"epoch={epoch} train_loss={train_loss:.4f} valid_skipped=1 epoch_time_sec={epoch_time_sec:.1f}", flush=True)
        if should_eval and score >= best_score:
            best_score = score
            checkpoint = build_checkpoint(model, optim, args, epoch, best_score, history)
            torch.save(checkpoint, out_dir / "checkpoints" / "best.pt")
            torch.save(checkpoint, out_dir / "checkpoints" / "last.pt")
    metrics = {}
    for split in ["valid", "test"]:
        events, detail = collect_predictions(model, loaders[split], device, desc=f"export {split}")
        metrics[split] = export_prediction_files(out_dir, split, events, detail)
    write_json(out_dir / "reports" / "metrics.json", metrics)
    write_json(out_dir / "reports" / "loss_breakdown.json", loss_breakdown)
    if writer is not None:
        writer.close()
    return {"metrics": metrics, "history": history}


def resolve_pos_weight_arg(value: Any, train_dataset: Any) -> float:
    if not isinstance(value, str):
        return float(value)
    key = value.lower()
    if key not in {"auto", "sqrt_auto", "soft"}:
        return float(value)
    labels = [float(sample.get("y", 0.0)) for sample in getattr(train_dataset, "samples", [])]
    pos = max(sum(1 for label in labels if label >= 0.5), 1)
    neg = max(len(labels) - pos, 1)
    ratio = neg / pos
    if key == "auto":
        return float(ratio)
    return float(ratio ** 0.5)

class NodeBucketBatchSampler:
    """Batch samples with similar node counts to reduce padding waste."""

    def __init__(
        self,
        node_counts: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        bucket_size_multiplier: int = 50,
        max_nodes_per_batch: int | None = None,
    ) -> None:
        self.node_counts = list(node_counts)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size * max(int(bucket_size_multiplier), 1), self.batch_size)
        self.max_nodes_per_batch = int(max_nodes_per_batch or 0)
        self.epoch = 0
        self._length_cache = self._build_batches(random.Random(self.seed), shuffle_buckets=False, shuffle_batches=False)
        self.largest_batch_padded_nodes = self._largest_padded_nodes(self._length_cache)

    @property
    def dynamic_nodes(self) -> bool:
        return self.max_nodes_per_batch > 0

    @staticmethod
    def _batch_padded_nodes(batch: Sequence[int], node_counts: Sequence[int]) -> int:
        if not batch:
            return 0
        return len(batch) * max(int(node_counts[idx]) for idx in batch)

    def _largest_padded_nodes(self, batches: Sequence[Sequence[int]]) -> int:
        if not batches:
            return 0
        return max(self._batch_padded_nodes(batch, self.node_counts) for batch in batches)

    def _flush_dynamic_batch(self, batches: List[List[int]], batch: List[int]) -> None:
        if batch:
            batches.append(list(batch))
            batch.clear()

    def _append_dynamic_bucket(self, batches: List[List[int]], bucket: Sequence[int]) -> None:
        batch: List[int] = []
        max_nodes = 0
        for idx in bucket:
            nodes = int(self.node_counts[idx])
            if nodes >= self.max_nodes_per_batch:
                self._flush_dynamic_batch(batches, batch)
                batches.append([idx])
                max_nodes = 0
                continue

            next_size = len(batch) + 1
            next_max_nodes = max(max_nodes, nodes)
            next_padded_nodes = next_size * next_max_nodes
            if batch and (next_size > self.batch_size or next_padded_nodes > self.max_nodes_per_batch):
                self._flush_dynamic_batch(batches, batch)
                max_nodes = 0

            batch.append(idx)
            max_nodes = max(max_nodes, nodes)
            if len(batch) >= self.batch_size:
                self._flush_dynamic_batch(batches, batch)
                max_nodes = 0

        self._flush_dynamic_batch(batches, batch)

    def _build_batches(self, rng: random.Random, *, shuffle_buckets: bool, shuffle_batches: bool) -> List[List[int]]:
        indices = sorted(range(len(self.node_counts)), key=lambda idx: self.node_counts[idx])
        batches: List[List[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            if shuffle_buckets and not self.dynamic_nodes:
                rng.shuffle(bucket)
            if self.dynamic_nodes:
                self._append_dynamic_bucket(batches, bucket)
            else:
                for batch_start in range(0, len(bucket), self.batch_size):
                    batches.append(bucket[batch_start : batch_start + self.batch_size])
        if shuffle_batches:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = self._build_batches(rng, shuffle_buckets=self.shuffle, shuffle_batches=self.shuffle)
        self.largest_batch_padded_nodes = self._largest_padded_nodes(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._length_cache)

def build_loaders(datasets: Mapping[str, Any], args: Any, loader_kwargs: Dict[str, Any]) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    use_buckets = bool(getattr(args, "bucket_by_nodes", False))
    bucket_multiplier = int(getattr(args, "bucket_size_multiplier", 50) or 50)
    max_nodes_per_batch = int(getattr(args, "max_nodes_per_batch", 0) or 0)
    for split, ds in datasets.items():
        if use_buckets:
            node_counts = [int(sample["node_x"].shape[1]) for sample in ds.samples]
            sampler = NodeBucketBatchSampler(
                node_counts,
                args.batch_size,
                shuffle=(split == "train"),
                seed=int(getattr(args, "seed", 0)),
                bucket_size_multiplier=bucket_multiplier,
                max_nodes_per_batch=max_nodes_per_batch,
            )
            print(
                f"{split} bucket_by_nodes=1 dynamic_nodes={int(sampler.dynamic_nodes)} batches={len(sampler)} "
                f"min_nodes={min(node_counts) if node_counts else 0} max_nodes={max(node_counts) if node_counts else 0} "
                f"max_nodes_per_batch={max_nodes_per_batch if max_nodes_per_batch > 0 else 'none'} "
                f"largest_batch_padded_nodes={sampler.largest_batch_padded_nodes}",
                flush=True,
            )
            loaders[split] = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, **loader_kwargs)
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=(split == "train"),
                collate_fn=collate_fn,
                **loader_kwargs,
            )
    return loaders
def build_dataloader_kwargs(args: Any, device: torch.device) -> Dict[str, Any]:
    num_workers = max(int(getattr(args, "num_workers", 0) or 0), 0)
    pin_memory_arg = getattr(args, "pin_memory", None)
    pin_memory = bool(device.type == "cuda") if pin_memory_arg is None else bool(pin_memory_arg)
    persistent_arg = getattr(args, "persistent_workers", None)
    persistent_workers = bool(num_workers > 0) if persistent_arg is None else bool(persistent_arg and num_workers > 0)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    prefetch_factor = getattr(args, "prefetch_factor", None)
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return kwargs


def use_non_blocking_transfer(loader: DataLoader) -> bool:
    return bool(getattr(loader, "pin_memory", False) and torch.cuda.is_available())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    weights: Dict[str, float],
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in progress_iter(loader, total=len(loader), desc=f"train epoch {epoch}/{epochs}", every=max(len(loader) // 20, 1)):
        batch = move_batch_to_device(batch, device, non_blocking=use_non_blocking_transfer(loader))
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
def evaluate_loss_and_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    weights: Dict[str, float],
    device: torch.device,
    *,
    desc: str = "valid",
) -> tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    y_true = []
    y_prob = []
    breakdown_sum: Dict[str, float] = {}
    count = 0
    for batch in progress_iter(loader, total=len(loader), desc=desc, every=max(len(loader) // 10, 1)):
        batch = move_batch_to_device(batch, device, non_blocking=use_non_blocking_transfer(loader))
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_epoch_metrics(path: Path, *, append: bool) -> None:
    if append and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_metric_fields())
        writer.writeheader()
        f.flush()


def append_epoch_metrics(
    path: Path,
    *,
    epoch: int,
    train_loss: float,
    valid_loss: float | None,
    valid_metrics: Mapping[str, float],
    lr: float,
    epoch_time_sec: float,
) -> None:
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss if valid_loss is not None else "",
        "valid_accuracy": valid_metrics.get("accuracy", ""),
        "valid_precision": valid_metrics.get("precision", ""),
        "valid_recall": valid_metrics.get("recall", ""),
        "valid_f1": valid_metrics.get("f1", ""),
        "valid_auc": valid_metrics.get("auc", ""),
        "valid_ap": valid_metrics.get("ap", ""),
        "valid_mcc": valid_metrics.get("mcc", ""),
        "lr": lr,
        "epoch_time_sec": epoch_time_sec,
    }
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_metric_fields())
        writer.writerow(row)
        f.flush()


def epoch_metric_fields() -> list[str]:
    return [
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_accuracy",
        "valid_precision",
        "valid_recall",
        "valid_f1",
        "valid_auc",
        "valid_ap",
        "valid_mcc",
        "lr",
        "epoch_time_sec",
    ]


def current_lr(optim: torch.optim.Optimizer) -> float:
    return float(optim.param_groups[0].get("lr", 0.0)) if optim.param_groups else 0.0


def create_tensorboard_writer(args: Any, out_dir: Path) -> Any:
    if not getattr(args, "tensorboard", False):
        return None
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard is not installed. Run: pip install tensorboard")
    tb_dir = Path(args.tb_log_dir) if getattr(args, "tb_log_dir", None) else out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"tensorboard log_dir={tb_dir}", flush=True)
    return SummaryWriter(log_dir=str(tb_dir))


def write_tensorboard_epoch(
    writer: Any,
    *,
    epoch: int,
    train_loss: float,
    valid_loss: float | None,
    valid_metrics: Mapping[str, float],
    valid_breakdown: Mapping[str, float],
    lr: float,
    epoch_time_sec: float,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/lr", lr, epoch)
    writer.add_scalar("train/epoch_time_sec", epoch_time_sec, epoch)
    if valid_loss is not None:
        writer.add_scalar("valid/loss", valid_loss, epoch)
    for key in ["accuracy", "precision", "recall", "f1", "auc", "ap", "mcc"]:
        value = valid_metrics.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"valid/{key}", float(value), epoch)
    for key, value in valid_breakdown.items():
        if isinstance(value, (int, float)):
            name = key.removeprefix("loss_")
            writer.add_scalar(f"loss/{name}", float(value), epoch)
    writer.flush()


def build_checkpoint(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    args: Any,
    epoch: int,
    best_score: float,
    history: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    checkpoint: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optim.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "args": vars(args),
        "history": history,
        "rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return checkpoint


def load_checkpoint(path: Path, model: torch.nn.Module, optim: torch.optim.Optimizer, device: torch.device) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optim.load_state_dict(checkpoint["optimizer_state"])
    else:
        print("resume checkpoint has no optimizer_state; optimizer starts from current args", flush=True)
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    return checkpoint
