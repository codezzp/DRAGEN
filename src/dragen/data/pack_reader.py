"""Read pickle-stream DRAGEN packs and collate variable-size cascades."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dragen.data.feature_schema import DEFAULT_SCHEMA, FeatureSchema, schema_from_meta


class PickleStreamDataset(Dataset):
    """A small-index wrapper around the current pickle-stream .pt pack format."""

    def __init__(self, path: Path | str, max_samples: Optional[int] = None, split: str | None = None) -> None:
        self.path = Path(path)
        self.samples: List[Dict[str, Any]] = []
        for sample in iter_pickle_stream(self.path):
            self.samples.append(sample)
            if max_samples is not None and len(self.samples) >= max_samples:
                break
        label = split or self.path.stem
        limit = f", limit={max_samples}" if max_samples is not None else ""
        print(f"loaded {label}: {len(self.samples)} samples from {self.path}{limit}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def iter_pickle_stream(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def read_pack_meta(pack_dir: Path | str) -> tuple[Mapping[str, Any], FeatureSchema]:
    meta_path = Path(pack_dir) / "meta.json"
    if not meta_path.exists():
        return {}, DEFAULT_SCHEMA
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta, schema_from_meta(meta)


def make_datasets(pack_dir: Path | str, max_train: int | None = None, max_valid: int | None = None, max_test: int | None = None) -> Dict[str, PickleStreamDataset]:
    base = Path(pack_dir)
    return {
        "train": PickleStreamDataset(base / "train.pt", max_train, "train"),
        "valid": PickleStreamDataset(base / "valid.pt", max_valid, "valid"),
        "test": PickleStreamDataset(base / "test.pt", max_test, "test"),
    }


def collate_fn(samples: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("empty batch")
    batch_size = len(samples)
    T = int(samples[0]["window_x"].shape[0])
    d_w = int(samples[0]["window_x"].shape[1])
    d_n = int(samples[0]["node_x"].shape[2])
    n_max = max(int(sample["node_x"].shape[1]) for sample in samples)

    window_x = torch.zeros(batch_size, T, d_w, dtype=torch.float32)
    node_x = torch.zeros(batch_size, T, n_max, d_n, dtype=torch.float32)
    node_mask = torch.zeros(batch_size, T, n_max, dtype=torch.bool)
    y = torch.zeros(batch_size, dtype=torch.float32)
    cascade_idx = torch.zeros(batch_size, dtype=torch.long)
    edge_index_current: List[List[torch.Tensor]] = []
    edge_index_context: List[List[torch.Tensor]] = []
    global_candidate_edge_index: List[torch.Tensor] = []
    global_candidate_edge_weight: List[torch.Tensor] = []
    has_node_text = any(sample.get("node_text_x") is not None for sample in samples)
    has_window_text = any(sample.get("window_text_x") is not None for sample in samples)
    d_node_text = infer_optional_dim(samples, "node_text_x") if has_node_text else 0
    d_window_text = infer_optional_dim(samples, "window_text_x") if has_window_text else 0
    node_text_x = torch.zeros(batch_size, T, n_max, d_node_text, dtype=torch.float32) if has_node_text else None
    window_text_x = torch.zeros(batch_size, T, d_window_text, dtype=torch.float32) if has_window_text else None

    for b, sample in enumerate(samples):
        n = int(sample["node_x"].shape[1])
        window_x[b] = stabilize_features(torch.as_tensor(np.asarray(sample["window_x"]), dtype=torch.float32))
        node_x[b, :, :n, :] = stabilize_features(torch.as_tensor(np.asarray(sample["node_x"]), dtype=torch.float32))
        node_mask[b, :, :n] = torch.as_tensor(np.asarray(sample["node_mask"]), dtype=torch.bool)
        y[b] = float(sample["y"])
        cascade_idx[b] = int(sample["cascade_idx"])
        edge_index_current.append(_edge_list_to_tensors(sample["edge_index_current"], T))
        edge_index_context.append(_edge_list_to_tensors(sample["edge_index_context"], T))
        global_candidate_edge_index.append(_single_edge_tensor(sample.get("global_candidate_edge_index")))
        global_candidate_edge_weight.append(_weight_tensor(sample.get("global_candidate_edge_weight")))
        if node_text_x is not None and sample.get("node_text_x") is not None:
            text_arr = np.asarray(sample["node_text_x"], dtype=np.float32)
            node_text_x[b, :, :n, :] = torch.as_tensor(text_arr, dtype=torch.float32)
        if window_text_x is not None and sample.get("window_text_x") is not None:
            window_text_x[b] = torch.as_tensor(np.asarray(sample["window_text_x"], dtype=np.float32), dtype=torch.float32)

    batch = {
        "cascade_idx": cascade_idx,
        "window_x": window_x,
        "node_x": node_x,
        "edge_index_current": edge_index_current,
        "edge_index_context": edge_index_context,
        "global_candidate_edge_index": global_candidate_edge_index,
        "global_candidate_edge_weight": global_candidate_edge_weight,
        "node_mask": node_mask,
        "y": y,
    }
    if node_text_x is not None:
        batch["node_text_x"] = node_text_x
    if window_text_x is not None:
        batch["window_text_x"] = window_text_x
    return batch


def infer_optional_dim(samples: List[Mapping[str, Any]], key: str) -> int:
    for sample in samples:
        value = sample.get(key)
        if value is not None:
            arr = np.asarray(value)
            if arr.ndim >= 1:
                return int(arr.shape[-1])
    return 0


def _edge_list_to_tensors(edge_list: Iterable[Any], T: int) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for edge in list(edge_list)[:T]:
        arr = np.asarray(edge, dtype=np.int64)
        if arr.size == 0:
            arr = np.zeros((2, 0), dtype=np.int64)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        tensors.append(torch.as_tensor(arr, dtype=torch.long))
    while len(tensors) < T:
        tensors.append(torch.zeros(2, 0, dtype=torch.long))
    return tensors


def _single_edge_tensor(edge: Any) -> torch.Tensor:
    if edge is None:
        return torch.zeros(2, 0, dtype=torch.long)
    arr = np.asarray(edge, dtype=np.int64)
    if arr.size == 0:
        arr = np.zeros((2, 0), dtype=np.int64)
    if arr.shape[0] != 2:
        arr = arr.reshape(2, -1)
    return torch.as_tensor(arr, dtype=torch.long)


def _weight_tensor(weight: Any) -> torch.Tensor:
    if weight is None:
        return torch.zeros(0, dtype=torch.float32)
    arr = np.asarray(weight, dtype=np.float32).reshape(-1)
    return torch.as_tensor(arr, dtype=torch.float32)


def stabilize_features(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.sign(x) * torch.log1p(torch.abs(x))
