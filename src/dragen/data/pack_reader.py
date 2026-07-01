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

    def __init__(self, path: Path | str, max_samples: Optional[int] = None) -> None:
        self.path = Path(path)
        self.samples: List[Dict[str, Any]] = []
        for sample in iter_pickle_stream(self.path):
            self.samples.append(sample)
            if max_samples is not None and len(self.samples) >= max_samples:
                break

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
        "train": PickleStreamDataset(base / "train.pt", max_train),
        "valid": PickleStreamDataset(base / "valid.pt", max_valid),
        "test": PickleStreamDataset(base / "test.pt", max_test),
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

    for b, sample in enumerate(samples):
        n = int(sample["node_x"].shape[1])
        window_x[b] = stabilize_features(torch.as_tensor(np.asarray(sample["window_x"]), dtype=torch.float32))
        node_x[b, :, :n, :] = stabilize_features(torch.as_tensor(np.asarray(sample["node_x"]), dtype=torch.float32))
        node_mask[b, :, :n] = torch.as_tensor(np.asarray(sample["node_mask"]), dtype=torch.bool)
        y[b] = float(sample["y"])
        cascade_idx[b] = int(sample["cascade_idx"])
        edge_index_current.append(_edge_list_to_tensors(sample["edge_index_current"], T))
        edge_index_context.append(_edge_list_to_tensors(sample["edge_index_context"], T))

    return {
        "cascade_idx": cascade_idx,
        "window_x": window_x,
        "node_x": node_x,
        "edge_index_current": edge_index_current,
        "edge_index_context": edge_index_context,
        "node_mask": node_mask,
        "y": y,
    }


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


def stabilize_features(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.sign(x) * torch.log1p(torch.abs(x))
