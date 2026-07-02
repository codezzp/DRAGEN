from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

import _bootstrap  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    window_dir = args.window_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    root = load_embedding(args.text_emb_dir, f"root_text_emb{args.dim}")
    retweet = load_embedding(args.text_emb_dir, f"retweet_text_emb{args.dim}")
    rows = read_text_window_rows(window_dir / "text_window_table.csv")
    node_index, node_x, window_index, window_x, diagnostics = aggregate_text_semantics(rows, root, retweet)
    np.save(out_dir / "node_text_features.npy", node_x)
    np.save(out_dir / "window_text_features.npy", window_x)
    write_json(out_dir / "node_text_feature_index.json", node_index)
    write_json(out_dir / "window_text_feature_index.json", window_index)
    write_json(
        out_dir / "text_semantic_feature_meta.json",
        {
            "run_id": args.run_id,
            "window_dir": str(window_dir),
            "text_emb_dir": str(args.text_emb_dir),
            "dim": int(args.dim),
            "node_rows": len(node_index),
            "window_rows": len(window_index),
            "aggregation": "visible_text_mean_by_node_window_and_window",
            "causality": "uses text_window_table visibility; root is visible in every window, retweets only when post_offset < end_offset upstream",
            "diagnostics": diagnostics,
        },
    )
    print(f"Wrote semantic text features to {out_dir}: node={node_x.shape} window={window_x.shape}")
    return 0


class EmbeddingTable:
    def __init__(self, emb: np.ndarray, keys: List[int]) -> None:
        self.emb = emb.astype(np.float32)
        self.keys = keys
        self.pos = {int(key): i for i, key in enumerate(keys)}
        self.dim = int(self.emb.shape[1]) if self.emb.ndim == 2 else 0

    def get(self, key: Any) -> np.ndarray | None:
        if key in (None, ""):
            return None
        pos = self.pos.get(int(key))
        if pos is None:
            return None
        return self.emb[pos]


def load_embedding(path: Path, stem: str) -> EmbeddingTable:
    emb_path = path / f"{stem}.npy"
    idx_path = path / f"{stem}.idx.json"
    if not emb_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Missing reduced embedding files: {emb_path}, {idx_path}")
    emb = np.load(emb_path)
    keys = json.loads(idx_path.read_text(encoding="utf-8"))
    if len(keys) != emb.shape[0]:
        raise ValueError(f"Index length does not match embedding rows for {stem}: {len(keys)} vs {emb.shape[0]}")
    return EmbeddingTable(emb, [int(key) for key in keys])


def read_text_window_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_text_semantics(
    rows: List[Mapping[str, str]],
    root: EmbeddingTable,
    retweet: EmbeddingTable,
) -> tuple[List[Dict[str, int]], np.ndarray, List[Dict[str, int]], np.ndarray, Dict[str, Any]]:
    dim = root.dim or retweet.dim
    node_sum: Dict[Tuple[int, int, int], np.ndarray] = {}
    node_count: Dict[Tuple[int, int, int], int] = defaultdict(int)
    window_sum: Dict[Tuple[int, int], np.ndarray] = {}
    window_count: Dict[Tuple[int, int], int] = defaultdict(int)
    missing_root = 0
    missing_retweet = 0
    used_rows = 0
    for row in rows:
        cascade_idx = int(row["cascade_idx"])
        window_idx = int(row["window_idx"])
        user_idx = int(row["user_idx"])
        post_type = str(row.get("post_type", "")).lower()
        if post_type == "root":
            emb = root.get(cascade_idx)
            if emb is None:
                missing_root += 1
                continue
        else:
            emb = retweet.get(row.get("tweet_idx"))
            if emb is None:
                missing_retweet += 1
                continue
        node_key = (cascade_idx, window_idx, user_idx)
        window_key = (cascade_idx, window_idx)
        if node_key not in node_sum:
            node_sum[node_key] = np.zeros(dim, dtype=np.float32)
        if window_key not in window_sum:
            window_sum[window_key] = np.zeros(dim, dtype=np.float32)
        node_sum[node_key] += emb
        window_sum[window_key] += emb
        node_count[node_key] += 1
        window_count[window_key] += 1
        used_rows += 1
    node_keys = sorted(node_sum)
    window_keys = sorted(window_sum)
    node_x = np.zeros((len(node_keys), dim), dtype=np.float32)
    window_x = np.zeros((len(window_keys), dim), dtype=np.float32)
    node_index: List[Dict[str, int]] = []
    window_index: List[Dict[str, int]] = []
    for i, key in enumerate(node_keys):
        node_x[i] = normalize(node_sum[key] / max(node_count[key], 1))
        node_index.append({"cascade_idx": key[0], "window_idx": key[1], "user_idx": key[2], "num_texts": node_count[key]})
    for i, key in enumerate(window_keys):
        window_x[i] = normalize(window_sum[key] / max(window_count[key], 1))
        window_index.append({"cascade_idx": key[0], "window_idx": key[1], "num_texts": window_count[key]})
    diagnostics = {
        "input_text_window_rows": len(rows),
        "used_text_window_rows": used_rows,
        "missing_root_embedding_rows": missing_root,
        "missing_retweet_embedding_rows": missing_retweet,
        "node_feature_rows": len(node_index),
        "window_feature_rows": len(window_index),
    }
    return node_index, node_x, window_index, window_x, diagnostics


def normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        return x.astype(np.float32)
    return (x / norm).astype(np.float32)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cached text embeddings into causal window-level semantic text features.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--text-emb-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
