from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

import _bootstrap  # noqa: F401


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reduce_one(args.in_dir, args.out_dir, "root_text_emb", args.dim, args.seed, args.force)
    reduce_one(args.in_dir, args.out_dir, "retweet_text_emb", args.dim, args.seed, args.force)
    print(f"Wrote reduced text embeddings to {args.out_dir}")
    return 0


def reduce_one(in_dir: Path, out_dir: Path, stem: str, dim: int, seed: int, force: bool) -> None:
    in_npy = in_dir / f"{stem}.npy"
    in_idx = in_dir / f"{stem}.idx.json"
    in_meta = in_dir / f"{stem}.meta.json"
    out_stem = f"{stem}{dim}"
    out_npy = out_dir / f"{out_stem}.npy"
    out_idx = out_dir / f"{out_stem}.idx.json"
    out_meta = out_dir / f"{out_stem}.meta.json"
    if not force and out_npy.exists() and out_idx.exists() and out_meta.exists() and meta_matches(out_meta, dim, seed, in_npy):
        print(f"skip existing reduced embedding: {out_npy}")
        return
    emb = np.load(in_npy).astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding: {in_npy}")
    if dim > emb.shape[1]:
        raise ValueError(f"--dim {dim} cannot exceed input dim {emb.shape[1]}")
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(emb.shape[1], dim)).astype(np.float32)
    reduced = emb @ proj
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.maximum(norms, 1e-12)
    np.save(out_npy, reduced.astype(np.float32))
    out_idx.write_bytes(in_idx.read_bytes())
    source_meta: Dict[str, Any] = json.loads(in_meta.read_text(encoding="utf-8")) if in_meta.exists() else {}
    write_json(
        out_meta,
        {
            "method": "gaussian_random_projection",
            "reducer_type": "fixed_random_projection",
            "reducer_fit_scope": "none_no_fit",
            "leakage_note": "Projection matrix is sampled from a fixed seed and is not fit on train/valid/test embeddings.",
            "source_embedding_dim": int(emb.shape[1]),
            "reduced_dim": dim,
            "dim": dim,
            "seed": seed,
            "source": str(in_npy),
            "source_file_hash": sha256_file(in_npy),
            "source_idx_hash": sha256_file(in_idx),
            "source_meta": source_meta,
            "num_samples": int(reduced.shape[0]),
            "created_at": utc_now(),
        },
    )
    print(f"wrote {out_npy} shape={reduced.shape}")


def meta_matches(path: Path, dim: int, seed: int, source: Path) -> bool:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("dim") == dim and meta.get("seed") == seed and meta.get("source") == str(source) and meta.get("reducer_fit_scope") == "none_no_fit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce cached RoBERTa text embeddings to a compact dimension for window aggregation.")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
