from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import _bootstrap  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    text_dir = args.text_dir or run_dir / "processed" / "text"
    out_dir = args.out_dir or run_dir / "text_embeddings" / safe_model_dir(args.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "pooling": "cls",
        "normalize": bool(args.normalize),
        "window_text_table": str(args.window_text_table) if args.window_text_table else None,
    }
    outputs = [
        out_dir / "root_text_emb.npy",
        out_dir / "root_text_emb.idx.json",
        out_dir / "root_text_emb.meta.json",
        out_dir / "retweet_text_emb.npy",
        out_dir / "retweet_text_emb.idx.json",
        out_dir / "retweet_text_emb.meta.json",
    ]
    if not args.force and all(path.exists() for path in outputs) and cache_matches(out_dir, expected):
        print(f"RoBERTa text embeddings already exist and meta matches; skip: {out_dir}")
        return 0

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional preprocessing dependency.
        raise SystemExit("Missing optional dependency 'transformers'. Install it only on the preprocessing machine that runs RoBERTa encoding.") from exc

    device = resolve_device(args.device)
    print(f"Loading {args.model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    if args.window_text_table:
        root_rows, retweet_rows = read_text_rows_from_window_table(args.window_text_table)
        print(
            f"Using window text table directly: roots={len(root_rows)} retweets={len(retweet_rows)} from {args.window_text_table}",
            flush=True,
        )
        encode_rows(
            rows=root_rows,
            out_npy=out_dir / "root_text_emb.npy",
            out_idx=out_dir / "root_text_emb.idx.json",
            out_meta=out_dir / "root_text_emb.meta.json",
            model=model,
            tokenizer=tokenizer,
            device=device,
            expected_meta={**expected, "align_key": "cascade_idx", "text_type": "root"},
            max_length=args.max_length,
            batch_size=args.batch_size,
            normalize=args.normalize,
            input_source=args.window_text_table,
        )
        encode_rows(
            rows=retweet_rows,
            out_npy=out_dir / "retweet_text_emb.npy",
            out_idx=out_dir / "retweet_text_emb.idx.json",
            out_meta=out_dir / "retweet_text_emb.meta.json",
            model=model,
            tokenizer=tokenizer,
            device=device,
            expected_meta={**expected, "align_key": "tweet_idx", "text_type": "retweet"},
            max_length=args.max_length,
            batch_size=args.batch_size,
            normalize=args.normalize,
            input_source=args.window_text_table,
        )
    else:
        encode_jsonl(
            input_jsonl=text_dir / "root_text.jsonl",
            out_npy=out_dir / "root_text_emb.npy",
            out_idx=out_dir / "root_text_emb.idx.json",
            out_meta=out_dir / "root_text_emb.meta.json",
            key_field="cascade_idx",
            model=model,
            tokenizer=tokenizer,
            device=device,
            expected_meta={**expected, "align_key": "cascade_idx", "text_type": "root"},
            max_length=args.max_length,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
        encode_jsonl(
            input_jsonl=text_dir / "retweet_text.jsonl",
            out_npy=out_dir / "retweet_text_emb.npy",
            out_idx=out_dir / "retweet_text_emb.idx.json",
            out_meta=out_dir / "retweet_text_emb.meta.json",
            key_field="tweet_idx",
            model=model,
            tokenizer=tokenizer,
            device=device,
            expected_meta={**expected, "align_key": "tweet_idx", "text_type": "retweet"},
            max_length=args.max_length,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
    print(f"Wrote RoBERTa text embeddings to {out_dir}")
    return 0


def encode_jsonl(
    *,
    input_jsonl: Path,
    out_npy: Path,
    out_idx: Path,
    out_meta: Path,
    key_field: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    expected_meta: Dict[str, Any],
    max_length: int,
    batch_size: int,
    normalize: bool,
) -> None:
    rows = read_text_rows(input_jsonl, key_field)
    encode_rows(
        rows=rows,
        out_npy=out_npy,
        out_idx=out_idx,
        out_meta=out_meta,
        model=model,
        tokenizer=tokenizer,
        device=device,
        expected_meta=expected_meta,
        max_length=max_length,
        batch_size=batch_size,
        normalize=normalize,
        input_source=input_jsonl,
    )


def encode_rows(
    *,
    rows: List[Tuple[int, str]],
    out_npy: Path,
    out_idx: Path,
    out_meta: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    expected_meta: Dict[str, Any],
    max_length: int,
    batch_size: int,
    normalize: bool,
    input_source: Path,
) -> None:
    if not rows:
        raise SystemExit(f"No text rows found: {input_source}")
    rows.sort(key=lambda item: item[0])
    keys = [key for key, _ in rows]
    texts = [text for _, text in rows]
    embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"encode {expected_meta['align_key']}"):
            batch_text = texts[start : start + batch_size]
            inputs = tokenizer(batch_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)
            embs.append(emb.cpu().numpy().astype(np.float32))
    final = np.concatenate(embs, axis=0)
    np.save(out_npy, final)
    write_json(out_idx, keys)
    write_json(
        out_meta,
        {
            **expected_meta,
            "tokenizer_name": expected_meta["model_name"],
            "embedding_dim": int(final.shape[1]),
            "dim": int(final.shape[1]),
            "num_samples": len(keys),
            "input_source": str(input_source),
            "source_file_hash": sha256_file(input_source),
            "created_at": utc_now(),
        },
    )
    print(f"wrote {out_npy} shape={final.shape}", flush=True)


def read_text_rows_from_window_table(path: Path) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    root: Dict[int, str] = {}
    retweet: Dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            post_type = str(row.get("post_type", "")).lower()
            text = str(row.get("text") or "")
            if post_type == "root":
                value = row.get("cascade_idx")
                if value not in (None, "") and int(value) not in root:
                    root[int(value)] = text
            else:
                value = row.get("tweet_idx")
                if value not in (None, "") and int(value) not in retweet:
                    retweet[int(value)] = text
    return list(root.items()), list(retweet.items())


def read_text_rows(path: Path, key_field: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    if not path.exists():
        return rows
    bad_json = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if obj.get(key_field) in (None, ""):
                continue
            rows.append((int(obj[key_field]), str(obj.get("text") or "")))
    if bad_json:
        print(f"[WARN] skipped malformed JSON rows in {path}: {bad_json}", flush=True)
    return rows


def cache_matches(out_dir: Path, expected: Dict[str, Any]) -> bool:
    for name in ["root_text_emb.meta.json", "retweet_text_emb.meta.json"]:
        path = out_dir / name
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        for key, value in expected.items():
            if meta.get(key) != value:
                return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def safe_model_dir(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode raw root/retweet text with Chinese RoBERTa once and cache the embeddings.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--text-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--window-text-table", type=Path, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
