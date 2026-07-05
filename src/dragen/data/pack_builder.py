"""Build streamable event-level packs from feature, label, and edge tables."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from dragen.data.feature_schema import NODE_FEATURE_COLUMNS, WINDOW_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"

def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    feature_dir = args.feature_dir or run_dir / "features_v2" / "obs_1800_step300_multiscale_hybrid_tree"
    window_dir = args.window_dir or run_dir / "windows" / "obs_1800_step300_multiscale_hybrid_tree"
    labels_path = args.labels or run_dir / "labels" / "weak_event_labels.csv"
    global_candidate_edges = args.global_candidate_edges
    text_semantic_dir = args.text_semantic_dir
    non_text_evidence_dir = args.non_text_evidence_dir
    if text_semantic_dir is None:
        raise SystemExit("RoBERTa-only pack build requires --text-semantic-dir.")
    if not text_semantic_dir.exists():
        raise SystemExit(f"Text semantic directory does not exist: {text_semantic_dir}")
    if global_candidate_edges is None:
        default_candidate_edges = run_dir / "global_graph" / "obs_1800_step300_multiscale_hybrid_tree" / "global_candidate_edge_table.csv"
        global_candidate_edges = default_candidate_edges if default_candidate_edges.exists() else None
    out_dir = args.out_dir or run_dir / "packs" / "obs_1800_step300_multiscale_hybrid_tree_feature_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(labels_path)
    handles = {split: (out_dir / f"{split}.pt").open("wb") for split in ["train", "valid", "test"]}
    diagnostics = init_diagnostics(feature_dir, window_dir, labels_path)
    diagnostics["text_semantic_dir"] = str(text_semantic_dir) if text_semantic_dir is not None else ""
    diagnostics["non_text_evidence_dir"] = str(non_text_evidence_dir) if non_text_evidence_dir is not None else ""
    try:
        build_packs(feature_dir, window_dir, labels, handles, diagnostics, global_candidate_edges, text_semantic_dir, non_text_evidence_dir)
    finally:
        for handle in handles.values():
            handle.close()

    sample_keys = [
        "cascade_idx",
        "window_x",
        "node_x",
        "edge_index_current",
        "edge_index_context",
        "global_candidate_edge_index",
        "global_candidate_edge_weight",
        "node_mask",
        "y",
    ]
    text_semantic_dim = read_text_semantic_dim(text_semantic_dir)
    if text_semantic_dim:
        sample_keys.extend(["node_text_x", "window_text_x"])
    evidence_schema = read_evidence_schema(non_text_evidence_dir)
    if evidence_schema:
        sample_keys.extend(["node_evidence_x", "window_evidence_x"])
    meta = {
        "format": "pickle_stream",
        "reader_note": "Read records with repeated pickle.load(fp) until EOFError.",
        "sample_keys": sample_keys,
        "window_feature_columns": WINDOW_FEATURE_COLUMNS,
        "node_feature_columns": NODE_FEATURE_COLUMNS,
        "T": 6,
        "text_semantic_dim": text_semantic_dim,
        "evidence_v2_schema": evidence_schema,
    }
    write_json(out_dir / "meta.json", meta)
    finalize_diagnostics(diagnostics)
    write_json(out_dir / "pack_diagnostics.json", diagnostics)
    print(
        f"Wrote packs to {out_dir} "
        f"train={diagnostics['split_counts'].get('train', 0)} "
        f"valid={diagnostics['split_counts'].get('valid', 0)} "
        f"test={diagnostics['split_counts'].get('test', 0)}"
    )
    return 0


def read_evidence_schema(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    schema_path = path / "feature_schema.json"
    if not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text(encoding="utf-8"))


def read_text_semantic_dim(path: Optional[Path]) -> int:
    if path is None:
        return 0
    meta_path = path / "text_semantic_feature_meta.json"
    if meta_path.exists():
        try:
            return int(json.loads(meta_path.read_text(encoding="utf-8")).get("dim", 0))
        except Exception:
            return 0
    node_path = path / "node_text_features.npy"
    if node_path.exists():
        arr = np.load(node_path, mmap_mode="r")
        return int(arr.shape[1]) if arr.ndim == 2 else 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build streamable DRAGEN event packs.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--window-dir", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--global-candidate-edges", type=Path, default=None)
    parser.add_argument("--text-semantic-dir", type=Path, default=None)
    parser.add_argument("--non-text-evidence-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def read_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label_value = row.get("weak_label", row.get("label"))
            if label_value is None:
                raise ValueError(f"label row for cascade {row.get('cascade_idx')} has neither weak_label nor label")
            labels[str(row["cascade_idx"])] = {
                "weak_label": int(label_value),
                "split": row["split"],
                "weak_score": float(row["weak_score"]),
            }
    return labels


def build_packs(
    feature_dir: Path,
    window_dir: Path,
    labels: Mapping[str, Mapping[str, Any]],
    handles: Mapping[str, Any],
    diagnostics: Dict[str, Any],
    global_candidate_edges: Optional[Path] = None,
    text_semantic_dir: Optional[Path] = None,
    non_text_evidence_dir: Optional[Path] = None,
) -> None:
    window_groups = CsvGroupByCascade(feature_dir / "window_features.csv")
    node_groups = CsvGroupByCascade(feature_dir / "node_window_features.csv")
    edge_groups = CsvGroupByCascade(window_dir / "edge_window_table.csv")
    global_groups = CsvGroupByCascade(global_candidate_edges) if global_candidate_edges is not None else None
    text_semantic = TextSemanticFeatures(text_semantic_dir) if text_semantic_dir is not None else None
    evidence_v2 = EvidenceV2Features(non_text_evidence_dir) if non_text_evidence_dir is not None else None
    diagnostics["global_candidate_edges"] = str(global_candidate_edges) if global_candidate_edges is not None else ""
    try:
        for cascade_idx in sorted(labels, key=lambda x: int(x)):
            label = labels[cascade_idx]
            if int(label["weak_label"]) < 0:
                diagnostics["ignored_cascades"] += 1
                continue
            window_rows = window_groups.get(cascade_idx)
            node_rows = node_groups.get(cascade_idx)
            if not window_rows or not node_rows:
                diagnostics["skipped_missing_feature"] += 1
                continue
            edge_rows = edge_groups.get(cascade_idx)
            global_rows = global_groups.get(cascade_idx) if global_groups is not None else []
            sample = make_sample(
                cascade_idx,
                window_rows,
                node_rows,
                edge_rows,
                int(label["weak_label"]),
                global_rows,
                text_semantic=text_semantic,
                evidence_v2=evidence_v2,
            )
            split = str(label["split"])
            pickle.dump(sample, handles[split], protocol=pickle.HIGHEST_PROTOCOL)
            update_diagnostics(diagnostics, split, sample)
    finally:
        window_groups.close()
        node_groups.close()
        edge_groups.close()
        if global_groups is not None:
            global_groups.close()


def make_sample(
    cascade_idx: str,
    window_rows: List[Mapping[str, str]],
    node_rows: List[Mapping[str, str]],
    edge_rows: List[Mapping[str, str]],
    y: int,
    global_rows: Optional[List[Mapping[str, str]]] = None,
    text_semantic: Optional["TextSemanticFeatures"] = None,
    evidence_v2: Optional["EvidenceV2Features"] = None,
) -> Dict[str, Any]:
    window_rows = sorted(window_rows, key=lambda row: int(row["window_idx"]))
    window_ids = [int(row["window_idx"]) for row in window_rows]
    window_pos = {window_idx: i for i, window_idx in enumerate(window_ids)}
    T = len(window_rows)
    window_x = np.asarray([[to_float(row[col]) for col in WINDOW_FEATURE_COLUMNS] for row in window_rows], dtype=np.float32)

    users = sorted({str(row["user_idx"]) for row in node_rows}, key=lambda x: int(x))
    user_pos = {user_idx: i for i, user_idx in enumerate(users)}
    node_x = np.zeros((T, len(users), len(NODE_FEATURE_COLUMNS)), dtype=np.float32)
    node_mask = np.zeros((T, len(users)), dtype=np.bool_)
    for row in node_rows:
        t = window_pos.get(int(row["window_idx"]))
        n = user_pos.get(str(row["user_idx"]))
        if t is None or n is None:
            continue
        node_x[t, n, :] = np.asarray([to_float(row[col]) for col in NODE_FEATURE_COLUMNS], dtype=np.float32)
        node_mask[t, n] = True

    edge_current: List[np.ndarray] = [np.zeros((2, 0), dtype=np.int64) for _ in range(T)]
    edge_context: List[np.ndarray] = [np.zeros((2, 0), dtype=np.int64) for _ in range(T)]
    current_lists: List[List[List[int]]] = [[[], []] for _ in range(T)]
    context_lists: List[List[List[int]]] = [[[], []] for _ in range(T)]
    for row in edge_rows:
        t = window_pos.get(int(row["window_idx"]))
        src = user_pos.get(str(row["src_user_idx"]))
        dst = user_pos.get(str(row["dst_user_idx"]))
        if t is None or src is None or dst is None:
            continue
        target = context_lists if row.get("window_scope") == "context" else current_lists
        target[t][0].append(src)
        target[t][1].append(dst)
    for i in range(T):
        if current_lists[i][0]:
            edge_current[i] = np.asarray(current_lists[i], dtype=np.int64)
        if context_lists[i][0]:
            edge_context[i] = np.asarray(context_lists[i], dtype=np.int64)

    global_edge_index, global_edge_weight = make_global_candidate_edges(global_rows or [], user_pos)
    node_text_x, window_text_x = make_text_semantic_arrays(text_semantic, cascade_idx, window_ids, users)
    node_evidence_x, window_evidence_x = make_evidence_v2_arrays(evidence_v2, cascade_idx, window_ids, users)

    sample = {
        "cascade_idx": int(cascade_idx),
        "window_x": window_x,
        "node_x": node_x,
        "edge_index_current": edge_current,
        "edge_index_context": edge_context,
        "global_candidate_edge_index": global_edge_index,
        "global_candidate_edge_weight": global_edge_weight,
        "node_mask": node_mask,
        "y": int(y),
    }
    if node_text_x is not None and window_text_x is not None:
        sample["node_text_x"] = node_text_x
        sample["window_text_x"] = window_text_x
    if node_evidence_x is not None and window_evidence_x is not None:
        sample["node_evidence_x"] = node_evidence_x
        sample["window_evidence_x"] = window_evidence_x
    return sample


def make_global_candidate_edges(
    rows: List[Mapping[str, str]],
    user_pos: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    edges: List[List[int]] = [[], []]
    weights: List[float] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        src = parse_local_idx(row, "src", user_pos)
        dst = parse_local_idx(row, "dst", user_pos)
        if src is None or dst is None or src == dst:
            continue
        pair = (src, dst)
        if pair in seen:
            continue
        seen.add(pair)
        edges[0].append(src)
        edges[1].append(dst)
        weights.append(to_float(row.get("edge_weight", 1.0)))
    if not weights:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.asarray(edges, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def make_text_semantic_arrays(
    text_semantic: Optional["TextSemanticFeatures"],
    cascade_idx: str,
    window_ids: List[int],
    users: List[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if text_semantic is None:
        return None, None
    T = len(window_ids)
    N = len(users)
    dim = text_semantic.dim
    node_text_x = np.zeros((T, N, dim), dtype=np.float32)
    window_text_x = np.zeros((T, dim), dtype=np.float32)
    user_pos = {int(user): i for i, user in enumerate(users)}
    window_pos = {int(window_idx): i for i, window_idx in enumerate(window_ids)}
    c = int(cascade_idx)
    for (window_idx, user_idx), vec in text_semantic.node_by_cascade.get(c, {}).items():
        t = window_pos.get(int(window_idx))
        n = user_pos.get(int(user_idx))
        if t is not None and n is not None:
            node_text_x[t, n] = vec
    for window_idx, vec in text_semantic.window_by_cascade.get(c, {}).items():
        t = window_pos.get(int(window_idx))
        if t is not None:
            window_text_x[t] = vec
    return node_text_x, window_text_x


class TextSemanticFeatures:
    def __init__(self, path: Path) -> None:
        self.path = path
        node_x = np.load(path / "node_text_features.npy").astype(np.float32)
        window_x = np.load(path / "window_text_features.npy").astype(np.float32)
        node_index = json.loads((path / "node_text_feature_index.json").read_text(encoding="utf-8"))
        window_index = json.loads((path / "window_text_feature_index.json").read_text(encoding="utf-8"))
        self.dim = int(node_x.shape[1] if node_x.size else window_x.shape[1])
        self.node_by_cascade: Dict[int, Dict[tuple[int, int], np.ndarray]] = defaultdict(dict)
        self.window_by_cascade: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        for row, vec in zip(node_index, node_x):
            self.node_by_cascade[int(row["cascade_idx"])][(int(row["window_idx"]), int(row["user_idx"]))] = vec
        for row, vec in zip(window_index, window_x):
            self.window_by_cascade[int(row["cascade_idx"])][int(row["window_idx"])] = vec


def make_evidence_v2_arrays(
    evidence_v2: Optional["EvidenceV2Features"],
    cascade_idx: str,
    window_ids: List[int],
    users: List[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if evidence_v2 is None:
        return None, None
    T = len(window_ids)
    N = len(users)
    node_x = np.zeros((T, N, evidence_v2.node_dim), dtype=np.float32)
    window_x = np.zeros((T, evidence_v2.window_dim), dtype=np.float32)
    c = int(cascade_idx)
    user_pos = {int(user): i for i, user in enumerate(users)}
    window_pos = {int(window_idx): i for i, window_idx in enumerate(window_ids)}
    for (window_idx, user_idx), vec in evidence_v2.node_by_cascade.get(c, {}).items():
        t = window_pos.get(int(window_idx))
        n = user_pos.get(int(user_idx))
        if t is not None and n is not None:
            node_x[t, n] = vec
    for window_idx, vec in evidence_v2.window_by_cascade.get(c, {}).items():
        t = window_pos.get(int(window_idx))
        if t is not None:
            window_x[t] = vec
    return node_x, window_x


class EvidenceV2Features:
    def __init__(self, path: Path) -> None:
        self.path = path
        schema = json.loads((path / "feature_schema.json").read_text(encoding="utf-8"))
        self.node_columns = list(schema.get("node_evidence_columns", []))
        self.window_columns = list(schema.get("window_evidence_columns", []))
        self.node_dim = len(self.node_columns)
        self.window_dim = len(self.window_columns)
        self.schema = schema
        self.node_by_cascade: Dict[int, Dict[tuple[int, int], np.ndarray]] = defaultdict(dict)
        self.window_by_cascade: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        with (path / "node_evidence_features.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                c = int(row["cascade_idx"])
                key = (int(row["window_idx"]), int(row["user_idx"]))
                self.node_by_cascade[c][key] = np.asarray([to_float(row.get(col, 0.0)) for col in self.node_columns], dtype=np.float32)
        with (path / "window_evidence_features.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                c = int(row["cascade_idx"])
                self.window_by_cascade[c][int(row["window_idx"])] = np.asarray([to_float(row.get(col, 0.0)) for col in self.window_columns], dtype=np.float32)


def parse_local_idx(row: Mapping[str, str], side: str, user_pos: Mapping[str, int]) -> Optional[int]:
    user_key = f"{side}_user_idx"
    if row.get(user_key) not in (None, ""):
        mapped = user_pos.get(str(row[user_key]))
        if mapped is not None:
            return mapped
    local_key = f"{side}_local_idx"
    if row.get(local_key) not in (None, ""):
        return int(row[local_key])
    return None


class CsvGroupByCascade:
    def __init__(self, path: Path) -> None:
        self._file = path.open("r", encoding="utf-8-sig", newline="")
        self._reader = csv.DictReader(self._file)
        self._pending: Optional[Dict[str, str]] = None
        self._current_key: Optional[int] = None
        self._current_rows: List[Dict[str, str]] = []
        self._done = False

    def get(self, cascade_idx: str) -> List[Dict[str, str]]:
        target = int(cascade_idx)
        while not self._done and (self._current_key is None or self._current_key < target):
            self._load_next()
        if self._current_key == target:
            return self._current_rows
        return []

    def close(self) -> None:
        self._file.close()

    def _load_next(self) -> None:
        first = self._pending
        if first is None:
            try:
                first = next(self._reader)
            except StopIteration:
                self._done = True
                self._current_key = None
                self._current_rows = []
                return
        self._pending = None
        key = int(first["cascade_idx"])
        rows = [first]
        for row in self._reader:
            row_key = int(row["cascade_idx"])
            if row_key != key:
                self._pending = row
                break
            rows.append(row)
        self._current_key = key
        self._current_rows = rows


def init_diagnostics(feature_dir: Path, window_dir: Path, labels_path: Path) -> Dict[str, Any]:
    return {
        "feature_dir": str(feature_dir),
        "window_dir": str(window_dir),
        "labels": str(labels_path),
        "format": "pickle_stream",
        "split_counts": defaultdict(int),
        "split_label_counts": defaultdict(lambda: defaultdict(int)),
        "ignored_cascades": 0,
        "skipped_missing_feature": 0,
        "total_samples": 0,
        "min_T": None,
        "max_T": 0,
        "min_nodes": None,
        "max_nodes": 0,
        "empty_node_mask": 0,
        "edge_alignment_errors": 0,
        "global_candidate_alignment_errors": 0,
        "samples_with_global_candidates": 0,
        "total_global_candidate_edges": 0,
        "samples_with_text_semantic": 0,
        "text_semantic_dim": 0,
        "samples_with_evidence_v2": 0,
        "node_evidence_dim": 0,
        "window_evidence_dim": 0,
    }


def update_diagnostics(diagnostics: Dict[str, Any], split: str, sample: Mapping[str, Any]) -> None:
    T = int(sample["window_x"].shape[0])
    nodes = int(sample["node_x"].shape[1])
    diagnostics["split_counts"][split] += 1
    diagnostics["split_label_counts"][split][str(sample["y"])] += 1
    diagnostics["total_samples"] += 1
    diagnostics["min_T"] = T if diagnostics["min_T"] is None else min(diagnostics["min_T"], T)
    diagnostics["max_T"] = max(diagnostics["max_T"], T)
    diagnostics["min_nodes"] = nodes if diagnostics["min_nodes"] is None else min(diagnostics["min_nodes"], nodes)
    diagnostics["max_nodes"] = max(diagnostics["max_nodes"], nodes)
    if not bool(sample["node_mask"].any()):
        diagnostics["empty_node_mask"] += 1
    for edge_list in [sample["edge_index_current"], sample["edge_index_context"]]:
        for edge_index in edge_list:
            if edge_index.size and (edge_index.max() >= nodes or edge_index.min() < 0):
                diagnostics["edge_alignment_errors"] += 1
    if sample.get("node_text_x") is not None:
        diagnostics["samples_with_text_semantic"] += 1
        diagnostics["text_semantic_dim"] = int(sample["node_text_x"].shape[-1])
    if sample.get("node_evidence_x") is not None:
        diagnostics["samples_with_evidence_v2"] += 1
        diagnostics["node_evidence_dim"] = int(sample["node_evidence_x"].shape[-1])
        diagnostics["window_evidence_dim"] = int(sample["window_evidence_x"].shape[-1])
    global_edges = sample.get("global_candidate_edge_index")
    if global_edges is not None and global_edges.size:
        diagnostics["samples_with_global_candidates"] += 1
        diagnostics["total_global_candidate_edges"] += int(global_edges.shape[1])
        if global_edges.max() >= nodes or global_edges.min() < 0:
            diagnostics["global_candidate_alignment_errors"] += 1


def finalize_diagnostics(diagnostics: Dict[str, Any]) -> None:
    diagnostics["split_counts"] = dict(diagnostics["split_counts"])
    diagnostics["split_label_counts"] = {
        split: dict(counts) for split, counts in diagnostics["split_label_counts"].items()
    }


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
