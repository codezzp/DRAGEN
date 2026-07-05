from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm


def iter_pickle_stream(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def dump_pickle_stream(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for sample in samples:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().cpu()
    else:
        out = torch.as_tensor(np.asarray(value))
    return out.to(dtype=dtype) if dtype is not None else out


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.numel() == 0:
        return x
    valid = torch.isfinite(x)
    if int(valid.sum()) == 0:
        return torch.zeros_like(x)
    xv = x[valid]
    mn = xv.min()
    mx = xv.max()
    if float(mx - mn) < 1e-8:
        return torch.zeros_like(x)
    out = (x - mn) / (mx - mn)
    return torch.where(valid, out, torch.zeros_like(out))


def get_node_mask(sample: dict[str, Any]) -> torch.Tensor:
    if "node_mask" in sample:
        return as_tensor(sample["node_mask"]).bool()
    node_x = as_tensor(sample["node_x"], torch.float32)
    return node_x.abs().sum(dim=-1) > 0


def compute_seed_score(sample: dict[str, Any]) -> torch.Tensor:
    node_x = as_tensor(sample["node_x"], torch.float32)
    node_mask = get_node_mask(sample)
    abs_mean = node_x.abs().mean(dim=-1)
    abs_max = node_x.abs().amax(dim=-1)
    score = 0.6 * minmax_norm(abs_mean) + 0.4 * minmax_norm(abs_max)
    return score.masked_fill(~node_mask, -1.0)


def coerce_edge_index(edge_index: Any) -> torch.Tensor:
    if edge_index is None:
        return torch.zeros(2, 0, dtype=torch.long)
    if isinstance(edge_index, list):
        parts = [as_tensor(x, torch.long).reshape(2, -1) for x in edge_index if x is not None and as_tensor(x).numel() > 0]
        return torch.cat(parts, dim=1) if parts else torch.zeros(2, 0, dtype=torch.long)
    arr = as_tensor(edge_index, torch.long)
    if arr.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return arr.reshape(2, -1)


def coerce_edge_weight(edge_weight: Any, edge_count: int) -> torch.Tensor:
    if edge_count <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if edge_weight is None:
        return torch.ones(edge_count, dtype=torch.float32)
    if isinstance(edge_weight, list):
        parts = [as_tensor(x, torch.float32).reshape(-1) for x in edge_weight if x is not None and as_tensor(x).numel() > 0]
        out = torch.cat(parts, dim=0) if parts else torch.ones(edge_count, dtype=torch.float32)
    else:
        out = as_tensor(edge_weight, torch.float32).reshape(-1)
    if out.numel() != edge_count:
        return torch.ones(edge_count, dtype=torch.float32)
    return out


def build_adj(edge_index: Any, edge_weight: Any, num_nodes: int, *, undirected: bool) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    edges = coerce_edge_index(edge_index)
    if edges.numel() == 0:
        return adj
    weights = coerce_edge_weight(edge_weight, edges.shape[1])
    for u, v, w in zip(edges[0].tolist(), edges[1].tolist(), weights.tolist()):
        u = int(u)
        v = int(v)
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            adj[u].append((v, float(w)))
            if undirected:
                adj[v].append((u, float(w)))
    return adj


def build_key_user_pool_for_sample(
    sample: dict[str, Any],
    *,
    max_hops: int,
    key_users_per_window: int,
    seed_budget: int,
    rho: float,
    undirected: bool,
) -> dict[str, Any]:
    node_mask = get_node_mask(sample)
    T, N = node_mask.shape
    adj = build_adj(sample.get("global_candidate_edge_index"), sample.get("global_candidate_edge_weight"), int(N), undirected=undirected)
    seed_score = compute_seed_score(sample)

    key_user_idx = torch.zeros(T, key_users_per_window, dtype=torch.long)
    key_user_weight = torch.zeros(T, key_users_per_window, dtype=torch.float32)
    key_user_hop = torch.zeros(T, key_users_per_window, dtype=torch.long)
    key_user_mask = torch.zeros(T, key_users_per_window, dtype=torch.bool)

    for t in range(T):
        visible = node_mask[t]
        visible_count = int(visible.sum().item())
        if visible_count <= 0:
            continue

        scores_t = seed_score[t].clone().masked_fill(~visible, -1.0)
        k_seed = min(int(seed_budget), visible_count)
        seed_vals, seed_nodes = torch.topk(scores_t, k=k_seed)
        seeds = [int(node) for node, val in zip(seed_nodes.tolist(), seed_vals.tolist()) if float(val) >= 0.0]
        if not seeds:
            continue

        cand_path_score: dict[int, float] = defaultdict(float)
        cand_min_hop: dict[int, int] = {}
        for s in seeds:
            cand_path_score[s] += float(max(seed_score[t, s].item(), 0.0))
            cand_min_hop[s] = 0

        frontier = set(seeds)
        visited = set(seeds)
        for hop in range(1, int(max_hops) + 1):
            next_frontier: set[int] = set()
            for u in frontier:
                for v, w in adj.get(u, []):
                    if not bool(visible[v]):
                        continue
                    cand_path_score[v] += (float(rho) ** (hop - 1)) * float(w)
                    cand_min_hop[v] = min(cand_min_hop.get(v, hop), hop)
                    if v not in visited:
                        next_frontier.add(v)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier

        final: list[tuple[int, float, int]] = []
        for u, path_score in cand_path_score.items():
            if not bool(visible[u]):
                continue
            ss = float(max(seed_score[t, u].item(), 0.0))
            degree_score = min(1.0, len(adj.get(u, [])) / 50.0)
            hop = int(cand_min_hop.get(u, max_hops))
            hop_score = 1.0 / (1.0 + hop)
            final_score = 0.45 * float(path_score) + 0.30 * ss + 0.15 * degree_score + 0.10 * hop_score
            final.append((int(u), float(final_score), hop))

        final.sort(key=lambda x: x[1], reverse=True)
        final = final[:key_users_per_window]
        if not final:
            continue

        min_score = min(x[1] for x in final)
        max_score = max(x[1] for x in final)
        denom = max(max_score - min_score, 1e-8)
        for r, (u, score, hop) in enumerate(final):
            key_user_idx[t, r] = int(u)
            key_user_weight[t, r] = float((score - min_score) / denom)
            key_user_hop[t, r] = int(hop)
            key_user_mask[t, r] = True

    sample["key_user_idx"] = key_user_idx.numpy()
    sample["key_user_weight"] = key_user_weight.numpy()
    sample["key_user_hop"] = key_user_hop.numpy()
    sample["key_user_mask"] = key_user_mask.numpy()
    return sample


def process_split(in_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    def converted() -> Iterable[dict[str, Any]]:
        for sample in tqdm(iter_pickle_stream(in_path), desc=f"key-user pool {in_path.name}"):
            yield build_key_user_pool_for_sample(
                sample,
                max_hops=args.max_hops,
                key_users_per_window=args.key_users_per_window,
                seed_budget=args.seed_budget,
                rho=args.rho,
                undirected=not args.directed,
            )

    dump_pickle_stream(out_path, converted())


def copy_meta(in_pack: Path, out_pack: Path, args: argparse.Namespace) -> None:
    for name in ["meta.json", "pack_diagnostics.json"]:
        src = in_pack / name
        dst = out_pack / name
        if not src.exists():
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            data["key_user_pool"] = {
                "enabled": True,
                "max_hops": args.max_hops,
                "key_users_per_window": args.key_users_per_window,
                "seed_budget": args.seed_budget,
                "rho": args.rho,
                "directed": args.directed,
                "fields": ["key_user_idx", "key_user_weight", "key_user_hop", "key_user_mask"],
            }
            if name == "meta.json":
                keys = list(data.get("sample_keys", []))
                for field in ["key_user_idx", "key_user_weight", "key_user_hop", "key_user_mask"]:
                    if field not in keys:
                        keys.append(field)
                data["sample_keys"] = keys
            dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add window-level key-user pools to existing DRAGEN packs.")
    parser.add_argument("--in-pack", required=True)
    parser.add_argument("--out-pack", required=True)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--key-users-per-window", type=int, default=32)
    parser.add_argument("--seed-budget", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--directed", action="store_true")
    args = parser.parse_args()

    in_pack = Path(args.in_pack)
    out_pack = Path(args.out_pack)
    out_pack.mkdir(parents=True, exist_ok=True)
    for split in ["train.pt", "valid.pt", "test.pt"]:
        process_split(in_pack / split, out_pack / split, args)
    copy_meta(in_pack, out_pack, args)
    print(f"wrote key-user pack: {out_pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
