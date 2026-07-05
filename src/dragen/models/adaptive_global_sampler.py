"""Learnable adaptive global prior sampler for DRAGEN-Full."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveGlobalSampler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 20,
        eta: Tuple[float, float, float, float] = (0.2, 0.4, 0.4, 0.1),
        sim_candidate_k: int = 20,
        sim_pool_limit: int = 512,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.sim_candidate_k = sim_candidate_k
        self.sim_pool_limit = sim_pool_limit
        self.eta_follow, self.eta_context, self.eta_sim, self.eta_distance = eta
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6 + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else 1),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else 1, 1),
        )

    def forward(
        self,
        node_repr: torch.Tensor,
        context_edges: List[torch.Tensor],
        node_mask: torch.Tensor,
        global_edges: Optional[List[torch.Tensor]] = None,
        global_edge_weights: Optional[List[torch.Tensor]] = None,
        current_edges: Optional[List[torch.Tensor]] = None,
        evidence_repr: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        adaptive: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
        B, N, H = node_repr.shape
        k = int(top_k or self.top_k)
        evidence = node_repr if evidence_repr is None else evidence_repr
        sampled_edges: List[torch.Tensor] = []
        sample_weights: List[torch.Tensor] = []
        neighbors: List[List[Dict[str, Any]]] = []
        edge_losses: List[torch.Tensor] = []
        hub_losses: List[torch.Tensor] = []
        device = node_repr.device
        dtype = node_repr.dtype

        for b in range(B):
            ctx_b = context_edges[b].to(device)
            cur_b = torch.zeros(2, 0, dtype=torch.long, device=device) if current_edges is None else current_edges[b].to(device)
            glob_b = None if global_edges is None else global_edges[b].to(device)
            glob_w = None if global_edge_weights is None else global_edge_weights[b].to(device, dtype=dtype)
            repr_b = F.normalize(node_repr[b], dim=-1)
            evidence_b = evidence[b]
            candidates = build_candidate_sets(N, ctx_b, cur_b, glob_b, repr_b, node_mask[b], self.sim_candidate_k, self.sim_pool_limit)
            degrees = candidate_degrees(N, ctx_b, cur_b, glob_b, device=device, dtype=dtype)
            edges_out: List[List[int]] = [[], []]
            weights_out: List[torch.Tensor] = []
            rows: List[Dict[str, Any]] = []

            for i in range(N):
                if not bool(node_mask[b, i]):
                    continue
                cand = [j for j in candidates[i] if j != i and bool(node_mask[b, j])]
                if not cand:
                    continue
                cand_t = torch.as_tensor(cand, device=device, dtype=torch.long)
                sim = (repr_b[i].unsqueeze(0) * repr_b[cand_t]).sum(dim=-1)
                dist = (cand_t.float() - float(i)).abs() / max(N - 1, 1)
                follow_score = edge_values(i, cand, glob_b, glob_w, dtype=dtype) if glob_b is not None else torch.zeros_like(sim)
                follow_flag = (follow_score > 0).to(dtype)
                context_flag = edge_membership(i, cand, ctx_b, dtype=dtype)
                current_flag = edge_membership(i, cand, cur_b, dtype=dtype)

                if adaptive:
                    h_i = node_repr[b, i].unsqueeze(0).expand(len(cand), H)
                    h_j = node_repr[b, cand_t]
                    e_i = evidence_b[i].unsqueeze(0).expand(len(cand), H)
                    e_j = evidence_b[cand_t]
                    pair_x = torch.cat(
                        [
                            h_i,
                            h_j,
                            e_i,
                            e_j,
                            (e_i - e_j).abs(),
                            h_i * h_j,
                            follow_flag.unsqueeze(-1),
                            context_flag.unsqueeze(-1),
                            current_flag.unsqueeze(-1),
                            sim.unsqueeze(-1),
                            torch.log1p(degrees[cand_t]).unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    score = self.score_mlp(pair_x).squeeze(-1)
                else:
                    score = self.eta_follow * follow_score + self.eta_context * context_flag + self.eta_sim * sim - self.eta_distance * dist

                pos = (follow_flag > 0) | (context_flag > 0) | (current_flag > 0)
                if adaptive and bool(pos.any()) and bool((~pos).any()):
                    labels = pos.to(dtype)
                    edge_losses.append(F.binary_cross_entropy_with_logits(score, labels))

                take = min(k, score.numel())
                top_scores, idx = torch.topk(score, take)
                alpha = torch.softmax(top_scores, dim=0)
                top_cand = cand_t[idx]
                hub_losses.append((alpha * torch.log1p(degrees[top_cand])).sum())

                for rank, (j_tensor, score_tensor, weight_tensor) in enumerate(zip(top_cand, top_scores, alpha), start=1):
                    j = int(j_tensor.detach().cpu())
                    edges_out[0].append(j)
                    edges_out[1].append(i)
                    weights_out.append(weight_tensor)
                    prior_source = source_type(i, j, glob_b, ctx_b, cur_b)
                    rows.append(
                        {
                            "target_local_node_idx": i,
                            "neighbor_local_node_idx": j,
                            "sampled_score": float(score_tensor.detach().cpu()),
                            "sampled_weight": float(weight_tensor.detach().cpu()),
                            "prior_source": prior_source,
                            "sample_rank": rank,
                            # Backward-compatible names for older readers.
                            "local_node_idx": i,
                            "sample_weight": float(weight_tensor.detach().cpu()),
                            "source_type": prior_source,
                        }
                    )

            sampled_edges.append(torch.as_tensor(edges_out, device=device, dtype=torch.long))
            if weights_out:
                sample_weights.append(torch.stack(weights_out).to(device=device, dtype=dtype))
            else:
                sample_weights.append(torch.zeros(0, device=device, dtype=dtype))
            neighbors.append(rows)

        aux = {
            "sampler_edge_loss": torch.stack(edge_losses).mean() if edge_losses else node_repr.new_tensor(0.0),
            "sampler_hub_loss": torch.stack(hub_losses).mean() if hub_losses else node_repr.new_tensor(0.0),
        }
        return sampled_edges, sample_weights, neighbors, aux


def build_candidate_sets(
    N: int,
    context_edges: torch.Tensor,
    current_edges: torch.Tensor,
    global_edges: Optional[torch.Tensor],
    repr_b: torch.Tensor,
    node_mask_b: torch.Tensor,
    sim_candidate_k: int,
    sim_pool_limit: int,
) -> List[set[int]]:
    candidates = [set() for _ in range(N)]
    for edges in [context_edges, current_edges, global_edges]:
        if edges is None or edges.numel() == 0:
            continue
        src, dst = edges[0].long().tolist(), edges[1].long().tolist()
        for s, d in zip(src, dst):
            if 0 <= s < N and 0 <= d < N:
                candidates[d].add(s)
                candidates[s].add(d)
    if sim_candidate_k > 0 and N > 1:
        active = torch.where(node_mask_b.bool())[0]
        if active.numel() > 1:
            if active.numel() > sim_pool_limit:
                pool_pos = torch.linspace(0, active.numel() - 1, steps=sim_pool_limit, device=active.device).long()
                pool = active[pool_pos]
            else:
                pool = active
            sim = repr_b.matmul(repr_b[pool].t())
            take = min(sim_candidate_k, max(pool.numel() - 1, 1))
            _, idx = torch.topk(sim, take, dim=-1)
            for i in active.tolist():
                added = 0
                for pool_idx in idx[int(i)].tolist():
                    j = int(pool[pool_idx])
                    if j != int(i) and bool(node_mask_b[j]):
                        candidates[int(i)].add(j)
                        added += 1
                    if added >= sim_candidate_k:
                        break
    return candidates


def candidate_degrees(
    N: int,
    context_edges: torch.Tensor,
    current_edges: torch.Tensor,
    global_edges: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    deg = torch.zeros(N, device=device, dtype=dtype)
    for edges in [context_edges, current_edges, global_edges]:
        if edges is None or edges.numel() == 0:
            continue
        src, dst = edges[0].long(), edges[1].long()
        valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        src, dst = src[valid], dst[valid]
        if src.numel() == 0:
            continue
        one = torch.ones_like(src, dtype=dtype, device=device)
        deg.index_add_(0, src, one)
        deg.index_add_(0, dst, one)
    return deg


def edge_membership(i: int, cand: List[int], edges: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.zeros(len(cand), device=edges.device, dtype=dtype)
    pairs = {(int(s), int(d)) for s, d in zip(edges[0].tolist(), edges[1].tolist())}
    return torch.as_tensor([1.0 if (j, i) in pairs or (i, j) in pairs else 0.0 for j in cand], device=edges.device, dtype=dtype)


def edge_values(
    i: int,
    cand: List[int],
    edges: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if edges is None or edges.numel() == 0:
        device = weights.device if weights is not None else torch.device("cpu")
        return torch.zeros(len(cand), device=device, dtype=dtype)
    if weights is None or weights.numel() != edges.shape[1]:
        weights = torch.ones(edges.shape[1], device=edges.device, dtype=dtype)
    else:
        weights = weights.to(device=edges.device, dtype=dtype)
    values: Dict[Tuple[int, int], float] = {}
    for idx, (s, d) in enumerate(zip(edges[0].tolist(), edges[1].tolist())):
        w = float(weights[idx])
        values[(int(s), int(d))] = max(values.get((int(s), int(d)), 0.0), w)
        values[(int(d), int(s))] = max(values.get((int(d), int(s)), 0.0), w)
    return torch.as_tensor([values.get((j, i), 0.0) for j in cand], device=edges.device, dtype=dtype)


def source_type(i: int, j: int, global_edges: Optional[torch.Tensor], context_edges: torch.Tensor, current_edges: torch.Tensor) -> str:
    if has_edge(i, j, current_edges):
        return "current"
    if has_edge(i, j, context_edges):
        return "context"
    if global_edges is not None and has_edge(i, j, global_edges):
        return "follow"
    return "similarity"


def has_edge(i: int, j: int, edges: torch.Tensor) -> bool:
    if edges.numel() == 0:
        return False
    for s, d in zip(edges[0].tolist(), edges[1].tolist()):
        if (int(s) == i and int(d) == j) or (int(s) == j and int(d) == i):
            return True
    return False
