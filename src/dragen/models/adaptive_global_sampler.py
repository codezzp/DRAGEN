"""Adaptive global prior sampler for DRAGEN-Full."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveGlobalSampler(nn.Module):
    def __init__(self, top_k: int = 20, eta: Tuple[float, float, float, float] = (0.2, 0.4, 0.4, 0.1)) -> None:
        super().__init__()
        self.top_k = top_k
        self.eta_follow, self.eta_context, self.eta_sim, self.eta_distance = eta

    def forward(
        self,
        node_repr: torch.Tensor,
        context_edges: List[torch.Tensor],
        node_mask: torch.Tensor,
        global_edges: Optional[List[torch.Tensor]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[Dict[str, Any]]]]:
        B, N, _ = node_repr.shape
        k = int(top_k or self.top_k)
        sampled_edges: List[torch.Tensor] = []
        sample_weights: List[torch.Tensor] = []
        neighbors: List[List[Dict[str, Any]]] = []
        for b in range(B):
            candidates = build_candidate_sets(N, context_edges[b], None if global_edges is None else global_edges[b])
            edges_out: List[List[int]] = [[], []]
            weights_out: List[float] = []
            rows: List[Dict[str, Any]] = []
            repr_b = F.normalize(node_repr[b], dim=-1)
            for i in range(N):
                if not bool(node_mask[b, i]):
                    continue
                cand = [j for j in candidates[i] if j != i and bool(node_mask[b, j])]
                if not cand:
                    continue
                cand_t = torch.as_tensor(cand, device=node_repr.device, dtype=torch.long)
                sim = (repr_b[i].unsqueeze(0) * repr_b[cand_t]).sum(dim=-1)
                dist = (cand_t.float() - float(i)).abs() / max(N - 1, 1)
                context_flag = edge_membership(i, cand, context_edges[b].to(node_repr.device))
                global_flag = (
                    torch.zeros_like(sim)
                    if global_edges is None
                    else edge_membership(i, cand, global_edges[b].to(node_repr.device))
                )
                score = self.eta_follow * global_flag + self.eta_context * context_flag + self.eta_sim * sim - self.eta_distance * dist
                weights = torch.softmax(score, dim=0)
                take = min(k, weights.numel())
                vals, idx = torch.topk(weights, take)
                for val, local_idx in zip(vals.tolist(), idx.tolist()):
                    j = cand[local_idx]
                    edges_out[0].append(j)
                    edges_out[1].append(i)
                    weights_out.append(float(val))
                    rows.append(
                        {
                            "local_node_idx": i,
                            "neighbor_local_node_idx": j,
                            "sample_weight": float(val),
                            "source_type": "global" if global_edges is not None else "context_fallback",
                        }
                    )
            sampled_edges.append(torch.as_tensor(edges_out, device=node_repr.device, dtype=torch.long))
            sample_weights.append(torch.as_tensor(weights_out, device=node_repr.device, dtype=node_repr.dtype))
            neighbors.append(rows)
        return sampled_edges, sample_weights, neighbors


def build_candidate_sets(N: int, context_edges: torch.Tensor, global_edges: Optional[torch.Tensor]) -> List[set[int]]:
    candidates = [set() for _ in range(N)]
    for edges in [context_edges, global_edges]:
        if edges is None or edges.numel() == 0:
            continue
        src, dst = edges[0].long().tolist(), edges[1].long().tolist()
        for s, d in zip(src, dst):
            if 0 <= s < N and 0 <= d < N:
                candidates[d].add(s)
                candidates[s].add(d)
    return candidates


def edge_membership(i: int, cand: List[int], edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.zeros(len(cand), device=edges.device)
    pairs = {(int(s), int(d)) for s, d in zip(edges[0].tolist(), edges[1].tolist())}
    return torch.as_tensor([1.0 if (j, i) in pairs or (i, j) in pairs else 0.0 for j in cand], device=edges.device)
