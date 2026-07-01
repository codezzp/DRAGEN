"""Global structure prior encoder."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class GlobalPriorEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        e_obs_t: torch.Tensor,
        sampled_edges: List[torch.Tensor],
        sample_weights: List[torch.Tensor],
        node_mask_t: torch.Tensor,
    ) -> torch.Tensor:
        agg = weighted_aggregate(e_obs_t, sampled_edges, sample_weights)
        out = self.norm(self.update(torch.cat([e_obs_t, agg], dim=-1)) + e_obs_t)
        return out * node_mask_t.unsqueeze(-1).float()


def weighted_aggregate(h: torch.Tensor, edge_batch: List[torch.Tensor], weight_batch: List[torch.Tensor]) -> torch.Tensor:
    B, N, H = h.shape
    out = torch.zeros_like(h)
    deg = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
    for b, edges in enumerate(edge_batch):
        edges = edges.to(h.device)
        weights = weight_batch[b].to(h.device).to(h.dtype)
        if edges.numel() == 0 or weights.numel() == 0:
            continue
        src, dst = edges[0].long(), edges[1].long()
        valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        src, dst, weights = src[valid], dst[valid], weights[valid]
        if src.numel() == 0:
            continue
        msg = h[b, src] * weights.unsqueeze(-1)
        out[b].index_add_(0, dst, msg)
        deg[b].index_add_(0, dst, weights.unsqueeze(-1))
    return out / deg.clamp_min(1e-6)
