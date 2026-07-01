"""Sliding-window local role encoder based on context GraphSAGE."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class LocalRoleEncoder(nn.Module):
    def __init__(self, hidden_dim: int, window_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(hidden_dim + window_dim + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        e_local_t: torch.Tensor,
        window_x_t: torch.Tensor,
        edge_index_context_t: List[torch.Tensor],
        node_mask_t: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = e_local_t.shape
        pos = torch.linspace(0.0, 1.0, N, device=e_local_t.device).view(1, N, 1).expand(B, N, 1)
        window = window_x_t.unsqueeze(1).expand(B, N, window_x_t.shape[-1])
        h = self.input(torch.cat([e_local_t, window, pos], dim=-1))
        for layer, norm in zip(self.layers, self.norms):
            neigh = mean_aggregate(h, edge_index_context_t)
            h = norm(torch.relu(layer(torch.cat([h, neigh], dim=-1))) + h)
            h = h * node_mask_t.unsqueeze(-1).float()
        return h


def mean_aggregate(h: torch.Tensor, edge_batch: List[torch.Tensor]) -> torch.Tensor:
    B, N, H = h.shape
    out = torch.zeros_like(h)
    deg = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
    for b, edges in enumerate(edge_batch):
        edges = edges.to(h.device)
        if edges.numel() == 0:
            continue
        src, dst = edges[0].long(), edges[1].long()
        valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        src, dst = src[valid], dst[valid]
        if src.numel() == 0:
            continue
        out[b].index_add_(0, dst, h[b, src])
        deg[b].index_add_(0, dst, torch.ones(src.numel(), 1, device=h.device, dtype=h.dtype))
    return out / deg.clamp_min(1.0)
