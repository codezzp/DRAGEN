from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class KeyUserGlobalPrior(nn.Module):
    """Window-level key-user pool global prior.

    The module consumes a fixed-size key-user pool per window and avoids the old
    variable edge-list/index_add global branch.
    """

    def __init__(self, hidden_dim: int, max_hops: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_hops = int(max_hops)
        self.hop_emb = nn.Embedding(self.max_hops + 1, self.hidden_dim)
        self.weight_proj = nn.Linear(1, self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        node_repr: torch.Tensor,
        key_user_idx: torch.Tensor,
        key_user_weight: torch.Tensor,
        key_user_hop: torch.Tensor,
        key_user_mask: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if node_repr.ndim == 3:
            return self.forward_step(node_repr, key_user_idx, key_user_weight, key_user_hop, key_user_mask, node_mask)
        if node_repr.ndim != 4:
            raise ValueError(f"node_repr must have shape [B,N,H] or [B,T,N,H], got {tuple(node_repr.shape)}")
        B, T, N, H = node_repr.shape
        flat_node = node_repr.reshape(B * T, N, H)
        flat_mask = node_mask.reshape(B * T, N) if node_mask is not None else None
        out = self.forward_step(
            flat_node,
            key_user_idx.reshape(B * T, -1),
            key_user_weight.reshape(B * T, -1),
            key_user_hop.reshape(B * T, -1),
            key_user_mask.reshape(B * T, -1),
            flat_mask,
        )
        prior = out["global_prior"].reshape(B, T, N, H)
        attn = out["key_user_attention"].reshape(B, T, N, -1)
        return {"global_prior": prior, "key_user_attention": attn}

    def forward_step(
        self,
        node_repr: torch.Tensor,
        key_user_idx: torch.Tensor,
        key_user_weight: torch.Tensor,
        key_user_hop: torch.Tensor,
        key_user_mask: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        B, N, H = node_repr.shape
        R = int(key_user_idx.shape[-1]) if key_user_idx.ndim >= 2 else 0
        if R == 0 or N == 0:
            zero = torch.zeros_like(node_repr)
            return {"global_prior": zero, "key_user_attention": node_repr.new_zeros(B, N, 0)}

        key_user_idx = key_user_idx.long().clamp(min=0, max=max(N - 1, 0))
        key_user_weight = key_user_weight.to(device=node_repr.device, dtype=node_repr.dtype)
        key_user_hop = key_user_hop.long().clamp(min=0, max=self.max_hops).to(node_repr.device)
        key_user_mask = key_user_mask.bool().to(node_repr.device)
        key_user_idx = key_user_idx.to(node_repr.device)

        gather_idx = key_user_idx.unsqueeze(-1).expand(-1, -1, H)
        key_repr = torch.gather(node_repr, dim=1, index=gather_idx)
        key_repr = key_repr + self.hop_emb(key_user_hop) + self.weight_proj(key_user_weight.unsqueeze(-1))

        q = self.q_proj(node_repr)
        k = self.k_proj(key_repr)
        v = self.v_proj(key_repr)
        attn_score = torch.einsum("bnh,brh->bnr", q, k) / math.sqrt(float(H))
        key_mask = key_user_mask.unsqueeze(1)
        attn_score = attn_score.masked_fill(~key_mask, -1e9)
        attn = F.softmax(attn_score, dim=-1) * key_mask.to(dtype=node_repr.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prior = torch.einsum("bnr,brh->bnh", attn, v)
        out = self.norm(self.out_proj(torch.cat([node_repr, prior], dim=-1)) + node_repr)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(dtype=node_repr.dtype, device=node_repr.device)
        return {"global_prior": out, "key_user_attention": attn}
