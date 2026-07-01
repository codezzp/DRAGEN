"""Cross-window node memory."""

from __future__ import annotations

import torch
from torch import nn


class TemporalMemory(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(hidden_dim * 3, hidden_dim)

    def forward_step(
        self,
        local_role: torch.Tensor,
        global_prior: torch.Tensor,
        e_obs: torch.Tensor,
        prev_state: torch.Tensor,
        node_mask: torch.Tensor,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return torch.tanh(local_role + global_prior + e_obs) * node_mask.unsqueeze(-1).float()
        B, N, H = prev_state.shape
        x = torch.cat([local_role, global_prior, e_obs], dim=-1).reshape(B * N, H * 3)
        prev = prev_state.reshape(B * N, H)
        updated = self.cell(x, prev).reshape(B, N, H)
        mask = node_mask.unsqueeze(-1).float()
        return updated * mask + prev_state * (1.0 - mask)
