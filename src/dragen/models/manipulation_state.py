"""Historical manipulation state accumulation."""

from __future__ import annotations

import torch
from torch import nn


class ManipulationState(nn.Module):
    def __init__(self, hidden_dim: int, window_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1 + window_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward_step(
        self,
        e_obs: torch.Tensor,
        local_role: torch.Tensor,
        global_prior: torch.Tensor,
        history_state: torch.Tensor,
        prev_state: torch.Tensor,
        shock: torch.Tensor,
        window_x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = e_obs.shape
        window = window_x.unsqueeze(1).expand(B, N, window_x.shape[-1])
        shock_u = shock.unsqueeze(-1)
        c_hat = self.candidate(torch.cat([e_obs, local_role, global_prior, history_state, shock_u, window], dim=-1))
        beta = torch.sigmoid(self.gate(torch.cat([c_hat, prev_state, shock_u], dim=-1))).squeeze(-1)
        state = beta.unsqueeze(-1) * c_hat + (1.0 - beta.unsqueeze(-1)) * prev_state
        mask = node_mask.unsqueeze(-1).float()
        state = state * mask + prev_state * (1.0 - mask)
        beta = beta * node_mask.float()
        return state, beta
