"""Prior-observation Bayesian gate."""

from __future__ import annotations

import torch
from torch import nn


class BayesianGate(nn.Module):
    def __init__(self, hidden_dim: int, window_dim: int, role_num: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.prior = mlp(hidden_dim * 3 + window_dim, 1, hidden_dim, dropout)
        self.obs = mlp(hidden_dim * 2 + role_num + 1, 1, hidden_dim, dropout)
        self.gate = mlp(hidden_dim * 5 + window_dim + 1, 1, hidden_dim, dropout)
        self.uncertainty = mlp(hidden_dim * 5 + window_dim + 1, 1, hidden_dim, dropout)
        self.direct = mlp(hidden_dim * 5 + window_dim + role_num + 1, 1, hidden_dim, dropout)

    def forward(
        self,
        e_obs: torch.Tensor,
        local_role: torch.Tensor,
        global_prior: torch.Tensor,
        history_prev: torch.Tensor,
        manip_prev: torch.Tensor,
        role_prob: torch.Tensor,
        shock: torch.Tensor,
        window_x: torch.Tensor,
        use_gate: bool = True,
        use_uncertainty: bool = True,
    ) -> dict[str, torch.Tensor]:
        B, N, _ = e_obs.shape
        window = window_x.unsqueeze(1).expand(B, N, window_x.shape[-1])
        shock_u = shock.unsqueeze(-1)
        prior_logit = self.prior(torch.cat([global_prior, manip_prev, history_prev, window], dim=-1)).squeeze(-1)
        obs_logit = self.obs(torch.cat([e_obs, local_role, role_prob, shock_u], dim=-1)).squeeze(-1)
        gate_input = torch.cat([e_obs, local_role, global_prior, history_prev, manip_prev, shock_u, window], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        if use_gate:
            node_logit = alpha * obs_logit + (1.0 - alpha) * prior_logit
        else:
            node_logit = self.direct(torch.cat([gate_input, role_prob], dim=-1)).squeeze(-1)
            alpha = torch.ones_like(node_logit)
        log_var = self.uncertainty(gate_input).squeeze(-1).clamp(-6.0, 3.0) if use_uncertainty else torch.zeros_like(node_logit)
        return {
            "node_logit": node_logit,
            "gate_obs_weight": alpha,
            "gate_prior_weight": 1.0 - alpha,
            "uncertainty_log_var": log_var,
        }


def mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
