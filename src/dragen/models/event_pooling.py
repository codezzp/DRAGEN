"""Event-level attention pooling for node-window outputs."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EventPooling(nn.Module):
    def __init__(self, hidden_dim: int, window_dim: int, role_num: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.unit_dim = 2 + role_num + hidden_dim * 2 + window_dim
        self.attn = nn.Sequential(nn.Linear(self.unit_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.event_logit = nn.Linear(self.unit_dim, 1)
        self.event_strength = nn.Linear(self.unit_dim, 1)

    def forward(
        self,
        node_prob: torch.Tensor,
        node_strength: torch.Tensor,
        role_prob: torch.Tensor,
        history_state: torch.Tensor,
        manip_state: torch.Tensor,
        window_x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T, N = node_prob.shape
        window = window_x.unsqueeze(2).expand(B, T, N, window_x.shape[-1])
        unit = torch.cat(
            [node_prob.unsqueeze(-1), node_strength.unsqueeze(-1), role_prob, history_state, manip_state, window],
            dim=-1,
        )
        raw = self.attn(unit).squeeze(-1).masked_fill(~node_mask, -1e9)
        attention = torch.softmax(raw.reshape(B, T * N), dim=-1).reshape(B, T, N)
        z_event = (unit * attention.unsqueeze(-1)).sum(dim=(1, 2))
        event_logit = self.event_logit(z_event).squeeze(-1)
        return {
            "event_logit": event_logit,
            "event_prob": torch.sigmoid(event_logit),
            "event_strength": F.softplus(self.event_strength(z_event).squeeze(-1)),
            "event_attention": attention,
        }
