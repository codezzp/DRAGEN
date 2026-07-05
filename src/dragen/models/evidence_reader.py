"""Selective source evidence reader."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn

from dragen.data.feature_schema import SOURCE_NAMES


class EvidenceReader(nn.Module):
    MODES: Dict[str, List[str]] = {
        "local_role": ["text", "behavior", "structure"],
        "gate_obs": ["text", "emotion", "behavior", "structure"],
        "shock": ["text", "emotion", "behavior", "structure"],
    }

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.source_to_idx = {name: i for i, name in enumerate(SOURCE_NAMES)}
        self.mode_weights = nn.ParameterDict(
            {mode: nn.Parameter(torch.zeros(len(names))) for mode, names in self.MODES.items()}
        )
        self.proj = nn.ModuleDict({mode: nn.Linear(hidden_dim, hidden_dim) for mode in self.MODES})

    def forward(self, source_evidence: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in self.MODES:
            raise ValueError(f"unknown evidence read mode: {mode}")
        names = self.MODES[mode]
        idx = torch.as_tensor([self.source_to_idx[name] for name in names], device=source_evidence.device)
        selected = source_evidence.index_select(dim=3, index=idx)
        weights = torch.softmax(self.mode_weights[mode], dim=0).view(1, 1, 1, len(names), 1)
        return self.proj[mode]((selected * weights).sum(dim=3))
