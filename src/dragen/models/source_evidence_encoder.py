"""Multi-source abnormal evidence encoder for DRAGEN-Full."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from dragen.data.feature_schema import FeatureSchema, SOURCE_NAMES


class SourceEvidenceEncoder(nn.Module):
    def __init__(self, schema: FeatureSchema, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.schema = schema
        self.source_names = SOURCE_NAMES
        self.group_indices: Dict[str, List[int]] = {name: schema.group_indices(name) for name in self.source_names}
        self.encoders = nn.ModuleDict()
        for name in self.source_names:
            in_dim = max(len(self.group_indices[name]), 1)
            self.encoders[name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, node_x: torch.Tensor) -> torch.Tensor:
        """Encode node features into [B, T, N, M, H]."""
        outputs = []
        for name in self.source_names:
            indices = self.group_indices[name]
            if indices:
                x = node_x[..., indices]
            else:
                x = torch.zeros(*node_x.shape[:-1], 1, device=node_x.device, dtype=node_x.dtype)
            outputs.append(self.encoders[name](x))
        return torch.stack(outputs, dim=3)
