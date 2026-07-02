"""Multi-source abnormal evidence encoder for DRAGEN-Full."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from dragen.data.feature_schema import FeatureSchema, SOURCE_NAMES


class SourceEvidenceEncoder(nn.Module):
    def __init__(self, schema: FeatureSchema, hidden_dim: int, dropout: float = 0.1, text_semantic_dim: int = 64) -> None:
        super().__init__()
        self.schema = schema
        self.source_names = SOURCE_NAMES
        self.group_indices: Dict[str, List[int]] = {name: schema.group_indices(name) for name in self.source_names}
        self.encoders = nn.ModuleDict()
        self.text_semantic_dim = int(text_semantic_dim or 0)
        for name in self.source_names:
            in_dim = max(len(self.group_indices[name]), 1)
            self.encoders[name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.text_semantic_encoder = (
            nn.Sequential(
                nn.Linear(self.text_semantic_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.text_semantic_dim > 0
            else None
        )

    def forward(self, node_x: torch.Tensor, node_text_x: torch.Tensor | None = None) -> torch.Tensor:
        """Encode node features into [B, T, N, M, H]."""
        outputs = []
        for name in self.source_names:
            indices = self.group_indices[name]
            if indices:
                x = node_x[..., indices]
            else:
                x = torch.zeros(*node_x.shape[:-1], 1, device=node_x.device, dtype=node_x.dtype)
            encoded = self.encoders[name](x)
            if name == "text" and node_text_x is not None and self.text_semantic_encoder is not None:
                encoded = encoded + self.text_semantic_encoder(node_text_x.to(device=node_x.device, dtype=node_x.dtype))
            outputs.append(encoded)
        return torch.stack(outputs, dim=3)
