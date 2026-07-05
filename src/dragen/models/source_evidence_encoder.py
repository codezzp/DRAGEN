"""Multi-source abnormal evidence encoder for DRAGEN-Full."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from dragen.data.feature_schema import FeatureSchema, SOURCE_NAMES
from dragen.features.evidence_schema import DEFAULT_EVIDENCE_SCHEMA


class EvidenceBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max(input_dim, 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        if self.text_semantic_dim <= 0:
            raise ValueError("RoBERTa-only SourceEvidenceEncoder requires text_semantic_dim > 0.")
        self.text_semantic_encoder = EvidenceBranch(self.text_semantic_dim, hidden_dim, dropout)

        self.evidence_v2_slices = DEFAULT_EVIDENCE_SCHEMA.node_slices()
        self.evidence_v2_branches = nn.ModuleDict(
            {
                name: EvidenceBranch(len(indices), hidden_dim, dropout)
                for name, indices in self.evidence_v2_slices.items()
            }
        )
        self.evidence_v2_gate = nn.Sequential(
            nn.Linear(hidden_dim * len(self.evidence_v2_branches), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.evidence_v2_branches)),
        )
        self.evidence_v2_fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(
        self,
        node_x: torch.Tensor,
        node_text_x: torch.Tensor | None = None,
        node_evidence_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode node features into [B, T, N, M, H]."""
        outputs = []
        for name in self.source_names:
            indices = self.group_indices[name]
            if indices:
                x = node_x[..., indices]
            else:
                x = torch.zeros(*node_x.shape[:-1], 1, device=node_x.device, dtype=node_x.dtype)
            encoded = self.encoders[name](x)
            if name == "text":
                if node_text_x is None:
                    raise ValueError("RoBERTa-only SourceEvidenceEncoder requires node_text_x.")
                encoded = encoded + self.text_semantic_encoder(node_text_x.to(device=node_x.device, dtype=node_x.dtype))
            outputs.append(encoded)

        if node_evidence_x is not None:
            evidence = node_evidence_x.to(device=node_x.device, dtype=node_x.dtype)
            reps = []
            for group, indices in self.evidence_v2_slices.items():
                if indices:
                    reps.append(self.evidence_v2_branches[group](evidence[..., indices]))
            if reps:
                stacked = torch.stack(reps, dim=-2)
                gate = torch.softmax(self.evidence_v2_gate(torch.cat(reps, dim=-1)), dim=-1)
                fused = self.evidence_v2_fusion((stacked * gate.unsqueeze(-1)).sum(dim=-2))
                group_rep = {name: rep for name, rep in zip(self.evidence_v2_slices.keys(), reps)}
                name_to_pos = {name: i for i, name in enumerate(self.source_names)}
                if "behavior" in name_to_pos:
                    outputs[name_to_pos["behavior"]] = outputs[name_to_pos["behavior"]] + group_rep.get("behavior", 0) + group_rep.get("temporal", 0) + 0.5 * group_rep.get("coordination", 0)
                if "structure" in name_to_pos:
                    outputs[name_to_pos["structure"]] = outputs[name_to_pos["structure"]] + group_rep.get("structure", 0) + group_rep.get("global_relation", 0) + 0.5 * group_rep.get("coordination", 0)
                if "emotion" in name_to_pos:
                    outputs[name_to_pos["emotion"]] = outputs[name_to_pos["emotion"]] + 0.25 * fused
        return torch.stack(outputs, dim=3)
