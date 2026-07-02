"""DRAGEN-Full model aligned with thesis Chapter 4."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from dragen.data.feature_schema import DEFAULT_SCHEMA, ROLE_NAMES
from dragen.models.adaptive_global_sampler import AdaptiveGlobalSampler
from dragen.models.bayesian_gate import BayesianGate
from dragen.models.evidence_reader import EvidenceReader
from dragen.models.event_pooling import EventPooling
from dragen.models.global_prior_encoder import GlobalPriorEncoder
from dragen.models.local_role_encoder import LocalRoleEncoder
from dragen.models.manipulation_state import ManipulationState
from dragen.models.source_evidence_encoder import SourceEvidenceEncoder
from dragen.models.temporal_memory import TemporalMemory


class DRAGENFull(nn.Module):
    role_names = ROLE_NAMES

    def __init__(
        self,
        node_input_dim: int = 27,
        window_input_dim: int = 12,
        hidden_dim: int = 64,
        role_num: int = 5,
        top_k_global: int = 20,
        dropout: float = 0.1,
        use_global_prior: bool = True,
        use_adaptive_sampler: bool = True,
        use_memory: bool = True,
        use_gate: bool = True,
        use_uncertainty: bool = True,
        use_role: bool = True,
    ) -> None:
        super().__init__()
        if role_num != len(ROLE_NAMES):
            raise ValueError(f"DRAGEN-Full role_num must be {len(ROLE_NAMES)}")
        if node_input_dim != DEFAULT_SCHEMA.node_input_dim or window_input_dim != DEFAULT_SCHEMA.window_input_dim:
            raise ValueError("input dimensions do not match fixed pack schema")
        self.hidden_dim = hidden_dim
        self.role_num = role_num
        self.top_k_global = top_k_global
        self.use_global_prior = use_global_prior
        self.use_adaptive_sampler = use_adaptive_sampler
        self.use_memory = use_memory
        self.use_gate = use_gate
        self.use_uncertainty = use_uncertainty
        self.use_role = use_role

        self.source_encoder = SourceEvidenceEncoder(DEFAULT_SCHEMA, hidden_dim, dropout)
        self.reader = EvidenceReader(hidden_dim)
        self.local_role_encoder = LocalRoleEncoder(hidden_dim, window_input_dim, dropout=dropout)
        self.sampler = AdaptiveGlobalSampler(hidden_dim=hidden_dim, top_k=top_k_global)
        self.global_encoder = GlobalPriorEncoder(hidden_dim, dropout)
        self.memory = TemporalMemory(hidden_dim)
        self.manipulation_state = ManipulationState(hidden_dim, window_input_dim, dropout)
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + window_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, role_num),
        )
        self.shock_head = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Softplus())
        self.gate = BayesianGate(hidden_dim, window_input_dim, role_num, dropout)
        self.strength = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.event_pooling = EventPooling(hidden_dim, window_input_dim, role_num, dropout)

    def forward(self, batch: Dict[str, Any], global_edge_index: Optional[Any] = None) -> Dict[str, Any]:
        node_x = batch["node_x"]
        window_x = batch["window_x"]
        node_mask = batch["node_mask"].bool()
        edge_context = batch["edge_index_context"]
        edge_current = batch.get("edge_index_current", edge_context)
        global_edges = global_edge_index if global_edge_index is not None else batch.get("global_candidate_edge_index")
        global_edge_weights = batch.get("global_candidate_edge_weight")
        B, T, N, _ = node_x.shape
        H = self.hidden_dim

        source_evidence = self.source_encoder(node_x)
        e_local = self.reader(source_evidence, "local_role")
        e_obs = self.reader(source_evidence, "gate_obs")

        local_steps: List[torch.Tensor] = []
        global_steps: List[torch.Tensor] = []
        history_steps: List[torch.Tensor] = []
        manip_steps: List[torch.Tensor] = []
        role_steps: List[torch.Tensor] = []
        shock_steps: List[torch.Tensor] = []
        node_logit_steps: List[torch.Tensor] = []
        node_strength_steps: List[torch.Tensor] = []
        gate_obs_steps: List[torch.Tensor] = []
        gate_prior_steps: List[torch.Tensor] = []
        log_var_steps: List[torch.Tensor] = []
        beta_steps: List[torch.Tensor] = []
        sampled_neighbors: List[List[Dict[str, Any]]] = []
        sampled_edges_all: List[List[torch.Tensor]] = []
        sampled_weights_all: List[List[torch.Tensor]] = []
        sampler_edge_losses: List[torch.Tensor] = []
        sampler_hub_losses: List[torch.Tensor] = []

        history_prev = torch.zeros(B, N, H, device=node_x.device)
        manip_prev = torch.zeros(B, N, H, device=node_x.device)
        prev_source = None
        prev_role = None
        prev_global = None
        prev_window = None

        for t in range(T):
            local_t = self.local_role_encoder(e_local[:, t], window_x[:, t], [edge_context[b][t] for b in range(B)], node_mask[:, t])
            if self.use_global_prior:
                sampled_edges, sample_weights, neigh, sampler_aux = self.sampler(
                    e_obs[:, t],
                    [edge_context[b][t] for b in range(B)],
                    node_mask[:, t],
                    global_edges=global_edges,
                    global_edge_weights=global_edge_weights,
                    current_edges=[edge_current[b][t] for b in range(B)],
                    evidence_repr=e_obs[:, t],
                    top_k=self.top_k_global,
                    adaptive=self.use_adaptive_sampler,
                )
                sampler_edge_losses.append(sampler_aux["sampler_edge_loss"])
                sampler_hub_losses.append(sampler_aux["sampler_hub_loss"])
                global_t = self.global_encoder(e_obs[:, t], sampled_edges, sample_weights, node_mask[:, t])
            else:
                sampled_edges = [torch.zeros(2, 0, dtype=torch.long, device=node_x.device) for _ in range(B)]
                sample_weights = [torch.zeros(0, device=node_x.device) for _ in range(B)]
                neigh = [[] for _ in range(B)]
                sampler_edge_losses.append(node_x.new_tensor(0.0))
                sampler_hub_losses.append(node_x.new_tensor(0.0))
                global_t = torch.zeros_like(local_t)
            history_t = self.memory.forward_step(local_t, global_t, e_obs[:, t], history_prev, node_mask[:, t], self.use_memory)
            role_prob_t = self._role_prob(local_t, history_t, global_t, window_x[:, t], node_mask[:, t])
            if not self.use_role:
                role_prob_t = torch.zeros_like(role_prob_t)
                role_prob_t[..., ROLE_NAMES.index("ordinary")] = node_mask[:, t].float()
            shock_t = self._compute_shock(
                source_evidence[:, t],
                prev_source,
                role_prob_t,
                prev_role,
                global_t,
                prev_global,
                window_x[:, t],
                prev_window,
                node_mask[:, t],
            )
            manip_t, beta_t = self.manipulation_state.forward_step(
                e_obs[:, t], local_t, global_t, history_t, manip_prev, shock_t, window_x[:, t], node_mask[:, t]
            )
            gate_out = self.gate(
                e_obs[:, t],
                local_t,
                global_t,
                history_prev,
                manip_prev,
                role_prob_t,
                shock_t,
                window_x[:, t],
                use_gate=self.use_gate,
                use_uncertainty=self.use_uncertainty,
            )
            strength_t = F.softplus(
                self.strength(torch.cat([e_obs[:, t], local_t, history_t, global_t, manip_t, shock_t.unsqueeze(-1)], dim=-1)).squeeze(-1)
            )
            local_steps.append(local_t)
            global_steps.append(global_t)
            history_steps.append(history_t)
            manip_steps.append(manip_t)
            role_steps.append(role_prob_t)
            shock_steps.append(shock_t)
            node_logit_steps.append(gate_out["node_logit"] * node_mask[:, t].float())
            node_strength_steps.append(strength_t * node_mask[:, t].float())
            gate_obs_steps.append(gate_out["gate_obs_weight"] * node_mask[:, t].float())
            gate_prior_steps.append(gate_out["gate_prior_weight"] * node_mask[:, t].float())
            log_var_steps.append(gate_out["uncertainty_log_var"] * node_mask[:, t].float())
            beta_steps.append(beta_t)
            sampled_neighbors.append(neigh)
            sampled_edges_all.append(sampled_edges)
            sampled_weights_all.append(sample_weights)
            history_prev = history_t
            manip_prev = manip_t
            prev_source = source_evidence[:, t]
            prev_role = role_prob_t
            prev_global = global_t
            prev_window = window_x[:, t]

        local_role_repr = torch.stack(local_steps, dim=1)
        global_prior = torch.stack(global_steps, dim=1)
        history_state = torch.stack(history_steps, dim=1)
        manip_state = torch.stack(manip_steps, dim=1)
        role_prob = torch.stack(role_steps, dim=1)
        shock = torch.stack(shock_steps, dim=1)
        node_logit = torch.stack(node_logit_steps, dim=1)
        node_prob = torch.sigmoid(node_logit) * node_mask.float()
        node_strength = torch.stack(node_strength_steps, dim=1)
        gate_obs_weight = torch.stack(gate_obs_steps, dim=1)
        gate_prior_weight = torch.stack(gate_prior_steps, dim=1)
        uncertainty_log_var = torch.stack(log_var_steps, dim=1)
        state_update_gate = torch.stack(beta_steps, dim=1)
        pooled = self.event_pooling(node_prob, node_strength, role_prob, history_state, manip_state, window_x, node_mask)
        return {
            **pooled,
            "node_logit": node_logit,
            "node_prob": node_prob,
            "node_strength": node_strength,
            "source_evidence": source_evidence,
            "local_role_repr": local_role_repr,
            "global_prior": global_prior,
            "history_state": history_state,
            "manip_state": manip_state,
            "state_update_gate": state_update_gate,
            "role_prob": role_prob,
            "dominant_role": role_prob.argmax(dim=-1),
            "shock": shock,
            "gate_obs_weight": gate_obs_weight,
            "gate_prior_weight": gate_prior_weight,
            "uncertainty_log_var": uncertainty_log_var,
            "sampled_global_edges": sampled_edges_all,
            "sampled_global_weights": sampled_weights_all,
            "sampled_global_neighbors": sampled_neighbors,
            "sampler_edge_loss": torch.stack(sampler_edge_losses).mean() if sampler_edge_losses else node_x.new_tensor(0.0),
            "sampler_hub_loss": torch.stack(sampler_hub_losses).mean() if sampler_hub_losses else node_x.new_tensor(0.0),
            "node_mask": node_mask,
        }

    def _role_prob(
        self,
        local_t: torch.Tensor,
        history_t: torch.Tensor,
        global_t: torch.Tensor,
        window_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = local_t.shape
        window = window_t.unsqueeze(1).expand(B, N, window_t.shape[-1])
        logits = self.role_head(torch.cat([local_t, history_t, global_t, window], dim=-1))
        prob = torch.softmax(logits, dim=-1)
        ordinary = torch.zeros_like(prob)
        ordinary[..., ROLE_NAMES.index("ordinary")] = 1.0
        return prob * mask_t.unsqueeze(-1).float() + ordinary * (~mask_t).unsqueeze(-1).float()

    def _compute_shock(
        self,
        source_t: torch.Tensor,
        prev_source: Optional[torch.Tensor],
        role_t: Optional[torch.Tensor],
        prev_role: Optional[torch.Tensor],
        global_t: torch.Tensor,
        prev_global: Optional[torch.Tensor],
        window_t: torch.Tensor,
        prev_window: Optional[torch.Tensor],
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        if prev_source is None or prev_global is None or prev_window is None:
            return torch.zeros(source_t.shape[0], source_t.shape[1], device=source_t.device)
        delta_e = (source_t - prev_source).pow(2).mean(dim=(2, 3)).sqrt()
        delta_role = torch.zeros_like(delta_e) if role_t is None or prev_role is None else (role_t - prev_role).pow(2).mean(dim=-1).sqrt()
        delta_stage = (window_t - prev_window).pow(2).mean(dim=-1).sqrt().unsqueeze(-1).expand_as(delta_e)
        delta_struct = (global_t - prev_global).pow(2).mean(dim=-1).sqrt()
        x = torch.stack([delta_e, delta_role, delta_stage, delta_struct], dim=-1)
        return self.shock_head(x).squeeze(-1) * mask_t.float()
