"""Joint DRAGEN-Full training losses."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def dragen_full_loss(outputs: Dict[str, Any], batch: Dict[str, Any], weights: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
    y = batch["y"].float().to(outputs["event_logit"].device)
    event = event_classification_loss(outputs["event_logit"], y, weights)
    jump = temporal_jump_loss(outputs)
    struct = structure_prior_loss(outputs)
    align = coral_alignment_loss(outputs["source_evidence"], outputs["node_mask"])
    uncertainty = uncertainty_loss(outputs, y)
    role = pseudo_role_loss(outputs, batch) if weights.get("lambda_role", 0.0) > 0 else outputs["event_logit"].new_tensor(0.0)
    sampler_edge = outputs.get("sampler_edge_loss", outputs["event_logit"].new_tensor(0.0))
    sampler_hub = outputs.get("sampler_hub_loss", outputs["event_logit"].new_tensor(0.0))
    sampler_temp = sampler_temporal_loss(outputs)
    raw_losses = {
        "event": event,
        "jump": jump,
        "struct": struct,
        "align": align,
        "uncertainty": uncertainty,
        "role": role,
        "sampler_edge": sampler_edge,
        "sampler_hub": sampler_hub,
        "sampler_temp": sampler_temp,
    }
    loss_weights = {
        "event": 1.0,
        "jump": float(weights.get("lambda_jump", 0.01)),
        "struct": float(weights.get("lambda_struct", 0.005)),
        "align": float(weights.get("lambda_align", 0.001)),
        "uncertainty": float(weights.get("lambda_uncertainty", 0.001)),
        "role": float(weights.get("lambda_role", 0.0)),
        "sampler_edge": float(weights.get("lambda_sampler_edge", 0.005)),
        "sampler_hub": float(weights.get("lambda_sampler_hub", 0.001)),
        "sampler_temp": float(weights.get("lambda_sampler_temp", 0.005)),
    }
    weighted_losses = {name: raw_losses[name] * loss_weights[name] for name in raw_losses}
    total = sum(weighted_losses.values())
    total_value = float(total.detach().cpu())
    denom = max(abs(total_value), 1e-12)
    breakdown = {"loss_total": total_value}
    for name, raw in raw_losses.items():
        weighted = weighted_losses[name]
        weighted_value = float(weighted.detach().cpu())
        breakdown[f"loss_{name}"] = float(raw.detach().cpu())
        breakdown[f"loss_weight_{name}"] = loss_weights[name]
        breakdown[f"weighted_loss_{name}"] = weighted_value
        breakdown[f"loss_contribution_{name}"] = weighted_value / denom
    breakdown["loss_sampler"] = breakdown["loss_sampler_edge"] + breakdown["loss_sampler_hub"] + breakdown["loss_sampler_temp"]
    breakdown["weighted_loss_sampler"] = (
        breakdown["weighted_loss_sampler_edge"] + breakdown["weighted_loss_sampler_hub"] + breakdown["weighted_loss_sampler_temp"]
    )
    return total, breakdown


def event_classification_loss(logits: torch.Tensor, y: torch.Tensor, weights: Dict[str, Any]) -> torch.Tensor:
    loss_name = str(weights.get("event_loss", "bce") or "bce").lower()
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, y)
    if loss_name == "weighted_bce":
        pos_weight = resolve_pos_weight(y, weights.get("pos_weight", "auto")).to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    if loss_name == "focal":
        alpha = float(weights.get("focal_alpha", 0.75))
        gamma = float(weights.get("focal_gamma", 2.0))
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * y + (1.0 - prob) * (1.0 - y)
        alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
        return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()
    raise ValueError(f"Unsupported event_loss: {loss_name}")


def resolve_pos_weight(y: torch.Tensor, value: Any) -> torch.Tensor:
    return y.new_tensor(float(value))

def sampler_temporal_loss(outputs: Dict[str, Any]) -> torch.Tensor:
    g = outputs["global_prior"]
    mask = outputs["node_mask"].float()
    if g.shape[1] < 2:
        return g.new_tensor(0.0)
    pair_mask = mask[:, 1:] * mask[:, :-1]
    shock = outputs.get("shock")
    if shock is None:
        shock_weight = torch.ones_like(pair_mask)
    else:
        shock_weight = torch.exp(-shock[:, 1:])
    diff = (g[:, 1:] - g[:, :-1]).pow(2).mean(dim=-1)
    loss = shock_weight * diff * pair_mask
    return loss.sum() / pair_mask.sum().clamp_min(1.0)


def temporal_jump_loss(outputs: Dict[str, Any]) -> torch.Tensor:
    mask = outputs["node_mask"].float()
    if mask.shape[1] < 2:
        return mask.new_tensor(0.0)
    pair_mask = mask[:, 1:] * mask[:, :-1]
    shock = outputs["shock"][:, 1:]
    role_jump = (outputs["role_prob"][:, 1:] - outputs["role_prob"][:, :-1]).pow(2).mean(dim=-1)
    node_jump = (outputs["node_prob"][:, 1:] - outputs["node_prob"][:, :-1]).pow(2)
    state_jump = (outputs["manip_state"][:, 1:] - outputs["manip_state"][:, :-1]).pow(2).mean(dim=-1)
    loss = torch.exp(-shock) * (role_jump + node_jump + state_jump) * pair_mask
    return loss.sum() / pair_mask.sum().clamp_min(1.0)


def structure_prior_loss(outputs: Dict[str, Any]) -> torch.Tensor:
    g = outputs["global_prior"]
    edges_all = outputs.get("sampled_global_edges", [])
    if not edges_all:
        return g.new_tensor(0.0)
    losses = []
    T = g.shape[1]
    N = g.shape[2]
    for t in range(T):
        for b, edges in enumerate(edges_all[t]):
            edges = edges.to(g.device)
            if edges.numel() == 0:
                continue
            src, dst = edges[0].long(), edges[1].long()
            valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
            src, dst = src[valid], dst[valid]
            if src.numel() == 0:
                continue
            neg = torch.randint(0, N, dst.shape, device=g.device)
            score_pos = F.cosine_similarity(g[b, t, src], g[b, t, dst], dim=-1)
            score_neg = F.cosine_similarity(g[b, t, src], g[b, t, neg], dim=-1)
            losses.append(-F.logsigmoid(score_pos - score_neg).mean())
    return torch.stack(losses).mean() if losses else g.new_tensor(0.0)


def coral_alignment_loss(source_evidence: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    B, T, N, M, H = source_evidence.shape
    mask = node_mask.reshape(B * T * N).bool()
    if int(mask.sum()) < 2:
        return source_evidence.new_tensor(0.0)
    sources = source_evidence.reshape(B * T * N, M, H)[mask]
    covs = [covariance(sources[:, m, :]) for m in range(M)]
    losses = []
    for i in range(M):
        for j in range(i + 1, M):
            losses.append((covs[i] - covs[j]).pow(2).mean())
    return torch.stack(losses).mean() if losses else source_evidence.new_tensor(0.0)


def covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return x.t().matmul(x) / max(x.shape[0] - 1, 1)


def uncertainty_loss(outputs: Dict[str, Any], y: torch.Tensor) -> torch.Tensor:
    logits = outputs["node_logit"]
    mask = outputs["node_mask"].float()
    target = y.view(-1, 1, 1).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    log_var = outputs["uncertainty_log_var"]
    loss = (torch.exp(-log_var) * bce + log_var) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def pseudo_role_loss(outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
    node_x = batch["node_x"].to(outputs["role_prob"].device)
    mask = outputs["node_mask"]
    # Column order from feature_schema: is_root, ..., out_degree_ctx, ..., num_texts_visible, ...
    is_root = node_x[..., 0] > 0.5
    out_ctx = node_x[..., 9]
    text_visible = node_x[..., 14]
    labels = torch.full(mask.shape, 4, dtype=torch.long, device=node_x.device)
    labels = torch.where(is_root | (text_visible > text_visible.mean()), torch.zeros_like(labels), labels)
    labels = torch.where(out_ctx > torch.quantile(out_ctx[mask], 0.8) if bool(mask.any()) else out_ctx > 1e9, torch.ones_like(labels), labels)
    role_prob = outputs["role_prob"].clamp_min(1e-8)
    ce = F.nll_loss(role_prob.log().reshape(-1, role_prob.shape[-1]), labels.reshape(-1), reduction="none").reshape_as(labels)
    return (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
