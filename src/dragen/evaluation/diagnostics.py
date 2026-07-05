"""Post-training diagnostic metrics for DRAGEN-Full prediction exports."""

from __future__ import annotations

import csv
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


ROLE_NAMES = ("producer", "amplifier", "suppressor", "reframer", "ordinary")


def compute_temporal_stability(path: Path) -> Dict[str, float]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("cascade_idx", ""), row.get("local_node_idx", ""))].append(row)

    prob_jumps: list[float] = []
    strength_jumps: list[float] = []
    role_changes = 0
    role_pairs = 0
    shock_weighted: list[float] = []
    high_risk_both = 0
    high_risk_union = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda r: to_int(r.get("window_idx"), 0))
        for prev, cur in zip(ordered, ordered[1:]):
            p_prev = to_float(prev.get("node_prob"))
            p_cur = to_float(cur.get("node_prob"))
            s_prev = to_float(prev.get("node_strength"))
            s_cur = to_float(cur.get("node_strength"))
            prob_jump = abs(p_cur - p_prev)
            prob_jumps.append(prob_jump)
            strength_jumps.append(abs(s_cur - s_prev))
            shock_weighted.append(prob_jump * math.exp(-max(to_float(cur.get("shock")), 0.0)))
            if prev.get("dominant_role") and cur.get("dominant_role"):
                role_pairs += 1
                role_changes += int(prev.get("dominant_role") != cur.get("dominant_role"))
            prev_high = p_prev >= 0.5
            cur_high = p_cur >= 0.5
            if prev_high or cur_high:
                high_risk_union += 1
                high_risk_both += int(prev_high and cur_high)

    return {
        "prob_jump_mean": mean(prob_jumps),
        "prob_jump_std": std(prob_jumps),
        "strength_jump_mean": mean(strength_jumps),
        "role_transition_rate": role_changes / max(role_pairs, 1),
        "shock_weighted_jump": mean(shock_weighted),
        "high_risk_persistence": high_risk_both / max(high_risk_union, 1),
    }


def compute_role_statistics(path: Path) -> Dict[str, float]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    counts = Counter(row.get("dominant_role", "") for row in rows)
    invalid = sorted(role for role in counts if role and role not in ROLE_NAMES)
    if invalid:
        warnings.warn(f"Unknown role names in {path}: {invalid}")
    total = max(len(rows), 1)
    out = {f"role_{role}_ratio": counts.get(role, 0) / total for role in ROLE_NAMES}
    entropy_values = []
    for row in rows:
        probs = [max(to_float(row.get(f"role_{role}")), 1e-12) for role in ROLE_NAMES]
        entropy_values.append(-sum(p * math.log(p) for p in probs) / math.log(len(ROLE_NAMES)))
    out["role_entropy"] = mean(entropy_values)
    out["dominant_role_diversity"] = len([role for role in ROLE_NAMES if counts.get(role, 0) > 0]) / len(ROLE_NAMES)
    return out


def compute_gate_statistics(path: Path) -> Dict[str, float]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    obs = [to_float(row.get("gate_obs_weight")) for row in rows]
    prior = [to_float(row.get("gate_prior_weight")) for row in rows]
    entropy = []
    for o, p in zip(obs, prior):
        total = max(o + p, 1e-12)
        o_norm = min(max(o / total, 1e-12), 1.0)
        p_norm = min(max(p / total, 1e-12), 1.0)
        entropy.append(-(o_norm * math.log(o_norm) + p_norm * math.log(p_norm)) / math.log(2.0))
    return {
        "mean_gate_obs": mean(obs),
        "mean_gate_prior": mean(prior),
        "obs_dominant_ratio": sum(1 for o, p in zip(obs, prior) if o >= p) / max(len(obs), 1),
        "prior_dominant_ratio": sum(1 for o, p in zip(obs, prior) if p > o) / max(len(obs), 1),
        "gate_entropy": mean(entropy),
    }


def compute_uncertainty_statistics(uncertainty_path: Path, event_predictions_path: Path) -> Dict[str, float]:
    uncertainty_rows = read_csv_rows(uncertainty_path)
    event_rows = read_csv_rows(event_predictions_path)
    if not uncertainty_rows:
        return {}
    values = [to_float(row.get("uncertainty_score", row.get("uncertainty_log_var"))) for row in uncertainty_rows]
    out = {"mean_uncertainty": mean(values)}
    if not event_rows:
        return out

    by_cascade: dict[str, list[float]] = defaultdict(list)
    for row in uncertainty_rows:
        by_cascade[row.get("cascade_idx", "")].append(to_float(row.get("uncertainty_score", row.get("uncertainty_log_var"))))
    correct: list[float] = []
    wrong: list[float] = []
    for row in event_rows:
        cascade = row.get("cascade_idx", "")
        if cascade not in by_cascade:
            continue
        value = mean(by_cascade[cascade])
        if to_int(row.get("y_true"), 0) == to_int(row.get("y_pred"), 0):
            correct.append(value)
        else:
            wrong.append(value)
    out["uncertainty_correct_mean"] = mean(correct)
    out["uncertainty_wrong_mean"] = mean(wrong)
    return out


def compute_attention_statistics(path: Path) -> Dict[str, object]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    by_cascade: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_cascade[row.get("cascade_idx", "")].append(row)

    entropy_values: list[float] = []
    top1_values: list[float] = []
    top5_values: list[float] = []
    key_windows = Counter()
    for group in by_cascade.values():
        weights = [max(to_float(row.get("event_attention")), 0.0) for row in group]
        total = sum(weights)
        if total <= 0:
            continue
        probs = [w / total for w in weights]
        denom = math.log(max(len(probs), 2))
        entropy_values.append(-sum(p * math.log(max(p, 1e-12)) for p in probs) / denom)
        sorted_probs = sorted(probs, reverse=True)
        top1_values.append(sorted_probs[0])
        top5_values.append(sum(sorted_probs[:5]))
        top_idx = max(range(len(group)), key=lambda i: weights[i])
        key_windows[str(to_int(group[top_idx].get("window_idx"), 0))] += 1
    total_events = max(sum(key_windows.values()), 1)
    return {
        "attention_entropy": mean(entropy_values),
        "top1_attention_mean": mean(top1_values),
        "top5_attention_mass_mean": mean(top5_values),
        "key_window_distribution": {k: v / total_events for k, v in sorted(key_windows.items(), key=lambda item: int(item[0]))},
    }


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        warnings.warn(f"Missing prediction file, skip diagnostics: {path}")
        return []
    if path.stat().st_size == 0:
        warnings.warn(f"Empty prediction file, skip diagnostics: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def std(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    mu = mean(data)
    return math.sqrt(sum((v - mu) ** 2 for v in data) / len(data))


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
