"""Labeling functions for Label-v3."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

LabelFunction = Callable[[Mapping[str, Any]], int]


def lf_burst_positive(item: Mapping[str, Any]) -> int:
    return POSITIVE if float(item.get("burst_score", 0.0)) >= 0.8 else ABSTAIN


def lf_temporal_sync_positive(item: Mapping[str, Any]) -> int:
    return POSITIVE if float(item.get("temporal_sync_score", 0.0)) >= 0.8 else ABSTAIN


def lf_structure_concentration_positive(item: Mapping[str, Any]) -> int:
    return POSITIVE if float(item.get("structure_score", 0.0)) >= 0.8 else ABSTAIN


def lf_text_repetition_positive(item: Mapping[str, Any]) -> int:
    return POSITIVE if float(item.get("text_score", 0.0)) >= 0.8 else ABSTAIN


def lf_follow_dense_positive(item: Mapping[str, Any]) -> int:
    return POSITIVE if float(item.get("follow_density_score", 0.0)) >= 0.8 else ABSTAIN


def lf_natural_spread_negative(item: Mapping[str, Any]) -> int:
    return NEGATIVE if float(item.get("natural_spread_score", 0.0)) >= 0.7 and float(item.get("coordination_score", 0.0)) <= 0.5 else ABSTAIN


def lf_low_coordination_negative(item: Mapping[str, Any]) -> int:
    return NEGATIVE if float(item.get("coordination_score", 0.0)) <= 0.3 and float(item.get("burst_score", 0.0)) <= 0.7 else ABSTAIN


def lf_low_structure_negative(item: Mapping[str, Any]) -> int:
    return NEGATIVE if float(item.get("structure_score", 0.0)) <= 0.3 and float(item.get("follow_density_score", 0.0)) <= 0.4 else ABSTAIN


LABEL_FUNCTIONS: Dict[str, LabelFunction] = {
    "LF_burst_positive": lf_burst_positive,
    "LF_temporal_sync_positive": lf_temporal_sync_positive,
    "LF_structure_concentration_positive": lf_structure_concentration_positive,
    "LF_text_repetition_positive": lf_text_repetition_positive,
    "LF_follow_dense_positive": lf_follow_dense_positive,
    "LF_natural_spread_negative": lf_natural_spread_negative,
    "LF_low_coordination_negative": lf_low_coordination_negative,
    "LF_low_structure_negative": lf_low_structure_negative,
}


def apply_label_functions(item: Mapping[str, Any]) -> Dict[str, int]:
    return {name: fn(item) for name, fn in LABEL_FUNCTIONS.items()}


def vote_counts(votes: Mapping[str, int]) -> tuple[int, int, int]:
    pos = sum(1 for value in votes.values() if value == POSITIVE)
    neg = sum(1 for value in votes.values() if value == NEGATIVE)
    abstain = sum(1 for value in votes.values() if value == ABSTAIN)
    return pos, neg, abstain


def confidence_from_votes(pos: int, neg: int) -> float:
    active = pos + neg
    return max(pos, neg) / active if active else 0.0
