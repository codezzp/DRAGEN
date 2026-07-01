"""Metrics without sklearn dependency."""

from __future__ import annotations

from typing import Dict, Iterable, List


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    y = list(map(int, y_true))
    p = list(map(float, y_prob))
    pred = [1 if v >= threshold else 0 for v in p]
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    total = max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score(y, p),
    }


def auc_score(y_true: List[int], y_prob: List[float]) -> float:
    pos = [(p, y) for p, y in zip(y_prob, y_true) if y == 1]
    neg = [(p, y) for p, y in zip(y_prob, y_true) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p_pos, _ in pos:
        for p_neg, _ in neg:
            if p_pos > p_neg:
                wins += 1.0
            elif p_pos == p_neg:
                wins += 0.5
    return wins / (len(pos) * len(neg))
