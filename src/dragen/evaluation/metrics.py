"""Metrics without sklearn dependency."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    y = list(map(int, y_true))
    p = [min(max(float(v), 0.0), 1.0) for v in y_prob]
    pred = [1 if v >= threshold else 0 for v in p]
    tp, tn, fp, fn = confusion_counts(y, pred)
    total = max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    neg_precision = tn / max(tn + fn, 1)
    f1 = f1_from_pr(precision, recall)
    neg_f1 = f1_from_pr(neg_precision, specificity)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    metrics = {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "macro_f1": (f1 + neg_f1) / 2.0,
        "auc": auc_score(y, p),
        "ap": average_precision_score(y, p),
        "mcc": ((tp * tn) - (fp * fn)) / denom,
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
    }
    metrics.update(risk_retrieval_metrics(y, p))
    return metrics


def risk_retrieval_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    return {
        "precision_at_100": precision_at_k(y_true, y_prob, 100),
        "precision_at_500": precision_at_k(y_true, y_prob, 500),
        "recall_at_500": recall_at_k(y_true, y_prob, 500),
        "precision_at_1pct": precision_at_percent(y_true, y_prob, 0.01),
        "recall_at_1pct": recall_at_percent(y_true, y_prob, 0.01),
        "precision_at_5pct": precision_at_percent(y_true, y_prob, 0.05),
        "recall_at_5pct": recall_at_percent(y_true, y_prob, 0.05),
    }


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return tp, tn, fp, fn


def f1_from_pr(precision: float, recall: float) -> float:
    return 2 * precision * recall / max(precision + recall, 1e-12)


def auc_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    pairs = sorted(zip(y_prob, y_true), key=lambda item: item[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(1 for _, y in pairs[i:j] if y == 1)
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    pairs = sorted(zip(y_prob, y_true), key=lambda item: item[0], reverse=True)
    total_pos = sum(1 for _, y in pairs if y == 1)
    if total_pos == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_pos


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    return sum((float(p) - int(y)) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def expected_calibration_error(y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    total = len(y_true)
    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        idx = [i for i, p in enumerate(y_prob) if (lo <= p < hi) or (bin_idx == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(float(y_prob[i]) for i in idx) / len(idx)
        acc = sum(1 for i in idx if (1 if y_prob[i] >= 0.5 else 0) == int(y_true[i])) / len(idx)
        ece += (len(idx) / total) * abs(acc - conf)
    return ece


def precision_at_k(y_true: Sequence[int], y_prob: Sequence[float], k: int) -> float:
    top = top_indices(y_prob, k)
    if not top:
        return 0.0
    return sum(int(y_true[i]) for i in top) / len(top)


def recall_at_k(y_true: Sequence[int], y_prob: Sequence[float], k: int) -> float:
    total_pos = sum(int(v) for v in y_true)
    if total_pos == 0:
        return 0.0
    top = top_indices(y_prob, k)
    return sum(int(y_true[i]) for i in top) / total_pos


def precision_at_percent(y_true: Sequence[int], y_prob: Sequence[float], percent: float) -> float:
    return precision_at_k(y_true, y_prob, percent_to_k(len(y_true), percent))


def recall_at_percent(y_true: Sequence[int], y_prob: Sequence[float], percent: float) -> float:
    return recall_at_k(y_true, y_prob, percent_to_k(len(y_true), percent))


def top_indices(y_prob: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    return [i for i, _ in sorted(enumerate(y_prob), key=lambda item: item[1], reverse=True)[: min(k, len(y_prob))]]


def percent_to_k(total: int, percent: float) -> int:
    if total <= 0:
        return 0
    pct = percent / 100.0 if percent > 1.0 else percent
    return max(1, int(math.ceil(total * pct)))
