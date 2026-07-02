"""Label fusion for multi-version weak labels."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from dragen.labeling.label_features import make_label_row, read_label_csv


def ensemble_consensus(label_paths: Mapping[str, Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_method = {method: {str(row["cascade_idx"]): row for row in read_label_csv(path)} for method, path in label_paths.items()}
    cascade_ids = sorted(set.intersection(*(set(rows) for rows in by_method.values())), key=lambda x: int(x))
    out: List[Dict[str, Any]] = []
    method_names = list(by_method)
    agreement: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    conflicts = 0
    for cascade_idx in cascade_ids:
        labels = {method: int(by_method[method][cascade_idx]["label"]) for method in method_names}
        pos = sum(1 for value in labels.values() if value == 1)
        neg = sum(1 for value in labels.values() if value == 0)
        for a in method_names:
            for b in method_names:
                agreement[f"{a}:{by_method[a][cascade_idx]['label']}"][f"{b}:{by_method[b][cascade_idx]['label']}"] += 1
        if pos >= 2 and neg == 0:
            label = 1
            confidence = pos / len(method_names)
        elif neg >= 2 and pos == 0:
            label = 0
            confidence = neg / len(method_names)
        else:
            label = -1
            confidence = max(pos, neg) / len(method_names)
            if pos > 0 and neg > 0:
                conflicts += 1
        base = next(iter(by_method.values()))[cascade_idx]
        scores = [float(by_method[m][cascade_idx].get("weak_score", 0.0)) for m in method_names]
        item = {
            "cascade_idx": cascade_idx,
            "split": base["split"],
            "size_bucket": base["size_bucket"],
            "observed_retweet_count": base["observed_retweet_count"],
            "final_retweet_count": base["final_retweet_count"],
        }
        row = make_label_row(item, label, confidence, sum(scores) / max(len(scores), 1), "ensemble_consensus")
        row.update({f"{method}_label": labels[method] for method in method_names})
        out.append(row)
    diagnostics = {
        "method_agreement_matrix": {k: dict(v) for k, v in agreement.items()},
        "method_conflict_rate": conflicts / max(len(cascade_ids), 1),
    }
    return out, diagnostics
