"""Infer time-consistent proxy propagation trees from cascade posts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"

TREE_FIELDS = [
    "cascade_idx",
    "parent_tweet_idx",
    "child_tweet_idx",
    "parent_user_idx",
    "child_user_idx",
    "parent_time",
    "child_time",
    "time_gap",
    "parent_depth",
    "child_depth",
    "parent_children_before",
    "candidate_count",
    "time_score",
    "follow_score",
    "text_score",
    "activity_score",
    "exposure_score",
    "depth_penalty",
    "load_penalty",
    "root_score",
    "parent_score",
    "parent_source",
    "root_fallback_flag",
    "text_missing_flag",
    "follow_checked_flag",
]


@dataclass(frozen=True)
class ParentChoice:
    parent: Dict[str, str]
    parent_source: str
    candidate_count: int
    time_score: float
    follow_score: float
    text_score: float
    activity_score: float
    exposure_score: float
    depth_penalty: float
    load_penalty: float
    root_score: float
    parent_score: float
    root_fallback_flag: int
    text_missing_flag: int
    follow_checked_flag: int


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    post_table = args.input_post_table or run_dir / "org_task" / "post_table.csv"
    follow_edges = args.follow_edges
    out_dir = args.out_dir or run_dir / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)

    cascades = read_cascades(
        post_table,
        max_cascades=args.max_cascades,
        cascade_id=args.cascade_id,
        max_observation_seconds=args.max_observation_seconds,
    )
    relevant_users = {str(post["user_idx"]) for posts in cascades.values() for post in posts}
    follow_pairs: set[Tuple[str, str]] = set()
    if args.method in {"follow_time", "hybrid", "hybrid_no_text"} and follow_edges is not None and follow_edges.exists():
        follow_pairs = read_relevant_follow_pairs(follow_edges, relevant_users)

    tree_rows, diagnostics = infer_trees(
        cascades,
        follow_pairs=follow_pairs,
        method=args.method,
        tau_seconds=args.tau_seconds,
        max_candidate_lookback=args.max_candidate_lookback,
        max_parent_gap=args.max_parent_gap,
        depth_penalty_weight=args.depth_penalty,
        activity_weight=args.activity_weight,
        root_bias=args.root_bias,
        child_penalty=args.child_penalty,
        follow_weight=args.follow_weight,
        text_weight=args.text_weight,
        exposure_weight=args.exposure_weight,
        root_threshold=args.root_threshold,
        window_seconds=args.window_seconds,
    )

    tree_path = out_dir / "inferred_tree_edge_table.csv"
    write_csv(tree_path, TREE_FIELDS, tree_rows)
    write_json(out_dir / "tree_diagnostics.json", diagnostics)
    write_readme(out_dir, args.method)

    source_star = run_dir / "org_task" / "cascade_edge_table.csv"
    if source_star.exists():
        shutil.copyfile(source_star, out_dir / "star_edge_table.csv")

    print(f"Wrote inferred tree edges to {tree_path}")
    print(
        f"cascades={diagnostics['num_cascades']} edges={diagnostics['num_tree_edges']} "
        f"follow_parent_ratio={diagnostics['follow_parent_ratio']:.6f} "
        f"root_fallback_ratio={diagnostics['root_fallback_ratio']:.6f}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a time-consistent proxy propagation tree.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--input-post-table", type=Path, default=None)
    parser.add_argument("--follow-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=[
            "time_only",
            "branching_time",
            "follow_time",
            "hybrid",
            "hybrid_no_text",
            "hybrid_no_follow",
        ],
        default="hybrid",
    )
    parser.add_argument("--tau-seconds", type=float, default=300.0)
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--max-candidate-lookback", type=int, default=100)
    parser.add_argument("--max-parent-gap", type=int, default=1800)
    parser.add_argument("--depth-penalty", type=float, default=0.05)
    parser.add_argument("--activity-weight", type=float, default=0.15)
    parser.add_argument("--root-bias", type=float, default=0.0)
    parser.add_argument("--child-penalty", type=float, default=0.05)
    parser.add_argument("--follow-weight", type=float, default=0.20)
    parser.add_argument("--text-weight", type=float, default=0.20)
    parser.add_argument("--exposure-weight", type=float, default=0.0)
    parser.add_argument("--root-threshold", type=float, default=-0.25)
    parser.add_argument("--cascade-id", default=None)
    parser.add_argument("--max-observation-seconds", type=int, default=None)
    parser.add_argument("--max-cascades", type=int, default=None)
    return parser.parse_args()


def read_cascades(
    path: Path,
    max_cascades: Optional[int],
    cascade_id: Optional[str] = None,
    max_observation_seconds: Optional[int] = None,
) -> Dict[str, List[Dict[str, str]]]:
    cascades: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    selected: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cascade_idx = str(row["cascade_idx"])
            if cascade_id is not None and cascade_idx != str(cascade_id):
                continue
            if max_observation_seconds is not None and int(row["relative_time"]) >= max_observation_seconds:
                continue
            if cascade_idx not in selected:
                if max_cascades is not None and len(selected) >= max_cascades:
                    break
                selected.add(cascade_idx)
            cascades[cascade_idx].append(row)
    return dict(cascades)


def read_relevant_follow_pairs(path: Path, relevant_users: set[str]) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            src = row[0].strip()
            dst = row[1].strip()
            if src in relevant_users and dst in relevant_users:
                pairs.add((src, dst))
                pairs.add((dst, src))
    return pairs


def infer_trees(
    cascades: Dict[str, List[Dict[str, str]]],
    *,
    follow_pairs: set[Tuple[str, str]],
    method: str,
    tau_seconds: float,
    max_candidate_lookback: int,
    max_parent_gap: int,
    depth_penalty_weight: float,
    activity_weight: float,
    root_bias: float,
    child_penalty: float,
    follow_weight: float,
    text_weight: float,
    exposure_weight: float,
    root_threshold: float,
    window_seconds: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    depth_values: List[int] = []
    root_child_count = 0
    source_counts: Counter = Counter()
    time_gaps: List[int] = []
    text_scores: List[float] = []
    random_text_scores: List[float] = []
    follow_supported = 0
    same_or_adjacent_window = 0
    invalid_time_edges = 0
    branching_parent_total = 0
    parent_child_counter: Counter = Counter()

    for cascade_idx, posts in sorted(cascades.items(), key=lambda item: _sort_key(item[0])):
        ordered = sorted(posts, key=lambda post: (int(post["relative_time"]), _sort_key(post["tweet_idx"])))
        if not ordered:
            continue
        root = next((post for post in ordered if _truthy(post.get("is_root"))), ordered[0])
        prior_posts: List[Dict[str, str]] = []
        tweet_depth: Dict[str, int] = {str(root["tweet_idx"]): 0}
        child_counts: Counter = Counter()

        for post in ordered:
            if str(post["tweet_idx"]) == str(root["tweet_idx"]):
                prior_posts.append(post)
                continue

            choice = choose_parent(
                child=post,
                root=root,
                prior_posts=prior_posts,
                follow_pairs=follow_pairs,
                method=method,
                max_candidate_lookback=max_candidate_lookback,
                max_parent_gap=max_parent_gap,
                tweet_depth=tweet_depth,
                child_counts=child_counts,
                tau_seconds=tau_seconds,
                depth_penalty_weight=depth_penalty_weight,
                activity_weight=activity_weight,
                root_bias=root_bias,
                child_penalty=child_penalty,
                follow_weight=follow_weight,
                text_weight=text_weight,
                exposure_weight=exposure_weight,
                root_threshold=root_threshold,
                window_seconds=window_seconds,
            )
            parent = choice.parent
            parent_time = int(parent["relative_time"])
            child_time = int(post["relative_time"])
            time_gap = child_time - parent_time
            if time_gap < 0:
                invalid_time_edges += 1
            if str(parent["tweet_idx"]) == str(root["tweet_idx"]):
                root_child_count += 1

            parent_tweet = str(parent["tweet_idx"])
            parent_depth = tweet_depth.get(parent_tweet, 0)
            child_depth = parent_depth + 1
            tweet_depth[str(post["tweet_idx"])] = child_depth
            child_counts[parent_tweet] += 1
            parent_child_counter[parent_tweet] += 1
            depth_values.append(child_depth)
            source_counts[choice.parent_source] += 1
            time_gaps.append(max(0, time_gap))
            text_scores.append(choice.text_score)
            if choice.follow_score > 0:
                follow_supported += 1
            if window_seconds > 0 and (max(0, time_gap) // window_seconds) <= 1:
                same_or_adjacent_window += 1
            if prior_posts:
                baseline_parent = prior_posts[(len(prior_posts) * 1103515245 + 12345) % len(prior_posts)]
                baseline_text_score, _missing = text_similarity(baseline_parent.get("text", ""), post.get("text", ""))
                random_text_scores.append(baseline_text_score)

            rows.append(
                {
                    "cascade_idx": cascade_idx,
                    "parent_tweet_idx": parent["tweet_idx"],
                    "child_tweet_idx": post["tweet_idx"],
                    "parent_user_idx": parent["user_idx"],
                    "child_user_idx": post["user_idx"],
                    "parent_time": parent_time,
                    "child_time": child_time,
                    "time_gap": time_gap,
                    "parent_depth": parent_depth,
                    "child_depth": child_depth,
                    "parent_children_before": child_counts[parent_tweet] - 1,
                    "candidate_count": choice.candidate_count,
                    "time_score": f"{choice.time_score:.8f}",
                    "follow_score": f"{choice.follow_score:.8f}",
                    "text_score": f"{choice.text_score:.8f}",
                    "activity_score": f"{choice.activity_score:.8f}",
                    "exposure_score": f"{choice.exposure_score:.8f}",
                    "depth_penalty": f"{choice.depth_penalty:.8f}",
                    "load_penalty": f"{choice.load_penalty:.8f}",
                    "root_score": f"{choice.root_score:.8f}",
                    "parent_score": f"{choice.parent_score:.8f}",
                    "parent_source": choice.parent_source,
                    "root_fallback_flag": choice.root_fallback_flag,
                    "text_missing_flag": choice.text_missing_flag,
                    "follow_checked_flag": choice.follow_checked_flag,
                }
            )
            prior_posts.append(post)
        branching_parent_total += sum(1 for count in child_counts.values() if count > 1)

    num_edges = len(rows)
    diagnostics = {
        "method": method,
        "num_cascades": len(cascades),
        "num_tree_edges": num_edges,
        "num_follow_pairs_loaded": len(follow_pairs) // 2,
        "source_counts": dict(source_counts),
        "avg_depth": _mean(depth_values),
        "depth_p50": _percentile(depth_values, 0.50),
        "depth_p90": _percentile(depth_values, 0.90),
        "depth_p95": _percentile(depth_values, 0.95),
        "max_depth": max(depth_values) if depth_values else 0,
        "root_child_ratio": root_child_count / num_edges if num_edges else 0.0,
        "root_fallback_ratio": source_counts["root_fallback"] / num_edges if num_edges else 0.0,
        "follow_parent_ratio": source_counts["follow_time"] / num_edges if num_edges else 0.0,
        "follow_supported_edge_ratio": follow_supported / num_edges if num_edges else 0.0,
        "parent_child_text_sim_mean": _mean_float(text_scores),
        "random_pair_text_sim_mean": _mean_float(random_text_scores),
        "text_sim_lift": _mean_float(text_scores) - _mean_float(random_text_scores),
        "same_or_adjacent_window_edge_ratio": same_or_adjacent_window / num_edges if num_edges else 0.0,
        "time_gap_mean": _mean(time_gaps),
        "time_gap_median": _percentile(time_gaps, 0.50),
        "time_gap_p90": _percentile(time_gaps, 0.90),
        "num_branching_parents": branching_parent_total,
        "top1_parent_child_ratio": _top_parent_ratio(parent_child_counter, 1, num_edges),
        "top5_parent_child_ratio": _top_parent_ratio(parent_child_counter, 5, num_edges),
        "branch_entropy": _entropy(parent_child_counter),
        "invalid_time_edges": invalid_time_edges,
        "cycle_count": 0,
        "orphan_node_count": 0,
        "missing_parent_count": 0,
        "tree_valid_ratio": 1.0,
        "notes": (
            "This is a time-consistent proxy propagation tree, not a recovered true retweet tree. "
            "HybridTree combines time, follow, text, activity, exposure, and shape regularization."
        ),
    }
    return rows, diagnostics


def choose_parent(
    *,
    child: Dict[str, str],
    root: Dict[str, str],
    prior_posts: Sequence[Dict[str, str]],
    follow_pairs: set[Tuple[str, str]],
    method: str,
    max_candidate_lookback: int,
    max_parent_gap: int,
    tweet_depth: Dict[str, int],
    child_counts: Counter,
    tau_seconds: float,
    depth_penalty_weight: float,
    activity_weight: float,
    root_bias: float,
    child_penalty: float,
    follow_weight: float,
    text_weight: float,
    exposure_weight: float,
    root_threshold: float,
    window_seconds: int,
) -> ParentChoice:
    child_time = int(child["relative_time"])
    candidates = [post for post in prior_posts if int(post["relative_time"]) < child_time]
    if max_parent_gap > 0:
        gap_candidates = [post for post in candidates if child_time - int(post["relative_time"]) <= max_parent_gap]
        if gap_candidates:
            candidates = gap_candidates
    if not candidates:
        return make_choice(
            parent=root,
            child=child,
            root=root,
            parent_source="root_fallback",
            candidate_count=0,
            follow_pairs=follow_pairs,
            tweet_depth=tweet_depth,
            child_counts=child_counts,
            tau_seconds=tau_seconds,
            depth_penalty_weight=depth_penalty_weight,
            activity_weight=activity_weight,
            root_bias=root_bias,
            child_penalty=child_penalty,
            follow_weight=follow_weight,
            text_weight=text_weight,
            exposure_weight=exposure_weight,
            window_seconds=window_seconds,
            force_root_fallback=True,
        )

    lookback = candidates[-max_candidate_lookback:] if max_candidate_lookback > 0 else candidates
    if method == "time_only":
        return make_choice(
            parent=candidates[-1],
            child=child,
            root=root,
            parent_source="time_only",
            candidate_count=len(candidates),
            follow_pairs=follow_pairs,
            tweet_depth=tweet_depth,
            child_counts=child_counts,
            tau_seconds=tau_seconds,
            depth_penalty_weight=depth_penalty_weight,
            activity_weight=activity_weight,
            root_bias=root_bias,
            child_penalty=child_penalty,
            follow_weight=follow_weight,
            text_weight=text_weight,
            exposure_weight=exposure_weight,
            window_seconds=window_seconds,
        )

    if method == "follow_time":
        child_user = str(child["user_idx"])
        for post in reversed(lookback):
            if (str(post["user_idx"]), child_user) in follow_pairs:
                return make_choice(
                    parent=post,
                    child=child,
                    root=root,
                    parent_source="follow_time",
                    candidate_count=len(candidates),
                    follow_pairs=follow_pairs,
                    tweet_depth=tweet_depth,
                    child_counts=child_counts,
                    tau_seconds=tau_seconds,
                    depth_penalty_weight=depth_penalty_weight,
                    activity_weight=activity_weight,
                    root_bias=root_bias,
                    child_penalty=child_penalty,
                    follow_weight=follow_weight,
                    text_weight=text_weight,
                    exposure_weight=exposure_weight,
                    window_seconds=window_seconds,
                )

    active = sorted(candidates, key=lambda post: child_counts[str(post["tweet_idx"])], reverse=True)[:20]
    use_follow = method in {"hybrid", "hybrid_no_text", "follow_time"}
    use_text = method in {"hybrid", "hybrid_no_follow"}
    follow_candidates = (
        [post for post in candidates if (str(post["user_idx"]), str(child["user_idx"])) in follow_pairs][:100]
        if use_follow
        else []
    )
    candidate_map = {str(post["tweet_idx"]): post for post in [*lookback, *active, *follow_candidates]}
    scored = [
        make_choice(
            parent=post,
            child=child,
            root=root,
            parent_source=method if method.startswith("hybrid") else "branching_time",
            candidate_count=len(candidates),
            follow_pairs=follow_pairs,
            tweet_depth=tweet_depth,
            child_counts=child_counts,
            tau_seconds=tau_seconds,
            depth_penalty_weight=depth_penalty_weight,
            activity_weight=activity_weight,
            root_bias=root_bias,
            child_penalty=child_penalty,
            follow_weight=follow_weight if use_follow else 0.0,
            text_weight=text_weight if use_text else 0.0,
            exposure_weight=exposure_weight if method.startswith("hybrid") else 0.0,
            window_seconds=window_seconds,
        )
        for post in candidate_map.values()
    ]
    best = max(scored, key=lambda choice: choice.parent_score)
    if method.startswith("hybrid") and str(best.parent["tweet_idx"]) != str(root["tweet_idx"]) and best.parent_score < root_threshold:
        return make_choice(
            parent=root,
            child=child,
            root=root,
            parent_source="root_fallback",
            candidate_count=len(candidates),
            follow_pairs=follow_pairs,
            tweet_depth=tweet_depth,
            child_counts=child_counts,
            tau_seconds=tau_seconds,
            depth_penalty_weight=depth_penalty_weight,
            activity_weight=activity_weight,
            root_bias=root_bias,
            child_penalty=child_penalty,
            follow_weight=follow_weight,
            text_weight=text_weight,
            exposure_weight=exposure_weight,
            window_seconds=window_seconds,
            force_root_fallback=True,
        )
    return best


def make_choice(
    *,
    parent: Dict[str, str],
    child: Dict[str, str],
    root: Dict[str, str],
    parent_source: str,
    candidate_count: int,
    follow_pairs: set[Tuple[str, str]],
    tweet_depth: Dict[str, int],
    child_counts: Counter,
    tau_seconds: float,
    depth_penalty_weight: float,
    activity_weight: float,
    root_bias: float,
    child_penalty: float,
    follow_weight: float,
    text_weight: float,
    exposure_weight: float,
    window_seconds: int,
    force_root_fallback: bool = False,
) -> ParentChoice:
    parent_tweet = str(parent["tweet_idx"])
    time_gap = int(child["relative_time"]) - int(parent["relative_time"])
    time_score = math.exp(-max(0, time_gap) / tau_seconds) if tau_seconds > 0 else 0.0
    follow_score = 1.0 if (str(parent["user_idx"]), str(child["user_idx"])) in follow_pairs else 0.0
    text_score, text_missing = text_similarity(parent.get("text", ""), child.get("text", ""))
    parent_depth = tweet_depth.get(parent_tweet, 0)
    parent_children = child_counts[parent_tweet]
    activity_score = math.log1p(parent_children)
    exposure_score = exposure_similarity(parent, child, window_seconds)
    depth_penalty = math.log1p(parent_depth) * depth_penalty_weight
    load_penalty = math.log1p(parent_children) * child_penalty
    root_score = root_bias if parent_tweet == str(root["tweet_idx"]) else 0.0
    score = (
        time_score
        + follow_weight * follow_score
        + text_weight * text_score
        + activity_weight * activity_score
        + exposure_weight * exposure_score
        + root_score
        - depth_penalty
        - load_penalty
    )
    return ParentChoice(
        parent=parent,
        parent_source=parent_source,
        candidate_count=candidate_count,
        time_score=time_score,
        follow_score=follow_score,
        text_score=text_score,
        activity_score=activity_score,
        exposure_score=exposure_score,
        depth_penalty=depth_penalty,
        load_penalty=load_penalty,
        root_score=root_score,
        parent_score=score,
        root_fallback_flag=1 if force_root_fallback else 0,
        text_missing_flag=text_missing,
        follow_checked_flag=1 if follow_pairs else 0,
    )


def text_similarity(parent_text: str, child_text: str) -> Tuple[float, int]:
    parent_tokens = text_tokens(parent_text)
    child_tokens = text_tokens(child_text)
    if not parent_tokens or not child_tokens:
        return 0.0, 1
    inter = len(parent_tokens & child_tokens)
    union = len(parent_tokens | child_tokens)
    return (inter / union if union else 0.0), 0


def text_tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() for ch in text if not ch.isspace())
    if len(cleaned) <= 2:
        return {cleaned} if cleaned else set()
    return {cleaned[idx : idx + 2] for idx in range(len(cleaned) - 1)}


def exposure_similarity(parent: Dict[str, str], child: Dict[str, str], window_seconds: int) -> float:
    if window_seconds <= 0:
        return 0.0
    parent_window = int(parent["relative_time"]) // window_seconds
    child_window = int(child["relative_time"]) // window_seconds
    return math.exp(-abs(child_window - parent_window))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(out_dir: Path, method: str) -> None:
    text = f"""# Edge Tables

This directory keeps original star edges and inferred proxy tree edges separately.

- `star_edge_table.csv`: copied from `org_task/cascade_edge_table.csv`.
- `inferred_tree_edge_table.csv`: tweet-level time-consistent proxy propagation tree.
- `tree_diagnostics.json`: structural diagnostics for the inferred tree.

Current method: `{method}`.

Important: the inferred tree is not a recovered true retweet tree. It is a proxy structure built from time order, text/follow evidence when available, activity, exposure, and shape regularization.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "root"}


def _sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_float(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def _top_parent_ratio(counter: Counter, k: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(count for _key, count in counter.most_common(k)) / total


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy
