"""Render inferred propagation trees as dependency-free SVG files."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_0002"


def main() -> int:
    args = parse_args()
    run_dir = PROJECT_ROOT / "work" / "runs" / args.run_id
    tree_path = args.tree_edges or run_dir / "edges" / "inferred_tree_edge_table.csv"
    post_path = args.post_table or run_dir / "org_task" / "post_table.csv"
    out_path = args.out or run_dir / "edges" / f"tree_cascade_{args.cascade_id}.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    posts = read_posts(post_path, args.cascade_id)
    edges = read_edges(tree_path, args.cascade_id)
    if not posts:
        raise SystemExit(f"No posts found for cascade {args.cascade_id}")

    root = next((tweet for tweet, row in posts.items() if row.get("is_root") == "1"), None)
    if root is None:
        root = min(posts, key=lambda tweet: int(posts[tweet]["relative_time"]))

    selected_nodes = select_nodes(root, posts, edges, args.max_nodes)
    svg = render_svg(root, posts, edges, selected_nodes, args.cascade_id)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote tree SVG to {out_path}")
    print(f"cascade={args.cascade_id} nodes={len(selected_nodes)} edges={len(edges)} rendered_edges={count_rendered_edges(edges, selected_nodes)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an inferred cascade tree as SVG.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--cascade-id", required=True)
    parser.add_argument("--tree-edges", type=Path, default=None)
    parser.add_argument("--post-table", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-nodes", type=int, default=160)
    return parser.parse_args()


def read_posts(path: Path, cascade_id: str) -> Dict[str, Dict[str, str]]:
    posts: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row["cascade_idx"]) == str(cascade_id):
                posts[str(row["tweet_idx"])] = row
    return posts


def read_edges(path: Path, cascade_id: str) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row["cascade_idx"]) == str(cascade_id):
                edges.append(row)
    return edges


def select_nodes(
    root: str,
    posts: Dict[str, Dict[str, str]],
    edges: Sequence[Dict[str, str]],
    max_nodes: int,
) -> Set[str]:
    by_time = sorted(posts, key=lambda tweet: (int(posts[tweet]["relative_time"]), int(tweet) if tweet.isdigit() else tweet))
    selected = set(by_time[: max(1, max_nodes)])
    selected.add(root)
    parent_by_child = {str(edge["child_tweet_idx"]): str(edge["parent_tweet_idx"]) for edge in edges}
    for tweet in list(selected):
        parent = parent_by_child.get(tweet)
        while parent and parent not in selected:
            selected.add(parent)
            parent = parent_by_child.get(parent)
    return selected


def render_svg(
    root: str,
    posts: Dict[str, Dict[str, str]],
    edges: Sequence[Dict[str, str]],
    selected_nodes: Set[str],
    cascade_id: str,
) -> str:
    children: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        parent = str(edge["parent_tweet_idx"])
        child = str(edge["child_tweet_idx"])
        if parent in selected_nodes and child in selected_nodes:
            children[parent].append(child)
    for parent in children:
        children[parent].sort(key=lambda tweet: (int(posts.get(tweet, {}).get("relative_time", 0)), tweet))

    depth: Dict[str, int] = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in children.get(node, []):
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)

    for node in selected_nodes:
        depth.setdefault(node, 0 if node == root else 1)

    levels: Dict[int, List[str]] = defaultdict(list)
    for node, node_depth in depth.items():
        if node in selected_nodes:
            levels[node_depth].append(node)
    for node_depth in levels:
        levels[node_depth].sort(key=lambda tweet: (int(posts.get(tweet, {}).get("relative_time", 0)), tweet))

    x_gap = 210
    y_gap = 34
    margin = 80
    width = max(900, margin * 2 + (max(levels) + 1) * x_gap)
    height = max(500, margin * 2 + max(len(nodes) for nodes in levels.values()) * y_gap)
    pos: Dict[str, Tuple[int, int]] = {}
    for node_depth, nodes in levels.items():
        level_height = (len(nodes) - 1) * y_gap
        start_y = margin + max(0, (height - 2 * margin - level_height) // 2)
        for idx, node in enumerate(nodes):
            pos[node] = (margin + node_depth * x_gap, start_y + idx * y_gap)

    edge_elems: List[str] = []
    for edge in edges:
        parent = str(edge["parent_tweet_idx"])
        child = str(edge["child_tweet_idx"])
        if parent not in pos or child not in pos:
            continue
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        edge_elems.append(
            f'<path d="M{x1 + 12},{y1} C{x1 + 90},{y1} {x2 - 90},{y2} {x2 - 12},{y2}" '
            'fill="none" stroke="#8b9bb0" stroke-width="1.1" opacity="0.65"/>'
        )

    node_elems: List[str] = []
    for node, (x, y) in pos.items():
        row = posts.get(node, {})
        is_root = node == root
        fill = "#d94841" if is_root else "#2f6f9f"
        radius = 8 if is_root else 5
        label = f"{node} / {format_seconds(int(row.get('relative_time', 0)))}"
        text = html.escape(row.get("text", "")[:42])
        node_elems.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"><title>{html.escape(label)} {text}</title></circle>')
        if is_root or len(node_elems) < 45:
            node_elems.append(f'<text x="{x + 12}" y="{y + 4}" font-size="11" fill="#253041">{html.escape(label)}</text>')

    title = html.escape(f"Cascade {cascade_id}: inferred branching tree preview")
    subtitle = html.escape(f"rendered nodes={len(pos)}, rendered edges={len(edge_elems)}")
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            f'<text x="24" y="32" font-size="20" font-weight="700" fill="#172033">{title}</text>',
            f'<text x="24" y="54" font-size="13" fill="#516173">{subtitle}</text>',
            *edge_elems,
            *node_elems,
            "</svg>",
        ]
    )


def count_rendered_edges(edges: Sequence[Dict[str, str]], selected_nodes: Set[str]) -> int:
    return sum(
        1
        for edge in edges
        if str(edge["parent_tweet_idx"]) in selected_nodes and str(edge["child_tweet_idx"]) in selected_nodes
    )


def format_seconds(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
