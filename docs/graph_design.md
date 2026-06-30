# 图结构与代理传播树设计

DRAGEN 的实验分成两条线：

- 主实验线：证明 DRAGEN 对组织化操控识别有效。
- 结构重构线：证明将星形级联构建为时间一致的代理传播树有必要，并且不会引入明显结构错误。

## 基本原则

当前 `org_task/cascade_edge_table.csv` 是星形边，即 root 指向所有转发节点。星形结构可以表示哪些用户参与了传播，但不能表示谁影响了谁。

如果原始数据没有真实父转发关系，不能声称恢复了真实传播树。本文工程中使用的说法应是：

```text
基于转发时间、关注关系和传播活跃性构建时间一致的代理传播树。
```

不要写成：

```text
恢复真实传播路径。
```

## 边表保留策略

不要覆盖原始星形边。结构实验需要同时保留：

```text
work/runs/<run_id>/edges/
  star_edge_table.csv
  inferred_tree_edge_table.csv
  tree_diagnostics.json
  README.md
```

星形窗口和树形窗口也要分开输出：

```text
work/runs/<run_id>/windows/obs_1800_win300_step300_star/
work/runs/<run_id>/windows/obs_1800_win300_step300_tree/
```

## Tree-v1

当前实现位置：

```text
scripts/06_build_inferred_tree.py
src/dragen/graph/infer_tree.py
```

当前支持三种方法：

- `time_only`：每个转发帖选择时间最近的前序帖作为父节点。该方法容易把大级联构造成接近线性的深链，只保留为对照或反例。
- `branching_time`：默认方法。综合时间接近、深度惩罚、父节点已有分支活跃度、root 轻微偏置和分支负载惩罚选择父节点，避免树退化成线性链。
- `follow_time`：优先在前序帖中选择存在关注证据且时间最近的父节点；没有关注候选时回退到 `branching_time`。
- `hybrid`：轻量 HybridTree，综合时间、关注、文本、父节点活跃、深度惩罚和负载惩罚。默认不扫描全量关注图，只有显式传入 `--follow-edges` 时才使用关注证据。
- `hybrid_no_text`：Hybrid 消融，关闭文本相似度。
- `hybrid_no_follow`：Hybrid 消融，关闭关注证据。

注意：`graph/follow_edges.tsv` 当前约 7.3GB，`follow_time` 会流式扫描该文件。推荐先用 `branching_time` 跑全量结构基线，再在 dev list 或小样本上验证 `follow_time`。

## 非线性约束

树构建不能简单使用“最近前序节点”规则，否则大级联会退化为线性链。例如 10 个级联调试样本中：

```text
time_only:
  avg_depth = 2592.58
  max_depth = 5405

branching_time:
  avg_depth = 7.62
  max_depth = 18
```

因此正式实验使用 `branching_time` 或后续的 `follow_time / hybrid`，不要把 `time_only` 当作主树结构。

## 轻量 HybridTree

当前已经实现轻量 HybridTree，不直接上 RoBERTa，也不默认扫描 7.3GB 关注全图。第一版打分为：

```text
Score(u, v)
= 0.35 * S_time
+ 0.20 * S_follow
+ 0.20 * S_text
+ 0.15 * S_activity
- 0.05 * R_depth
- 0.05 * R_load
```

其中：

- `S_time`：`exp(-(t_v - t_u) / tau)`，默认 `tau = 300s`。
- `S_follow`：关注边证据，显式传入 `--follow-edges` 时启用。
- `S_text`：轻量字符 bigram Jaccard，相当于第一版文本相似度。
- `S_activity`：`log(1 + children_count_before)`。
- `R_depth`：`log(1 + depth)`。
- `R_load`：`log(1 + children_count_before)`。

候选集合不是全量前序节点，而是：

- 时间最近的前 `max_candidate_lookback` 个节点，默认 100。
- 当前已有子节点最多的 top 20 活跃节点。
- 有关注证据的候选节点，最多 100。

root 不参与常规排序，只作为证据不足时的 fallback。这样避免 root 由于文本相似或早期优势吸附过多节点。

## 构建命令

全量分支时间树：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method branching_time --out-dir work/runs/run_0002/edges
```

小样本关注增强树：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method follow_time --max-cascades 10 --out-dir work/runs/run_0002/edges/_debug_tree_follow
```

使用树边构建窗口：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree
```

轻量 HybridTree 小样本调试：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --max-cascades 10 --max-observation-seconds 1800 --out-dir work/runs/run_0002/edges/_debug_tree_hybrid_light_obs1800
```

如果要启用关注证据，先构建或指定 run 级关注子图，再传给 HybridTree：

```powershell
python scripts/09_build_follow_subgraph.py --run-id run_0002 --out-dir work/runs/run_0002/graphs/follow_subgraph

python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --follow-edges work/runs/run_0002/graphs/follow_subgraph/follow_edges_run.tsv --out-dir work/runs/run_0002/edges/hybrid_tree
```

注意：`scripts/09_build_follow_subgraph.py` 会扫描全量 `graph/follow_edges.tsv`，不要在每次调试时运行。

## 可视化

树形结构可视化入口：

```text
scripts/07_visualize_tree.py
src/dragen/graph/visualize_tree.py
```

示例：构建 `cascade_idx = 78857` 在 30 分钟观测期内的分支树并导出 SVG。

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --cascade-id 78857 --max-observation-seconds 1800 --method branching_time --max-parent-gap 1800 --out-dir work/runs/run_0002/edges/_viz_cascade_78857_branching

python scripts/07_visualize_tree.py --run-id run_0002 --cascade-id 78857 --tree-edges work/runs/run_0002/edges/_viz_cascade_78857_branching/inferred_tree_edge_table.csv --out work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg --max-nodes 180
```

当前样例输出：

```text
work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg
```

诊断摘要：

```text
cascade_idx = 78857
observed_edges = 7471
avg_depth = 8.82
max_depth = 16
num_branching_parents = 258
time_gap_mean = 60.55s
invalid_time_edges = 0
tree_valid_ratio = 1.0
```

## 输出字段

`inferred_tree_edge_table.csv` 字段分为四类。

基础字段：

- `cascade_idx`
- `parent_tweet_idx`
- `child_tweet_idx`
- `parent_user_idx`
- `child_user_idx`
- `parent_time`
- `child_time`
- `time_gap`

结构状态字段：

- `parent_depth`
- `child_depth`
- `parent_children_before`
- `candidate_count`

分项得分字段：

- `time_score`
- `follow_score`
- `text_score`
- `activity_score`
- `depth_penalty`
- `load_penalty`
- `parent_score`

诊断标记字段：

- `parent_source`
- `root_fallback_flag`
- `text_missing_flag`
- `follow_checked_flag`

先构建 tweet-level tree，再在窗口阶段转成 user-level graph。这样可以避免同一用户多次转发导致用户级树不严格的问题。

## 结构诊断

`tree_diagnostics.json` 至少检查：

- `avg_depth`
- `max_depth`
- `root_child_ratio`
- `root_fallback_ratio`
- `follow_parent_ratio`
- `time_gap_mean`
- `random_pair_text_sim_mean`
- `text_sim_lift`
- `same_or_adjacent_window_edge_ratio`
- `num_branching_parents`
- `top1_parent_child_ratio`
- `top5_parent_child_ratio`
- `branch_entropy`
- `invalid_time_edges`
- `cycle_count`
- `orphan_node_count`
- `missing_parent_count`
- `tree_valid_ratio`

结构重构线的核心表：

```text
Table 2. Comparison of edge construction strategies.
```

建议比较：

- Star
- TimeTree，作为线性化对照
- BranchingTimeTree，当前默认树结构
- FollowTree
- HybridTree
