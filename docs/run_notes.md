# 实验记录

这个文件用于记录简短实验备注。每次重要实验、参数变更或结构调整都单独写一段。

## 模板

```text
日期：
Run ID：
Data config：
Window config：
Model config：
Train config：
命令：
输出：
备注：
```

## 记录

- 已整理工程结构，`scripts/` 只保留入口脚本。
- 已有数据继续保留在 `work/runs/`，不要移动或重命名。
- 已实现窗口划分模块和入口 `scripts/05_build_windows.py`。

```text
日期：2026-06-29
Run ID：run_0002
Window config：configs/window/obs_30m_step5m.yaml
命令：python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml
输出：work/runs/run_0002/windows/obs_1800_win300_step300/
结果：
  cascades = 85263
  windows_per_cascade = 6
  window_rows = 511578
  node_window_rows = 5940259
  edge_window_rows = 1392078
  text_window_rows = 6032866
  root_text_window_rows = 511578
  retweet_text_early_violations = 0
验证：
  python -m py_compile 通过
  直接调用 tests/test_window_builder.py 中的窗口测试函数通过
备注：
  当前环境没有安装 pytest，因此未运行 python -m pytest。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：Tree-v1 调试
命令：python scripts/06_build_inferred_tree.py --run-id run_0002 --method time_only --max-cascades 10 --out-dir work/runs/run_0002/edges/_debug_tree_time
输出：work/runs/run_0002/edges/_debug_tree_time/
结果：
  cascades = 10
  tree_edges = 5685
  follow_parent_ratio = 0.000000
  root_fallback_ratio = 0.000000
备注：
  该调试版本未扫描 7.3GB follow_edges.tsv，只验证 time_only 代理树字段和结构。
```

```text
日期：2026-06-29
Run ID：run_0002
Window config：configs/window/obs_30m_step5m.yaml
Edge mode：inferred_tree
命令：python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/_debug_tree_time/inferred_tree_edge_table.csv --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_win300_step300_tree
输出：work/runs/run_0002/windows/_debug_obs_1800_win300_step300_tree/
结果：
  cascades = 10
  windows = 60
  node_windows = 214
  edge_windows = 43
  text_windows = 224
备注：
  树形边已经可以进入窗口构建链路，后续可生成正式 tree 窗口目录。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：避免线性树
结论：
  time_only 会将大级联构造成接近线性的深链，不适合作为正式树结构。
  branching_time 已设为默认推荐方法。
对比结果：
  time_only 调试样本：
    avg_depth = 2592.58
    max_depth = 5405
  branching_time 调试样本：
    avg_depth = 7.62
    max_depth = 18
备注：
  time_only 保留为结构实验中的线性化对照或反例。
```

```text
日期：2026-06-29
Run ID：run_0002
结构可视化：cascade_idx = 78857
命令：
  python scripts/06_build_inferred_tree.py --run-id run_0002 --cascade-id 78857 --max-observation-seconds 1800 --method branching_time --max-parent-gap 1800 --out-dir work/runs/run_0002/edges/_viz_cascade_78857_branching
  python scripts/07_visualize_tree.py --run-id run_0002 --cascade-id 78857 --tree-edges work/runs/run_0002/edges/_viz_cascade_78857_branching/inferred_tree_edge_table.csv --out work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg --max-nodes 180
输出：
  work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg
结果：
  observed_edges = 7471
  avg_depth = 8.82
  max_depth = 16
  num_branching_parents = 258
  time_gap_mean = 60.55s
  invalid_time_edges = 0
  tree_valid_ratio = 1.0
备注：
  该可视化证明当前 branching_time 结构不是线性链。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：轻量 HybridTree 小样本
命令：
  python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --max-cascades 10 --max-observation-seconds 1800 --out-dir work/runs/run_0002/edges/_debug_tree_hybrid_light_obs1800
输出：
  work/runs/run_0002/edges/_debug_tree_hybrid_light_obs1800/
结果：
  num_tree_edges = 43
  avg_depth = 5.67
  depth_p90 = 10
  max_depth = 11
  root_child_ratio = 0.163
  parent_child_text_sim_mean = 0.258
  random_pair_text_sim_mean = 0.085
  text_sim_lift = 0.173
  same_or_adjacent_window_edge_ratio = 0.953
  time_gap_mean = 188.02s
  time_gap_p90 = 482s
  top1_parent_child_ratio = 0.116
  top5_parent_child_ratio = 0.372
  branch_entropy = 3.086
  invalid_time_edges = 0
  tree_valid_ratio = 1.0
备注：
  这是轻量 HybridTree 的可运行诊断版本，默认未加载关注边，因此 follow_supported_edge_ratio = 0。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：同批 10 cascade / 30m 观测期结构对照
输出：
  work/runs/run_0002/edges/_debug_tree_compare_obs1800.csv
对照设置：
  time_only
  branching_time
  hybrid_no_text
  hybrid
结果摘要：
  time_only:
    avg_depth = 9.19
    max_depth = 22
    text_sim_lift = 0.019
    top1_parent_child_ratio = 0.023
    branch_entropy = 3.761
  branching_time:
    avg_depth = 6.51
    max_depth = 12
    text_sim_lift = 0.080
    top1_parent_child_ratio = 0.116
    branch_entropy = 3.195
  hybrid_no_text:
    avg_depth = 6.51
    max_depth = 12
    text_sim_lift = 0.080
    top1_parent_child_ratio = 0.116
    branch_entropy = 3.195
  hybrid:
    avg_depth = 5.67
    max_depth = 11
    text_sim_lift = 0.173
    top1_parent_child_ratio = 0.116
    branch_entropy = 3.086
结论：
  在当前未加载关注边的小样本中，hybrid 与 hybrid_no_text 的主要差异来自文本相似项。
  hybrid 的 text_sim_lift 高于 hybrid_no_text，说明文本项在该小样本中提升了父子边文本一致性。
  该样本只有 43 条边，不能作为正式结论，只作为链路和指标验证。
```

```text
日期：2026-06-29
Run ID：run_0002
窗口实验：Causal MultiScale 小样本
配置：configs/window/obs_30m_step5m_multiscale.yaml
命令：
  python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode star --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_step300_multiscale_star
输出：
  work/runs/run_0002/windows/_debug_obs_1800_step300_multiscale_star/
结果：
  cascades = 10
  windows = 60
  node_windows = 214
  edge_windows = 124
  text_windows = 224
  current_edges = 43
  context_edges = 81
  retweet_text_early_violations = 0
备注：
  MultiScale 保持 6 个推断端点；window/node 表用 cur/ctx/cum 多尺度字段，edge 表用 current/context 两种 window_scope。
```
