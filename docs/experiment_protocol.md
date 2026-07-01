# 实验流程

本文档记录 DRAGEN 实验的流程约定。这里不实现训练逻辑，只说明每一步应该依赖哪些输入和输出。

## 1. 选择 run

从数据配置中选择一个 run：

```text
configs/data/run_0001.yaml
configs/data/run_0002.yaml
configs/data/run_0003.yaml
```

每个配置对应一个实验数据单元：

```text
work/runs/<run_id>/
```

## 2. 检查 processed 数据

确认该 run 下至少包含：

```text
processed/
processed/events/
processed/mapping/
processed/text/
processed/user/
```

不要修改或移动已有 `processed/` 数据。

## 3. 构建或复用组织化任务表

入口脚本：

```text
scripts/01_build_org_tables.py
```

输出位置：

```text
work/runs/<run_id>/org_task/
```

## 4. 分析转发时间分布

入口脚本：

```text
scripts/02_analyze_time_distribution.py
```

该阶段用于观察不同观测期下的可用级联数量，为窗口配置和 dev cascade list 提供依据。

## 5. 构建窗口

窗口构建已经有正式入口。实现位置：

```text
src/dragen/windowing/window_builder.py
src/dragen/windowing/node_window_builder.py
src/dragen/windowing/edge_window_builder.py
src/dragen/windowing/text_window_builder.py
```

窗口参数来自：

```text
configs/window/
```

当前可运行命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode star
```

输出位置：

```text
work/runs/<run_id>/windows/obs_<obs>_win<window>_step<step>/
```

每次正式构建后至少检查：

- `window_diagnostics.json` 中 `retweet_text_early_violations` 是否为 0。
- `root_text_window_rows` 是否等于 `num_cascades * windows_per_cascade`。
- 四张 CSV 是否均存在。

## 6. 结构重构实验线

结构重构线独立于主模型效果实验，目标是比较星形边和代理传播树。

先构建代理传播树：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method branching_time --out-dir work/runs/run_0002/edges
```

再构建树形窗口：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree
```

窗口目录必须分开：

```text
windows/obs_1800_win300_step300_star/
windows/obs_1800_win300_step300_tree/
```

结构实验至少比较：

- Star
- TimeTree，作为线性化对照
- BranchingTimeTree，当前默认树结构
- FollowTree
- HybridTree，轻量多证据主结构
- HybridTree-w/o Text
- HybridTree-w/o Follow

报告指标包括 `avg_depth`、`max_depth`、`root_child_ratio`、`root_fallback_ratio`、`follow_parent_ratio`、`follow_supported_edge_ratio`、`parent_child_text_sim_mean`、`random_pair_text_sim_mean`、`text_sim_lift`、`same_or_adjacent_window_edge_ratio`、`top1_parent_child_ratio`、`top5_parent_child_ratio`、`branch_entropy`、`time_gap_mean`、`num_branching_parents` 和 `tree_valid_ratio`。

## 7. 窗口策略实验线

窗口实验和结构实验分开做。

固定边结构为 HybridTree，比较窗口策略：

- `Fixed-5m`：当前非重叠 5 分钟窗口。
- `Causal-10m`：只使用端点对齐 10 分钟上下文窗口，后续可加。
- `MultiScale`：current 5m + context 10m + cumulative history，主窗口策略。

当前 MultiScale 调试命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode star --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_step300_multiscale_star
```

验收项：

- `num_window_rows = cascades * 6`
- `edge_window_table.csv` 中同时存在 `current` 和 `context`
- `retweet_text_early_violations = 0`

## 8. 训练和评估

训练和评估时必须记录：

- data config
- window config
- model config
- train config
- random seed
- checkpoint path
- metrics

没有记录这些配置的结果，不应该拿来做正式对比。

## 9. 记录实验

每次重要实验或结构调整都记录到 [run_notes.md](run_notes.md)。
