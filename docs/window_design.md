# 窗口划分说明

窗口划分是 DRAGEN 的核心模块之一。当前窗口链路已经冻结，后续训练优先复用现有产物，不再扩展新窗口策略。

## 模块划分

```text
src/dragen/windowing/window_builder.py
src/dragen/windowing/node_window_builder.py
src/dragen/windowing/edge_window_builder.py
src/dragen/windowing/text_window_builder.py
```

## 职责边界

`window_builder.py`：

- 读取 window 配置和命令行参数。
- 协调节点、边、文本三个窗口构建模块。
- 从 `work/runs/<run_id>/org_task/` 读取 `post_table.csv` 和 `cascade_edge_table.csv`。
- 当 `edge_mode = inferred_tree` 时，从 `work/runs/<run_id>/edges/inferred_tree_edge_table.csv` 读取代理树边。
- 写出 `window_table.csv`、`node_window_table.csv`、`edge_window_table.csv`、`text_window_table.csv`。
- 写出 `cascade_window_index.json`。
- 写出窗口构建诊断信息。

`node_window_builder.py`：

- 构建 `node_window_table`。
- 统计当前窗口和累计窗口中的节点活跃情况。
- 按 `cascade_idx`、`window_id`、`user_idx` 对齐。

`edge_window_builder.py`：

- 构建窗口内可见的传播边。
- 保留边的时间戳和相对时间。
- 后续可支持“当前窗口边”和“累计窗口边”两种策略。

`text_window_builder.py`：

- 根文本从第一个观测窗口开始进入。
- 转发文本只有在对应转发事件进入窗口后才可见。
- 保持 `tweet_idx` 对齐，避免只依赖行顺序。

## 窗口配置

窗口参数放在：

```text
configs/window/
```

当前保留配置包括：

```text
obs_30m_step5m.yaml
obs_30m_step5m_multiscale.yaml
obs_1h_step5m.yaml
obs_2h_step10m.yaml
```

每个配置至少应包含：

- `obs_seconds`：观测总时长。
- `window_size_seconds`：窗口长度。
- `step_seconds`：滑动步长。
- `edge_mode`：边结构，当前支持 `star` 和 `inferred_tree`。

如果配置中没有 `window_size_seconds`，代码默认使用 `step_seconds` 作为窗口长度。当前 30m/5m 配置对应：

```text
obs_seconds = 1800
window_size_seconds = 300
step_seconds = 300
edge_mode = star
```

后续如果需要累计/非累计策略、文本进入策略等字段，也应写进配置，不要写死在代码中。

## 输出位置

窗口产物保留在对应 run 下面，例如：

```text
work/runs/<run_id>/windows/obs_<obs>_win<window>_step<step>_<edge_mode>/
```

不要把窗口表写到项目根目录，也不要打乱 `work/runs/<run_id>/` 的数据组织。

## 窗口版本

当前保留两个窗口版本：

- `Fixed-5m`：非重叠 5 分钟窗口，工程 baseline。
- `Causal MultiScale`：端点对齐多尺度因果窗口，推荐作为 DRAGEN 主窗口策略。

当前正式输出：

```text
work/runs/run_0002/windows/obs_1800_win300_step300/
work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
```

其中：

```text
Fixed-5m Star              -> baseline / w/o Tree
Fixed-5m HybridTree        -> w/o MultiScale
MultiScale HybridTree      -> DRAGEN-Full 主输入
```

## 端点对齐多尺度因果窗口

配置：

```text
configs/window/obs_30m_step5m_multiscale.yaml
```

窗口端点：

```text
e_t = t * 300, t = 1, 2, ..., 6
```

每个端点构造三种范围：

```text
current    [max(0, e_t - 300), e_t)
context    [max(0, e_t - 600), e_t)
cumulative [0, e_t)
```

语义：

- `current`：短时异常冲击。
- `context`：局部传播角色图。
- `cumulative`：历史状态和跨窗口记忆。

第一版 MultiScale 输出策略：

- `window_table.csv`：一行一个 `cascade_idx + window_idx`，用 `_cur/_ctx/_cum` 字段保存多尺度统计。
- `node_window_table.csv`：一行一个 `cascade_idx + window_idx + user_idx`，用 `_cur/_ctx/_cum` 字段保存多尺度节点统计。
- `edge_window_table.csv`：增加 `window_scope`，当前只写 `current` 和 `context` 两种边图，不写 cumulative 全量边。
- `text_window_table.csv`：沿用现有因果文本可见规则，root 全窗口可见，retweet 出现后可见。

正式 MultiScale HybridTree 命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv --out-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree
```

正式结果：

```text
cascades = 85263
window_rows = 511578
node_window_rows = 5940259
edge_window_rows = 4019952
text_window_rows = 6032866
current_edges = 1392078
context_edges = 2627874
retweet_text_early_violations = 0
```

当前可运行命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode star
```

使用代理树边构建窗口：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree
```

调试时可以限制级联数量：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_win300_step300
```

## 输出表

`window_table.csv`：每个级联每个窗口一行，字段包括：

- `cascade_idx`
- `window_idx`
- `start_offset`
- `end_offset`
- `num_retweets`
- `cum_retweets`
- `num_active_users`
- `num_edges`
- `window_heat`
- `delta_heat`

`node_window_table.csv`：每个级联、窗口、可见用户一行，字段包括：

- `cascade_idx`
- `window_idx`
- `user_idx`
- `first_seen_time`
- `is_root`
- `num_posts_in_window`
- `cum_posts`
- `in_degree_window`
- `out_degree_window`
- `cum_in_degree`
- `cum_out_degree`
- `time_since_root`
- `time_since_first_seen`

`edge_window_table.csv`：当前窗口内传播边，字段包括：

- `cascade_idx`
- `window_idx`
- `src_user_idx`
- `dst_user_idx`
- `src_tweet_idx`
- `dst_tweet_idx`
- `edge_time`
- `edge_offset`
- `edge_type`

`text_window_table.csv`：窗口内可见文本，字段包括：

- `cascade_idx`
- `window_idx`
- `user_idx`
- `tweet_idx`
- `post_type`
- `text`
- `text_visible_type`
- `post_offset`

## 文本进入规则

- 根文本从第一个窗口开始，在每个窗口都可见，`text_visible_type = root_always_visible`。
- 转发文本只有在 `post_offset < window_end` 时才可见。
- 转发文本位于当前窗口时，`text_visible_type = current_window`。
- 转发文本早于当前窗口时，`text_visible_type = history_visible`。

`run_0002` 的完整 30m/5m 星形边构建诊断：

```text
cascades: 85263
windows_per_cascade: 6
window_rows: 511578
node_window_rows: 5940259
edge_window_rows: 1392078
text_window_rows: 6032866
root_text_window_rows: 511578
retweet_text_early_violations: 0
```
