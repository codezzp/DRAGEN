# 数据字段说明

本文档记录后续窗口、特征、训练和评估阶段需要依赖的稳定字段。已有生成文件中可以包含额外诊断字段，但核心字段应保持稳定。

## 级联表

典型位置：

```text
work/runs/<run_id>/org_task/cascade_table.csv
```

核心字段：

- `cascade_idx`：级联索引。
- `root_tweet_idx`：根帖子的 tweet 索引。
- `root_user_idx`：根用户索引。
- `root_time_epoch`：根帖子发布时间。
- `final_retweet_count`：最终转发数。
- `observed_retweet_count`：观测期内转发数。
- `duration`：完整传播持续时间。
- `observed_duration`：观测期内持续时间。
- `valid_for_training`：是否可用于训练。
- `drop_reason`：不可用原因。

## 帖子表

典型位置：

```text
work/runs/<run_id>/org_task/post_table.csv
```

核心字段：

- `cascade_idx`：所属级联。
- `tweet_idx`：帖子索引。
- `user_idx`：发帖或转发用户。
- `parent_tweet_idx`：父帖子索引。
- `parent_user_idx`：父帖子用户索引。
- `time_epoch`：事件时间。
- `relative_time`：相对根帖子的时间。
- `text`：文本内容。
- `is_root`：是否为根帖子。
- `in_observation`：是否落在观测期内。

## 用户表

典型位置：

```text
work/runs/<run_id>/org_task/user_table.csv
```

核心字段：

- `user_idx`：用户索引。
- `profile_text`：用户画像文本。
- `is_active_in_events`：是否在事件中活跃。
- `is_root_user`：是否作为根用户出现。
- `is_retweet_user`：是否作为转发用户出现。
- `num_root_posts`：根帖数量。
- `num_retweets`：转发数量。

## 传播边表

典型位置：

```text
work/runs/<run_id>/org_task/cascade_edge_table.csv
```

核心字段：

- `cascade_idx`：所属级联。
- `src_user_idx`：源用户。
- `dst_user_idx`：目标用户。
- `src_tweet_idx`：源帖子。
- `dst_tweet_idx`：目标帖子。
- `edge_type`：边类型。
- `time_epoch`：边对应事件时间。
- `relative_time`：相对根帖子的时间。
- `in_observation`：是否落在观测期内。

## 代理传播树边表

典型位置：

```text
work/runs/<run_id>/edges/inferred_tree_edge_table.csv
```

该表是 tweet-level 的时间一致代理传播树，不是真实传播路径恢复结果。

核心字段：

- `cascade_idx`：级联索引。
- `parent_tweet_idx`：推断父帖。
- `child_tweet_idx`：子帖。
- `parent_user_idx`：推断父用户。
- `child_user_idx`：子用户。
- `parent_time`：父帖相对根帖时间。
- `child_time`：子帖相对根帖时间。
- `time_gap`：父子时间差。
- `time_score`：时间接近度。
- `follow_score`：关注关系证据。
- `text_score`：轻量文本相似度。
- `activity_score`：父节点已有扩散活跃度。
- `depth_penalty`：深度惩罚项。
- `load_penalty`：分支负载惩罚项。
- `parent_score`：最终父节点得分。
- `parent_source`：父节点来源，当前包括 `time_only`、`branching_time`、`follow_time`、`hybrid`、`hybrid_no_text`、`hybrid_no_follow`、`root_fallback`。
- `root_fallback_flag`：是否因为证据不足回退 root。
- `text_missing_flag`：父子文本是否缺失。
- `follow_checked_flag`：该次构树是否加载了关注边证据。
- `candidate_count`：当前子帖可选的前序候选父节点数。

配套诊断文件：

```text
work/runs/<run_id>/edges/tree_diagnostics.json
```

关键字段包括：

- `avg_depth`
- `max_depth`
- `root_child_ratio`
- `root_fallback_ratio`
- `follow_parent_ratio`
- `follow_supported_edge_ratio`
- `parent_child_text_sim_mean`
- `random_pair_text_sim_mean`
- `text_sim_lift`
- `same_or_adjacent_window_edge_ratio`
- `time_gap_mean`
- `num_branching_parents`
- `top1_parent_child_ratio`
- `top5_parent_child_ratio`
- `branch_entropy`
- `invalid_time_edges`
- `cycle_count`
- `orphan_node_count`
- `missing_parent_count`
- `tree_valid_ratio`

## 窗口表

典型位置：

```text
work/runs/<run_id>/windows/obs_<obs>_win<window>_step<step>/
```

星形边和树形边窗口应分开保存：

```text
work/runs/<run_id>/windows/obs_1800_win300_step300_star/
work/runs/<run_id>/windows/obs_1800_win300_step300_tree/
```

多尺度窗口保存为：

```text
work/runs/<run_id>/windows/obs_1800_step300_multiscale_star/
work/runs/<run_id>/windows/obs_1800_step300_multiscale_hybrid_tree/
```

MultiScale 的 `window_table.csv` 使用 `_cur/_ctx/_cum` 后缀区分当前窗口、上下文窗口和累计窗口。MultiScale 的 `edge_window_table.csv` 增加 `window_scope` 字段，当前取值为 `current` 或 `context`。

### `window_table.csv`

每个级联每个窗口一行。

- `cascade_idx`：级联索引。
- `window_idx`：窗口编号，从 1 开始。
- `start_offset`：窗口开始时间，相对根帖，单位秒。
- `end_offset`：窗口结束时间，相对根帖，单位秒。
- `num_retweets`：当前窗口内转发数，不包含 root。
- `cum_retweets`：截至当前窗口结束前累计转发数，不包含 root。
- `num_active_users`：当前窗口活跃转发用户数。
- `num_edges`：当前窗口传播边数。
- `window_heat`：当前实现中等于 `num_retweets`。
- `delta_heat`：当前窗口热度相对上一窗口的变化。

### `node_window_table.csv`

每个级联、窗口、可见用户一行。

- `cascade_idx`：级联索引。
- `window_idx`：窗口编号。
- `user_idx`：用户索引。
- `first_seen_time`：该用户在该级联中首次出现的相对时间。
- `is_root`：是否根用户。
- `num_posts_in_window`：当前窗口内该用户帖子数。
- `cum_posts`：截至当前窗口结束前该用户累计帖子数。
- `in_degree_window`：当前窗口入度。
- `out_degree_window`：当前窗口出度。
- `cum_in_degree`：截至当前窗口结束前累计入度。
- `cum_out_degree`：截至当前窗口结束前累计出度。
- `time_since_root`：当前实现中等于 `first_seen_time`。
- `time_since_first_seen`：窗口结束时间与首次出现时间的差值。

### `edge_window_table.csv`

当前窗口内传播边。后续如果需要累计边，应新增策略字段或单独输出，不覆盖当前语义。

- `cascade_idx`：级联索引。
- `window_idx`：窗口编号。
- `src_user_idx`：源用户。
- `dst_user_idx`：目标用户。
- `src_tweet_idx`：源帖子。
- `dst_tweet_idx`：目标帖子。
- `edge_time`：边对应的绝对时间。
- `edge_offset`：边对应的相对时间。
- `edge_type`：边类型。

### `text_window_table.csv`

窗口内可见文本。

- `cascade_idx`：级联索引。
- `window_idx`：窗口编号。
- `user_idx`：文本发布用户。
- `tweet_idx`：帖子索引。
- `post_type`：`root` 或 `retweet`。
- `text`：文本内容。
- `text_visible_type`：`root_always_visible`、`current_window` 或 `history_visible`。
- `post_offset`：帖子相对根帖的发布时间。

文本进入规则：

- 根文本从第一个窗口开始可见，并在所有窗口保留。
- 转发文本只有在 `post_offset < end_offset` 时可见。
- 转发文本不会提前进入窗口。

### `cascade_window_index.json`

按 `cascade_idx` 保存窗口索引，便于后续 pack 构建和调试定位。

### `window_diagnostics.json`

保存窗口构建统计和基本约束检查。关键字段包括：

- `num_cascades`
- `windows_per_cascade`
- `num_window_rows`
- `num_node_window_rows`
- `num_edge_window_rows`
- `num_text_window_rows`
- `root_text_window_rows`
- `retweet_text_early_violations`

## 特征表

典型位置：

```text
work/runs/<run_id>/features/<window_variant>/
```

当前正式变体：

```text
obs_1800_win300_step300_star
obs_1800_win300_step300_hybrid_tree
obs_1800_step300_multiscale_hybrid_tree
```

每个目录包含：

```text
window_features.csv
node_window_features.csv
feature_diagnostics.json
```

### `window_features.csv`

每个级联每个窗口一行，三套输入保持同一列集。

- `cascade_idx`
- `window_idx`
- `num_retweets_cur`
- `num_retweets_ctx`
- `num_retweets_cum`
- `num_active_users_cur`
- `num_active_users_ctx`
- `num_active_users_cum`
- `num_edges_cur`
- `num_edges_ctx`
- `heat_cur`
- `heat_ctx`
- `heat_cum`
- `delta_heat_cur`

Fixed-5m 输入没有真实 context 字段时，`*_ctx` 使用当前窗口值补齐；累计字段使用已有累计值或当前值补齐，以保证特征维度一致。

### `node_window_features.csv`

每个级联、窗口、用户一行，稳定键为：

```text
cascade_idx + window_idx + user_idx
```

核心字段：

- `cascade_idx`
- `window_idx`
- `user_idx`
- `is_root`
- `first_seen_time`
- `time_since_first_seen`
- `num_posts_cur`
- `num_posts_ctx`
- `num_posts_cum`
- `in_degree_cur`
- `out_degree_cur`
- `in_degree_ctx`
- `out_degree_ctx`
- `in_degree_cum`
- `out_degree_cum`
- `active_window_count`
- `num_texts_cur`
- `num_texts_visible`
- `avg_text_len_cur`
- `avg_text_len_visible`
- `has_tree_feature`
- `depth`
- `parent_time_gap`
- `parent_score`
- `time_score`
- `text_score`
- `activity_score`
- `depth_penalty`
- `load_penalty`
- `root_fallback_flag`

训练和 pack 构建必须按稳定键 join，不依赖行号或 CSV 原始顺序。

## 弱监督标签

典型位置：

```text
work/runs/<run_id>/labels/
```

输出：

```text
weak_event_labels.csv
label_diagnostics.json
```

### `weak_event_labels.csv`

每个 cascade 一行：

- `cascade_idx`
- `weak_score`
- `weak_label`：`1` 表示 positive，`0` 表示 negative，`-1` 表示 ignore。
- `label_confidence`
- `split`：`train`、`valid` 或 `test`。
- `burst_score`
- `text_repetition_score`
- `coordination_score`
- `structure_concentration_score`
- `persistence_score`

split 必须按 `cascade_idx` 划分，避免同一 cascade 的不同窗口泄露到多个数据集。

## Pack

典型位置：

```text
work/runs/<run_id>/packs/obs_1800_step300_multiscale_hybrid_tree/
```

文件：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

当前 `.pt` 使用 pickle stream 格式，逐条保存 sample。读取方式：

```python
import pickle

with open("train.pt", "rb") as f:
    while True:
        try:
            sample = pickle.load(f)
        except EOFError:
            break
```

sample 结构：

```python
sample = {
    "cascade_idx": cascade_idx,
    "window_x": window_x,              # [T, d_w]
    "node_x": node_x,                  # [T, N, d_n]
    "edge_index_current": edge_cur,    # list[T], each [2, E_t]
    "edge_index_context": edge_ctx,    # list[T], each [2, E_t]
    "node_mask": node_mask,            # [T, N]
    "y": event_label,
}
```

当前约定：

- `T = 6`
- `d_w = 12`
- `d_n = 27`
- `y` 只包含 `0` 和 `1`，ignore cascade 不进入 pack。


## 2026-07-02 Schema Update: Global Candidate Edges and Label Versions

Pack samples may now include offline global follow candidate fields:

```text
global_candidate_edge_index:  [2, E_cand]
global_candidate_edge_weight: [E_cand]
```

These fields are candidate pools only. They are not final sampled neighbors. Final sampled neighbors are produced inside `AdaptiveGlobalSampler.forward()` for each batch/window.

The default Label-v1 file remains:

```text
work/runs/<run_id>/labels/weak_event_labels.csv
```

Formal Label-v2 should be written separately:

```text
work/runs/<run_id>/labels_v2_stratified_score/weak_event_labels.csv
work/runs/<run_id>/labels_v2_stratified_score/label_diagnostics.json
```

Label-v2 rows should contain:

```text
cascade_idx
weak_score
burst_score
coordination_score
structure_score
text_score
evidence_hit_count
size_bucket
label
split
```

`13_build_packs.py` accepts either `weak_label` or `label`: `1` positive, `0` negative, `-1` ignore. Label-v1 uses `weak_label`; Label-v2 may use `label` as the public field name.
