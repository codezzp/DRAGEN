# 多源证据特征设计

DRAGEN-Full 已经进入模型实验阶段。当前特征构建采用统计特征 v1，不使用 RoBERTa 或复杂情绪模型；模型内部通过 `feature_schema.py` 将节点窗口特征拆分为多源证据。

## 模块位置

```text
src/dragen/features/build_features.py
src/dragen/data/feature_schema.py
src/dragen/models/source_evidence_encoder.py
src/dragen/models/evidence_reader.py
```

## 证据类型

当前分源证据：

```python
FEATURE_GROUPS = {
    "text": [
        "num_texts_cur",
        "num_texts_visible",
        "avg_text_len_cur",
        "avg_text_len_visible",
    ],
    "emotion": [],
    "behavior": [
        "num_posts_cur",
        "num_posts_ctx",
        "num_posts_cum",
        "active_window_count",
        "time_since_first_seen",
    ],
    "structure": [
        "in_degree_cur",
        "out_degree_cur",
        "in_degree_ctx",
        "out_degree_ctx",
        "in_degree_cum",
        "out_degree_cum",
        "depth",
        "parent_time_gap",
        "parent_score",
        "time_score",
        "text_score",
        "activity_score",
        "depth_penalty",
        "load_penalty",
        "root_fallback_flag",
    ],
}
```

`emotion` 组当前为空，作为论文模块接口保留。后续若加入情绪模型，只能在特征层补字段和 schema，不应破坏 pack 结构。

## 当前输出

特征构建入口：

```bash
python scripts/11_build_features.py --run-id run_0002 --tree-edges work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv
```

输出：

```text
work/runs/run_0002/features/obs_1800_win300_step300_star/
work/runs/run_0002/features/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
```

每个目录包含：

```text
window_features.csv
node_window_features.csv
feature_diagnostics.json
```

当前验收：

```text
window_features = 511578
node_window_features = 5940259
nan_count = 0
inf_count = 0
```

## 对齐规则

每个特征文件都必须明确自己的对齐键，不能只依赖行顺序。

允许的对齐键包括：

- `cascade_idx`
- `tweet_idx`
- `user_idx`
- `cascade_idx + window_idx`
- `cascade_idx + window_idx + user_idx`

当前窗口构建已经固定输出：

```text
window_table.csv
node_window_table.csv
edge_window_table.csv
text_window_table.csv
```

特征阶段应优先从这些表读取：

- 文本证据来自 `text_window_table.csv`。
- 行为证据来自 `node_window_table.csv`。
- 结构证据来自 `node_window_table.csv` 和 `edge_window_table.csv`。
- 窗口级舆情状态来自 `window_table.csv`。

结构证据必须记录边结构来源：

- `star`：原始星形边窗口。
- `inferred_tree`：时间一致代理传播树窗口。

当前 pack 路径保留结构和窗口策略后缀：

```text
packs/obs_1800_win300_step300_star/
packs/obs_1800_win300_step300_hybrid_tree/
packs/obs_1800_step300_multiscale_hybrid_tree/
```

## 输出规则

和单个 run 绑定的特征应写在该 run 下面，或在文件名中显式包含 `run_id`。跨 run 的统计、报告、图表放到 `work/artifacts/`。
