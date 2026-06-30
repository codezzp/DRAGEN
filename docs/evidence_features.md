# 多源证据特征设计

DRAGEN 后续会使用多种异常传播证据。特征构建要模块化，并且所有特征都必须能通过稳定 ID 对齐。

## 模块位置

```text
src/dragen/features/text_features.py
src/dragen/features/emotion_features.py
src/dragen/features/behavior_features.py
src/dragen/features/structure_features.py
src/dragen/features/evidence_align.py
```

## 证据类型

文本证据：

- 根文本按 `cascade_idx` 对齐。
- 转发文本按 `tweet_idx` 对齐。

情绪证据：

- 从根文本或转发文本中抽取情绪、极性、强度等信息。
- 在进入窗口聚合前，要先和 root/retweet 记录对齐。

行为证据：

- 转发时间。
- 用户活跃次数。
- 窗口内重复转发、集中爆发等行为模式。

结构证据：

- 局部传播入度、出度。
- 当前窗口度数和累计度数。
- 全局图邻居统计或角色信息。

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

后续 pack 路径也应保留结构后缀，例如：

```text
packs/obs_1800_win300_step300_star/
packs/obs_1800_win300_step300_tree/
```

## 输出规则

和单个 run 绑定的特征应写在该 run 下面，或在文件名中显式包含 `run_id`。跨 run 的统计、报告、图表放到 `work/artifacts/`。
