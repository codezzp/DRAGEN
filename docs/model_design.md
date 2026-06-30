# DRAGEN 模型结构

DRAGEN 的模型代码放在：

```text
src/dragen/models/
```

当前这些文件只是结构占位。模型模块只负责网络结构和前向计算，不负责读取原始文件、构建窗口或写报告。

## 组件划分

```text
evidence_encoder.py       # 多源证据编码。
local_role_encoder.py     # 局部传播角色编码。
temporal_memory.py        # 窗口级时间记忆。
global_prior.py           # 全局图先验。
bayesian_gate.py          # 局部证据和全局先验之间的门控。
event_pooling.py          # 节点/窗口/事件级聚合。
dragen.py                 # 组装完整 DRAGEN 模型。
```

## 消融版本

模型消融由 `configs/model/` 控制：

- `dragen_base`：完整模型。
- `dragen_no_global_prior`：去掉全局图先验。
- `dragen_no_memory`：去掉时间记忆模块。
- `dragen_no_gate`：去掉贝叶斯门控模块。

## 模块边界

模型模块不应该：

- 直接读取 CSV/JSONL 原始文件。
- 硬编码 `work/runs/<run_id>` 路径。
- 构建窗口。
- 写评估报告。

这些职责分别属于 `data`、`windowing`、`training` 和 `evaluation` 模块。
