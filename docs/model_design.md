# DRAGEN-Full 模型结构

当前实现采用论文完整版 `DRAGEN-Full`，不再使用 light/v0 口径。模型代码放在：

```text
src/dragen/models/
```

模型模块只负责网络结构和前向计算，不直接读取 CSV、不构建窗口、不写报告。这些职责分别属于 `data`、`windowing`、`training` 和 `evaluation`。

## 论文模块对应关系

```text
source_evidence_encoder.py   # 4.1 多源异常证据分源编码
evidence_reader.py           # 4.1/4.3 选择性证据读取
local_role_encoder.py        # 4.2 滑动窗口局部角色建模
adaptive_global_sampler.py   # 4.2 全局结构先验自适应采样
global_prior_encoder.py      # 4.2 全局结构先验编码
temporal_memory.py           # 4.2 跨窗口节点记忆
manipulation_state.py        # 4.3 历史操控状态累积
bayesian_gate.py             # 4.3 先验观测贝叶斯门控与不确定性
event_pooling.py             # 4.3 事件级注意力聚合
dragen_full.py               # 组装完整 DRAGEN-Full
```

## 输入

DRAGEN-Full 使用 pack 输入：

```text
work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
  train.pt
  valid.pt
  test.pt
  meta.json
```

当前 `.pt` 是 pickle stream，不是 `torch.save` 单对象。读取逻辑在：

```text
src/dragen/data/pack_reader.py
```

collate 后的 batch：

```text
window_x: [B, T, 12]
node_x: [B, T, N, 27]
edge_index_current: list[B][T]
edge_index_context: list[B][T]
node_mask: [B, T, N]
y: [B]
```

当前固定：

```text
T = 6
role_num = 5
```

## 角色集合

角色顺序固定为：

```text
producer
amplifier
suppressor
reframer
ordinary
```

代码、配置、导出字段和论文图例都必须使用这五类。

## Forward 输出

`DRAGENFull.forward()` 返回完整解释字典：

```text
event_logit
event_prob
event_strength
node_logit
node_prob
node_strength
source_evidence
local_role_repr
global_prior
history_state
manip_state
state_update_gate
role_prob
dominant_role
shock
gate_obs_weight
gate_prior_weight
uncertainty_log_var
event_attention
sampled_global_edges
sampled_global_weights
sampled_global_neighbors
node_mask
```

节点窗口级解释结果由 `src/dragen/evaluation/export_predictions.py` 导出。

## 联合训练目标

实现位置：

```text
src/dragen/training/losses.py
```

当前损失：

```text
L_event          事件级 BCE
L_jump           证据冲击加权的状态连续性损失
L_struct         sampled edge 正负对比结构约束
L_align          多源证据 CORAL 对齐
L_uncertainty    异方差不确定性正则
L_role           可选伪角色损失，默认关闭
```

总损失：

```text
L = L_event
  + lambda_jump * L_jump
  + lambda_struct * L_struct
  + lambda_align * L_align
  + lambda_uncertainty * L_uncertainty
  + lambda_role * L_role
```

默认：

```text
lambda_jump = 0.01
lambda_struct = 0.005
lambda_align = 0.001
lambda_uncertainty = 0.001
lambda_role = 0.0
```

## 训练入口

正式入口：

```text
scripts/16_train_dragen_full.py
```

Debug 命令：

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_debug \
  --epochs 1 \
  --batch-size 2 \
  --max-train-samples 200 \
  --max-valid-samples 100 \
  --max-test-samples 100 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --device auto
```

当前 debug 已通过：

```text
train_loss = 0.3926
valid_auc = 0.9113
test_auc = 0.9005
```

输出：

```text
work/artifacts/dragen_full_debug/reports/metrics.json
work/artifacts/dragen_full_debug/reports/loss_breakdown.json
work/artifacts/dragen_full_debug/predictions/*.csv
work/artifacts/dragen_full_debug/checkpoints/best.pt
```

## 消融

配置位于：

```text
configs/model/
```

当前消融：

```text
ablation_no_tree.yaml
ablation_no_multiscale.yaml
ablation_no_role.yaml
ablation_no_memory.yaml
ablation_no_global_prior.yaml
ablation_no_adaptive_sampling.yaml
ablation_no_gate.yaml
ablation_no_uncertainty.yaml
```

`w/o Tree` 使用 `obs_1800_win300_step300_star` pack。  
`w/o MultiScale` 使用 `obs_1800_win300_step300_hybrid_tree` pack。  
其他消融使用 MultiScale HybridTree pack，并通过命令行关闭模块。
