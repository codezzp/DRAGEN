# run_0002 Loss 生效诊断

本文档记录 run_0002 主模型三种子训练中的 loss 生效情况，用于说明论文训练目标、YAML 配置和实际训练日志是否一致。

## 输入来源

诊断使用以下三个主模型历史运行的 `loss_breakdown.json`：

```text
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/reports/loss_breakdown.json
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1/reports/loss_breakdown.json
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2/reports/loss_breakdown.json
```

对应训练配置为：

```text
configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml
```

## 判断规则

```text
原始 loss 非零，且配置权重非零：真实进入总损失，可认为生效。
原始 loss 非零，但配置权重为 0：代码有计算，但当前配置关闭。
原始 loss 长期为 0，且配置权重非零：当前运行中没有实际贡献，需要谨慎表述。
原始 loss 为 0，且配置权重为 0：配置关闭或没有对应监督信号。
```

## 配置权重

| Loss | 配置权重 | 说明 |
|---|---:|---|
| `loss_event` | 1.000 | 主事件分类 BCE loss |
| `loss_jump` | 0.010 | 跳变/时序变化辅助 loss |
| `loss_struct` | 0.005 | 结构辅助 loss |
| `loss_align` | 0.001 | 对齐辅助 loss |
| `loss_uncertainty` | 0.001 | 不确定性辅助 loss |
| `loss_role` | 0.000 | 角色监督 loss，当前配置关闭 |
| `loss_sampler_edge` | 0.005 | sampler edge 辅助 loss |
| `loss_sampler_hub` | 0.001 | sampler hub 辅助 loss |
| `loss_sampler_temp` | 0.005 | sampler temperature 辅助 loss |

## 最后一轮三种子均值

下表统计三种子第 10 epoch 的 validation loss。`weighted_mean_final` 表示乘以配置权重后对 `loss_total` 的实际贡献。

| Loss | Weight | Raw mean, epoch 10 | Weighted mean, epoch 10 | Raw min, all epochs | Raw max, all epochs | 诊断结论 |
|---|---:|---:|---:|---:|---:|---|
| `loss_event` | 1.000 | 0.2954906951 | 0.2954906951 | 0.2744824224 | 0.4858725148 | 主事件分类 loss，正常生效 |
| `loss_jump` | 0.010 | 0.0034751963 | 0.0000347520 | 0.0000925080 | 0.0066818250 | 原始值非零，进入加权总损失，生效但贡献很小 |
| `loss_struct` | 0.005 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 当前运行中未产生实际贡献，不应写成已生效训练目标 |
| `loss_align` | 0.001 | 0.0002301644 | 0.0000002302 | 0.0001074739 | 0.0003360021 | 原始值非零，进入加权总损失，贡献极小 |
| `loss_uncertainty` | 0.001 | 0.3197140773 | 0.0003197141 | 0.0979906325 | 0.8351479862 | 原始值非零，进入加权总损失，生效 |
| `loss_role` | 0.000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 当前配置关闭；没有进入主模型训练目标 |
| `loss_sampler_edge` | 0.005 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 当前运行中未产生实际贡献，不应写成已生效训练目标 |
| `loss_sampler_hub` | 0.001 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 当前运行中未产生实际贡献，不应写成已生效训练目标 |
| `loss_sampler_temp` | 0.005 | 0.0002034871 | 0.0000010174 | 0.0000379307 | 0.0005342298 | 原始值非零，进入加权总损失，贡献很小 |

## 第 10 Epoch 总损失核对

| Seed | Epoch | `loss_total` | `loss_event` | 加权辅助 loss 合计 | `train_loss` |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.2746363664 | 0.2744824224 | 0.0001539457 | 0.3089049507 |
| 1 | 10 | 0.3328323345 | 0.3326493520 | 0.0001829828 | 0.3242321578 |
| 2 | 10 | 0.2800705230 | 0.2793403110 | 0.0007302124 | 0.3296995077 |

三种子的 `loss_total` 与 `loss_event + weighted auxiliary losses` 基本一致，说明已记录的非零辅助项确实进入了总损失，而不是只被单独打印。

## 最终结论

主模型训练中可以明确写为生效的 loss：

```text
loss_event
loss_jump
loss_align
loss_uncertainty
loss_sampler_temp
```

其中 `loss_event` 是主导项；`loss_uncertainty` 有稳定的加权贡献；`loss_jump`、`loss_align`、`loss_sampler_temp` 的加权贡献较小，论文中应表述为弱辅助正则项，不宜夸大。

当前不能写成已生效训练目标的 loss：

```text
loss_struct
loss_sampler_edge
loss_sampler_hub
```

这些项在三种子全部 epoch 中原始值均为 0。若论文需要强调这些模块，需要先检查对应实现、输入 mask、触发条件或监督信号，并在新实验中确认非零后再写入正式训练目标说明。

当前明确关闭的 loss：

```text
loss_role
```

`loss_role` 的配置权重为 0，日志中原始值也为 0。主模型结果中不要表述为使用了角色监督训练；如果保留角色相关模块，只能表述为模型结构或特征侧设计，不应表述为角色标签监督 loss 生效。

## 论文写法建议

建议在方法或实验设置中写：

```text
主模型采用事件分类 BCE 作为主要训练目标，并加入跳变约束、表示对齐、不确定性约束和 sampler temperature 正则等弱辅助项。结构 loss、sampler edge/hub loss 在当前 run_0002 配置下未产生非零贡献；角色监督 loss 在主模型中关闭。
```

如果后续修复 `loss_struct`、`loss_sampler_edge` 或 `loss_sampler_hub`，应重新运行三种子或至少重新做 seed1 诊断，并更新本文档。
