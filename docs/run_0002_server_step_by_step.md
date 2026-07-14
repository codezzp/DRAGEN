# run_0002 服务器逐步实验执行手册

> 本文档根据 Obsidian 中的《DRAGEN 后续实验执行计划（修改后）》整理，用于服务器按步执行。

## 0. 服务器快速开始

先更新代码：

```bash
cd /path/to/DRAGEN
git fetch codezzp
git checkout experiment/run-0002-calibration-diagnostics
git pull
export PYTHONPATH=src
```

确认当前分支：

```bash
git branch --show-current
git log --oneline -3
```

确认主实验 pack 存在：

```bash
ls packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

当前服务器第一步从概率校准开始：

```bash
python scripts/23_calibrate_probabilities.py
```

---

# DRAGEN 后续实验执行计划（修改后）

> 适用范围：当前 `run_0002` 性能优化与实验闭环。  
> 当前主线：`Feature-v2 + RoBERTa Text + Label-v2 + Key-user Pool Global Prior`。  
> 核心原则：所有方案选择只使用验证集；测试集只在配置冻结后用于最终报告。

---

## 1. 当前状态

### 1.1 已完成

1. `run_0002` 的窗口、HybridTree、Feature-v2、RoBERTa 文本语义、Label-v2、Global Candidate 和 Key-user Pool Pack 已完成。
2. DRAGEN 已完成 `seed0 / seed1 / seed2` 三次训练。
3. 默认阈值下三种子测试结果已经汇总：
   - AUC：`0.9220 ± 0.0086`
   - AP：`0.7927 ± 0.0086`
   - F1：`0.5896 ± 0.0316`
   - MCC：`0.5679 ± 0.0231`
   - P@100：`0.9767 ± 0.0115`
   - P@500：`0.7620 ± 0.0122`
4. 阈值校准已经完成：
   - Valid-best F1：测试 F1 约为 `0.6999`
   - Valid-best MCC：测试 F1 约为 `0.6998`，MCC 约为 `0.6216`
5. Epoch 曲线分析已经完成。现有旧实验未完整保存各轮 checkpoint，因此当前不能声称已按最佳 epoch 重新评估测试集。
6. Weighted BCE、Focal Loss、概率校准和损失生效诊断的命令入口已经准备好。

### 1.2 尚未完成

1. 概率校准结果。
2. 辅助损失是否真实生效的诊断。
3. BCE、Weighted BCE、Focal Loss 的 seed1 对比。
4. 最终主模型配置冻结。
5. 冻结配置下的三随机种子正式复跑。
6. Adapted baselines。
7. 核心消融实验。
8. 小模块替换实验。
9. Label-v5 严格标签鲁棒性实验。
10. 论文第五章结果表与分析文字。

---

## 2. 总体执行路线

```text
阶段 A：完成低成本性能优化
    概率校准
    → 辅助损失诊断
    → seed1 loss probe
    → 必要时 learning-rate probe
    → 确定 checkpoint 与阈值策略

阶段 B：冻结主模型
    固定 pack、loss、threshold、checkpoint、seed、输出命名

阶段 C：正式主结果
    最终配置 seed0 / seed1 / seed2
    → 汇总 mean ± std

阶段 D：主实验
    实现并运行 adapted baselines

阶段 E：核心消融
    验证主要模块是否有效

阶段 F：模块替换
    验证具体小模块优于常规替代方案

阶段 G：鲁棒性与补充实验
    Label-v5、参数敏感性、效率分析

阶段 H：论文结果整理
    生成主表、消融表、替换表、曲线和分析文字
```

---

# 3. 阶段 A：完成低成本性能优化

## 3.1 概率校准

### 目标

当前模型排序能力较强，但默认阈值明显偏保守。概率校准用于判断输出概率是否可信，并降低不同 seed 间最佳阈值差异。

### 执行命令

```bash
python scripts/23_calibrate_probabilities.py
```

需要保存逐样本校准概率时：

```bash
python scripts/23_calibrate_probabilities.py --write-predictions
```

### 对比方法

| 方法 | 是否重训 | 用途 |
|---|---:|---|
| None | 否 | 原始概率 |
| Temperature Scaling | 否 | 单参数稳定校准 |
| Platt Scaling | 否 | 逻辑回归式校准 |
| Isotonic Regression | 否 | 非参数单调校准 |

### 主要指标

```text
NLL
Brier Score
ECE
```

同时记录校准后的：

```text
Precision
Recall
F1
MCC
```

### 选择规则

只根据验证集选择校准器：

1. Valid ECE、Brier Score 或 NLL 明显降低；
2. Valid AUC/AP 不变；
3. 校准后的最佳阈值在不同 seed 间更稳定；
4. 不因校准导致 F1/MCC 明显下降。

优先考虑 Temperature Scaling。Isotonic 只作为补充，因为验证集规模有限时可能过拟合。

---

## 3.2 辅助损失生效诊断

### 目标

确认论文中的训练目标是否真实进入总损失并参与反向传播，避免公式、配置和代码不一致。

### 检查命令

```bash
cat work/artifacts/<run>/reports/loss_breakdown.json
```

重点检查：

```text
loss_event
weighted_loss_event
loss_contribution_event

loss_jump
weighted_loss_jump
loss_contribution_jump

loss_struct
weighted_loss_struct
loss_contribution_struct

loss_sampler
weighted_loss_sampler
loss_contribution_sampler

loss_uncertainty
weighted_loss_uncertainty
loss_contribution_uncertainty

loss_role
weighted_loss_role
loss_contribution_role
```

### 判断规则

| 状态 | 处理方式 |
|---|---|
| 原始损失和加权损失均正常非零 | 可保留为有效训练目标 |
| 原始损失非零但权重为 0 | 配置关闭，论文中说明未启用 |
| 原始损失长期为 0 | 检查输入、mask、分支条件或实现 |
| 加权贡献低于总损失的极小比例 | 视为弱正则，不应夸大作用 |
| `loss_role=0` 且无角色标签 | 属于合理关闭，不写成角色监督训练 |

### 输出

新增一份诊断文档：

```text
docs/run_0002_loss_effectiveness_analysis.md
```

内容至少包括：

```text
每个损失的均值
配置权重
加权贡献
相对贡献率
是否启用
是否需要修复
论文中的最终表述
```

---

## 3.3 类别不平衡 Loss Probe

### 目标

判断修改事件级分类损失能否在不损害 AUC/AP 的前提下进一步提高 Valid F1/MCC。

### 第一轮只跑 seed1

#### Weighted BCE soft

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss weighted_bce \
  --pos-weight soft \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_weighted_bce_soft_seed1
```

#### Weighted BCE auto

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss weighted_bce \
  --pos-weight auto \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_weighted_bce_auto_seed1
```

#### Focal Loss

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss focal \
  --focal-alpha 0.75 \
  --focal-gamma 2.0 \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_focal_seed1
```

### 对比表

| Loss | Valid AUC | Valid AP | Valid F1 | Valid MCC | Valid Precision | Valid Recall | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE |  |  |  |  |  |  |  |
| Weighted BCE soft |  |  |  |  |  |  |  |
| Weighted BCE auto |  |  |  |  |  |  |  |
| Focal Loss |  |  |  |  |  |  |  |

每个 loss 都需要同时报告：

1. 默认阈值 0.5 结果；
2. Valid-best MCC 阈值结果；
3. AUC/AP；
4. NLL/Brier/ECE。

### 升级为三 seed 的条件

某个 loss 只有同时满足以下条件，才进入三种子正式复跑：

```text
1. seed1 Valid F1 或 MCC 提升 >= 0.02；
2. Valid AUC/AP 下降不超过 0.01；
3. Precision 不出现明显崩塌；
4. 概率校准指标不明显恶化；
5. 提升不只是由默认阈值输出尺度变化造成。
```

如果 Weighted BCE/Focal 只改善默认阈值 F1，但 Valid-best MCC 后的结果没有提升，则继续使用 BCE。

---

## 3.4 Learning-rate Probe：仅在必要时执行

只有当 loss probe 结果不稳定或训练曲线波动明显时，再运行学习率实验。

建议只试：

```text
1e-3
5e-4
3e-4
```

不做大规模网格搜索。若任一配置没有明显超过当前 `lr=1e-3`，立即结束学习率调试。

---

## 3.5 Checkpoint 选择

后续最终训练必须开启：

```bash
--save-every-epoch
```

至少保存：

```text
last.pt
best_valid_f1.pt
best_valid_mcc.pt
```

最终建议：

- 主表强调类别不平衡综合性能：优先 `best_valid_mcc.pt`；
- 若最佳 checkpoint 与 last 的差异很小：使用 last，保证流程简单；
- 不得用测试集选择 checkpoint。

---

# 4. 阶段 B：冻结主模型

性能优化结束后，新建：

```text
docs/run_0002_frozen_main_config.md
```

## 4.1 冻结配置表

| 配置项 | 冻结值 |
|---|---|
| Dataset line | run_0002 |
| Pack | `packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool` |
| Label | Label-v2 |
| Window | 30 min observation, 5 min step, MultiScale HybridTree |
| Window steps | T=6 |
| Window feature dim | 24 |
| Node feature dim | 47 |
| Text | RoBERTa semantic 64-dim |
| Global prior | Key-user Pool |
| Key users per window | 32 |
| Event loss | 待实验确定 |
| Threshold | 优先 Valid-best MCC |
| Probability calibrator | 待实验确定 |
| Checkpoint rule | 待实验确定 |
| Seeds | 0 / 1 / 2 |
| Main output | `work/artifacts/run_0002_final_dragen_seed*` |

## 4.2 冻结规则

冻结后禁止修改：

```text
pack
label
窗口配置
主模型隐藏维度
事件损失
阈值选择规则
checkpoint 选择规则
评价指标
数据划分
```

消融实验只能关闭指定模块，不能同时改变其他配置。

---

# 5. 阶段 C：最终主模型三种子复跑

只有当 event loss、checkpoint 或训练配置发生改变时，才需要重新跑 seed0/1/2。

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed <SEED> \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_final_dragen_seed<SEED>
```

正式结果统一报告：

```text
mean ± std over seed0 / seed1 / seed2
```

主表指标：

```text
Acc
Precision
Recall
F1
AUC
AP
MCC
P@100
P@500
```

补充指标可放正文或附表：

```text
Balanced Accuracy
Macro-F1
Brier Score
ECE
```

---

# 6. 阶段 D：Adapted Baselines

## 6.1 实现顺序

1. `TDCB-adapt`
2. `UWD-FSN-adapt`
3. `IOHunter-adapt`
4. `LEN-GNN-adapt`
5. `EDCOC-adapt`
6. `X-CoIA-adapt`

## 6.2 公平性要求

所有 baseline 必须统一：

```text
同一 Label-v2 数据划分
同一 train / valid / test
同一事件级评价指标
同一 valid 阈值选择原则
同一测试集报告原则
```

不能让 DRAGEN 使用 Valid-best MCC，而 baseline 固定使用 0.5。每个模型都应在自己的 valid 预测上选择阈值，再固定到 test。

## 6.3 主实验表

| 方法 | Acc | Prec. | Rec. | F1 | AUC | AP | MCC | P@100 | P@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IOHunter-adapt |  |  |  |  |  |  |  |  |  |
| LEN-GNN-adapt |  |  |  |  |  |  |  |  |  |
| EDCOC-adapt |  |  |  |  |  |  |  |  |  |
| X-CoIA-adapt |  |  |  |  |  |  |  |  |  |
| UWD-FSN-adapt |  |  |  |  |  |  |  |  |  |
| TDCB-adapt |  |  |  |  |  |  |  |  |  |
| DRAGEN |  |  |  |  |  |  |  |  |  |

---

# 7. 阶段 E：核心消融实验

## 7.1 第一批必须完成

```text
w/o Global Prior
w/o Adaptive Sampling
w/o Memory
w/o Role
w/o Gate
w/o Uncertainty
```

## 7.2 第二批按代码实际支持情况补充

```text
w/o RoBERTa Text
w/o Jump Loss
w/o MultiScale
w/o HybridTree
```

注意：

- `w/o Text` 必须明确是关闭 RoBERTa 语义，还是关闭全部文本特征；
- `w/o Jump Loss` 只有在 Jump Loss 实际非零并参与训练时才有意义；
- 若某辅助损失始终未生效，不应设计对应消融。

## 7.3 消融执行前检查

现有部分消融 YAML 历史上可能指向其他 pack。执行前统一检查：

```text
pack_dir
label version
loss
threshold strategy
checkpoint rule
output directory
```

不得出现主模型使用 Label-v2，而消融误用 Label-v4 的情况。

## 7.4 消融结果表

| 模型 | Acc | F1 | AUC | AP | MCC | ΔF1 | ΔAUC | ΔAP | ΔMCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o Global Prior |  |  |  |  |  |  |  |  |  |
| w/o Adaptive Sampling |  |  |  |  |  |  |  |  |  |
| w/o Memory |  |  |  |  |  |  |  |  |  |
| w/o Role |  |  |  |  |  |  |  |  |  |
| w/o Gate |  |  |  |  |  |  |  |  |  |
| w/o Uncertainty |  |  |  |  |  |  |  |  |  |
| DRAGEN |  |  |  |  |  | — | — | — | — |

## 7.5 种子数量

- 完整模型：3 seeds；
- 核心消融：至少 1 seed，最好 3 seeds；
- 资源有限时，先用 seed1 跑全部消融，再对差异较小或结论关键的消融补 seed0/2。

---

# 8. 阶段 F：模块替换实验

消融回答“模块有没有用”，替换实验回答“本文设计是否优于普通实现”。优先保留三组。

## 8.1 跨窗口记忆模块替换

| 方法 | 说明 |
|---|---|
| Last Window | 仅使用当前窗口 |
| Mean Pooling | 历史窗口平均 |
| EMA | 指数滑动平均 |
| LSTM | 序列建模替代 |
| GRU Memory | 本文方法 |

重点指标：

```text
F1
AUC
AP
MCC
prob_jump_mean
role_transition_rate
训练时间
```

## 8.2 先验—观测融合模块替换

| 方法 | 说明 |
|---|---|
| Concat | 直接拼接后分类 |
| Fixed Weight | 固定 0.5/0.5 |
| Attention Fusion | 普通注意力 |
| Prior–Observation Gate | 本文方法 |

重点指标：

```text
F1
AUC
AP
MCC
mean_gate_obs
mean_gate_prior
```

## 8.3 全局候选/采样策略替换

| 方法 | 说明 |
|---|---|
| Random Top-K | 随机候选 |
| Degree Top-K | 按节点度排序 |
| Static Similarity Top-K | 固定相似度排序 |
| Key-user Pool / Adaptive Sampling | 本文方法 |

重点指标：

```text
AUC
AP
P@100
P@500
Epoch Time
GPU Memory
```

## 8.4 可选：文本表示替换

| 文本表示 | 说明 |
|---|---|
| StatText | 数量、长度、轻量相似度 |
| RoBERTa-CLS | CLS 表示 |
| RoBERTa-Mean | Mean Pooling |
| RoBERTa Window Semantic | 当前方案 |

仅在输入文本覆盖和实现口径清楚时执行。

---

# 9. 阶段 G：鲁棒性与补充实验

## 9.1 Label-v5 严格标签鲁棒性

当前已有 Label-v5 pack，可用于检验严格弱标签条件下模型是否仍保持稳定排序能力。

| Label | AUC | AP | F1 | MCC | P@100 |
|---|---:|---:|---:|---:|---:|
| Label-v2 |  |  |  |  |  |
| Label-v5 |  |  |  |  |  |

定位：这是标签严格度鲁棒性实验，不是第二数据集实验。

## 9.2 K 值敏感性

只在主模型冻结后做：

```text
K = 16 / 32 / 64
```

同时报告：

```text
AUC
AP
F1
P@100
epoch time
显存
```

## 9.3 效率分析

至少记录：

```text
参数量
每 epoch 时间
峰值显存
单样本或单批推理时间
```

重点比较：

```text
w/o Global Prior
Static Top-K
Key-user Pool / Adaptive Sampling
DRAGEN-Full
```

---

# 10. 论文第五章建议结构

```text
5 实验验证及结果分析

5.1 实验设置
  5.1.1 实验数据
  5.1.2 基线方法
  5.1.3 评价指标
  5.1.4 参数设置

5.2 实验结果与分析
  5.2.1 主实验结果
  5.2.2 阈值与概率校准分析
  5.2.3 消融实验
  5.2.4 模块替换实验
  5.2.5 标签鲁棒性与参数敏感性
  5.2.6 效率分析

5.3 本章小结
```

若篇幅有限，可将阈值校准和概率校准合并为“决策阈值与概率质量分析”，将 K 值和效率放入同一小节。

---

# 11. 结果文件组织

```text
work/artifacts/
  _analysis/
    run_0002_threshold_calibration/
    run_0002_probability_calibration/
    run_0002_epoch_selection/
    run_0002_loss_effectiveness/
    run_0002_loss_comparison/

  main/
    run_0002_final_dragen_seed0/
    run_0002_final_dragen_seed1/
    run_0002_final_dragen_seed2/

  baselines/
    iohunter_adapt/
    len_gnn_adapt/
    edcoc_adapt/
    x_coia_adapt/
    uwd_fsn_adapt/
    tdcb_adapt/

  ablations/
    no_global_prior/
    no_adaptive_sampling/
    no_memory/
    no_role/
    no_gate/
    no_uncertainty/

  replacements/
    memory/
    fusion/
    sampling/
    text/

  robustness/
    label_v5/
    key_user_k/
```

每个正式运行目录必须保留：

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
reports/metrics.json
reports/epoch_metrics.csv
reports/loss_breakdown.json
predictions/valid_event_predictions.csv
predictions/test_event_predictions.csv
checkpoints/
```

---

# 12. 立即执行清单

```text
[ ] 1. 运行概率校准
[ ] 2. 汇总 None / Temperature / Platt / Isotonic 的 Valid 指标
[ ] 3. 跑一次 loss 生效诊断
[ ] 4. 明确 struct / sampler / jump / uncertainty 是否真实生效
[ ] 5. 运行 seed1 Weighted BCE soft
[ ] 6. 运行 seed1 Weighted BCE auto
[ ] 7. 运行 seed1 Focal Loss
[ ] 8. 使用 Valid-best MCC 统一比较四种 loss
[ ] 9. 判断是否需要 learning-rate probe
[ ] 10. 冻结主模型配置
[ ] 11. 最终配置运行 seed0 / seed1 / seed2
[ ] 12. 实现 adapted baselines
[ ] 13. 运行核心消融
[ ] 14. 完成 Memory / Fusion / Sampling 三组模块替换
[ ] 15. 运行 Label-v5 鲁棒性
[ ] 16. 汇总第五章表格和分析
```

---

# 13. 停止调参条件

满足以下条件后必须停止继续优化，进入消融：

```text
1. 最终 loss 已由 valid 指标确定；
2. 阈值策略已冻结；
3. checkpoint 规则已冻结；
4. 三种子 AUC/AP 稳定；
5. F1/MCC 已达到当前可接受水平；
6. 新增实验连续两次没有带来有效提升；
7. 所有关键辅助损失状态已经解释清楚。
```

不要因为单个 seed 偶然提高继续无限调参。

---

# 14. 最终实验论证链

```text
主实验：
DRAGEN 相比 adapted baselines 整体性能更好。

阈值与概率分析：
模型排序能力稳定，校准后分类性能更加平衡。

消融实验：
全局先验、动态采样、跨窗口记忆、角色感知、门控和不确定性模块均具有贡献。

模块替换：
本文的 GRU Memory、Prior–Observation Gate 和 Key-user Pool/Adaptive Sampling
优于简单平均、固定融合和静态采样。

鲁棒性实验：
在更严格的 Label-v5 条件下，模型仍保持稳定识别能力。

效率实验：
模型增加的全局和时序模块带来可接受的计算成本。
```

---

## 当前最关键的一句话

> 现在先完成概率校准、辅助损失诊断和 seed1 loss probe；确定最终配置后立即冻结主模型，再开始 baseline、消融和三组模块替换实验。
