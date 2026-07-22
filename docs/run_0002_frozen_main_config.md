# run_0002 主模型冻结配置

本文档用于冻结完成 run_0002 诊断后的 DRAGEN 主模型配置。后续生成最终主模型结果、消融实验结果和论文表格时，除非新开一个 run id，否则不要再修改本文档中固定的配置项。

## 冻结结论

| 配置项 | 冻结值 |
|---|---|
| Pack | `packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool` |
| 训练配置 | `configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml` |
| Label | Label-v2 分层打分弱标签，以冻结 Pack 中实际保存的标签为准 |
| Window | 观测窗口 `1800s`，步长 `300s`，multiscale hybrid tree |
| Text | RoBERTa 文本特征，`text_semantic_dim=64` |
| Key-user pool | 启用；`key_users_per_window=32`，`key_user_max_hops=4`，`global_sampling_mode=key_user_pool` |
| Event Loss | BCE，也就是 `event_loss=bce`；主模型不使用 weighted BCE 或 focal |
| Threshold Strategy | Valid-best MCC；每个 seed 只用该 seed 的 valid 预测选择阈值 |
| Probability Calibrator | none / identity；主分类结果不做概率校准 |
| Checkpoint Rule | 固定 10 epoch 训练后的 final checkpoint，即 `checkpoints/last.pt`；不要把 `best.pt` 当作 MCC/F1 最优 checkpoint |
| Seeds | `0`、`1`、`2` |
| Output Naming | `work/artifacts/main/run_0002_final_dragen_seed{seed}` |

## 冻结依据

seed 1 上的 loss probe 表明，在测试过的事件级损失中，原始 BCE 的验证集分类指标最好：

| Loss | Valid F1 | Valid MCC | Valid AUC | Valid AP |
|---|---:|---:|---:|---:|
| BCE | 0.5782 | 0.5709 | 0.9166 | 0.7980 |
| Weighted BCE, soft | 0.5187 | 0.5268 | 0.9060 | 0.7737 |
| Weighted BCE, auto | 0.5011 | 0.4982 | 0.9006 | 0.7509 |
| Focal | 0.4983 | 0.4821 | 0.8868 | 0.7245 |

阈值校准应进入主结果口径，因为它是目前收益最大的、只依赖验证集的改进。三种子已有结果中，Valid-best MCC 将 test mean F1 从 `0.5896` 提升到 `0.6998`，将 test mean MCC 从 `0.5679` 提升到 `0.6216`。

概率校准不进入主分类结果口径。Platt / isotonic 等校准方法可以改善 NLL、Brier、ECE 等概率质量指标，但在主分类指标 F1/MCC 上没有超过 `identity + Valid-best MCC`。如果论文中报告概率质量，应作为单独的概率校准表，并明确校准器只在 valid 预测上拟合，再冻结后应用到 test。

当前 trainer 中的 `checkpoints/best.pt` 优先按 validation AUC 保存，因此它不是 valid-best-MCC 或 valid-best-F1 checkpoint。冻结主模型统一报告固定 10 epoch 的 final checkpoint。最终命令中保留 `--save-every-epoch` 是为了审计和画曲线，但主结果不要做事后 epoch 选择；除非先冻结新的 checkpoint 选择协议。

## 每个 Seed 的阈值

以下阈值来自已完成的阈值校准分析。它们只用于记录当前完成实验的选择结果：

| Seed | Valid-best MCC threshold | Valid F1 | Valid MCC | Valid Precision | Valid Recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.225 | 0.7490 | 0.6815 | 0.7427 | 0.7555 |
| 1 | 0.140 | 0.7058 | 0.6321 | 0.7351 | 0.6787 |
| 2 | 0.196 | 0.7525 | 0.6863 | 0.7496 | 0.7555 |

如果后续正式三种子重新训练，需要对每个新 run 的 valid 预测重新按同一条 Valid-best MCC 规则选择阈值。禁止使用 test 调阈值。

## 最终主模型命令

Seed 0:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 0 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/main/run_0002_final_dragen_seed0
```

Seed 1:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/main/run_0002_final_dragen_seed1
```

Seed 2:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 2 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/main/run_0002_final_dragen_seed2
```

## 报告规则

- 主表报告 seeds `0,1,2` 的 mean/std。
- 主表阈值使用 Valid-best MCC：只在 valid 上选择，再一次性应用到 test。
- 主表概率使用原始模型概率，不做概率校准。
- 主表 checkpoint 使用 epoch 10 final checkpoint。
- adapted baselines 和模块替换实验在实现与配置单独冻结之前，不写入正式最终结果。
