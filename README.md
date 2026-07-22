# DRAGEN

本仓库用于组织化级联传播预测实验。本文档只保留当前有效信息，旧分支和过时命令已移除。

## 当前主线

```text
Feature-v2 + RoBERTa Text + Key-user Pool Global Prior
```

当前优化分支：

```text
experiment/run-0002-calibration-diagnostics
```

当前阶段只做主模型性能和稳定性优化，不再扩展复杂模型结构。

## 主实验数据包

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

该 pack 必须包含：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

训练阶段只读取 pack，不重新运行 RoBERTa、文本降维或 key-user pool 构建。

## 主训练配置

```text
configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml
```

正式训练模板：

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

## 训练后必做分析

```bash
python scripts/21_calibrate_thresholds.py
python scripts/22_summarize_epoch_selection.py
python scripts/23_calibrate_probabilities.py
```

概率校准必须遵循：在 valid 上拟合，固定参数，再应用到 test。不要用 test 选校准方法或阈值。

## Loss 生效诊断

每次训练后检查：

```bash
cat work/artifacts/<run>/reports/loss_breakdown.json
```

重点看：

```text
loss_<name>
loss_weight_<name>
weighted_loss_<name>
loss_contribution_<name>
```

如果某项 loss 长期为 0，论文中不应宣称该训练目标已生效。

## 受控优化顺序

```text
1. 阈值校准
2. 概率校准
3. loss 生效诊断
4. seed1 上比较 BCE / Weighted BCE / Focal Loss
5. 必要时做 learning-rate probe
6. checkpoint 选择或概率平均
7. 冻结最终主模型配置
8. 只对最终配置做三种子复跑
9. 冻结后再做核心消融
```

## 相关文档

```text
docs/training_commands.md
docs/experiment_guide.md
docs/run_0002_performance_improvement_plan.md
docs/run_0002_performance_runbook.md
docs/run_0002_threshold_epoch_analysis.md
docs/seed_results_summary.md
```

## Git 与大文件约定

代码、配置和文档进 Git。以下内容不进 Git：

```text
work/
packs/
graph/follow_edges.tsv
*.zip
```
