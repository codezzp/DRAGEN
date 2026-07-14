# run_0002 实验指南

本文档定义当前实验边界，避免继续无边界扩展。

## 1. 当前主线

```text
Feature-v2 + RoBERTa Text + Key-user Pool Global Prior
```

当前优化分支：

```text
experiment/run-0002-calibration-diagnostics
```

## 2. 当前阶段只解决什么

```text
1. 默认阈值下 Recall/F1 偏低
2. 概率输出是否校准
3. 辅助 loss 是否真实参与训练
4. 类别不平衡 loss 是否改善 valid F1/MCC
5. checkpoint/epoch 选择是否影响稳定性
```

## 3. 主模型冻结前不做什么

```text
balanced sampler
hard-negative mining
事件聚合模块替换
Memory/Fusion/Candidate 模块替换
K=16/32/64 敏感性
特征标准化或缺失指示变量重构
```

## 4. 执行顺序

```text
1. 阈值校准
2. 概率校准
3. loss 生效诊断
4. seed1 上比较 BCE / Weighted BCE / Focal Loss
5. 必要时做 learning-rate probe
6. checkpoint 选择或概率平均
7. 冻结最终主模型配置
8. 三种子复跑最终配置
9. 冻结后做核心消融
10. 最后再做模块替换或敏感性实验
```

## 5. 主要参考文档

```text
docs/training_commands.md
docs/run_0002_performance_improvement_plan.md
docs/run_0002_performance_runbook.md
docs/run_0002_threshold_epoch_analysis.md
docs/seed_results_summary.md
```
