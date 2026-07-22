# 文档索引

本文档说明每个文档的用途。优先看前三个；其他文档主要作为历史记录或设计参考。

## 当前必看

| 文档 | 用途 |
|---|---|
| `../README.md` | 项目当前状态、主实验配置、最短执行路径 |
| `training_commands.md` | 训练、loss probe、阈值校准、概率校准的可执行命令 |
| `experiment_guide.md` | 当前 run_0002 优化边界，说明现在做什么、不做什么 |

## run_0002 当前分析

| 文档 | 用途 |
|---|---|
| `run_0002_threshold_epoch_analysis.md` | 已完成的阈值校准和 epoch 选择分析结果 |
| `seed_results_summary.md` | 三种子主结果汇总 |
| `run_0002_speed_diagnosis.md` | key-user pool 前的速度瓶颈诊断 |

## run_0002 计划和执行记录

| 文档 | 用途 |
|---|---|
| `run_0002_performance_improvement_plan.md` | 性能优化计划，内容较详细，作为参考 |
| `run_0002_server_step_by_step.md` | 服务器逐步实验执行手册，按阶段列出命令、输出目录和判断规则 |
| `run_0002_performance_runbook.md` | 历史执行手册，部分内容已被 `training_commands.md` 合并 |
| `server_experiment_guide.md` | 服务器迁移和运行记录，按需查看 |

## 设计参考

| 文档 | 用途 |
|---|---|
| `model_design.md` | 模型结构说明 |
| `data_schema.md` | 数据表和字段说明 |
| `evidence_features.md` | evidence 特征说明 |
| `graph_design.md` | 图构建说明 |
| `label_design.md` | 弱标签设计说明 |
| `window_design.md` | 窗口构建说明 |
| `text_embeddings.md` | RoBERTa 文本特征说明 |
| `configuration.md` | 配置文件规则 |
| `experiment_protocol.md` | 早期实验协议，历史参考 |
| `results_summary.md` | 早期结果摘要，历史参考 |
| `run_notes.md` | 历史运行笔记，较杂，除非追溯细节否则不建议优先看 |

## 当前原则

```text
README.md 讲当前状态。
training_commands.md 放命令。
experiment_guide.md 定边界。
run_0002_* 文档保留分析记录，不再作为入口。
```
