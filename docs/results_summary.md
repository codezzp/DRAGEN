# Results Summary

## 当前阶段

截至当前分支：

```text
experiment/run-0002-roberta-only
```

`run_0002` 的预处理、HybridTree、MultiScale 窗口、feature_v2、RoBERTa text、global follow candidate、Label-v2 到 Label-v5 pack 已经完成。当前重点已经转入服务器正式训练，不再扩展预处理结构。

Git 只提交代码、配置和文档。以下目录不进 Git：

```text
work/
packs/
graph/follow_edges.tsv
*.zip
```

## 已完成输入

### MultiScale HybridTree 窗口

```text
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
```

规模：

```text
cascades = 85263
window_rows = 511578
node_window_rows = 5940259
edge_window_rows = 4019952
current_edges = 1392078
context_edges = 2627874
text_window_rows = 6032866
retweet_text_early_violations = 0
```

### HybridTree Light

```text
work/runs/run_0002/edges/hybrid_tree_light/
```

规模：

```text
cascades = 85263
tree_edges = 1392078
tree_valid_ratio = 1.0
invalid_time_edges = 0
avg_depth = 7.50
max_depth = 32
root_child_ratio = 0.101
text_sim_lift = 0.00107
```

### Feature-v2

```text
work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree/
```

诊断：

```text
window_features rows = 511578
node_window_features rows = 5940259
window_nan_count = 0
window_inf_count = 0
node_nan_count = 0
node_inf_count = 0
```

输入维度：

```text
window_x = 24
node_x = 47
T = 6
```

### RoBERTa Text

原始 embedding：

```text
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext/
root_text_emb.npy    shape = (85263, 768)
retweet_text_emb.npy shape = (193331, 768)
```

64 维降维：

```text
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64/
root_text_emb64.npy    shape = (85263, 64)
retweet_text_emb64.npy shape = (193331, 64)
```

窗口聚合语义：

```text
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/
node_text_features.npy   shape = (567718, 64)
window_text_features.npy shape = (511578, 64)
```

注意：大量 retweet 行没有原始文本，因此无文本节点的 `node_text_x` 为零向量；每个窗口仍有 root 文本语义，`window_text_x` 完整覆盖窗口。

### Global Follow Candidate

```text
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv
```

已写入 pack 字段：

```text
global_candidate_edge_index
global_candidate_edge_weight
```

服务器正常训练不需要重新扫描：

```text
graph/follow_edges.tsv
```

## 标签版本

标签对比：

```text
work/runs/run_0002/label_comparison/label_version_comparison.csv
```

当前统计：

```text
v1 score_rank:              pos=17053 neg=42631 ignore=25579 corr_size=0.057
v2 stratified_score:        pos=4177  neg=15728 ignore=65358 corr_size=0.240
v3 lf_vote:                 pos=3179  neg=2974  ignore=79110 corr_size=0.128
v4 coordination_network:    pos=5848  neg=10974 ignore=68441 corr_size=0.203
v5 ensemble_consensus:      pos=1392  neg=3911  ignore=79960 corr_size=0.071
```

实验使用策略：

```text
v2 = 主实验标签
v5 = 严格标签鲁棒性
v3/v4 = 后续补充
```

## 当前正式 Pack

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

v2 pack 规模：

```text
train = 13940
valid = 3032
test = 2933
total_samples = 19905
```

v5 pack 规模：

```text
train = 3702
valid = 843
test = 758
total_samples = 5303
```

pack smoke test 已确认训练端字段：

```text
window_x      = (B, 6, 24)
node_x        = (B, 6, N, 47)
node_text_x   = (B, 6, N, 64)
window_text_x = (B, 6, 64)
```

## 当前可用训练配置

主模型：

```text
configs/train/dragen_full_label_v2_roberta_text.yaml
configs/train/dragen_full_label_v3_roberta_text.yaml
configs/train/dragen_full_label_v4_roberta_text.yaml
configs/train/dragen_full_label_v5_roberta_text.yaml
```

消融：

```text
configs/train/ablation_no_adaptive_sampling.yaml
configs/train/ablation_no_gate.yaml
configs/train/ablation_no_global_prior.yaml
configs/train/ablation_no_memory.yaml
configs/train/ablation_no_multiscale.yaml
configs/train/ablation_no_role.yaml
configs/train/ablation_no_tree.yaml
configs/train/ablation_no_uncertainty.yaml
```

注意：现有消融 YAML 历史上默认指向 v4 pack。若论文主消融使用 v2，需要用 CLI 覆盖 `--pack-dir` 和 `--out-dir`。

## 当前限制

以下 baseline 入口当前仍是占位实现，不能产出正式 baseline 结果：

```text
scripts/14_train_cac_stat.py
scripts/15_train_gnn_baselines.py
src/dragen/baselines/cac_stat.py
src/dragen/baselines/campaign_gnn.py
src/dragen/baselines/temporal_gnn.py
```

因此服务器当前优先任务是：

```text
1. v2 DRAGEN-Full seed0 smoke/full run
2. v2 DRAGEN-Full seed1/seed2
3. v2 模块消融
4. v5 严格标签鲁棒性
5. 回传 reports/ 和 predictions/
```

## 服务器训练最小命令

先跑 smoke：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_dragen_v2_roberta_text
```

v2 正式 seed0：

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v2_roberta_text.yaml
```

v5 鲁棒性：

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5_roberta_text.yaml
```

## 评估提醒

当前训练导出的分类指标默认使用：

```text
threshold = 0.5
```

论文正式报告建议后处理为：

```text
在 valid_event_predictions.csv 上选择 F1 最优 threshold，
再固定该 threshold 到 test_event_predictions.csv。
```

这一点后续应补进评估/导表脚本。
