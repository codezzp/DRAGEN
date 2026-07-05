# DRAGEN RoBERTa-only 训练命令

本文档是当前分支的命令索引。当前主线是：

```text
Feature-v2 + RoBERTa Text + Adaptive Global Sampling + Global Follow candidates
```

当前分支：

```text
experiment/run-0002-roberta-only
```

当前正式训练优先使用：

```text
Label-v2：主实验
Label-v5：严格标签鲁棒性
```

`v3/v4` pack 已生成，但暂不作为第一优先级。

## 1. 已完成的正式 pack

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

训练阶段只需要读取这些 pack，不会重新调用 RoBERTa。

## 2. 服务器优先 smoke test

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dragen.data.pack_reader import PickleStreamDataset, collate_fn
p = 'packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/train.pt'
ds = PickleStreamDataset(p, max_samples=2, split='train-smoke')
b = collate_fn([ds[0], ds[1]])
print('window_x', tuple(b['window_x'].shape))
print('node_x', tuple(b['node_x'].shape))
print('node_text_x', tuple(b['node_text_x'].shape))
print('window_text_x', tuple(b['window_text_x'].shape))
print('global edges', [tuple(x.shape) for x in b['global_candidate_edge_index']])
PY
```

小样本训练：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_dragen_v2_roberta_text
```

## 3. v2 主实验

seed 0：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml
```

seed 1：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed1
```

seed 2：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --seed 2 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed2
```

断点续训：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --resume work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed0/checkpoints/last.pt
```

## 4. v5 严格标签鲁棒性

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5_roberta_text.yaml
```

## 5. v2 核心消融

现有消融 YAML 早期默认指向 v4 pack。论文主消融如果使用 v2，需要 CLI 覆盖 `--pack-dir` 和 `--out-dir`。

v2 pack：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
```

命令：

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_global_prior.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_global_prior

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_role.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_role

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_memory.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_memory

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_gate.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_gate

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_uncertainty.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_uncertainty

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_adaptive_sampling.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_adaptive_sampling
```

暂时不要直接跑：

```text
w/o RoBERTa Text
w/o MultiScale Context
w/o HybridTree
```

原因：当前分支是 RoBERTa-only，模型要求 pack 有 `node_text_x`；MultiScale/HybridTree 类消融还需要额外构建匹配的 feature_v2 + roberta_text pack。

## 6. 当前 baseline 状态

以下入口当前仍是占位实现，不能作为正式 baseline：

```text
scripts/14_train_cac_stat.py
scripts/15_train_gnn_baselines.py
src/dragen/baselines/cac_stat.py
src/dragen/baselines/campaign_gnn.py
src/dragen/baselines/temporal_gnn.py
```

因此当前服务器优先跑：

```text
DRAGEN-Full v2
DRAGEN-Full v5
v2 模块消融
```

## 7. 结果检查

```bash
cat work/artifacts/<run>/reports/metrics.json
head work/artifacts/<run>/predictions/event_predictions.csv
ls work/artifacts/<run>/checkpoints
```

每个 run 至少回传：

```text
work/artifacts/<run>/reports/
work/artifacts/<run>/predictions/
```

checkpoint 太大时可以暂时不回传。

## 8. 离线预处理命令记录

这些命令本机已经跑完，服务器训练通常不需要再跑。

Feature-v2：

```bash
python scripts/11_build_features_v2.py \
  --run-id run_0002 \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --tree-edges work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv \
  --out-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree
```

RoBERTa 编码与降维：

```bash
python scripts/10_encode_text_roberta.py \
  --run-id run_0002 \
  --model-name hfl/chinese-roberta-wwm-ext \
  --max-length 128 \
  --batch-size 32 \
  --device cuda \
  --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext

python scripts/10b_reduce_text_embeddings.py \
  --in-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext \
  --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64 \
  --dim 64 \
  --seed 0

python scripts/11b_build_text_semantic_features.py \
  --run-id run_0002 \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --text-emb-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64 \
  --out-dir work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64 \
  --dim 64
```

构建 v2 pack 示例：

```bash
python scripts/13_build_packs.py \
  --run-id run_0002 \
  --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --labels work/runs/run_0002/labels_v2_stratified_score/weak_event_labels.csv \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --text-semantic-dir work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64 \
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
```

## Key-user Pool Global Prior Commands

This branch adds a faster global prior mode that replaces the old variable edge-list global branch with a window-level key-user pool and GPU cross-attention.

Build the Label-v2 key-user pack:

```bash
python scripts/13b_build_key_user_pool_packs.py \
  --in-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser \
  --max-hops 4 \
  --key-users-per-window 32 \
  --seed-budget 16 \
  --rho 0.6
```

Check pack shapes:

```bash
python -c "import sys; sys.path.insert(0,'src'); from dragen.data.pack_reader import PickleStreamDataset, collate_fn; p='packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser/train.pt'; ds=PickleStreamDataset(p,max_samples=2,split='train-smoke'); b=collate_fn([ds[0],ds[1]]); [print(k, tuple(b[k].shape), b[k].dtype) for k in ['window_x','node_x','node_text_x','window_text_x','key_user_idx','key_user_weight','key_user_hop','key_user_mask']]"
```

Expected key-user fields:

```text
key_user_idx     (B, 6, 32)
key_user_weight  (B, 6, 32)
key_user_hop     (B, 6, 32)
key_user_mask    (B, 6, 32)
```

Run a small end-to-end smoke:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 64 \
  --max-valid-samples 32 \
  --max-test-samples 32 \
  --out-dir work/artifacts/_smoke_v2_keyuser_pool_e2e
```

Run a speed test:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 512 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --out-dir work/artifacts/_speed_v2_keyuser_pool_bs8_bucket
```

Local speed result for the training epoch:

```text
old edge-list Full = 599.02s
no_global          = 36.05s
key_user_pool      = 42.07s
```
