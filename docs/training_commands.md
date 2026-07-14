# DRAGEN 训练与后处理命令

本文档只记录当前有效命令。

## 1. 主实验 pack

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

## 2. Pack 字段检查

```bash
python -c "import sys; sys.path.insert(0,'src'); from dragen.data.pack_reader import PickleStreamDataset, collate_fn; p='packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/train.pt'; ds=PickleStreamDataset(p,max_samples=2,split='train-smoke'); b=collate_fn([ds[0],ds[1]]); [print(k, tuple(b[k].shape), b[k].dtype) for k in ['window_x','node_x','node_text_x','window_text_x','key_user_idx','key_user_weight','key_user_hop','key_user_mask']]"
```

## 3. 小样本 smoke test

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 64 \
  --max-valid-samples 32 \
  --max-test-samples 32 \
  --out-dir work/artifacts/_smoke_v2_key_user_pool_e2e
```

## 4. 正式训练模板

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

## 5. Loss 对比命令

先只跑 seed1，用 valid 指标选方向。

Weighted BCE soft：

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

Weighted BCE auto：

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

Focal Loss：

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

## 6. 后处理命令

```bash
python scripts/21_calibrate_thresholds.py
python scripts/22_summarize_epoch_selection.py
python scripts/23_calibrate_probabilities.py
```

需要逐样本校准预测时才加：

```bash
python scripts/23_calibrate_probabilities.py --write-predictions
```

## 7. Loss 生效诊断

```bash
cat work/artifacts/<run>/reports/loss_breakdown.json
```

重点字段：

```text
loss_event
weighted_loss_event
loss_contribution_event
loss_jump
weighted_loss_jump
loss_struct
weighted_loss_struct
loss_sampler
weighted_loss_sampler
loss_uncertainty
weighted_loss_uncertainty
loss_role
weighted_loss_role
```
