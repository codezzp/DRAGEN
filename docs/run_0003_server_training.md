# run_0003 Server Training Checklist

Use this checklist when moving the formal Weibo experiment to the server.

## Status

```text
run_0003 formal training has not been run yet.
run_0003 pack and key-user pack are ready locally.
A tiny smoke run exists only for pipeline validation.
```

Do not use these old local artifacts as formal `run_0003` results:

```text
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2
```

Those runs used the older default pack path, not the explicit `run_0003` pack.

## Branch

```bash
git checkout experiment/run-0003-server-training-docs
```

## Required Pack

```text
packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Required files:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

## Upload Pack

```bash
rsync -av --progress \
  packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/ \
  user@server:/path/to/DRAGEN/packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/
```

## Smoke

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --pack-dir packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
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
  --out-dir work/artifacts/_smoke_run_0003_key_user_pool
```

## Formal Training

Seed0:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --pack-dir packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
  --seed 0 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0
```

Seed1:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --pack-dir packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed1
```

Seed2:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --pack-dir packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
  --seed 2 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed2
```

## Sync Back

```bash
rsync -av --progress \
  user@server:/path/to/DRAGEN/work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0/reports/ \
  work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0/reports/

rsync -av --progress \
  user@server:/path/to/DRAGEN/work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0/predictions/ \
  work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0/predictions/
```
