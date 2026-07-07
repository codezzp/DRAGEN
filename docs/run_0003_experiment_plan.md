# run_0003 Formal Experiment Plan

This document tracks the formal Weibo experiment pipeline for Chapter 5.
`run_0003` is the thesis data baseline. `run_0002` is kept only as a development and preliminary-experiment reference.

## 1. Current Status

As of 2026-07-07, the `run_0003` data and pack pipeline is complete, but formal `run_0003` training has not been run yet.

Completed local artifacts:

```text
work/runs/run_0003/processed/
work/runs/run_0003/org_task/
work/runs/run_0003/time_distribution/
work/runs/run_0003/edges/hybrid_tree_light/
work/runs/run_0003/windows/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0003/features_v2/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0003/global_graph/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0003/labels_v2_stratified_score/
work/runs/run_0003/text_embeddings/
work/runs/run_0003/text_semantic/
packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/
packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/
```

Local smoke artifact:

```text
work/artifacts/_smoke_run_0003_key_user_pool/
```

The smoke run used only 64 train samples, 32 valid samples, and 32 test samples. It is a pipeline check only and must not be reported as a formal result.

Not completed yet:

```text
work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0/
work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed1/
work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed2/
run_0003 ablation runs
run_0003 adapted baseline runs
run_0003 final result tables
```

## 2. Dataset Statistics

Available dataset statistics:

| Item | Value |
|---|---:|
| Raw cascades | 300,000 |
| Valid cascades | 232,978 |
| Users | 1,787,443 |
| Active users in events | 1,340,816 |
| Root users | 47,555 |
| Retweet users | 1,334,887 |
| Retweets | 33,160,671 |
| Root text coverage | 99.998% |
| Retweet text coverage | 99.999% |
| User profile coverage | 99.998% |

Current key-user pack diagnostics:

| Item | Value |
|---|---:|
| Train samples | 6,749 |
| Valid samples | 1,450 |
| Test samples | 1,348 |
| Total labeled samples | 9,547 |
| Positive labels | 2,043 |
| Negative labels | 7,504 |
| Ignored cascades | 223,431 |
| Samples with global candidates | 9,493 |
| Total global candidate edges | 199,257 |
| Samples with text semantic features | 9,547 |
| Text semantic dimension | 64 |
| T | 6 |
| Key users per window | 32 |

## 3. Server Training Branch

Use this branch for migration and training commands:

```text
experiment/run-0003-server-training-docs
```

The default training config still points to the older non-run-prefixed pack. Therefore every formal `run_0003` training command must explicitly pass `--pack-dir`.

Required pack:

```text
packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

## 4. Formal DRAGEN Runs

Run seed0:

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

Run seed1:

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

Run seed2:

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

Report the mean and standard deviation over the three seeds in the main experiment table.

## 5. Weibo Ablation Priority

Run these after the three main seeds:

```text
w/o Global Prior
w/o Adaptive Sampler
w/o Memory
w/o Role
w/o Gate
w/o Uncertainty
```

Then add:

```text
w/o Text
w/o Jump Loss
```

`w/o Text` may require either a compatible no-text pack or a model-side switch that masks `node_text_x` and `window_text_x`.

## 6. Chapter 5 Table Design

Use `run_0003` for Table 5-1:

```text
Table 5-1 Weibo Dataset Statistics
```

Main table candidates:

```text
IOHunter-adapt
LEN-GNN-adapt
EDCOC-adapt
X-CoIA-adapt
UWD-FSN-adapt
TDCB-adapt
DRAGEN
```

Metrics:

```text
Acc, Precision, Recall, F1, AUC, AP, MCC, P@100, P@500
```

Do not fill the main performance table until formal `run_0003` seed results exist.

## 7. Execution Priority

```text
1. Push and check out experiment/run-0003-server-training-docs on the server.
2. Upload the run_0003 key-user pack.
3. Run pack shape smoke.
4. Run tiny smoke training.
5. Run DRAGEN seed0, seed1, seed2.
6. Sync back reports/ and predictions/.
7. Export result tables after formal results exist.
8. Build adapted Weibo baselines.
9. Run Weibo ablations.
10. Add threshold calibration.
```
