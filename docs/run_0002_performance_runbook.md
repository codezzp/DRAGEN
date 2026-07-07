# run_0002 Performance Experiment Runbook

This runbook gives the exact execution order for the run_0002 performance-improvement phase.

Use this together with:

```text
docs/run_0002_performance_improvement_plan.md
docs/run_0002_threshold_epoch_analysis.md
```

## 1. Branch

Current implementation branch:

```text
experiment/run-0002-performance-plan
```

Push from local:

```bash
git push -u codezzp experiment/run-0002-performance-plan
```

Checkout on the server:

```bash
git fetch codezzp
git checkout experiment/run-0002-performance-plan
```

Verify:

```bash
git branch --show-current
git log --oneline -3
```

## 2. Dataset

Use the run_0002-aligned key-user pool pack:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Do not use the local run_0003 pack for this experiment line.

## 3. Step 1: Threshold Calibration

This does not retrain the model.

```bash
python scripts/21_calibrate_thresholds.py
```

Outputs:

```text
work/artifacts/_analysis/run_0002_threshold_calibration/threshold_by_seed.csv
work/artifacts/_analysis/run_0002_threshold_calibration/threshold_test_metrics.csv
work/artifacts/_analysis/run_0002_threshold_calibration/threshold_summary_mean_std.csv
work/artifacts/_analysis/run_0002_threshold_calibration/threshold_calibration_summary.md
```

Check:

```bash
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_summary_mean_std.csv
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_by_seed.csv
```

Current result from the existing three seeds:

```text
Default 0.5 F1      = 0.5896
Valid-best F1 F1    = 0.6999
Valid-best MCC F1   = 0.6998
Default 0.5 Recall  = 0.4426
Calibrated Recall   = about 0.69
Default 0.5 MCC     = 0.5679
Calibrated MCC      = about 0.621
```

Select threshold strategy using valid metrics only. Test is only for final reporting after the strategy is fixed.

## 4. Step 2: Epoch-selection Analysis

This does not retrain the model.

```bash
python scripts/22_summarize_epoch_selection.py
```

Outputs:

```text
work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_by_seed.csv
work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_summary_mean_std.csv
work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_summary.md
```

Check:

```bash
cat work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_summary_mean_std.csv
```

Important: if earlier epoch checkpoints were not saved, do not claim that test has been re-evaluated under best-valid-F1 or best-valid-MCC checkpoints. Treat this as validation-curve analysis only.

For future final training, add:

```bash
--save-every-epoch
```

## 5. Step 3: Seed1 Loss Probes

Run only seed1 first. Choose loss using valid metrics only.

### 5.1 BCE baseline

Use this if a clean comparable baseline is needed outside `_artifacts`:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_bce_seed1
```

### 5.2 Weighted BCE soft

`soft` means `sqrt(neg / pos)`, about `1.94` for run_0002 label-v2.

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

### 5.3 Weighted BCE auto

`auto` means `neg / pos`, about `3.77` for run_0002 label-v2.

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

### 5.4 Focal Loss

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

### 5.5 Loss Probe Selection

Read valid metrics:

```bash
cat work/artifacts/run_0002_loss_bce_seed1/reports/metrics.json
cat work/artifacts/run_0002_loss_weighted_bce_soft_seed1/reports/metrics.json
cat work/artifacts/run_0002_loss_weighted_bce_auto_seed1/reports/metrics.json
cat work/artifacts/run_0002_loss_focal_seed1/reports/metrics.json
```

Selection rule:

```text
Use valid metrics only.
A loss enters final three-seed rerun if:
1. valid F1 improves by >= 0.02, or valid MCC improves by >= 0.02;
2. valid AUC/AP do not clearly drop;
3. valid Precision does not collapse.
```

## 6. Step 4: Freeze Main Configuration

Freeze these values after threshold, epoch strategy, and loss are selected:

| Config item | Frozen value |
|---|---|
| Pack | `packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool` |
| Feature | feature_v2 |
| Text | RoBERTa semantic 64-dim |
| Label | label_v2 |
| Global prior | key-user pool |
| K | 32 |
| Loss | TBD |
| Threshold | TBD |
| Model selection | TBD |
| Seeds | 0 / 1 / 2 |
| Main output | `work/artifacts/run_0002_final_dragen_seed*` |

Create or update:

```text
docs/run_0002_frozen_main_config.md
```

## 7. Step 5: Final Three-seed Main Runs

Only run this after the final configuration is frozen.

Template:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  <LOSS_ARGS> \
  --seed <SEED> \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_final_dragen_seed<SEED>
```

Examples for `weighted_bce soft`:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss weighted_bce \
  --pos-weight soft \
  --seed 0 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_final_dragen_seed0
```

Repeat for seed1 and seed2.

## 8. Step 6: Core Ablations After Freeze

Run ablations only after the final main model is frozen.

First batch:

```text
w/o Global Prior
w/o Adaptive Sampling
w/o Memory
w/o Role
w/o Gate
w/o Uncertainty
```

Do not run module replacement experiments before this batch is complete.

Use the frozen main config as the baseline. If the final main config uses a new event loss, ablations should use the same event loss unless the ablation specifically studies loss.

## 9. Sync Back

For each run, sync back at least:

```text
work/artifacts/<run>/reports/
work/artifacts/<run>/predictions/
```

If checkpoint size is acceptable, also sync:

```text
work/artifacts/<run>/checkpoints/best.pt
work/artifacts/<run>/checkpoints/last.pt
work/artifacts/<run>/checkpoints/epoch_*.pt
```
