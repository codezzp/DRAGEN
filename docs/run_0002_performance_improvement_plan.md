# run_0002 Performance Improvement Plan

This document defines the short, controlled performance-improvement phase before freezing the DRAGEN main model and starting ablation experiments.

## 1. Objective

Current three-seed DRAGEN results already show stable ranking quality on the run_0002-aligned Weibo pack:

```text
Test AUC  = 0.9220 +/- 0.0086
Test AP   = 0.7927 +/- 0.0086
P@100     = 0.9767 +/- 0.0115
P@500     = 0.7620 +/- 0.0122
```

The main weakness is thresholded classification under the default threshold `0.5`:

```text
Test F1     = 0.5896 +/- 0.0316
Test Recall = 0.4426 +/- 0.0392
```

The model is conservative: precision and top-k precision are high, but recall is moderate. Therefore this phase focuses on low-cost changes that improve decision quality without changing the overall DRAGEN architecture.

## 2. Scope

Only three experiment types are allowed before freezing the main model:

```text
1. Threshold calibration
2. Best epoch / model selection
3. Class-imbalance loss
```

Do not start module replacement experiments in this phase. Do not keep tuning the model indefinitely. Once a final setting is selected, freeze it and use that frozen setting for all ablations.

## 3. Fixed Dataset and Baseline

Use the run_0002-aligned key-user pool pack:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Existing baseline runs:

```text
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2
```

Baseline summary document:

```text
docs/seed_results_summary.md
```

Important: these are run_0002-aligned results. Do not describe them as run_0003 results.

## 4. Experiment A: Threshold Calibration

### Goal

Improve F1, Recall, and MCC by replacing the fixed threshold `0.5` with thresholds selected on the validation split.

### Input Files

For each seed:

```text
work/artifacts/_artifacts/<run>/predictions/valid_event_predictions.csv
work/artifacts/_artifacts/<run>/predictions/test_event_predictions.csv
```

### Strategies

| Strategy | Selection rule | Applied to test |
|---|---|---|
| Default 0.5 | Use threshold 0.5 | Yes |
| Valid-best F1 | Choose threshold with max valid F1 | Yes |
| Valid-best MCC | Choose threshold with max valid MCC | Yes |

### Output Directory

```text
work/artifacts/_analysis/run_0002_threshold_calibration/
```

Expected outputs:

```text
threshold_by_seed.csv
threshold_test_metrics.csv
threshold_summary_mean_std.csv
```

### Reporting Table

| Strategy | Acc | Precision | Recall | F1 | AUC | AP | MCC | P@100 | P@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default 0.5 |  |  |  |  |  |  |  |  |  |
| Valid-best F1 |  |  |  |  |  |  |  |  |  |
| Valid-best MCC |  |  |  |  |  |  |  |  |  |

### Decision Rule

If threshold calibration improves mean test F1 or MCC without causing a large precision collapse, use calibrated thresholding in the final reporting pipeline.

Suggested acceptance condition:

```text
mean test F1 improves by >= 0.02
or mean test MCC improves by >= 0.02
and mean test Precision remains acceptable for the risk-screening setting.
```

## 5. Experiment B: Best Epoch / Model Selection

### Goal

Decide whether the final model should be selected by final epoch, best validation F1, best validation AUC, or best validation MCC.

Current observations from `epoch_metrics.csv`:

```text
seed0: best valid F1 at epoch 10; best valid AUC at epoch 8
seed1: best valid F1 at epoch 9;  best valid AUC at epoch 10
seed2: best valid F1 at epoch 6;  best valid AUC at epoch 10
```

### Candidate Selection Rules

| Rule | Advantage | Risk |
|---|---|---|
| Final epoch | Simple and reproducible | F1 may be suboptimal |
| Best valid F1 | Optimizes the main thresholded metric | May sacrifice ranking metrics |
| Best valid MCC | Better for class imbalance | May be less intuitive |
| Best valid AUC | Best for ranking-focused claims | May not improve F1/Recall |

### Execution

First, summarize existing validation curves without retraining:

```text
work/artifacts/_analysis/run_0002_epoch_selection/
```

If epoch checkpoints for earlier epochs are unavailable, this experiment can only report validation-curve evidence for the current runs. For future final runs, enable:

```bash
--save-every-epoch
```

### Decision Rule

For the thesis main table, prefer one of these two options:

```text
Best valid F1, if the goal is classification quality.
Best valid MCC, if the goal is robustness under class imbalance.
```

Keep AUC/AP/P@K in the table regardless of the selection rule.

## 6. Experiment C: Class-imbalance Loss

### Goal

Improve recall and F1 by handling positive/negative imbalance during training.

### Candidate Losses

| Loss | Purpose |
|---|---|
| BCE | Current baseline |
| Weighted BCE | Increase positive-class weight |
| Focal Loss | Focus training on hard examples |

### Initial Probe

Run seed1 first, because seed1 has relatively low recall/F1 under the default setting and is useful for detecting whether the loss direction helps.

Recommended output dirs:

```text
work/artifacts/run_0002_loss_bce_seed1
work/artifacts/run_0002_loss_weighted_bce_seed1
work/artifacts/run_0002_loss_focal_seed1
```

### Baseline Re-run Command

Use this only if a clean comparable seed1 baseline is needed outside `_artifacts`:

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

### Required Code/Config Check

Before running Weighted BCE or Focal Loss, verify that the training code supports loss selection. If not, add a minimal config field such as:

```yaml
loss:
  event_loss: weighted_bce
  pos_weight: auto
```

or:

```yaml
loss:
  event_loss: focal
  focal_alpha: 0.75
  focal_gamma: 2.0
```

Keep this change scoped to event classification loss. Do not change model architecture during this phase.

### Decision Rule

Run seed1 first. If a loss improves validation/test F1 or MCC and does not damage AUC/AP heavily, promote it to full three-seed training.

Suggested acceptance condition:

```text
seed1 test F1 improves by >= 0.02
or seed1 test MCC improves by >= 0.02
and test AUC drop is <= 0.01
```

If both Weighted BCE and Focal Loss help, choose the simpler and more stable option for the frozen main model.

## 7. Freeze Criteria

Freeze the main DRAGEN configuration after the following choices are fixed:

```text
1. Threshold strategy
2. Model selection strategy
3. Event loss configuration
4. Pack path
5. Seed list
6. Output naming convention
```

Record the frozen configuration in a separate document or section:

```text
docs/run_0002_frozen_main_config.md
```

After freezing, do not change the main model while running ablations. Any additional module replacement experiments must be clearly marked as post-freeze supplementary experiments.

## 8. Final Main Runs After Freeze

Once the final setting is selected, rerun the final main model for seed0/seed1/seed2 if the loss or model-selection procedure changed.

Suggested output naming:

```text
work/artifacts/run_0002_final_dragen_seed0
work/artifacts/run_0002_final_dragen_seed1
work/artifacts/run_0002_final_dragen_seed2
```

Recommended command template:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed <SEED> \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_final_dragen_seed<SEED>
```

Add loss-specific CLI/config overrides only after the loss implementation is finalized.

## 9. Ablation Starts Only After Freeze

The first ablation batch after freeze:

```text
w/o Global Prior
w/o Adaptive Sampling
w/o Memory
w/o Role
w/o Gate
w/o Uncertainty
```

Do not run these before the frozen main setting is selected, otherwise the ablation baseline will drift.

Postpone:

```text
w/o Text
w/o Jump Loss
module replacement experiments
```

## 10. Immediate Next Actions

```text
1. Implement or run threshold calibration on existing prediction files.
2. Produce threshold calibration tables and update docs/seed_results_summary.md.
3. Summarize epoch-selection evidence from existing epoch_metrics.csv files.
4. Check whether Weighted BCE / Focal Loss are already supported.
5. If not supported, add the smallest loss-selection patch.
6. Run seed1 loss probes.
7. Select the final setting and freeze the main model.
```

One-sentence rule:

> Tune the main model once, freeze it, then use ablations to explain that fixed model.
