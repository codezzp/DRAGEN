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

## 2. Core Rule

All scheme selection must use validation metrics only.

```text
Use valid to choose threshold, epoch strategy, and loss.
Use test only once for final reporting after the configuration is frozen.
```

This is the central rule for avoiding test-set leakage. During the performance-improvement phase, test metrics may be computed for documentation after a valid-selected strategy is fixed, but they must not be used to choose among strategies.

## 3. Scope

Only three experiment types are allowed before freezing the main model:

```text
1. Threshold calibration
2. Best epoch / model selection
3. Class-imbalance loss
```

Do not start module replacement experiments in this phase. Do not keep tuning the model indefinitely. Once a final setting is selected, freeze it and use that frozen setting for all ablations.

## 4. Fixed Dataset and Baseline

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

## 5. Experiment A: Threshold Calibration

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
| Default 0.5 | Fixed threshold | Yes |
| Valid-best F1 | Choose threshold with max valid F1 | Yes |
| Valid-best MCC | Choose threshold with max valid MCC | Yes |

### Important Metric Note

Threshold calibration changes thresholded metrics only:

```text
Precision / Recall / F1 / MCC / Accuracy / Balanced Accuracy
```

It does not change ranking metrics:

```text
AUC / AP / P@100 / P@500
```

AUC, AP, and P@K may remain in the reporting table for completeness, but the threshold-calibration conclusion should focus on Precision, Recall, F1, and MCC.

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

Select the threshold strategy using validation metrics only.

Suggested valid-set acceptance condition:

```text
A threshold strategy enters the frozen reporting pipeline if:
1. valid F1 improves by >= 0.02, or valid MCC improves by >= 0.02;
2. valid Precision does not collapse relative to default 0.5;
3. ranking metrics remain unchanged as expected.
```

After the threshold strategy is selected on valid, apply it to test once for final reporting.

## 6. Experiment B: Best Epoch / Model Selection

### Goal

Decide whether the final model should be selected by final epoch, best validation F1, best validation AUC, or best validation MCC.

Current observations from `epoch_metrics.csv`:

| Seed | Best valid F1 epoch | Best valid AUC epoch |
|---:|---:|---:|
| 0 | 10 | 8 |
| 1 | 9 | 10 |
| 2 | 6 | 10 |

This means F1-oriented selection may stop earlier, while AUC-oriented selection often favors later epochs.

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

If the current runs did not save per-epoch checkpoints, this stage can only report epoch-selection analysis from validation curves. Do not claim test has been re-evaluated under best-valid-F1 or best-valid-MCC checkpoints unless those checkpoints actually exist.

For future final training, add a best-checkpoint saving mechanism. Preferred checkpoint files:

```text
best_valid_f1.pt
best_valid_mcc.pt
last.pt
```

If the code does not yet support a CLI such as:

```bash
--save-best-metric valid_f1
```

then implement the minimal checkpoint-saving patch before final three-seed training, or use `--save-every-epoch` and select checkpoints offline.

### Decision Rule

Use validation metrics to choose the model-selection strategy. For the thesis main table, prefer one of these two options:

```text
Best valid F1, if the goal is classification quality.
Best valid MCC, if the goal is robustness under class imbalance.
```

Keep AUC/AP/P@K in the table regardless of the selection rule.

## 7. Experiment C: Class-imbalance Loss

### Goal

Improve recall and F1 by handling positive/negative imbalance during training.

### Candidate Losses

Only test these three losses in the first pass:

| Loss | Priority | Purpose |
|---|---:|---|
| BCE | Baseline | Current loss |
| Weighted BCE | Highest | Increase positive-class recall |
| Focal Loss | High | Focus training on hard examples |

Do not add Asymmetric Focal Loss, Class-balanced Loss, Dice Loss, or other variants in this phase. They will make the experiment line hard to control.

### Weighted BCE Settings

Use the run_0002 label-v2 effective sample counts:

```text
pos = 4177
neg = 15728
neg / pos = 3.77
sqrt(neg / pos) = 1.94
```

Probe two practical weighted-BCE settings if time allows:

| Setting | pos_weight | Rationale |
|---|---:|---|
| auto | 3.77 | Full imbalance correction |
| soft | 1.94 | More conservative, less likely to collapse precision |

If full `3.77` hurts Precision too much, prefer the softer `1.94` setting.

### Focal Loss Settings

Start with one setting only:

```yaml
loss:
  event_loss: focal
  focal_alpha: 0.75
  focal_gamma: 2.0
```

Do not run a focal-loss grid search in this phase.

### Initial Probe

Run seed1 first, because seed1 has relatively low recall/F1 under the default setting and is useful for detecting whether the loss direction helps.

Recommended output dirs:

```text
work/artifacts/run_0002_loss_bce_seed1
work/artifacts/run_0002_loss_weighted_bce_auto_seed1
work/artifacts/run_0002_loss_weighted_bce_soft_seed1
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

Run seed1 first. Select loss candidates using validation metrics only.

Suggested valid-set acceptance condition:

```text
A loss enters three-seed rerun if:
1. seed1 valid F1 improves by >= 0.02, or seed1 valid MCC improves by >= 0.02;
2. seed1 valid AUC/AP do not show a clear drop;
3. seed1 valid Precision does not collapse.
```

Test metrics are reported only after the final frozen loss configuration is selected.

If both Weighted BCE and Focal Loss help, choose the simpler and more stable option for the frozen main model.

## 8. Freeze Criteria

Freeze the main DRAGEN configuration after the following choices are fixed:

```text
1. Threshold strategy
2. Model selection strategy
3. Event loss configuration
4. Pack path
5. Seed list
6. Output naming convention
```

Add a frozen configuration record:

```text
docs/run_0002_frozen_main_config.md
```

Suggested freeze table:

| Config item | Frozen value |
|---|---|
| Pack | `packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool` |
| Feature | feature_v2 |
| Text | RoBERTa semantic 64-dim |
| Label | label_v2 |
| Global prior | key-user pool |
| K | 32 |
| Loss | TBD: BCE / Weighted BCE / Focal |
| Threshold | TBD: 0.5 / valid-best F1 / valid-best MCC |
| Model selection | TBD |
| Seeds | 0 / 1 / 2 |
| Main output | `work/artifacts/run_0002_final_dragen_seed*` |

After freezing, do not change the main model while running ablations. Any additional module replacement experiments must be clearly marked as post-freeze supplementary experiments.

## 9. Final Main Runs After Freeze

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

## 10. Ablation Starts Only After Freeze

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

## 11. Immediate Next Actions

```text
1. Run threshold calibration on existing prediction files.
2. Produce threshold_by_seed.csv, threshold_test_metrics.csv, and threshold_summary_mean_std.csv.
3. Summarize epoch-selection evidence from existing epoch_metrics.csv files.
4. Check whether Weighted BCE / Focal Loss are already supported.
5. If not supported, add the smallest loss-selection patch.
6. Run seed1 loss probes using valid metrics for selection.
7. Select the final setting on valid and freeze the main model.
8. Report test only for the frozen final configuration.
```

One-sentence rule:

> Tune the main model once using validation evidence, freeze it, then use test for final reporting and ablations to explain that fixed model.
