# run_0002 Threshold and Epoch-selection Analysis

This document records the first two low-cost performance-improvement analyses for the run_0002-aligned DRAGEN runs.

## 1. Inputs

Runs:

```text
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2
```

Generated analysis outputs:

```text
work/artifacts/_analysis/run_0002_threshold_calibration/
work/artifacts/_analysis/run_0002_epoch_selection/
```

Scripts:

```text
scripts/21_calibrate_thresholds.py
scripts/22_summarize_epoch_selection.py
```

## 2. Threshold Calibration

Thresholds were selected on each seed's validation predictions and then applied to that seed's test predictions. The selection rule uses validation metrics only.

### Selected Thresholds

| Seed | Default | Valid-best F1 | Valid-best MCC |
|---:|---:|---:|---:|
| 0 | 0.500 | 0.225 | 0.225 |
| 1 | 0.500 | 0.140 | 0.140 |
| 2 | 0.500 | 0.185 | 0.196 |

The selected thresholds are much lower than 0.5, confirming that the default threshold is conservative.

### Test Results After Applying Valid-selected Thresholds

| Strategy | Threshold | Acc | Precision | Recall | F1 | AUC | AP | MCC | P@100 | P@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default 0.5 | 0.5000 +/- 0.0000 | 0.8703 +/- 0.0060 | 0.8874 +/- 0.0293 | 0.4426 +/- 0.0392 | 0.5896 +/- 0.0316 | 0.9220 +/- 0.0086 | 0.7927 +/- 0.0086 | 0.5679 +/- 0.0231 | 0.9767 +/- 0.0115 | 0.7620 +/- 0.0122 |
| Valid-best F1 | 0.1833 +/- 0.0425 | 0.8748 +/- 0.0087 | 0.7069 +/- 0.0104 | 0.6936 +/- 0.0464 | 0.6999 +/- 0.0290 | 0.9220 +/- 0.0086 | 0.7927 +/- 0.0086 | 0.6211 +/- 0.0334 | 0.9767 +/- 0.0115 | 0.7620 +/- 0.0122 |
| Valid-best MCC | 0.1870 +/- 0.0432 | 0.8754 +/- 0.0096 | 0.7112 +/- 0.0163 | 0.6893 +/- 0.0417 | 0.6998 +/- 0.0290 | 0.9220 +/- 0.0086 | 0.7927 +/- 0.0086 | 0.6216 +/- 0.0341 | 0.9767 +/- 0.0115 | 0.7620 +/- 0.0122 |

### Interpretation

Threshold calibration is clearly useful:

```text
F1:     0.5896 -> 0.6999
Recall: 0.4426 -> 0.6936 / 0.6893
MCC:    0.5679 -> 0.6211 / 0.6216
```

Precision drops from about `0.8874` to about `0.707-0.711`, which is expected because the calibrated thresholds classify more cascades as positive. For the thesis, this can be framed as a tradeoff between high-confidence screening and balanced detection.

AUC, AP, P@100, and P@500 are unchanged because they are ranking metrics and do not depend on the classification threshold.

Recommended threshold strategy for the next phase:

```text
Use Valid-best MCC if the final table emphasizes balanced class performance.
Use Valid-best F1 if the final table emphasizes F1 directly.
```

The two strategies are nearly tied in F1. Valid-best MCC is slightly better on MCC and accuracy.

## 3. Epoch-selection Analysis

This analysis uses `epoch_metrics.csv` only. It does not re-export test predictions from earlier checkpoints.

### Validation Mean +/- Std

| Strategy | Epoch | Acc | Precision | Recall | F1 | AUC | AP | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best valid AUC | 9.33 +/- 1.15 | 0.8728 +/- 0.0036 | 0.8895 +/- 0.0332 | 0.4530 +/- 0.0322 | 0.5993 +/- 0.0229 | 0.9281 +/- 0.0106 | 0.8061 +/- 0.0101 | 0.5765 +/- 0.0137 |
| Best valid F1 | 8.33 +/- 2.08 | 0.8727 +/- 0.0187 | 0.8140 +/- 0.1024 | 0.5214 +/- 0.0324 | 0.6336 +/- 0.0406 | 0.9048 +/- 0.0213 | 0.7623 +/- 0.0599 | 0.5825 +/- 0.0616 |
| Best valid MCC | 9.67 +/- 0.58 | 0.8798 +/- 0.0112 | 0.9194 +/- 0.0099 | 0.4707 +/- 0.0649 | 0.6207 +/- 0.0532 | 0.9264 +/- 0.0087 | 0.8170 +/- 0.0165 | 0.6035 +/- 0.0414 |
| Final epoch | 10.00 +/- 0.00 | 0.8780 +/- 0.0126 | 0.9005 +/- 0.0316 | 0.4728 +/- 0.0639 | 0.6182 +/- 0.0546 | 0.9278 +/- 0.0105 | 0.8143 +/- 0.0150 | 0.5959 +/- 0.0470 |

### By-seed Best Epochs

| Seed | Best valid F1 epoch | Best valid AUC epoch | Best valid MCC epoch |
|---:|---:|---:|---:|
| 0 | 10 | 8 | 10 |
| 1 | 9 | 10 | 10 |
| 2 | 6 | 10 | 9 |

### Interpretation

The validation curves show that final epoch is not always the best F1 epoch. However, AUC/AP tend to remain strongest near the final epoch.

Current recommendation:

```text
Do not claim test improvements from best-valid-F1 or best-valid-MCC selection unless the matching checkpoints are available.
For future final training, save best_valid_f1.pt and best_valid_mcc.pt, or use --save-every-epoch.
```

For the current completed runs, final epoch remains the cleanest reported checkpoint. Threshold calibration already gives a large F1/Recall gain without needing checkpoint changes.

## 4. Decision for Next Step

Immediate next experiment:

```text
Run seed1 loss probes on valid metrics only:
1. BCE baseline, if a clean rerun is needed
2. Weighted BCE with pos_weight=auto
3. Weighted BCE with pos_weight=soft
4. Focal Loss with alpha=0.75, gamma=2.0
```

Selection rule:

```text
Choose the loss using valid F1/MCC, with valid AUC/AP as safeguards.
Report test only after the final loss/threshold/selection strategy is frozen.
```
