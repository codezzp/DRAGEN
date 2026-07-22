# Three-seed Experiment Summary

This document summarizes the three completed DRAGEN runs under `work/artifacts/_artifacts`.

## 1. Experiment Scope

These runs are the completed three-seed experiments for the current key-user pool branch:

```text
Feature-v2 + RoBERTa Text + Label-v2 + Key-user Pool Global Prior
```

They use the existing run_0002-aligned pack, not the problematic local run_0003 pack:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Run directories:

```text
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2
```

Input pack diagnostics:

| Item | Value |
|---|---:|
| Train samples | 13,940 |
| Valid samples | 3,032 |
| Test samples | 2,933 |
| Total labeled samples | 19,905 |
| Ignored cascades | 65,358 |
| Samples with global candidates | 19,833 |
| Total global candidate edges | 784,264 |
| Samples with text semantic features | 19,905 |
| Text semantic dimension | 64 |
| Time steps T | 6 |
| Window feature dim | 24 |
| Node feature dim | 47 |
| Key users per window | 32 |

Training setting from the saved commands/configs:

| Item | Value |
|---|---|
| Epochs | 10 |
| Learning rate | 0.001 |
| Main config | `configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml` |
| Batch mode | bucketed by node count |
| Max nodes per batch | 12,000 |
| TensorBoard | disabled for these runs |
| Plot every epoch | disabled for these runs |

## 2. Final Test Results

The table below uses `reports/metrics.json` from each run. These are the final exported test metrics.

| Seed | Acc | Precision | Recall | F1 | AUC | AP | MCC | P@100 | P@500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8773 | 0.8798 | 0.4847 | 0.6250 | 0.9192 | 0.7939 | 0.5944 | 0.9700 | 0.7680 |
| 1 | 0.8674 | 0.9197 | 0.4071 | 0.5644 | 0.9151 | 0.7836 | 0.5575 | 0.9900 | 0.7480 |
| 2 | 0.8663 | 0.8626 | 0.4362 | 0.5794 | 0.9316 | 0.8007 | 0.5519 | 0.9700 | 0.7700 |
| Mean | 0.8703 | 0.8874 | 0.4426 | 0.5896 | 0.9220 | 0.7927 | 0.5679 | 0.9767 | 0.7620 |
| Std | 0.0060 | 0.0293 | 0.0392 | 0.0316 | 0.0086 | 0.0086 | 0.0231 | 0.0115 | 0.0122 |

Main takeaways from test metrics:

- Ranking quality is stable across seeds: test AUC is `0.9220 ± 0.0086`, and AP is `0.7927 ± 0.0086`.
- Top-k retrieval is also stable: P@100 is `0.9767 ± 0.0115`, and P@500 is `0.7620 ± 0.0122`.
- Thresholded F1 varies more than ranking metrics: test F1 is `0.5896 ± 0.0316`.
- The model is conservative at the default threshold `0.5`: precision is high, but recall is moderate. This suggests threshold calibration on the validation set should improve F1/recall tradeoff before final reporting.

## 3. Final Validation Results

The table below also uses `reports/metrics.json`, but for the validation split.

| Seed | Acc | Precision | Recall | F1 | AUC | AP | MCC | P@100 | P@500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8925 | 0.9084 | 0.5439 | 0.6804 | 0.9292 | 0.8275 | 0.6501 | 0.9800 | 0.8080 |
| 1 | 0.8710 | 0.9273 | 0.4201 | 0.5782 | 0.9166 | 0.7980 | 0.5709 | 1.0000 | 0.7720 |
| 2 | 0.8704 | 0.8657 | 0.4545 | 0.5961 | 0.9375 | 0.8174 | 0.5666 | 0.9700 | 0.8140 |
| Mean | 0.8780 | 0.9005 | 0.4728 | 0.6182 | 0.9278 | 0.8143 | 0.5959 | 0.9833 | 0.7980 |
| Std | 0.0126 | 0.0316 | 0.0639 | 0.0546 | 0.0105 | 0.0150 | 0.0470 | 0.0153 | 0.0227 |

Validation is slightly stronger than test on F1/MCC, especially for seed0. Ranking metrics remain close between validation and test.

## 4. Best Validation Epochs

`epoch_metrics.csv` records validation metrics at the end of each epoch. The table below separates the epoch with best validation F1 from the epoch with best validation AUC.

| Seed | Best F1 Epoch | Best Valid F1 | Valid AUC at Best F1 | Valid AP at Best F1 | Valid MCC at Best F1 | Best AUC Epoch | Best Valid AUC | Valid F1 at Best AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.6804 | 0.9292 | 0.8275 | 0.6501 | 8 | 0.9303 | 0.6236 |
| 1 | 9 | 0.6113 | 0.8947 | 0.7499 | 0.5679 | 10 | 0.9166 | 0.5782 |
| 2 | 6 | 0.6091 | 0.8903 | 0.7096 | 0.5296 | 10 | 0.9375 | 0.5961 |

Observations:

- Seed0 improves steadily through epoch 10 and has the strongest final validation F1/MCC.
- Seed1 reaches its best F1 at epoch 9, then improves AUC at epoch 10 while F1 drops because the default threshold becomes more conservative.
- Seed2 reaches its best F1 early at epoch 6, but AUC continues improving until epoch 10. This is another sign that ranking improves even when the fixed 0.5 threshold is not optimal.

## 5. Training Dynamics

Final epoch validation metrics from `epoch_metrics.csv` match `metrics.json` validation metrics.

| Seed | Final Epoch | Train Loss | Valid Loss | Valid Acc | Valid Precision | Valid Recall | Valid F1 | Valid AUC | Valid AP | Valid MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.3089 | 0.2746 | 0.8925 | 0.9084 | 0.5439 | 0.6804 | 0.9292 | 0.8275 | 0.6501 |
| 1 | 10 | 0.3242 | 0.3328 | 0.8710 | 0.9273 | 0.4201 | 0.5782 | 0.9166 | 0.7980 | 0.5709 |
| 2 | 10 | 0.3297 | 0.2801 | 0.8704 | 0.8657 | 0.4545 | 0.5961 | 0.9375 | 0.8174 | 0.5666 |

Epoch time on this machine/server copy:

| Seed | Approx. Epoch Time Range |
|---:|---:|
| 0 | 451.6s to 460.9s |
| 1 | 465.9s to 484.4s |
| 2 | 450.7s to 464.0s |

The training loss decreases consistently in all three seeds. Validation AUC generally improves through later epochs, while validation F1 depends more strongly on the fixed classification threshold.

## 6. Interpretation for Reporting

Recommended headline numbers for the current completed three-seed run:

```text
Test Acc = 0.8703 ± 0.0060
Test F1  = 0.5896 ± 0.0316
Test AUC = 0.9220 ± 0.0086
Test AP  = 0.7927 ± 0.0086
Test MCC = 0.5679 ± 0.0231
P@100    = 0.9767 ± 0.0115
P@500    = 0.7620 ± 0.0122
```

For paper tables, report mean ± std over seed0/seed1/seed2. Use test metrics as the final performance table and validation metrics only for model selection or threshold calibration.

## 7. Caveats

- These results are for the run_0002-aligned pack. They should not be described as run_0003 results.
- Metrics currently use the default classification threshold `0.5` for precision/recall/F1/MCC.
- Ranking metrics AUC/AP/P@K are less sensitive to threshold and are more stable across seeds.
- Before final thesis reporting, run validation-set threshold calibration: choose the F1-optimal threshold on `valid_event_predictions.csv`, then apply that fixed threshold to `test_event_predictions.csv`.
- The local run_0003 pack was found to be inconsistent with the expected run_0003 data coverage, because its processed `events.jsonl` is smaller than run_0002. Do not use that pack for final training unless the run_0003 processed layer is rebuilt or resynced.

## 8. Source Files

Metrics used in this summary:

```text
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/reports/epoch_metrics.csv
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/reports/metrics.json
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1/reports/epoch_metrics.csv
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1/reports/metrics.json
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2/reports/epoch_metrics.csv
work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2/reports/metrics.json
```
