# run_0002 Frozen Main Config

This document freezes the main DRAGEN configuration after the completed run_0002 diagnostics. Do not change these items when producing the final main-model results, ablations, or thesis tables unless a new run id is opened.

## Frozen Decision

| Item | Frozen value |
|---|---|
| Pack | `packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool` |
| Config | `configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml` |
| Label | Label-v2 stratified score labels, as stored in the frozen pack |
| Window | observation `1800s`, step `300s`, multiscale hybrid tree |
| Text | RoBERTa text features, `text_semantic_dim=64` |
| Key-user pool | enabled, `key_users_per_window=32`, `key_user_max_hops=4`, `global_sampling_mode=key_user_pool` |
| Event Loss | BCE (`event_loss=bce`; do not use weighted BCE or focal for the main model) |
| Threshold Strategy | Valid-best MCC, selected per seed on validation predictions only |
| Probability Calibrator | none / identity for main classification results |
| Checkpoint Rule | final epoch checkpoint after 10 epochs (`checkpoints/last.pt`); do not use `best.pt` as a MCC/F1-selected checkpoint |
| Seeds | `0`, `1`, `2` |
| Output Naming | `work/artifacts/main/run_0002_final_dragen_seed{seed}` |

## Rationale

Loss probes on seed 1 showed the original BCE loss has the best validation classification metrics among the tested event losses:

| Loss | Valid F1 | Valid MCC | Valid AUC | Valid AP |
|---|---:|---:|---:|---:|
| BCE | 0.5782 | 0.5709 | 0.9166 | 0.7980 |
| Weighted BCE, soft | 0.5187 | 0.5268 | 0.9060 | 0.7737 |
| Weighted BCE, auto | 0.5011 | 0.4982 | 0.9006 | 0.7509 |
| Focal | 0.4983 | 0.4821 | 0.8868 | 0.7245 |

Threshold calibration should be used because it gives the largest clean validation-only improvement. On the three existing seeds, Valid-best MCC improved mean test F1 from `0.5896` to `0.6998` and mean test MCC from `0.5679` to `0.6216`.

Probability calibration is not part of the main classification setting. Platt/isotonic calibration improves probability-quality metrics such as NLL, Brier, or ECE, but it does not improve the main F1/MCC result over identity calibration with Valid-best MCC. If probability quality is reported, put it in a separate calibration table and state that calibrators are fitted on validation predictions.

The current trainer saves `checkpoints/best.pt` using validation AUC first. It is therefore not a valid-best-MCC or valid-best-F1 checkpoint. For the frozen main model, report the final epoch checkpoint from a fixed 10-epoch run. Keep `--save-every-epoch` in final commands for auditability, but do not use post-hoc epoch selection for the main result unless a new checkpoint-selection protocol is frozen first.

## Per-seed Thresholds

Use these thresholds only after the matching seed's validation predictions have selected them. They are recorded here from the completed calibration analysis:

| Seed | Valid-best MCC threshold | Valid F1 | Valid MCC | Valid Precision | Valid Recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.225 | 0.7490 | 0.6815 | 0.7427 | 0.7555 |
| 1 | 0.140 | 0.7058 | 0.6321 | 0.7351 | 0.6787 |
| 2 | 0.196 | 0.7525 | 0.6863 | 0.7496 | 0.7555 |

For new final reruns, recompute the threshold from that run's validation predictions using the same Valid-best MCC rule. Do not tune on test.

## Final Main Commands

Seed 0:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 0 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/main/run_0002_final_dragen_seed0
```

Seed 1:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/main/run_0002_final_dragen_seed1
```

Seed 2:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 2 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/main/run_0002_final_dragen_seed2
```

## Reporting Rules

- Main table: report mean/std over seeds `0,1,2`.
- Main threshold: Valid-best MCC, selected on validation predictions and applied once to test predictions.
- Main probabilities: raw model probabilities, no calibration.
- Main checkpoint: epoch 10 final checkpoint.
- Do not report adapted baselines or module-replacement experiments as final results until their implementations/configs are frozen separately.
