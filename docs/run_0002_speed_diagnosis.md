# run_0002 Label-v2 Speed Diagnosis

Date: 2026-07-05

## Current Status

The current experiment line is:

```text
Feature-v2 + RoBERTa Text + Adaptive Global Sampling + Global Follow candidates
```

The local preprocessing stage has completed for `run_0002`. The formal RoBERTa-text packs have been built under `packs/`:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

The main formal training target is Label-v2. Label-v5 is reserved for strict-label robustness. Label-v3 and Label-v4 are backup label versions.

Important Label-v2 pack diagnostics:

```text
train samples = 13940
valid samples = 3032
test samples  = 2933
max nodes     = 7310
T             = 6
node features = 47
window features = 24
text semantic dim = 64
```

## Training Pipeline

The training entry is:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml
```

Training reads only the packed samples:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

RoBERTa is not executed during training. RoBERTa encoding, dimensionality reduction, and text-window aggregation are all offline preprocessing steps. During training, the model only reads `node_text_x` and `window_text_x` from the pack.

The default formal config uses:

```text
epochs = 10
batch_size = 8
hidden_dim = 64
top_k_global = 20
text_semantic_dim = 64
num_workers = 8
pin_memory = true
persistent_workers = true
prefetch_factor = 2
eval_every = 1
```

A training epoch does:

```text
1. load pickle-stream samples from pack
2. collate variable-size cascades
3. dynamically pad each batch to the batch max node count
4. run DRAGEN-Full forward/backward over T=6 windows
5. run full validation when eval_every triggers
6. write metrics/checkpoints
7. after training, export valid/test predictions
```

## Why Training Is Slow

The bottleneck is not RoBERTa and not GPU memory capacity. The current bottleneck is the global graph branch and irregular graph computation.

Main reasons:

```text
1. The pack is a pickle stream, so sample loading and object materialization are Python-heavy.
2. Cascades have highly variable node counts; dynamic padding is expensive when large cascades enter a batch.
3. The batch contains irregular per-sample/per-window edge lists, not a single dense tensor workload.
4. DRAGEN-Full runs multiple modules for each of T=6 windows.
5. The global branch processes global candidate edges and global prior encoding.
6. Full validation and prediction export also repeat expensive forward passes.
```

Opening `--bucket-by-nodes` is required for speed tests and formal runs because it groups cascades with similar node counts and reduces padding waste.

Boolean CLI note:

```text
Use --no-plot-every-epoch, not --plot-every-epoch false.
Use --no-tensorboard if TensorBoard is not installed.
```

## Speed Diagnosis Setup

All speed runs used the same Label-v2 pack and the same sample limits:

```text
train samples = 512
valid samples = 256
test samples  = 256
batch_size    = 8
bucket        = yes
bucket multiplier = 50
num_workers = 8
pin_memory = true
persistent_workers = true
prefetch_factor = 2
plot_every_epoch = false
tensorboard = false
```

Commands used:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --num-workers 8 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 512 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --out-dir work/artifacts/_speed_v2_full_bs8_bucket
```

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_global_prior.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --num-workers 8 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 512 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --out-dir work/artifacts/_speed_v2_no_global_bs8_bucket
```

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_adaptive_sampling.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --num-workers 8 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 512 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --out-dir work/artifacts/_speed_v2_no_adaptive_bs8_bucket
```

## Speed Results

| Run | Batch | Bucket | Global | Adaptive | Top-K | Train Samples | Valid Samples | Epoch Time Sec | Valid AUC | Valid AP | Valid F1 | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full | 8 | yes | yes | yes | 20 | 512 | 256 | 599.02 | 0.7800 | 0.5956 | 0.0000 | Baseline speed run |
| no_global | 8 | yes | no | no | - | 512 | 256 | 36.05 | 0.7321 | 0.4710 | 0.0000 | Same DataLoader settings as Full |
| no_adaptive | 8 | yes | yes | no | 20 | 512 | 256 | 516.22 | 0.7801 | 0.5971 | 0.0000 | Keeps global prior, disables adaptive scoring |

Differences:

```text
Full - no_global        = 562.97s
Full - no_adaptive      = 82.80s
no_adaptive - no_global = 480.17s
```

## Diagnosis

Global-related computation is the dominant bottleneck on this speed subset.

Disabling the whole global branch reduces epoch time from 599.02s to 36.05s. Disabling only adaptive scoring reduces epoch time from 599.02s to 516.22s. Therefore, adaptive scoring is not the main cost by itself.

The largest cost appears to be non-adaptive global prior / global candidate edge handling / global encoder computation.

The speed profile corresponds to this case:

```text
Full is very slow.
no_global is very fast.
no_adaptive is still close to Full.
```

So the next performance target should be the global branch as a whole, not only the adaptive scorer.

## Recommended Next Steps

Do not immediately run 10 epochs of the current Full model.

Recommended next experiments:

```text
1. Test Full with top_k_global=10.
2. Test Full with top_k_global=5.
3. If top-k reduction makes Full acceptable, run formal Label-v2 Full with the selected top-k.
4. If Full is still too slow, run a formal no_global or Lite version first as a usable result.
5. Then optimize global candidate edge handling/global prior implementation before returning to Full.
```

Formal runs should keep:

```text
--bucket-by-nodes
--bucket-size-multiplier 50
--no-plot-every-epoch
```

Use `--no-tensorboard` unless TensorBoard is installed.

## Cleanup Note

Intermediate speed-run artifacts under `work/artifacts/` were temporary and can be deleted after this document is saved. This document is the retained trace for the speed diagnosis.

## Key-user Pool Follow-up

A new `key_user_pool` global sampling mode was implemented on branch `feature/key-user-pool-global-prior` to replace the slow edge-list global branch while preserving the old branch as the default.

New files and config:

```text
scripts/13b_build_key_user_pool_packs.py
src/dragen/models/key_user_global_prior.py
configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml
```

The key-user pack adds fixed window-level fields:

```text
key_user_idx    [T, R]
key_user_weight [T, R]
key_user_hop    [T, R]
key_user_mask   [T, R]
```

Verified shape smoke:

```text
key_user_idx     (2, 6, 32) torch.int64
key_user_weight  (2, 6, 32) torch.float32
key_user_hop     (2, 6, 32) torch.int64
key_user_mask    (2, 6, 32) torch.bool
```

A 64/32/32 end-to-end smoke run completed successfully, including valid/test prediction export.

Speed update on the 512/256/256 training epoch:

| Run | Epoch Time Sec |
|---|---:|
| old edge-list Full | 599.02 |
| no_global | 36.05 |
| no_adaptive | 516.22 |
| key_user_pool | 42.07 |

The key-user pool branch reduces the training epoch time close to the no-global lower bound. The 512/256/256 run exceeded the tool timeout during final prediction export, after `epoch_metrics.csv` had already been written. The training epoch itself finished and recorded `epoch_time_sec=42.07s`.
