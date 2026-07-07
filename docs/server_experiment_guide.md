# Server Experiment Guide

This guide is for migrating the current `run_0003` Weibo experiment to a GPU server for formal training.

Current branch:

```text
experiment/run-0003-server-training-docs
```

Training line:

```text
run_0003 + Feature-v2 + RoBERTa Text + Label-v2 + Key-user Pool Global Prior
```

Important status:

```text
run_0003 formal training has not been run yet.
Only the run_0003 pack pipeline and a tiny smoke run exist locally.
Do not report the old run_0002 seed results as run_0003 results.
```

## 1. Pull Code

Fresh clone:

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout experiment/run-0003-server-training-docs
```

Existing repo:

```bash
cd DRAGEN
git fetch origin
git checkout experiment/run-0003-server-training-docs
git pull
```

Verify:

```bash
git branch --show-current
git log --oneline -3
```

## 2. Environment

Recommended:

```text
Python >= 3.10
CUDA-enabled PyTorch
```

Install common dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas scipy scikit-learn tqdm pyyaml matplotlib networkx
python -m pip install transformers accelerate datasets sentencepiece tokenizers safetensors huggingface_hub
python -m pip install tensorboard
```

Install PyTorch according to the server CUDA version. CUDA 12.8 example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## 3. Required run_0003 Pack

Formal `run_0003` training must use this key-user pack:

```text
packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Each pack directory must contain:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

Local pack diagnostics:

```text
train = 6749
valid = 1450
test  = 1348
total = 9547
positive labels = 2043
negative labels = 7504
T = 6
window_x dim = 24
node_x dim = 47
text_semantic_dim = 64
key_users_per_window = 32
```

Upload the pack:

```bash
rsync -av --progress \
  packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/ \
  user@server:/path/to/DRAGEN/packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/
```

Normal training does not need raw/intermediate directories if this pack is already present on the server.

## 4. Pack Shape Smoke

Run this before training:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dragen.data.pack_reader import PickleStreamDataset, collate_fn

p = 'packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/train.pt'
ds = PickleStreamDataset(p, max_samples=2, split='train-smoke')
b = collate_fn([ds[0], ds[1]])

for k in ['window_x', 'node_x', 'node_text_x', 'window_text_x', 'key_user_idx', 'key_user_weight', 'key_user_hop', 'key_user_mask']:
    print(k, tuple(b[k].shape), b[k].dtype)
PY
```

Expected key-user shapes:

```text
key_user_idx     (2, 6, 32)
key_user_weight  (2, 6, 32)
key_user_hop     (2, 6, 32)
key_user_mask    (2, 6, 32)
```

## 5. End-to-end Smoke

Run a tiny `run_0003` smoke before any formal run:

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

The local smoke already completed with these validation values:

```text
valid_accuracy = 0.75
valid_auc = 0.4635
valid_ap = 0.4278
```

These are smoke-only values and must not be used as formal results.

## 6. Formal run_0003 Training

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

Report the mean and standard deviation over the three seeds.

## 7. Resume

Resume seed0:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --pack-dir packs/run_0003_obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
  --resume work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0/checkpoints/last.pt \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0003_dragen_label_v2_roberta_text_key_user_pool_seed0
```

## 8. Result Sync-back

For each formal run, sync back at least:

```text
work/artifacts/<run>/reports/
work/artifacts/<run>/predictions/
```

If checkpoint size is acceptable, also sync:

```text
work/artifacts/<run>/checkpoints/best.pt
work/artifacts/<run>/checkpoints/last.pt
```

## 9. Recommended Order

```text
1. Checkout experiment/run-0003-server-training-docs
2. Upload the run_0003 key-user pack
3. Run pack shape smoke
4. Run tiny smoke training
5. Run formal run_0003 seed0
6. Run formal run_0003 seed1 and seed2
7. Sync back reports/ and predictions/
8. Export tables only after formal run_0003 results exist
```
