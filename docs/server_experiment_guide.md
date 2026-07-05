# Server Experiment Guide

This guide is for the current active training branch:

```text
feature/key-user-pool-global-prior
```

Current recommended experiment line:

```text
Feature-v2 + RoBERTa Text + Key-user Pool Global Prior
```

Do not use the old edge-list Full run as the first formal server run. Speed diagnosis showed the old global edge-list branch is the main bottleneck.

## 1. Pull Code

Fresh clone:

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout feature/key-user-pool-global-prior
```

Existing repo:

```bash
cd DRAGEN
git fetch codezzp
git checkout feature/key-user-pool-global-prior
git pull
```

Verify:

```bash
git branch --show-current
git log --oneline -3
```

Expected branch:

```text
feature/key-user-pool-global-prior
```

## 2. Environment

Recommended:

```text
Python >= 3.10
CUDA-enabled PyTorch
```

Install common dependencies:

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
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

## 3. Required Pack

The formal v2 key-user run requires this pack:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Each pack directory must contain:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

If the key-user pack already exists locally, upload it:

```powershell
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool user@server:/path/to/DRAGEN/packs/
```

Linux/macOS or server-to-server transfer:

```bash
rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/
```

If the server only has the old v2 RoBERTa-text pack, build the key-user pack on the server:

```bash
python scripts/13b_build_key_user_pool_packs.py \
  --in-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
  --max-hops 4 \
  --key-users-per-window 32 \
  --seed-budget 16 \
  --rho 0.6
```

Normal training does not need these raw/intermediate directories:

```text
graph/follow_edges.tsv
work/runs/run_0002/windows/
work/runs/run_0002/features_v2/
work/runs/run_0002/text_embeddings/
```

## 4. Pack Shape Smoke

Check that the key-user fields are readable and collated correctly:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dragen.data.pack_reader import PickleStreamDataset, collate_fn

p = 'packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool/train.pt'
ds = PickleStreamDataset(p, max_samples=2, split='train-smoke')
b = collate_fn([ds[0], ds[1]])

for k in [
    'window_x',
    'node_x',
    'node_text_x',
    'window_text_x',
    'key_user_idx',
    'key_user_weight',
    'key_user_hop',
    'key_user_mask',
]:
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

Run a small train/valid/test export smoke before any formal run:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
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
  --out-dir work/artifacts/_smoke_v2_key_user_pool_e2e
```

Check outputs:

```bash
cat work/artifacts/_smoke_v2_key_user_pool_e2e/reports/metrics.json
ls work/artifacts/_smoke_v2_key_user_pool_e2e/predictions
```

## 6. Speed Test

Before formal training, run the 512/256/256 speed test:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 512 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --out-dir work/artifacts/_speed_v2_key_user_pool_bs8_bucket
```

Read training epoch time:

```bash
cat work/artifacts/_speed_v2_key_user_pool_bs8_bucket/reports/epoch_metrics.csv
```

Local reference:

```text
old edge-list Full = 599.02s
no_global          = 36.05s
no_adaptive        = 516.22s
key_user_pool      = 42.07s
```

`epoch_time_sec` records train + valid. Final prediction export may take extra time on larger samples.

## 7. Formal v2 Key-user Training

Run v2 seed0:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard
```

Default output:

```text
work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
```

Optional seed1:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1
```

Optional seed2:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 2 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2
```

## 8. Resume

Resume seed0 from `last.pt`:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --resume work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/checkpoints/last.pt \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard
```

## 9. Result Sync-back

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

Example:

```bash
rsync -av --progress user@server:/path/to/DRAGEN/work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/reports/ \
  work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/reports/

rsync -av --progress user@server:/path/to/DRAGEN/work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/predictions/ \
  work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/predictions/
```

## 10. Recommended Order

```text
1. Checkout feature/key-user-pool-global-prior
2. Upload or build v2 key_user_pool pack
3. Run pack shape smoke
4. Run 64/32/32 end-to-end smoke
5. Run 512/256/256 speed test
6. Run formal v2 key_user_pool seed0
7. Sync back reports/ and predictions/
8. Decide whether to add seed1/seed2 or v5 key_user_pool
```
