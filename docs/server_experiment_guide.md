# 服务器实验指南

本文档是当前服务器训练指南。当前有效分支：

```text
experiment/run-0002-roberta-only
```

当前有效实验线：

```text
Feature-v2 + RoBERTa Text + Adaptive Global Sampling + Global Follow candidates
```

如果与旧文档冲突，以本文档和根目录 `README.md` 为准。

## 1. 服务器拉代码

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout experiment/run-0002-roberta-only
```

已有仓库：

```bash
cd DRAGEN
git fetch codezzp
git checkout experiment/run-0002-roberta-only
git pull
```

确认：

```bash
git branch --show-current
git log --oneline -3
```

## 2. 环境

建议：

```text
Python >= 3.10
PyTorch CUDA 版
```

依赖：

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas scipy scikit-learn tqdm pyyaml matplotlib networkx
python -m pip install transformers accelerate datasets sentencepiece tokenizers safetensors huggingface_hub
python -m pip install tensorboard
```

PyTorch 需要按服务器 CUDA 版本安装。CUDA 12.8 示例：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

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

## 3. 需要传的数据

优先只传两个 pack：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

每个 pack 目录必须包含：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

后续如需补 v3/v4，再传：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
```

不需要传：

```text
graph/follow_edges.tsv
work/runs/run_0002/windows/
work/runs/run_0002/features_v2/
work/runs/run_0002/text_embeddings/
```

原因：训练只读 pack，RoBERTa text、feature_v2 和 global follow candidate 已经写入 pack。

PowerShell 传输示例：

```powershell
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text user@server:/path/to/DRAGEN/packs/
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text user@server:/path/to/DRAGEN/packs/
```

rsync 示例：

```bash
rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/
```

## 4. 先跑 smoke test

检查 pack 读取：

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dragen.data.pack_reader import PickleStreamDataset, collate_fn
p = 'packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/train.pt'
ds = PickleStreamDataset(p, max_samples=2, split='train-smoke')
b = collate_fn([ds[0], ds[1]])
print('window_x', tuple(b['window_x'].shape))
print('node_x', tuple(b['node_x'].shape))
print('node_text_x', tuple(b['node_text_x'].shape))
print('window_text_x', tuple(b['window_text_x'].shape))
print('global edges', [tuple(x.shape) for x in b['global_candidate_edge_index']])
PY
```

小样本训练：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_dragen_v2_roberta_text
```

检查：

```bash
cat work/artifacts/_smoke_dragen_v2_roberta_text/reports/metrics.json
head work/artifacts/_smoke_dragen_v2_roberta_text/predictions/event_predictions.csv
```

## 5. 正式训练顺序

```text
1. v2 DRAGEN-Full seed 0
2. v2 DRAGEN-Full seed 1/2，有时间再补
3. v2 模块消融
4. v5 严格标签鲁棒性
5. 导出表格，回传 reports/ 和 predictions/
```

v2 seed 0：

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v2_roberta_text.yaml
```

v2 seed 1：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed1
```

v5：

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5_roberta_text.yaml
```

## 6. 消融

现有消融 YAML 默认历史上指向 v4。若主消融使用 v2，必须覆盖 `--pack-dir` 和 `--out-dir`。

示例：

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_global_prior.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_global_prior
```

完整消融命令见：

```text
docs/training_commands.md
README.md
```

## 7. 当前不能直接跑的 baseline

以下 baseline 入口当前仍是占位实现，不能产出正式结果：

```text
scripts/14_train_cac_stat.py
scripts/15_train_gnn_baselines.py
src/dragen/baselines/cac_stat.py
src/dragen/baselines/campaign_gnn.py
src/dragen/baselines/temporal_gnn.py
```

当前服务器优先完成 DRAGEN-Full 与模块消融。

## 8. 结果回传

每个 run 至少回传：

```text
work/artifacts/<run>/reports/
work/artifacts/<run>/predictions/
```

如需恢复训练或保留最佳模型，再回传：

```text
work/artifacts/<run>/checkpoints/best.pt
work/artifacts/<run>/checkpoints/last.pt
```

## 9. 阈值提醒

当前训练脚本默认 `threshold=0.5`。论文正式结果建议后处理：

```text
在 valid_event_predictions.csv 上选择 F1 最优 threshold，
再固定该 threshold 到 test_event_predictions.csv。
```

后续需要把该逻辑补进评估/导表脚本。

## 10. Key-user pool 分支后续训练指导

当前推荐优先使用 key-user pool 分支继续训练，不再优先跑旧的 edge-list Full 分支。

代码分支：

```bash
git fetch codezzp
git checkout feature/key-user-pool-global-prior
git pull
```

该分支保留旧 global 分支，默认 `edge_list` 不变；新增快速分支：

```text
global_sampling_mode: key_user_pool
```

### 10.1 服务器需要的 pack

key-user 训练需要这个 pack：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser
```

如果本地已经构建好，直接传到服务器：

```powershell
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser user@server:/path/to/DRAGEN/packs/
```

或用 rsync：

```bash
rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser/
```

如果服务器上只有旧 v2 RoBERTa-text pack，也可以在服务器上构建 key-user pack：

```bash
python scripts/13b_build_key_user_pool_packs.py \
  --in-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser \
  --max-hops 4 \
  --key-users-per-window 32 \
  --seed-budget 16 \
  --rho 0.6
```

构建完成后，key-user pack 每个 split 仍然必须包含：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

### 10.2 pack shape 检查

先检查新字段是否能被读取和 collate：

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dragen.data.pack_reader import PickleStreamDataset, collate_fn

p = 'packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_keyuser/train.pt'
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

期望看到：

```text
key_user_idx     (2, 6, 32)
key_user_weight  (2, 6, 32)
key_user_hop     (2, 6, 32)
key_user_mask    (2, 6, 32)
```

### 10.3 端到端 smoke

先跑小样本端到端训练，确认 train、valid、test export 都能完成：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 64 \
  --max-valid-samples 32 \
  --max-test-samples 32 \
  --out-dir work/artifacts/_smoke_v2_keyuser_pool_e2e
```

检查：

```bash
cat work/artifacts/_smoke_v2_keyuser_pool_e2e/reports/metrics.json
ls work/artifacts/_smoke_v2_keyuser_pool_e2e/predictions
```

### 10.4 速度测试

服务器正式跑前建议先复测 512/256/256：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --epochs 1 \
  --batch-size 8 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --max-train-samples 512 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --out-dir work/artifacts/_speed_v2_keyuser_pool_bs8_bucket
```

本地记录的训练 epoch 速度：

```text
old edge-list Full = 599.02s
no_global          = 36.05s
no_adaptive        = 516.22s
key_user_pool      = 42.07s
```

注意：`epoch_time_sec` 只统计 train + valid，不包含最终 valid/test prediction export。512/256/256 的本地 run 在 final export 阶段被工具超时打断，但训练 epoch 已完成并写出 `epoch_metrics.csv`。服务器如果时间充足，可以让它跑完整导出。

查看速度结果：

```bash
cat work/artifacts/_speed_v2_keyuser_pool_bs8_bucket/reports/epoch_metrics.csv
```

### 10.5 正式 v2 key-user 训练

如果 smoke 和 speed 都正常，正式跑 v2 seed0：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard
```

默认输出目录：

```text
work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed0
```

如果要补 seed1：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed1
```

如果要补 seed2：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --seed 2 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed2
```

### 10.6 中断恢复

如果训练中断，用 `last.pt` 恢复：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_keyuser.yaml \
  --resume work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed0/checkpoints/last.pt \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --no-plot-every-epoch \
  --no-tensorboard
```

### 10.7 结果回传

每个正式 run 至少回传：

```text
work/artifacts/<run>/reports/
work/artifacts/<run>/predictions/
```

如果 checkpoint 不太大，也回传：

```text
work/artifacts/<run>/checkpoints/best.pt
work/artifacts/<run>/checkpoints/last.pt
```

推荐回传命令：

```bash
rsync -av --progress user@server:/path/to/DRAGEN/work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed0/reports/ \
  work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed0/reports/

rsync -av --progress user@server:/path/to/DRAGEN/work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed0/predictions/ \
  work/artifacts/dragen_follow_keyuser_label_v2_roberta_text_feature_v2_seed0/predictions/
```

### 10.8 当前推荐顺序

```text
1. 拉取 feature/key-user-pool-global-prior 分支
2. 确认或构建 v2 key-user pack
3. 运行 pack shape 检查
4. 运行 64/32/32 smoke
5. 运行 512/256/256 speed test
6. 正式运行 v2 key-user seed0
7. 回传 reports/ 和 predictions/
8. 再决定是否补 seed1/seed2 或 v5 key-user
```

当前不建议优先跑旧 `dragen_full_label_v2_roberta_text.yaml` 的 edge-list Full，因为已确认 global edge-list 分支是主要速度瓶颈。
