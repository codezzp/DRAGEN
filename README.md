# DRAGEN

DRAGEN 是组织化级联传播预测实验仓库。当前主线是：

```text
Feature-v2 + RoBERTa Text + Key-user Pool Global Prior
```

当前代码分支：

```text
feature/key-user-pool-global-prior
```

大文件和实验产物不进 Git：

```text
work/
packs/
graph/follow_edges.tsv
*.zip
```

代码、配置和文档走 Git；数据包、训练结果、checkpoint 用 `scp` / `rsync` 单独传输。

## 1. 当前状态

本机已经完成 `run_0002` 的正式 RoBERTa-text pack：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

实验策略：

```text
v2：主实验标签版本
v5：严格标签鲁棒性验证
v3/v4：先保留，后面有时间再补
```

已经验证过的输入维度：

```text
window_x      = (B, 6, 24)
node_x        = (B, 6, N, 47)
node_text_x   = (B, 6, N, 64)
window_text_x = (B, 6, 64)
```

RoBERTa 文本产物：

```text
root embedding 原始维度    = (85263, 768)
retweet embedding 原始维度 = (193331, 768)
降维后维度                 = 64
node_text_features         = (567718, 64)
window_text_features       = (511578, 64)
```

注意：很多 retweet 行没有原始文本，所以这些节点的 `node_text_x` 是零向量；但每个窗口都有 root 文本语义，所以 `window_text_x` 不为空。

## 2. 服务器部署

服务器拉代码：

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout feature/key-user-pool-global-prior
```

如果服务器已有仓库：

```bash
cd DRAGEN
git fetch codezzp
git checkout feature/key-user-pool-global-prior
git pull
```

推荐环境：

```text
Python >= 3.10
PyTorch CUDA 版
numpy pandas scipy scikit-learn tqdm pyyaml matplotlib networkx
transformers tokenizers safetensors huggingface_hub
```

安装依赖：

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install --upgrade pip setuptools wheel

python -m pip install numpy pandas scipy scikit-learn tqdm pyyaml matplotlib networkx
python -m pip install transformers accelerate datasets sentencepiece tokenizers safetensors huggingface_hub
python -m pip install tensorboard
```

PyTorch 根据服务器 CUDA 版本安装。CUDA 12.8 示例：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证 GPU：

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

## 3. 需要传到服务器的数据

第一批只传 v2 和 v5：

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

可选传输，后续补实验时再用：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
```

PowerShell 传输示例：

```powershell
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text user@server:/path/to/DRAGEN/packs/
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text user@server:/path/to/DRAGEN/packs/
```

Linux/macOS 或服务器间传输示例：

```bash
rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/

rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text/
```

正常训练不需要传：

```text
graph/follow_edges.tsv
work/runs/run_0002/windows/
work/runs/run_0002/features_v2/
work/runs/run_0002/text_embeddings/
```

因为训练只读 pack，global follow candidate 和 RoBERTa text 已经写进 `.pt`。

## 4. 第一件事：服务器 smoke test

先检查 v2 pack 能不能读取：

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

然后跑一个 1 epoch 小样本训练：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_v2_key_user_pool_e2e
```

检查输出：

```bash
ls work/artifacts/_smoke_v2_key_user_pool_e2e/reports
ls work/artifacts/_smoke_v2_key_user_pool_e2e/predictions
cat work/artifacts/_smoke_v2_key_user_pool_e2e/reports/metrics.json
```

如果 smoke test 报错，先不要跑正式实验。

## 5. 正式实验顺序

推荐顺序：

```text
1. v2 DRAGEN-Full seed 0
2. v2 DRAGEN-Full seed 1 / seed 2，有时间再补
3. v2 核心消融
4. v5 严格标签鲁棒性
5. 导出结果表和回传 predictions/reports
```

当前分支里的 baseline 入口还只是占位实现：

```text
src/dragen/baselines/cac_stat.py
src/dragen/baselines/campaign_gnn.py
src/dragen/baselines/temporal_gnn.py
```

因此现在不要直接跑：

```bash
python scripts/14_train_cac_stat.py
python scripts/15_train_gnn_baselines.py
```

它们不会产生正式 baseline 结果。当前服务器可直接执行的是 DRAGEN-Full 和 DRAGEN 模块消融。

## 6. v2 主实验

seed 0：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml
```

seed 1：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1
```

seed 2：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 2 \
  --out-dir work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2
```

断点续训：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --resume work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0/checkpoints/last.pt
```

## 7. v5 严格标签鲁棒性

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5_roberta_text.yaml
```

如果要补 seed：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5_roberta_text.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v5_roberta_text_feature_v2_seed1
```

## 8. v2 消融实验

现有消融 YAML 默认指向 v4 pack。如果论文主消融要用 v2，请用 CLI 覆盖 `--pack-dir` 和 `--out-dir`。

v2 主 pack：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
```

核心消融：

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_global_prior.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_global_prior

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_role.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_role

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_memory.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_memory

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_gate.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_gate

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_uncertainty.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_uncertainty

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_adaptive_sampling.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_adaptive_sampling
```

暂时不要直接跑以下消融，除非已经准备好对应 pack 或代码：

```text
w/o RoBERTa Text
w/o MultiScale Context
w/o HybridTree
```

原因：当前 key-user pool 分支仍使用 RoBERTa text pack，模型要求 pack 里有 `node_text_x`；`w/o MultiScale` 和 `w/o HybridTree` 也需要对应结构的 feature_v2 + roberta_text pack。

## 9. 结果检查

每个 run 结束后看：

```bash
cat work/artifacts/<run>/reports/metrics.json
head work/artifacts/<run>/predictions/event_predictions.csv
ls work/artifacts/<run>/checkpoints
```

重点文件：

```text
reports/metrics.json
reports/loss_breakdown.json
reports/epoch_metrics.csv
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
predictions/event_predictions.csv
predictions/valid_event_predictions.csv
predictions/test_event_predictions.csv
predictions/node_window_predictions.csv
predictions/role_distribution.csv
predictions/gate_weights.csv
predictions/uncertainty.csv
predictions/event_attention.csv
predictions/sampled_global_neighbors.csv
checkpoints/best.pt
checkpoints/last.pt
```

如果 checkpoint 太大，至少回传：

```text
reports/
predictions/
```

## 10. 阈值和正式指标

当前代码默认分类阈值仍是：

```text
threshold = 0.5
```

论文正式结果建议后处理为：

```text
在 valid_event_predictions.csv 上选择 F1 最优 threshold，
然后把这个 threshold 固定应用到 test_event_predictions.csv。
```

这样可以避免出现：

```text
AUC 高，但 Precision / Recall / F1 = 0
```

正式表至少看：

```text
AUC
AUPRC / AP
Precision
Recall
F1
MCC
Accuracy
Best threshold
Precision@K
Recall@K
```

## 11. 导出结果表

训练完成后运行预测分析：

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
```

导出表格：

```bash
python scripts/18_export_result_tables.py \
  --run-dirs \
    work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0 \
    work/artifacts/dragen_follow_adaptive_label_v5_roberta_text_feature_v2_seed0 \
  --ablation-run-dirs \
    work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0 \
    work/artifacts/label_v2_ablation_no_global_prior \
    work/artifacts/label_v2_ablation_no_role \
    work/artifacts/label_v2_ablation_no_memory \
    work/artifacts/label_v2_ablation_no_gate \
    work/artifacts/label_v2_ablation_no_uncertainty \
    work/artifacts/label_v2_ablation_no_adaptive_sampling \
  --full-run-dir work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0 \
  --out-dir work/artifacts/reports
```

输出：

```text
work/artifacts/reports/main_results.csv
work/artifacts/reports/risk_retrieval_results.csv
work/artifacts/reports/ablation_results.csv
```

## 12. 如果必须在服务器重建 pack

正常不建议服务器重建 pack。确实要重建时，需要先传：

```text
work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/
work/runs/run_0002/labels_v2_stratified_score/
work/runs/run_0002/labels_v5_ensemble_consensus/
```

重建 v2 pack：

```bash
python scripts/13_build_packs.py \
  --run-id run_0002 \
  --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --labels work/runs/run_0002/labels_v2_stratified_score/weak_event_labels.csv \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --text-semantic-dir work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64 \
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
```

不要在服务器重扫 `graph/follow_edges.tsv`，除非必须重建 global candidate table。

## 13. 相关文档

```text
docs/training_commands.md       RoBERTa 预处理和 pack 构建命令
docs/results_summary.md         当前实验进度
docs/label_design.md            弱标签版本设计
docs/configuration.md           YAML 配置规则
docs/server_experiment_guide.md 历史服务器迁移记录
```

## Key-user Pool Global Prior Branch

A new implementation branch is available for replacing the slow edge-list global prior path:

```text
feature/key-user-pool-global-prior
```

The goal of this branch is engineering speed, not a paper-text rewrite. It keeps the old `AdaptiveGlobalSampler + GlobalPriorEncoder` path as the default `edge_list` mode and adds a new `key_user_pool` mode.

### Why

The speed diagnosis on Label-v2 showed:

```text
old Full edge-list epoch_time_sec = 599.02s
no_global epoch_time_sec          = 36.05s
no_adaptive epoch_time_sec        = 516.22s
```

This means the bottleneck is the edge-list global branch as a whole, not only the adaptive scorer.

### New Pack Format

Build a key-user pack from an existing RoBERTa-text pack:

```bash
python scripts/13b_build_key_user_pool_packs.py \
  --in-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-pack packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool \
  --max-hops 4 \
  --key-users-per-window 32 \
  --seed-budget 16 \
  --rho 0.6
```

The script adds these per-sample fields:

```text
key_user_idx    [T, R]
key_user_weight [T, R]
key_user_hop    [T, R]
key_user_mask   [T, R]
```

Default values are `T=6`, `R=32`, `max_hops=4`.

### Training Config

Use:

```text
configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml
```

Main fields:

```yaml
model:
  global_sampling_mode: key_user_pool
  key_user_max_hops: 4
  key_users_per_window: 32
```

### Smoke Test

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

Verified locally: the 64/32/32 end-to-end smoke completed, including valid/test export.

### Speed Result

With 512/256/256 samples, `batch_size=8`, bucket enabled, the key-user mode reached:

```text
key_user_pool epoch_time_sec = 42.07s
```

This is close to `no_global=36.05s` and much faster than the old edge-list Full path `599.02s`.

### Server Training Guide

For server-side continuation, use the key-user branch and follow:

```text
docs/server_experiment_guide.md
```

Recommended server sequence:

```text
1. git fetch codezzp && git checkout feature/key-user-pool-global-prior && git pull
2. upload or build the v2 key-user pack
3. run key-user pack shape smoke
4. run 64/32/32 end-to-end smoke
5. run 512/256/256 speed test
6. run formal v2 key-user seed0
7. sync back reports/ and predictions/
8. decide whether to add seed1/seed2 or v5 key-user
```

Key-user pack required by the formal v2 run:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

Formal v2 key-user command:

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

The old edge-list Full config is no longer the recommended first formal run because the speed diagnosis showed the global edge-list branch is the main bottleneck.

## Run 0002 Calibration and Diagnostics Update

Current optimization branch:

```text
experiment/run-0002-calibration-diagnostics
```

This branch keeps the main DRAGEN architecture fixed and adds only bounded performance tools:

```text
1. loss-effectiveness diagnostics
2. probability calibration on existing predictions
3. NLL / Brier / ECE reporting for calibrated probabilities
```

Recommended post-training checks:

```bash
python scripts/21_calibrate_thresholds.py
python scripts/22_summarize_epoch_selection.py
python scripts/23_calibrate_probabilities.py
```

`reports/loss_breakdown.json` now includes raw, weighted, and relative loss contribution fields:

```text
loss_<name>
loss_weight_<name>
weighted_loss_<name>
loss_contribution_<name>
```

Use these fields to verify whether role, structure, sampler, jump, and uncertainty losses are actually active before describing them as effective training objectives.

Probability calibration is a validation-fitted post-processing step. It fits on `valid_event_predictions.csv`, freezes the calibrator, and applies it to `test_event_predictions.csv`. It should be used to improve probability interpretability and threshold stability, not to claim an AUC gain.

Detailed docs:

```text
docs/run_0002_performance_improvement_plan.md
docs/run_0002_performance_runbook.md
docs/run_0002_threshold_epoch_analysis.md
```
