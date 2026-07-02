# Server Experiment Guide

本文档说明如何把当前 DRAGEN-Full 实验迁移到服务器运行。`work/` 不进入 Git，因此服务器需要单独同步本地实验产物。

## 1. 代码分支

服务器上拉取：

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout experiment/run-0002-code
```

以远端分支最新提交为准。可在服务器确认：

```bash
git log --oneline -1
```

## 2. 环境

建议环境：

```text
Python >= 3.10
PyTorch >= 2.1
NumPy >= 1.24
tqdm >= 4.66
TensorBoard >= 2.14
PyYAML >= 6.0
```

CPU 环境：

```bash
python -m pip install -r requirements.txt
```

GPU 服务器建议按服务器 CUDA 版本安装 PyTorch。例如 CUDA 12.1：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy pytest tqdm tensorboard PyYAML
```

验证：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
```

## 3. 最小迁移文件

如果服务器只负责训练 DRAGEN-Full 和消融，不重建窗口/特征/标签，最小需要同步：

```text
work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/packs/obs_1800_win300_step300_star/
work/runs/run_0002/packs/obs_1800_win300_step300_hybrid_tree/
```

每个 pack 目录至少包含：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

服务器输出目录需要存在或由脚本自动创建：

```text
work/artifacts/
```

## 4. 可复现实验迁移文件

如果服务器需要从特征/标签重新构建 pack，额外同步：

```text
work/runs/run_0002/features/obs_1800_win300_step300_star/
work/runs/run_0002/features/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/labels/
```

如果服务器需要从窗口重新构建特征，额外同步：

```text
work/runs/run_0002/windows/obs_1800_win300_step300/
work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/edges/hybrid_tree_light/
```

如果服务器需要从原始组织化任务表重建窗口，额外同步：

```text
work/runs/run_0002/org_task/
```

不建议同步到 Git 的大文件仍然保持本地/服务器文件传输：

```text
work/
graph/follow_edges.tsv
```

当前 DRAGEN-Full 不需要 `graph/follow_edges.tsv`，因为正式 HybridTree Light 不扫描关注图。

## 5. 推荐传输方式

Linux/macOS 客户端：

```bash
rsync -av --progress \
  work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/ \
  user@server:/path/to/DRAGEN/work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
```

Windows PowerShell 可用 `scp`：

```powershell
scp -r work\runs\run_0002\packs\obs_1800_step300_multiscale_hybrid_tree user@server:/path/to/DRAGEN/work/runs/run_0002/packs/
scp -r work\runs\run_0002\packs\obs_1800_win300_step300_star user@server:/path/to/DRAGEN/work/runs/run_0002/packs/
scp -r work\runs\run_0002\packs\obs_1800_win300_step300_hybrid_tree user@server:/path/to/DRAGEN/work/runs/run_0002/packs/
```

## 6. Debug 训练

先在服务器跑小样本 debug：

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_debug \
  --epochs 1 \
  --batch-size 2 \
  --max-train-samples 200 \
  --max-valid-samples 100 \
  --max-test-samples 100 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto
```

验收输出：

```text
work/artifacts/dragen_full_debug/reports/metrics.json
work/artifacts/dragen_full_debug/reports/loss_breakdown.json
work/artifacts/dragen_full_debug/reports/epoch_metrics.csv
work/artifacts/dragen_full_debug/predictions/event_predictions.csv
work/artifacts/dragen_full_debug/predictions/node_window_predictions.csv
work/artifacts/dragen_full_debug/predictions/role_distribution.csv
work/artifacts/dragen_full_debug/predictions/gate_weights.csv
work/artifacts/dragen_full_debug/predictions/uncertainty.csv
work/artifacts/dragen_full_debug/predictions/event_attention.csv
work/artifacts/dragen_full_debug/predictions/sampled_global_neighbors.csv
work/artifacts/dragen_full_debug/checkpoints/best.pt
work/artifacts/dragen_full_debug/checkpoints/last.pt
```

## 6.1 服务器根目录 packs 快速命令

如果你已经把 pack 目录放在项目根目录下：

```text
packs/
  obs_1800_step300_multiscale_hybrid_tree/
  obs_1800_win300_step300_star/
  obs_1800_win300_step300_hybrid_tree/
```

则后续命令统一使用 `packs/...`，不再使用 `work/runs/run_0002/packs/...`。

先确认文件：

```bash
ls packs
ls packs/obs_1800_step300_multiscale_hybrid_tree
```

每个 pack 目录应包含：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

服务器先更新代码和依赖：

```bash
git pull
python -m pip install -r requirements.txt
```

确认 PyTorch 和 tqdm：

```bash
python - <<'PY'
import torch
import tqdm
import tensorboard
import yaml
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("tqdm", tqdm.__version__)
print("tensorboard", tensorboard.__version__)
print("yaml", yaml.__version__)
PY
```

## 6.2 配置驱动训练命令

服务器训练优先使用配置文件。参数优先级是：

```text
脚本默认值 < YAML 配置 < CLI 覆盖
```

DRAGEN-Full debug：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_debug.yaml
```

DRAGEN-Full 正式训练：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml
```

临时覆盖 seed、输出目录或 TensorBoard：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_full_run0002_seed1 \
  --no-tensorboard
```

从 `last.pt` 续训：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --resume work/artifacts/dragen_full_run0002_seed0/checkpoints/last.pt
```

消融训练：

```bash
python scripts/16_train_dragen_full.py --config configs/train/ablation_no_tree.yaml
python scripts/16_train_dragen_full.py --config configs/train/ablation_no_multiscale.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_role.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_memory.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_global_prior.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_adaptive_sampling.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_gate.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_uncertainty.yaml
```

每次训练开始会保存：

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
```

结果表：

```bash
python scripts/18_export_result_tables.py \
  --config configs/train/result_tables_run0002.yaml
```

训练后分析：

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_full_run0002_seed0
```

### Debug 备用展开命令

以下命令与配置文件等价，主要用于临时调参或排查配置覆盖问题。

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_debug \
  --epochs 1 \
  --batch-size 2 \
  --max-train-samples 200 \
  --max-valid-samples 100 \
  --max-test-samples 100 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto
```

### 保存和断点续训

训练脚本现在每个 epoch 结束后都会写：

```text
reports/epoch_metrics.csv
reports/loss_breakdown.json
checkpoints/last.pt
```

验证指标刷新历史最优时会写：

```text
checkpoints/best.pt
```

默认不会保存每个 epoch 的完整 checkpoint，避免磁盘占用过大。如果确实需要保留每轮模型，额外加：

```bash
--save-every-epoch
```

训练过程中查看每轮指标：

```bash
tail -f work/artifacts/dragen_full_run0002/reports/epoch_metrics.csv
```

查看 checkpoint：

```bash
ls -lh work/artifacts/dragen_full_run0002/checkpoints
```

从最近一轮继续训练：

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_run0002 \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto \
  --resume work/artifacts/dragen_full_run0002/checkpoints/last.pt
```

`--epochs` 表示目标总 epoch 数。例如 `last.pt` 已经保存到第 4 轮，设置 `--epochs 10` 会从第 5 轮继续到第 10 轮。

### TensorBoard

训练脚本支持 PyTorch TensorBoard 日志。训练时加：

```bash
--tensorboard
```

默认日志目录：

```text
<out-dir>/tb
```

也可以显式指定：

```bash
--tb-log-dir work/artifacts/tb/dragen_full_run0002
```

启动 TensorBoard：

```bash
tensorboard --logdir work/artifacts --host 0.0.0.0 --port 6006
```

浏览器访问：

```text
http://服务器IP:6006
```

如果服务器端口不能直接访问，用 SSH 转发：

```bash
ssh -L 6006:localhost:6006 user@server
```

然后本地浏览器打开：

```text
http://localhost:6006
```

记录的曲线包括：

```text
train/loss
train/lr
train/epoch_time_sec
valid/loss
valid/accuracy
valid/precision
valid/recall
valid/f1
valid/auc
valid/ap
valid/mcc
loss/total
loss/event
loss/jump
loss/struct
loss/align
loss/uncertainty
loss/role
```

### DRAGEN-Full 正式训练备用展开命令

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_run0002 \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto \
  --seed 0 \
  --tensorboard
```

### 显存不足版本

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_run0002_small \
  --epochs 10 \
  --batch-size 2 \
  --hidden-dim 32 \
  --role-num 5 \
  --top-k-global 10 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto
```

### w/o Tree

使用 Fixed-5m Star pack：

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir packs/obs_1800_win300_step300_star \
  --out-dir work/artifacts/ablation_no_tree \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto
```

### w/o MultiScale

使用 Fixed-5m HybridTree pack：

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir packs/obs_1800_win300_step300_hybrid_tree \
  --out-dir work/artifacts/ablation_no_multiscale \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto
```

### 模块消融

以下消融都使用 MultiScale HybridTree pack。

```bash
python scripts/17_run_ablation.py \
  --ablation no_role \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_role \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --device auto
```

```bash
python scripts/17_run_ablation.py \
  --ablation no_memory \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_memory \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --device auto
```

```bash
python scripts/17_run_ablation.py \
  --ablation no_global_prior \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_global_prior \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --device auto
```

```bash
python scripts/17_run_ablation.py \
  --ablation no_adaptive_sampling \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_adaptive_sampling \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --device auto
```

```bash
python scripts/17_run_ablation.py \
  --ablation no_gate \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_gate \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --device auto
```

```bash
python scripts/17_run_ablation.py \
  --ablation no_uncertainty \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_uncertainty \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --device auto
```

建议运行顺序：

```text
1. dragen_full_debug
2. dragen_full_run0002
3. ablation_no_tree
4. ablation_no_multiscale
5. ablation_no_role
6. ablation_no_memory
7. ablation_no_global_prior
8. ablation_no_adaptive_sampling
9. ablation_no_gate
10. ablation_no_uncertainty
```

## 7. 正式训练

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_run0002 \
  --epochs 10 \
  --batch-size 8 \
  --hidden-dim 64 \
  --role-num 5 \
  --top-k-global 20 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --lambda-jump 0.01 \
  --lambda-struct 0.005 \
  --lambda-align 0.001 \
  --lambda-uncertainty 0.001 \
  --lambda-role 0.0 \
  --device auto
```

如果显存不足：

```bash
python scripts/16_train_dragen_full.py \
  --pack-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/dragen_full_run0002_small \
  --epochs 10 \
  --batch-size 2 \
  --hidden-dim 32 \
  --role-num 5 \
  --top-k-global 10 \
  --lr 0.001 \
  --lambda-jump 0.01 \
  --device auto
```

## 8. 消融输入

`w/o Tree` 使用：

```text
work/runs/run_0002/packs/obs_1800_win300_step300_star/
```

`w/o MultiScale` 使用：

```text
work/runs/run_0002/packs/obs_1800_win300_step300_hybrid_tree/
```

其他消融使用 MultiScale HybridTree pack，并通过脚本参数关闭模块：

```bash
python scripts/17_run_ablation.py \
  --ablation no_gate \
  --pack-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree \
  --out-dir work/artifacts/ablation_no_gate \
  --epochs 10 \
  --batch-size 8 \
  --device auto
```

可选消融名：

```text
no_role
no_memory
no_global_prior
no_adaptive_sampling
no_gate
no_uncertainty
```

## 9. 结果回传

训练后建议回传：

```text
work/artifacts/dragen_full_run0002/reports/
work/artifacts/dragen_full_run0002/predictions/
work/artifacts/dragen_full_run0002/checkpoints/best.pt
```

如果 checkpoint 太大，只回传：

```text
reports/metrics.json
reports/loss_breakdown.json
predictions/*.csv
```

## 10. Post-training metrics and tables

After each training run, compute the event-level and DRAGEN explanation reports:

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_full_run0002
```

For ablations with DRAGEN-style explanation outputs, run the same command:

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/ablation_no_role

python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/ablation_no_memory

python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/ablation_no_gate

python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/ablation_no_uncertainty
```

The script writes:

```text
reports/event_metrics_extended.json
reports/risk_retrieval_metrics.json
reports/temporal_stability_metrics.json
reports/interpretability_metrics.json
reports/diagnostic_summary.csv
```

Fair comparison tables use only event-level predictions. All models must provide:

```text
predictions/event_predictions.csv
```

Required fields:

```text
cascade_idx
split
y_true
y_prob
y_pred
```

Export main, risk-retrieval, and ablation tables:

```bash
python scripts/18_export_result_tables.py \
  --run-dirs \
    work/artifacts/cac_stat_run0002 \
    work/artifacts/campaign_gnn_run0002 \
    work/artifacts/temporal_gnn_run0002 \
    work/artifacts/dragen_full_run0002 \
  --ablation-run-dirs \
    work/artifacts/dragen_full_run0002 \
    work/artifacts/ablation_no_tree \
    work/artifacts/ablation_no_multiscale \
    work/artifacts/ablation_no_role \
    work/artifacts/ablation_no_memory \
    work/artifacts/ablation_no_global_prior \
    work/artifacts/ablation_no_adaptive_sampling \
    work/artifacts/ablation_no_gate \
    work/artifacts/ablation_no_uncertainty \
  --full-run-dir work/artifacts/dragen_full_run0002 \
  --out-dir work/artifacts/reports
```

Outputs:

```text
work/artifacts/reports/main_results.csv
work/artifacts/reports/risk_retrieval_results.csv
work/artifacts/reports/ablation_results.csv
```

Do not put DRAGEN-only explanation metrics into the baseline comparison table. Role, gate, uncertainty, temporal stability, and attention metrics belong only in the explanation/stability analysis table.


## 2026-07-02 Update: Server Data Requirements

For runs with `use_global_prior=true`, the server pack should include global follow candidate fields. If packs are copied from the workstation after rebuild, no server-side `graph/follow_edges.tsv` scan is required.

Required rebuilt pack metadata should list:

```text
global_candidate_edge_index
global_candidate_edge_weight
```

If rebuilding packs on the server, copy `graph/follow_edges.tsv` and run before `13_build_packs.py`:

```bash
python scripts/10_build_global_candidate_edges.py --run-id run_0002 --follow-edges graph/follow_edges.tsv
python scripts/13_build_packs.py --run-id run_0002
```

The current workstation pack has already been rebuilt with candidate edges:

```text
samples_with_global_candidates = 41,289
total_global_candidate_edges   = 890,037
global_candidate_alignment_errors = 0
```

Sampler loss CLI/config keys are available:

```text
--lambda-sampler-edge 0.005
--lambda-sampler-hub 0.001
--lambda-sampler-temp 0.005
```


## Configuration-Driven Runs

Use YAML configs for formal runs. See `docs/configuration.md` for supported sections, field mapping, priority rules, label-version configs, ablation rules, and reproducibility metadata.

Label-version training configs are available at:

```text
configs/train/dragen_full_label_v2.yaml
configs/train/dragen_full_label_v3.yaml
configs/train/dragen_full_label_v4.yaml
configs/train/dragen_full_label_v5.yaml
```

Example:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5.yaml
```


## Current Server Migration Quickstart (run_0002-next)

Use this section as the current entry point for the next experiment version. Older sections above are kept for historical reference.

### 1. Code Branch

On the server:

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout experiment/run-0002-next
```

If the repository already exists:

```bash
cd DRAGEN
git fetch codezzp
git checkout experiment/run-0002-next
git pull
```

Confirm:

```bash
git branch --show-current
git log --oneline -1
```

### 2. Required Data for Training Only

If the workstation has already built packs, the server does not need to scan `graph/follow_edges.tsv`.

Recommended minimum transfer for Label-v2 through Label-v5 experiments:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v2/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v3/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v4/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5/
```

Each pack must contain:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

`meta.json` must list:

```text
global_candidate_edge_index
global_candidate_edge_weight
```

Optional but recommended transfer for diagnostics and audit:

```text
work/runs/run_0002/labels_v1_score_rank/
work/runs/run_0002/labels_v2_stratified_score/
work/runs/run_0002/labels_v3_lf_vote/
work/runs/run_0002/labels_v4_coordination_network/
work/runs/run_0002/labels_v5_ensemble_consensus/
work/runs/run_0002/label_comparison/
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_diagnostics.json
```

Do not commit or transfer through Git:

```text
packs/
work/
graph/follow_edges.tsv
*.zip
```

### 3. Transfer Commands

From workstation with `rsync`:

```bash
rsync -av --progress packs/ user@server:/path/to/DRAGEN/packs/
rsync -av --progress work/runs/run_0002/label_comparison/ user@server:/path/to/DRAGEN/work/runs/run_0002/label_comparison/
```

Windows PowerShell with `scp`:

```powershell
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v2 user@server:/path/to/DRAGEN/packs/
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v3 user@server:/path/to/DRAGEN/packs/
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v4 user@server:/path/to/DRAGEN/packs/
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5 user@server:/path/to/DRAGEN/packs/
scp -r work\runs\run_0002\label_comparison user@server:/path/to/DRAGEN/work/runs/run_0002/
```

### 4. Environment

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install tensorboard PyYAML tqdm
```

Install PyTorch according to the server CUDA version. Example for CUDA 12.1:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python - <<'PY'
import torch, yaml, tqdm
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('yaml', yaml.__version__)
print('tqdm', tqdm.__version__)
PY
```

### 5. Config-Driven Training Commands

Use config files for all formal runs.

Label-v2:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v2.yaml
```

Label-v3:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v3.yaml
```

Label-v4:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4.yaml
```

Label-v5:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5.yaml
```

Temporary seed override example:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v5_seed1
```

### 6. Recommended First Server Run

Start with Label-v5 but cap samples for a quick smoke test:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_label_v5
```

Then run full Label-v5 training:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5.yaml
```

服务器上可以不改 YAML，直接用 CLI 临时覆盖 DataLoader worker 参数：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --num-workers 8 \
  --prefetch-factor 2
```

训练曲线会写到：

```text
<out-dir>/reports/training_curves.png
```

如果服务器环境没有安装 `matplotlib`，DRAGEN 会自动写出 HTML fallback：

```text
<out-dir>/reports/training_curves.html
```

If Label-v5 is too small or unstable, run Label-v4 as the fallback formal label:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4.yaml
```

### 7. Output Files to Bring Back

After each run, copy back:

```text
work/artifacts/<run_name>/reports/
work/artifacts/<run_name>/predictions/
work/artifacts/<run_name>/checkpoints/best.pt
```

If checkpoint files are too large, copy at least:

```text
reports/metrics.json
reports/loss_breakdown.json
reports/epoch_metrics.csv
reports/training_curves.png 或 reports/training_curves.html
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
predictions/event_predictions.csv
predictions/sampled_global_neighbors.csv
```


### 文本语义增强额外文件

如果服务器只训练 `DRAGEN-Full-StatText`，不需要 RoBERTa 文本 embedding。若要训练 `DRAGEN-Full-RoBERTaText`，还需要同步：

```text
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64/
work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5_roberta_text/
```

RoBERTa 原始编码只应作为离线预处理缓存，训练脚本和 pack 构建脚本不会触发 RoBERTa。完整命令见 `docs/text_embeddings.md`。

### 8. Rebuilding Labels or Packs on Server

Only do this when necessary. If rebuilding packs from existing labels/features, transfer:

```text
work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv
work/runs/run_0002/labels_v2_stratified_score/
work/runs/run_0002/labels_v3_lf_vote/
work/runs/run_0002/labels_v4_coordination_network/
work/runs/run_0002/labels_v5_ensemble_consensus/
```

Example rebuild for Label-v5:

```bash
python scripts/13_build_packs.py \
  --run-id run_0002 \
  --labels work/runs/run_0002/labels_v5_ensemble_consensus/weak_event_labels.csv \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5
```

If the server must rebuild the global candidate table, transfer `graph/follow_edges.tsv` and run:

```bash
python scripts/10_build_global_candidate_edges.py --run-id run_0002 --follow-edges graph/follow_edges.tsv
```

This scans a large graph and should not be part of normal training runs.
