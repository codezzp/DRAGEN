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
```

CPU 环境：

```bash
python -m pip install -r requirements.txt
```

GPU 服务器建议按服务器 CUDA 版本安装 PyTorch。例如 CUDA 12.1：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy pytest tqdm
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
work/artifacts/dragen_full_debug/predictions/event_predictions.csv
work/artifacts/dragen_full_debug/predictions/node_window_predictions.csv
work/artifacts/dragen_full_debug/predictions/role_distribution.csv
work/artifacts/dragen_full_debug/predictions/gate_weights.csv
work/artifacts/dragen_full_debug/predictions/uncertainty.csv
work/artifacts/dragen_full_debug/predictions/event_attention.csv
work/artifacts/dragen_full_debug/predictions/sampled_global_neighbors.csv
work/artifacts/dragen_full_debug/checkpoints/best.pt
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
