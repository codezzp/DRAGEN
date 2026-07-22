# DRAGEN 服务器全部实验命令与统一查看手册

> 当前实验线：`run_0002`  
> 当前分支：`experiment/run-0002-calibration-diagnostics`  
> 当前主 Pack：`Feature-v2 + RoBERTa Text + Label-v2 + Key-user Pool`  
> 原则：只用 Valid 选择阈值、Loss 和 checkpoint；Test 只用于冻结配置后的最终报告。

---

# 0. 每次登录服务器先执行

```bash
cd /usr/src/code/DRAGEN

export PYTHONPATH=src
export BRANCH=experiment/run-0002-calibration-diagnostics
export PACK=packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool

mkdir -p logs
```

---

# 1. 更新代码

## 1.1 服务器已有该分支

```bash
git fetch codezzp
git switch "$BRANCH"
git pull --ff-only codezzp "$BRANCH"
```

## 1.2 服务器第一次切换到该分支

```bash
git fetch codezzp
git switch -c "$BRANCH" --track "codezzp/$BRANCH"
```

## 1.3 统一检查

```bash
echo "===== BRANCH ====="
git branch --show-current

echo "===== COMMITS ====="
git log --oneline -3

echo "===== STATUS ====="
git status --short
```

---

# 2. 检查环境与 Pack

## 2.1 GPU

```bash
nvidia-smi
```

## 2.2 PyTorch

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 2.3 Pack 文件

```bash
ls -lh "$PACK"
```

应至少包含：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

## 2.4 Pack 字段

```bash
python -c "import sys; sys.path.insert(0,'src'); from dragen.data.pack_reader import PickleStreamDataset, collate_fn; p='$PACK/train.pt'; ds=PickleStreamDataset(p,max_samples=2,split='train-smoke'); b=collate_fn([ds[0],ds[1]]); [print(k, tuple(b[k].shape), b[k].dtype) for k in ['window_x','node_x','node_text_x','window_text_x','key_user_idx','key_user_weight','key_user_hop','key_user_mask']]"
```

当前应接近：

```text
window_x        = (B, 6, 24)
node_x          = (B, 6, N, 47)
node_text_x     = (B, 6, N, 64)
window_text_x   = (B, 6, 64)
key_user_idx    = (B, 6, 32)
key_user_weight = (B, 6, 32)
key_user_hop    = (B, 6, 32)
key_user_mask   = (B, 6, 32)
```

---

# 3. Smoke Test

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

查看：

```bash
cat work/artifacts/_smoke_v2_key_user_pool_e2e/reports/metrics.json
cat work/artifacts/_smoke_v2_key_user_pool_e2e/reports/loss_breakdown.json
```

---

# 4. 阈值校准

## 4.1 运行

```bash
python scripts/21_calibrate_thresholds.py
```

## 4.2 统一查看

```bash
echo "===== THRESHOLDS ====="
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_by_seed.csv

echo "===== SUMMARY ====="
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_summary_mean_std.csv

echo "===== REPORT ====="
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_calibration_summary.md
```

当前已有结论：

```text
Default 0.5 F1 ≈ 0.5896
Valid-best F1 F1 ≈ 0.6999
Valid-best MCC F1 ≈ 0.6998
```

后续优先使用 Valid-best MCC。

---

# 5. Epoch 选择分析

## 5.1 运行

```bash
python scripts/22_summarize_epoch_selection.py
```

## 5.2 统一查看

```bash
echo "===== EPOCH BY SEED ====="
cat work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_by_seed.csv

echo "===== EPOCH SUMMARY ====="
cat work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_summary_mean_std.csv

echo "===== EPOCH REPORT ====="
cat work/artifacts/_analysis/run_0002_epoch_selection/epoch_selection_summary.md
```

旧实验未保存完整早期 checkpoint 时，只能做 Valid 曲线分析，不能声称已用最佳 epoch 重新评估 Test。

---

# 6. 概率校准

## 6.1 运行

```bash
python scripts/23_calibrate_probabilities.py
```

保存逐样本结果：

```bash
python scripts/23_calibrate_probabilities.py --write-predictions
```

## 6.2 统一查看

```bash
find work/artifacts/_analysis/run_0002_probability_calibration -maxdepth 1 -type f | sort
```

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("work/artifacts/_analysis/run_0002_probability_calibration")

for p in sorted(root.glob("*.csv")):
    print(f"\n===== {p.name} =====")
    try:
        print(pd.read_csv(p).to_string(index=False))
    except Exception as e:
        print("读取失败:", e)

for p in sorted(root.glob("*.md")):
    print(f"\n===== {p.name} =====")
    print(p.read_text())
PY
```

主要看：Valid NLL、Brier Score、ECE、F1 和 MCC。

---

# 7. Loss 生效诊断
```bash
python - <<'PY'
import json
from pathlib import Path

runs = [
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0"),
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1"),
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2"),
]

wanted = [
    "loss_event","weighted_loss_event","loss_contribution_event",
    "loss_jump","weighted_loss_jump","loss_contribution_jump",
    "loss_struct","weighted_loss_struct","loss_contribution_struct",
    "loss_align","weighted_loss_align","loss_contribution_align",
    "loss_sampler","weighted_loss_sampler","loss_contribution_sampler",
    "loss_sampler_edge","weighted_loss_sampler_edge",
    "loss_sampler_hub","weighted_loss_sampler_hub",
    "loss_sampler_temp","weighted_loss_sampler_temp",
    "loss_uncertainty","weighted_loss_uncertainty","loss_contribution_uncertainty",
    "loss_role","weighted_loss_role","loss_contribution_role",
]

for root in runs:
    p = root / "reports/loss_breakdown.json"
    print(f"\n===== {root.name} =====")
    if not p.exists():
        print("missing:", p)
        continue
    data = json.loads(p.read_text())
    for k in wanted:
        if k in data:
            print(f"{k:36s}: {data[k]}")
PY
```

```bash
python - <<'PY'
import json
from pathlib import Path

runs = [
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0"),
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1"),
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2"),
]

for root in runs:
    path = root / "reports/loss_breakdown.json"

    print(f"\n{'=' * 20} {root.name} {'=' * 20}")

    if not path.exists():
        print("文件不存在：", path)
        continue

    data = json.loads(path.read_text())

    history = data.get("history", [])
    if not history:
        print("history 为空")
        continue

    last = history[-1]
    valid_loss = last.get("valid_loss", {})

    print("epoch:", last.get("epoch"))
    print("train_loss:", last.get("train_loss"))

    for key, value in valid_loss.items():
        if isinstance(value, (int, float)):
            print(f"{key:32s}: {value:.10f}")
        else:
            print(f"{key:32s}: {value}")
PY
```

```bash
python - <<'PY'
import json
from pathlib import Path

runs = [
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0"),
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1"),
    Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed2"),
]

keys = [
    "loss_total",
    "loss_event",
    "loss_jump",
    "loss_struct",
    "loss_align",
    "loss_uncertainty",
    "loss_role",
    "loss_sampler_edge",
    "loss_sampler_hub",
    "loss_sampler_temp",
]

for root in runs:
    path = root / "reports/loss_breakdown.json"

    print(f"\n{'=' * 20} {root.name} {'=' * 20}")

    if not path.exists():
        print("文件不存在")
        continue

    data = json.loads(path.read_text())

    print("epoch | " + " | ".join(keys))
    print("-" * 190)

    for record in data.get("history", []):
        losses = record.get("valid_loss", {})

        values = []
        for key in keys:
            value = losses.get(key)
            values.append("-" if value is None else f"{value:.6f}")

        print(f"{record.get('epoch', '-')!s:>5} | " + " | ".join(values))
PY
```
判断：

```text
非零且进入 weighted loss：真实生效
原始值非零但权重为 0：配置关闭
长期为 0：检查实现、输入、mask 或触发条件
loss_role=0 且无角色标签：合理关闭
```

---

# 8. Loss Probe：只跑 Seed1

## 8.1 Weighted BCE soft

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss weighted_bce \
  --pos-weight soft \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_weighted_bce_soft_seed1
```

## 8.2 Weighted BCE auto

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss weighted_bce \
  --pos-weight auto \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_weighted_bce_auto_seed1
```

## 8.3 Focal Loss

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --event-loss focal \
  --focal-alpha 0.75 \
  --focal-gamma 2.0 \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_loss_focal_seed1
```

---

# 9. 统一查看所有 Loss Probe

```bash
python - <<'PY'
import json
from pathlib import Path

runs = {
    "BCE": Path("work/artifacts/_artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed1"),
    "WBCE-soft": Path("work/artifacts/run_0002_loss_weighted_bce_soft_seed1"),
    "WBCE-auto": Path("work/artifacts/run_0002_loss_weighted_bce_auto_seed1"),
    "Focal": Path("work/artifacts/run_0002_loss_focal_seed1"),
}

keys = ["accuracy","precision","recall","f1","auc","ap","mcc"]

for split in ["valid","test"]:
    print(f"\n===== {split.upper()} =====")
    print("loss | " + " | ".join(keys))
    print("-" * 110)

    for name, root in runs.items():
        p = root / "reports/metrics.json"
        if not p.exists():
            print(f"{name} | missing")
            continue

        data = json.loads(p.read_text())
        block = data.get(split, {})
        if not block and split == "valid":
            block = data.get("validation", {})

        vals = []
        for k in keys:
            v = block.get(k)
            vals.append("-" if v is None else f"{v:.4f}")

        print(name + " | " + " | ".join(vals))
PY
```

选择 Loss 时只使用 Valid 指标。
===== VALID =====
loss | accuracy | precision | recall | f1 | auc | ap | mcc
--------------------------------------------------------------------------------------------------------------
BCE | 0.8710 | 0.9273 | 0.4201 | 0.5782 | 0.9166 | 0.7980 | 0.5709
WBCE-soft | 0.8598 | 0.9347 | 0.3589 | 0.5187 | 0.9060 | 0.7737 | 0.5268
WBCE-auto | 0.8536 | 0.8849 | 0.3495 | 0.5011 | 0.9006 | 0.7509 | 0.4982
Focal | 0.8499 | 0.8401 | 0.3542 | 0.4983 | 0.8868 | 0.7245 | 0.4821

===== TEST =====
loss | accuracy | precision | recall | f1 | auc | ap | mcc
--------------------------------------------------------------------------------------------------------------
BCE | 0.8674 | 0.9197 | 0.4071 | 0.5644 | 0.9151 | 0.7836 | 0.5575
WBCE-soft | 0.8599 | 0.9262 | 0.3651 | 0.5238 | 0.9052 | 0.7696 | 0.5280
WBCE-auto | 0.8476 | 0.8644 | 0.3296 | 0.4772 | 0.8946 | 0.7273 | 0.4736
Focal | 0.8496 | 0.8371 | 0.3570 | 0.5006 | 0.8775 | 0.7073 | 0.4825
---

# 10. Learning-rate Probe（仅在必要时）

## 10.1 lr=5e-4

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --lr 0.0005 \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_lr_5e-4_seed1
```

## 10.2 lr=3e-4

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --lr 0.0003 \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/run_0002_lr_3e-4_seed1
```

当前基准为 lr=1e-3。

---

# 11. 冻结主模型

```bash
nano docs/run_0002_frozen_main_config.md
```

至少固定：

```text
Pack
Label
Window
Event Loss
Threshold Strategy
Probability Calibrator
Checkpoint Rule
Seeds
Output Naming
```

---

# 12. 最终主模型三种子

将最终 Loss 参数加入命令。

## Seed0

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

## Seed1

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

## Seed2

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

---

# 13. 核心消融实验

> 现有消融 YAML 历史上可能指向其他 Pack。必须使用 `--pack-dir "$PACK"` 覆盖。

```bash
mkdir -p work/artifacts/ablations
```

## w/o Global Prior

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_global_prior.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/ablations/no_global_prior_seed1
```

## w/o Adaptive Sampling

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_adaptive_sampling.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/ablations/no_adaptive_sampling_seed1
```

## w/o Memory

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_memory.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/ablations/no_memory_seed1
```

## w/o Role

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_role.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/ablations/no_role_seed1
```

## w/o Gate

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_gate.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/ablations/no_gate_seed1
```

## w/o Uncertainty

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_uncertainty.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/ablations/no_uncertainty_seed1
```

## 其他已有消融

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_multiscale.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --out-dir work/artifacts/ablations/no_multiscale_seed1
```

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_tree.yaml \
  --pack-dir "$PACK" \
  --seed 1 \
  --out-dir work/artifacts/ablations/no_tree_seed1
```

---

# 14. 统一查看所有消融结果

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("work/artifacts/ablations")
keys = ["accuracy","precision","recall","f1","auc","ap","mcc"]

print("run | " + " | ".join(keys))
print("-" * 120)

for run in sorted(root.glob("*")):
    p = run / "reports/metrics.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    test = data.get("test", data)
    vals = []
    for k in keys:
        v = test.get(k)
        vals.append("-" if v is None else f"{v:.4f}")
    print(run.name + " | " + " | ".join(vals))
PY
```

正式论文比较前，每个消融必须使用同一套 Valid 阈值选择规则。

---

# 15. Label-v5 严格标签鲁棒性

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5_roberta_text.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/robustness/label_v5_seed1
```

运行前检查：

```bash
grep -E "pack_dir|use_global_prior|use_adaptive|use_memory|use_gate" configs/train/dragen_full_label_v5_roberta_text.yaml
```

如果 Label-v5 配置与冻结主模型结构不同，只能作为补充鲁棒性结果，不能直接归因于标签变化。

---

# 16. 结果表、预测分析和曲线

## 16.1 导出结果表

```bash
python scripts/18_export_result_tables.py --config configs/train/result_tables_run0002.yaml
```

## 16.2 预测分析

```bash
python scripts/19_analyze_predictions.py --config configs/train/<analysis_config>.yaml
```

`<analysis_config>.yaml` 必须替换成仓库中真实存在的配置。

## 16.3 训练曲线

```bash
python scripts/20_plot_training_curves.py --artifact-dir work/artifacts/main/run_0002_final_dragen_seed0
```

---

# 17. Adapted Baselines 当前状态

当前以下入口仍是占位实现，不能产出正式 baseline 结果：

```text
scripts/14_train_cac_stat.py
scripts/15_train_gnn_baselines.py
src/dragen/baselines/cac_stat.py
src/dragen/baselines/campaign_gnn.py
src/dragen/baselines/temporal_gnn.py
```

因此目前不能给出可靠的正式 baseline 运行命令。需要先实现：

```text
TDCB-adapt
UWD-FSN-adapt
IOHunter-adapt
LEN-GNN-adapt
EDCOC-adapt
X-CoIA-adapt
```

实现后建议新增独立 YAML：

```text
configs/train/baseline_tdcb_adapt.yaml
configs/train/baseline_uwd_fsn_adapt.yaml
configs/train/baseline_iohunter_adapt.yaml
configs/train/baseline_len_gnn_adapt.yaml
configs/train/baseline_edcoc_adapt.yaml
configs/train/baseline_x_coia_adapt.yaml
```

在代码实现前，不要运行当前占位入口并把结果写进论文。

---

# 18. 模块替换实验当前状态

目前没有确认存在以下可运行配置：

```text
Memory：Last / Mean / EMA / LSTM / GRU
Fusion：Concat / Fixed / Attention / Gate
Sampling：Random / Degree / Similarity / Adaptive
Text：StatText / CLS / Mean / Window Semantic
```

需要先增加模块开关与 YAML。建议配置名：

```text
configs/train/replacement_memory_last.yaml
configs/train/replacement_memory_mean.yaml
configs/train/replacement_memory_ema.yaml
configs/train/replacement_memory_lstm.yaml

configs/train/replacement_fusion_concat.yaml
configs/train/replacement_fusion_fixed.yaml
configs/train/replacement_fusion_attention.yaml

configs/train/replacement_sampling_random.yaml
configs/train/replacement_sampling_degree.yaml
configs/train/replacement_sampling_similarity.yaml
```

实现后统一使用：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/<replacement_config>.yaml \
  --seed 1
```

当前不要虚构命令或结果。

---

# 19. 统一查看服务器运行状态

## GPU

```bash
watch -n 2 nvidia-smi
```

## 训练进程

```bash
ps -ef | grep -E "16_train_dragen_full|17_run_ablation" | grep -v grep
```

## 日志

```bash
tail -f logs/<experiment>.log
```

## 所有结果文件

```bash
find work/artifacts -maxdepth 4 -type f \
  \( -name "metrics.json" -o -name "epoch_metrics.csv" -o -name "loss_breakdown.json" \) \
  | sort
```

---

# 20. 后台运行通用格式

```bash
nohup bash -lc '
export PYTHONPATH=src
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
  --seed 1 \
  --bucket-by-nodes \
  --bucket-size-multiplier 50 \
  --max-nodes-per-batch 12000 \
  --save-every-epoch \
  --no-plot-every-epoch \
  --no-tensorboard \
  --out-dir work/artifacts/example_run
' > logs/example_run.log 2>&1 &
```

记录 PID：

```bash
echo $! > logs/example_run.pid
```

查看：

```bash
tail -f logs/example_run.log
```

停止：

```bash
kill "$(cat logs/example_run.pid)"
```

---

# 21. 全部执行顺序

```text
01. 更新并确认分支
02. 检查 GPU、环境、Pack
03. Smoke Test
04. 阈值校准
05. Epoch 分析
06. 概率校准
07. Loss 生效诊断
08. Weighted BCE soft
09. Weighted BCE auto
10. Focal Loss
11. 统一比较 Loss
12. 必要时做 Learning-rate Probe
13. 冻结主模型配置
14. 最终主模型 Seed0/1/2
15. 核心消融
16. Label-v5 鲁棒性
17. 导出结果表和预测分析
18. 实现 Adapted Baselines
19. 运行 Baselines
20. 实现模块替换配置
21. 运行 Memory/Fusion/Sampling 替换
22. 汇总第五章结果
```

---

# 22. 当前可运行状态

| 实验 | 当前状态 |
|---|---|
| 阈值校准 | 可直接运行 |
| Epoch 分析 | 可直接运行 |
| 概率校准 | 可直接运行 |
| Loss 生效诊断 | 可直接查看 |
| Weighted BCE | 可直接运行 |
| Focal Loss | 可直接运行 |
| Learning-rate Probe | 可直接运行 |
| 最终三种子 | 可直接运行 |
| 核心消融 | 有配置，运行前检查 Pack |
| Label-v5 | 有配置，需检查结构一致性 |
| 结果导出 | 有入口 |
| Adapted Baselines | 仍是占位，暂不可正式运行 |
| Memory/Fusion/Sampling 替换 | 尚未确认配置，需先实现 |
