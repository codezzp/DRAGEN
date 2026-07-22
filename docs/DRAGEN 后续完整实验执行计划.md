# DRAGEN 后续完整实验执行计划

## 先统一一个关键口径

你已经冻结的主实验配置是：

```text
Event Loss：BCE
Epoch：固定 10
Checkpoint：Final Epoch / last.pt
Probability Calibrator：None
Threshold：Valid-best MCC
Seeds：0、1、2
```

因此，**论文主表不使用 Platt Scaling**。Platt 只放在单独的概率校准实验中。Baseline 和消融也统一使用：

```text
原始概率
→ Valid 上选择 Best MCC 阈值
→ 固定阈值应用到 Test
```

不能再用 Platt 处理主表，否则会与冻结配置不一致。

你的主 Baseline 固定为：

```text
IOHunter-adapt
LEN-GNN-adapt
EDCOC-adapt
X-CoIA-adapt
UWD-FSN-adapt
TDCB-adapt
```

---

# 一、整体执行顺序

```text
阶段 1：最终 DRAGEN 三种子
阶段 2：统一 Baseline 框架
阶段 3：六个 Baseline 的 Seed1 预实验
阶段 4：冻结 Baseline 配置
阶段 5：六个 Baseline 三种子正式实验
阶段 6：核心消融 Seed1
阶段 7：关键消融补三种子
阶段 8：模块替换、鲁棒性和效率实验
阶段 9：统一汇总论文表格
```

只有一张 GPU 时，训练任务不要同时运行。

---

# 二、阶段0：服务器准备

## 0.1 更新代码

```bash
cd /usr/src/code/DRAGEN

git fetch codezzp
git switch experiment/run-0002-calibration-diagnostics
git pull --ff-only

export PYTHONPATH=src
```

## 0.2 创建统一目录

```bash
mkdir -p logs

mkdir -p work/artifacts/main
mkdir -p work/artifacts/baselines
mkdir -p work/artifacts/ablations
mkdir -p work/artifacts/replacements
mkdir -p work/artifacts/robustness
mkdir -p work/artifacts/_analysis
```

## 0.3 检查当前代码

```bash
echo "===== BRANCH ====="
git branch --show-current

echo "===== COMMIT ====="
git log --oneline -3

echo "===== STATUS ====="
git status --short

echo "===== GPU ====="
nvidia-smi
```

## 0.4 检查冻结文件

```bash
cat docs/run_0002_frozen_main_config.md
```

## 0.5 检查主 Pack

```bash
PACK=packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool

ls -lh "$PACK"
```

应至少有：

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

---

# 三、阶段1：最终 DRAGEN 三种子

## 1.1 创建三种子运行脚本

```bash
cat > scripts/run_final_dragen_3seeds.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /usr/src/code/DRAGEN
export PYTHONPATH=src

mkdir -p logs
mkdir -p work/artifacts/main

for SEED in 0 1 2
do
    RUN="run_0002_final_dragen_seed${SEED}"
    OUT="work/artifacts/main/${RUN}"
    LOG="logs/${RUN}.log"

    echo "========================================"
    echo "START ${RUN}"
    echo "OUTPUT: ${OUT}"
    echo "LOG: ${LOG}"
    echo "========================================"

    python scripts/16_train_dragen_full.py \
      --config configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml \
      --seed "${SEED}" \
      --bucket-by-nodes \
      --bucket-size-multiplier 50 \
      --max-nodes-per-batch 12000 \
      --save-every-epoch \
      --no-plot-every-epoch \
      --no-tensorboard \
      --out-dir "${OUT}" \
      > "${LOG}" 2>&1

    echo "FINISHED ${RUN}"
done
BASH

chmod +x scripts/run_final_dragen_3seeds.sh
```

这与冻结配置中的正式训练命令一致。

## 1.2 后台运行

```bash
nohup bash scripts/run_final_dragen_3seeds.sh \
  > logs/run_final_dragen_3seeds_all.log 2>&1 &

echo $! > logs/run_final_dragen_3seeds.pid
```

## 1.3 查看运行状态

查看总进程：

```bash
ps -fp "$(cat logs/run_final_dragen_3seeds.pid)"
```

查看当前训练进程：

```bash
ps -ef | grep 16_train_dragen_full.py | grep -v grep
```

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

查看总日志：

```bash
tail -f logs/run_final_dragen_3seeds_all.log
```

查看某一个 seed：

```bash
tail -f logs/run_0002_final_dragen_seed0.log
```

退出日志查看：

```text
Ctrl + C
```

---

## 1.4 检查每个主模型结果

```bash
for SEED in 0 1 2
do
    RUN=work/artifacts/main/run_0002_final_dragen_seed${SEED}

    echo
    echo "================ SEED ${SEED} ================"

    test -f "$RUN/reports/metrics.json" \
      && echo "[OK] metrics.json" \
      || echo "[MISSING] metrics.json"

    test -f "$RUN/reports/epoch_metrics.csv" \
      && echo "[OK] epoch_metrics.csv" \
      || echo "[MISSING] epoch_metrics.csv"

    test -f "$RUN/predictions/valid_event_predictions.csv" \
      && echo "[OK] valid predictions" \
      || echo "[MISSING] valid predictions"

    test -f "$RUN/predictions/test_event_predictions.csv" \
      && echo "[OK] test predictions" \
      || echo "[MISSING] test predictions"

    test -f "$RUN/checkpoints/last.pt" \
      && echo "[OK] last.pt" \
      || echo "[MISSING] last.pt"
done
```

## 1.5 查看最后几个 Epoch

```bash
for SEED in 0 1 2
do
    echo
    echo "================ SEED ${SEED} ================"
    tail -n 6 \
      work/artifacts/main/run_0002_final_dragen_seed${SEED}/reports/epoch_metrics.csv
done
```

## 1.6 查看原始指标

```bash
for SEED in 0 1 2
do
    echo
    echo "================ SEED ${SEED} ================"
    cat \
      work/artifacts/main/run_0002_final_dragen_seed${SEED}/reports/metrics.json
done
```

注意：这里的分类指标可能还是默认阈值 0.5，不能直接作为论文最终 F1/MCC。

---

# 四、阶段1.5：重新选择 Valid-best MCC 阈值

新复跑后，必须根据新生成的 Valid 预测重新选阈值，不能直接照搬旧阈值。冻结文件也明确要求新运行重新计算阈值。

## 2.1 先检查阈值脚本读取哪些目录

```bash
python scripts/21_calibrate_thresholds.py --help
```

然后检查脚本内是否还写着旧目录：

```bash
grep -nE \
"_artifacts|work/artifacts/main|seed0|seed1|seed2|RUNS|run_dirs" \
scripts/21_calibrate_thresholds.py
```

### 正确目标目录应为

```text
work/artifacts/main/run_0002_final_dragen_seed0
work/artifacts/main/run_0002_final_dragen_seed1
work/artifacts/main/run_0002_final_dragen_seed2
```

如果脚本仍读取旧的 `_artifacts`，需要先把脚本中的运行目录改成上面三个正式目录。

## 2.2 运行阈值分析

```bash
python scripts/21_calibrate_thresholds.py
```

## 2.3 查看阈值结果

```bash
echo "===== THRESHOLDS ====="
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_by_seed.csv

echo
echo "===== MEAN STD ====="
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_summary_mean_std.csv

echo
echo "===== REPORT ====="
cat work/artifacts/_analysis/run_0002_threshold_calibration/threshold_calibration_summary.md
```

## 2.4 正式主模型重点看

```text
valid_best_mcc
accuracy
precision
recall
f1
auc
ap
mcc
precision_at_100
precision_at_500
```

主模型最终需要记录：

|Seed|Threshold|Acc|Precision|Recall|F1|AUC|AP|MCC|
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|0|||||||||
|1|||||||||
|2|||||||||
|Mean ± Std|||||||||

---

# 五、阶段2：Baseline 代码准备

## 3.1 先检查当前仓库是否已有 Baseline 代码

```bash
find scripts src/dragen configs/train \
  -type f \
  | grep -Ei \
  "baseline|iohunter|len_gnn|edcoc|x_coia|uwd|tdcb" \
  | sort
```

再检查统一入口：

```bash
test -f scripts/24_train_baseline.py \
  && echo "[OK] baseline trainer exists" \
  || echo "[MISSING] scripts/24_train_baseline.py"
```

检查六个配置：

```bash
for MODEL in \
  iohunter_adapt \
  len_gnn_adapt \
  edcoc_adapt \
  x_coia_adapt \
  uwd_fsn_adapt \
  tdcb_adapt
do
    CFG="configs/train/baselines/${MODEL}.yaml"

    test -f "$CFG" \
      && echo "[OK] $CFG" \
      || echo "[MISSING] $CFG"
done
```

如果这些文件不存在，说明 Baseline 目前只是“方法设计完成”，还没有形成可运行代码。这时先完成统一入口，不要直接开始正式实验。

---

## 3.2 Baseline 统一接口要求

建议固定为：

```text
scripts/24_train_baseline.py
```

统一调用方式：

```bash
python scripts/24_train_baseline.py \
  --config configs/train/baselines/<MODEL>.yaml \
  --seed <SEED> \
  --out-dir work/artifacts/baselines/<MODEL>_seed<SEED>
```

每个 Baseline 必须输出：

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
reports/metrics.json
reports/epoch_metrics.csv

predictions/valid_event_predictions.csv
predictions/test_event_predictions.csv

checkpoints/last.pt
```

预测 CSV 至少统一为：

```text
event_id
label
prob
```

其中：

- `label`：真实事件标签；
    
- `prob`：原始事件级概率；
    
- 不提前进行 Platt 校准；
    
- 不在 Test 上调阈值。
    

---

# 六、阶段3：先跑 LEN-GNN 闭环

LEN-GNN 是最简单的静态事件图分类 Baseline，先用它验证整个 Baseline 流程。

## 4.1 Smoke Test

下面命令要求 Baseline 入口支持小样本参数。若暂未支持，应先补上这些参数。

```bash
python scripts/24_train_baseline.py \
  --config configs/train/baselines/len_gnn_adapt.yaml \
  --seed 1 \
  --epochs 1 \
  --max-train-samples 64 \
  --max-valid-samples 32 \
  --max-test-samples 32 \
  --out-dir work/artifacts/baselines/_smoke_len_gnn_seed1
```

## 4.2 查看 Smoke Test

```bash
find work/artifacts/baselines/_smoke_len_gnn_seed1 \
  -maxdepth 3 -type f | sort
```

```bash
cat \
  work/artifacts/baselines/_smoke_len_gnn_seed1/reports/metrics.json
```

```bash
head \
  work/artifacts/baselines/_smoke_len_gnn_seed1/predictions/valid_event_predictions.csv
```

必须确认：

```text
能正常训练
能输出 Valid 概率
能输出 Test 概率
预测数量与数据划分一致
概率不全相同
概率没有 NaN
```

检查预测数量：

```bash
wc -l \
  work/artifacts/baselines/_smoke_len_gnn_seed1/predictions/*.csv
```

检查概率：

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path(
    "work/artifacts/baselines/_smoke_len_gnn_seed1/predictions"
)

for name in [
    "valid_event_predictions.csv",
    "test_event_predictions.csv",
]:
    p = root / name
    df = pd.read_csv(p)

    print(f"\n===== {name} =====")
    print(df.head())
    print("rows:", len(df))
    print("nan:", df.isna().sum().to_dict())

    if "prob" in df.columns:
        print(df["prob"].describe())
PY
```

---

## 4.3 LEN-GNN Seed1 完整运行

```bash
mkdir -p logs/baselines

nohup python scripts/24_train_baseline.py \
  --config configs/train/baselines/len_gnn_adapt.yaml \
  --seed 1 \
  --out-dir work/artifacts/baselines/len_gnn_adapt_seed1 \
  > logs/baselines/len_gnn_adapt_seed1.log 2>&1 &

echo $! > logs/baselines/len_gnn_adapt_seed1.pid
```

查看：

```bash
tail -f logs/baselines/len_gnn_adapt_seed1.log
```

---

# 七、阶段4：六个 Baseline Seed1 预实验

## 5.1 建议顺序

```text
LEN-GNN
UWD-FSN
TDCB
EDCOC
IOHunter
X-CoIA
```

这个顺序从实现较简单逐步进入多视角和复杂图模型。

## 5.2 创建 Seed1 执行脚本

```bash
cat > scripts/run_baselines_seed1.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /usr/src/code/DRAGEN
export PYTHONPATH=src

mkdir -p logs/baselines
mkdir -p work/artifacts/baselines

MODELS=(
  len_gnn_adapt
  uwd_fsn_adapt
  tdcb_adapt
  edcoc_adapt
  iohunter_adapt
  x_coia_adapt
)

for MODEL in "${MODELS[@]}"
do
    CONFIG="configs/train/baselines/${MODEL}.yaml"
    OUT="work/artifacts/baselines/${MODEL}_seed1"
    LOG="logs/baselines/${MODEL}_seed1.log"

    if [[ ! -f "$CONFIG" ]]; then
        echo "[MISSING CONFIG] $CONFIG"
        exit 1
    fi

    echo "========================================"
    echo "START ${MODEL} seed1"
    echo "========================================"

    python scripts/24_train_baseline.py \
      --config "$CONFIG" \
      --seed 1 \
      --out-dir "$OUT" \
      > "$LOG" 2>&1

    echo "FINISHED ${MODEL} seed1"
done
BASH

chmod +x scripts/run_baselines_seed1.sh
```

后台运行：

```bash
nohup bash scripts/run_baselines_seed1.sh \
  > logs/baselines/all_seed1.log 2>&1 &

echo $! > logs/baselines/all_seed1.pid
```

查看总进度：

```bash
tail -f logs/baselines/all_seed1.log
```

查看正在训练的方法：

```bash
ps -ef | grep 24_train_baseline.py | grep -v grep
```

---

## 5.3 Seed1 结果完整性检查

```bash
for MODEL in \
  len_gnn_adapt \
  uwd_fsn_adapt \
  tdcb_adapt \
  edcoc_adapt \
  iohunter_adapt \
  x_coia_adapt
do
    RUN="work/artifacts/baselines/${MODEL}_seed1"

    echo
    echo "================ ${MODEL} ================"

    for FILE in \
      reports/metrics.json \
      reports/epoch_metrics.csv \
      predictions/valid_event_predictions.csv \
      predictions/test_event_predictions.csv \
      checkpoints/last.pt
    do
        test -f "$RUN/$FILE" \
          && echo "[OK] $FILE" \
          || echo "[MISSING] $FILE"
    done
done
```

---

# 八、Baseline 阈值选择

Baseline 必须和 DRAGEN 使用同一规则：

```text
原始概率
Valid-best MCC
Final Epoch
Test 不参与选择
```

建议将当前 `scripts/21_calibrate_thresholds.py` 改造成支持：

```bash
--artifact-root
--output-dir
```

目标命令应为：

```bash
python scripts/21_calibrate_thresholds.py \
  --artifact-root work/artifacts/baselines \
  --output-dir work/artifacts/_analysis/baseline_threshold_calibration
```

如果当前脚本没有这些参数，先执行：

```bash
python scripts/21_calibrate_thresholds.py --help
```

然后补充参数支持。不要为每个 Baseline 单独复制一套阈值代码。

输出应至少包括：

```text
threshold_by_run.csv
test_metrics_by_run.csv
summary_mean_std.csv
summary.md
```

查看：

```bash
cat \
  work/artifacts/_analysis/baseline_threshold_calibration/threshold_by_run.csv
```

```bash
cat \
  work/artifacts/_analysis/baseline_threshold_calibration/test_metrics_by_run.csv
```

Seed1 预实验重点只使用 **Valid 指标** 决定配置：

```text
Valid AUC
Valid AP
Valid F1
Valid MCC
Valid Precision
Valid Recall
```

不能根据 Test 结果修改模型。

---

# 九、阶段5：冻结每个 Baseline

每个 Baseline 完成 Seed1 后，固定：

```text
输入字段
图构建方式
节点特征
边定义
事件池化方式
隐藏维度
学习率
Dropout
Epoch
Checkpoint
Threshold Strategy
Output Naming
```

创建目录：

```bash
mkdir -p docs/baselines
```

建议文件：

```text
docs/baselines/len_gnn_adapt_frozen.md
docs/baselines/uwd_fsn_adapt_frozen.md
docs/baselines/tdcb_adapt_frozen.md
docs/baselines/edcoc_adapt_frozen.md
docs/baselines/iohunter_adapt_frozen.md
docs/baselines/x_coia_adapt_frozen.md
```

检查配置：

```bash
for CFG in configs/train/baselines/*.yaml
do
    echo
    echo "================ $CFG ================"
    grep -nE \
    "pack|label|epoch|lr|dropout|hidden|threshold|checkpoint|seed" \
    "$CFG"
done
```

---

# 十、阶段6：六个 Baseline 三种子正式实验

## 7.1 创建正式运行脚本

```bash
cat > scripts/run_all_baselines_3seeds.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /usr/src/code/DRAGEN
export PYTHONPATH=src

mkdir -p logs/baselines
mkdir -p work/artifacts/baselines

MODELS=(
  len_gnn_adapt
  uwd_fsn_adapt
  tdcb_adapt
  edcoc_adapt
  iohunter_adapt
  x_coia_adapt
)

for MODEL in "${MODELS[@]}"
do
    CONFIG="configs/train/baselines/${MODEL}.yaml"

    for SEED in 0 1 2
    do
        RUN="${MODEL}_seed${SEED}"
        OUT="work/artifacts/baselines/${RUN}"
        LOG="logs/baselines/${RUN}.log"

        echo "========================================"
        echo "START ${RUN}"
        echo "========================================"

        python scripts/24_train_baseline.py \
          --config "$CONFIG" \
          --seed "$SEED" \
          --out-dir "$OUT" \
          > "$LOG" 2>&1

        echo "FINISHED ${RUN}"
    done
done
BASH

chmod +x scripts/run_all_baselines_3seeds.sh
```

## 7.2 后台运行

```bash
nohup bash scripts/run_all_baselines_3seeds.sh \
  > logs/baselines/all_3seeds.log 2>&1 &

echo $! > logs/baselines/all_3seeds.pid
```

## 7.3 查看运行状态

```bash
tail -f logs/baselines/all_3seeds.log
```

```bash
ps -fp "$(cat logs/baselines/all_3seeds.pid)"
```

```bash
watch -n 2 nvidia-smi
```

---

# 十一、统一查看 Baseline 原始结果

```bash
python - work/artifacts/baselines test <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
split = sys.argv[2]

keys = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "ap",
    "mcc",
    "precision_at_100",
    "precision_at_500",
]

aliases = {
    "precision_at_100": ["precision_at_100", "p_at_100"],
    "precision_at_500": ["precision_at_500", "p_at_500"],
}

rows = []

for path in sorted(root.rglob("reports/metrics.json")):
    run = path.parent.parent.name
    data = json.loads(path.read_text())

    block = data.get(split, {})
    if split == "valid" and not block:
        block = data.get("validation", {})

    if not block:
        continue

    row = {"run": run}

    for key in keys:
        names = aliases.get(key, [key])
        value = None

        for name in names:
            if name in block:
                value = block[name]
                break

        row[key] = value

    rows.append(row)

print("run | " + " | ".join(keys))
print("-" * 150)

for row in rows:
    values = []

    for key in keys:
        value = row[key]
        values.append("-" if value is None else f"{value:.4f}")

    print(row["run"] + " | " + " | ".join(values))

print("\n===== MEAN ± STD =====")

groups = {}

for row in rows:
    group = re.sub(r"_seed\d+$", "", row["run"])
    groups.setdefault(group, []).append(row)

for group, items in sorted(groups.items()):
    print(f"\n{group}")

    for key in keys:
        values = [
            x[key] for x in items
            if isinstance(x[key], (int, float))
        ]

        if not values:
            continue

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        print(f"  {key:20s}: {mean:.4f} ± {std:.4f}")
PY
```

但是这里仍可能是默认阈值结果。论文主表必须使用阈值分析脚本产生的 `Valid-best MCC` 结果。

---

# 十二、阶段7：核心消融 Seed1

## 8.1 先检查消融入口

```bash
python scripts/17_run_ablation.py --help
```

查找已有配置：

```bash
find configs/train \
  -type f \
  | grep -Ei "ablation|no_global|no_memory|no_role|no_gate|no_uncertainty|no_adaptive" \
  | sort
```

## 8.2 第一批消融

```text
w/o Global Prior
w/o Adaptive Sampling
w/o Memory
w/o Role
w/o Gate
w/o Uncertainty
```

## 8.3 创建 Seed1 消融脚本

下面配置名需要与仓库真实文件一致。运行前先用上一条 `find` 命令确认。

```bash
cat > scripts/run_core_ablations_seed1.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /usr/src/code/DRAGEN
export PYTHONPATH=src

PACK=packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool

mkdir -p logs/ablations
mkdir -p work/artifacts/ablations

NAMES=(
  no_global_prior
  no_adaptive_sampling
  no_memory
  no_role
  no_gate
  no_uncertainty
)

CONFIGS=(
  configs/train/ablation_no_global_prior.yaml
  configs/train/ablation_no_adaptive_sampling.yaml
  configs/train/ablation_no_memory.yaml
  configs/train/ablation_no_role.yaml
  configs/train/ablation_no_gate.yaml
  configs/train/ablation_no_uncertainty.yaml
)

for INDEX in "${!NAMES[@]}"
do
    NAME="${NAMES[$INDEX]}"
    CONFIG="${CONFIGS[$INDEX]}"
    OUT="work/artifacts/ablations/${NAME}_seed1"
    LOG="logs/ablations/${NAME}_seed1.log"

    if [[ ! -f "$CONFIG" ]]; then
        echo "[MISSING CONFIG] $CONFIG"
        exit 1
    fi

    python scripts/17_run_ablation.py \
      --config "$CONFIG" \
      --pack-dir "$PACK" \
      --seed 1 \
      --bucket-by-nodes \
      --bucket-size-multiplier 50 \
      --max-nodes-per-batch 12000 \
      --save-every-epoch \
      --no-plot-every-epoch \
      --no-tensorboard \
      --out-dir "$OUT" \
      > "$LOG" 2>&1
done
BASH

chmod +x scripts/run_core_ablations_seed1.sh
```

## 8.4 后台运行

```bash
nohup bash scripts/run_core_ablations_seed1.sh \
  > logs/ablations/all_seed1.log 2>&1 &

echo $! > logs/ablations/all_seed1.pid
```

查看：

```bash
tail -f logs/ablations/all_seed1.log
```

---

# 十三、统一查看消融结果

```bash
python - work/artifacts/ablations test <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
split = sys.argv[2]

keys = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "ap",
    "mcc",
]

print("run | " + " | ".join(keys))
print("-" * 120)

for path in sorted(root.rglob("reports/metrics.json")):
    run = path.parent.parent.name
    data = json.loads(path.read_text())

    block = data.get(split, {})
    if split == "valid" and not block:
        block = data.get("validation", {})

    if not block:
        continue

    values = []

    for key in keys:
        value = block.get(key)
        values.append("-" if value is None else f"{value:.4f}")

    print(run + " | " + " | ".join(values))
PY
```

同样，正式消融表需要使用每个消融自己的：

```text
Valid-best MCC 阈值
```

不能直接套用完整 DRAGEN 的阈值。

---

# 十四、哪些消融补三种子

Seed1 全部完成后，优先补：

```text
w/o Global Prior
w/o Memory
w/o Role
w/o Gate
```

因为它们直接对应论文核心设计。

## 10.1 创建关键消融三种子脚本

```bash
cat > scripts/run_key_ablations_3seeds.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd /usr/src/code/DRAGEN
export PYTHONPATH=src

PACK=packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool

mkdir -p logs/ablations
mkdir -p work/artifacts/ablations

NAMES=(
  no_global_prior
  no_memory
  no_role
  no_gate
)

CONFIGS=(
  configs/train/ablation_no_global_prior.yaml
  configs/train/ablation_no_memory.yaml
  configs/train/ablation_no_role.yaml
  configs/train/ablation_no_gate.yaml
)

for INDEX in "${!NAMES[@]}"
do
    NAME="${NAMES[$INDEX]}"
    CONFIG="${CONFIGS[$INDEX]}"

    for SEED in 0 1 2
    do
        OUT="work/artifacts/ablations/${NAME}_seed${SEED}"
        LOG="logs/ablations/${NAME}_seed${SEED}.log"

        if [[ -f "$OUT/reports/metrics.json" ]]; then
            echo "[SKIP EXISTING] ${OUT}"
            continue
        fi

        python scripts/17_run_ablation.py \
          --config "$CONFIG" \
          --pack-dir "$PACK" \
          --seed "$SEED" \
          --bucket-by-nodes \
          --bucket-size-multiplier 50 \
          --max-nodes-per-batch 12000 \
          --save-every-epoch \
          --no-plot-every-epoch \
          --no-tensorboard \
          --out-dir "$OUT" \
          > "$LOG" 2>&1
    done
done
BASH

chmod +x scripts/run_key_ablations_3seeds.sh
```

运行：

```bash
nohup bash scripts/run_key_ablations_3seeds.sh \
  > logs/ablations/key_3seeds.log 2>&1 &

echo $! > logs/ablations/key_3seeds.pid
```

---

# 十五、最终主实验表

Baseline 主实验最终表：

|Model|Acc|Precision|Recall|F1|AUC|AP|MCC|P@100|P@500|
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|IOHunter-adapt||||||||||
|LEN-GNN-adapt||||||||||
|EDCOC-adapt||||||||||
|X-CoIA-adapt||||||||||
|UWD-FSN-adapt||||||||||
|TDCB-adapt||||||||||
|**DRAGEN**||||||||||

已有方法分别覆盖用户驱动识别、协调社区、静态图分类和时序协调分析，而 DRAGEN 进一步输出动态角色和事件级结果。

消融表：

|Model|F1|AUC|AP|MCC|ΔF1|ΔAUC|ΔAP|ΔMCC|
|---|--:|--:|--:|--:|--:|--:|--:|--:|
|w/o Global Prior|||||||||
|w/o Adaptive Sampling|||||||||
|w/o Memory|||||||||
|w/o Role|||||||||
|w/o Gate|||||||||
|w/o Uncertainty|||||||||
|DRAGEN|||||—|—|—|—|

---

# 十六、日常查看命令汇总

## 查看正在运行的所有实验

```bash
ps -ef \
  | grep -E \
  "16_train_dragen_full|17_run_ablation|24_train_baseline" \
  | grep -v grep
```

## 查看 GPU

```bash
watch -n 2 nvidia-smi
```

## 查找所有结果

```bash
find work/artifacts \
  -type f \
  \( \
    -name "metrics.json" \
    -o -name "epoch_metrics.csv" \
    -o -name "loss_breakdown.json" \
  \) \
  | sort
```

## 查看异常日志

```bash
grep -R -nE \
"Traceback|CUDA out of memory|RuntimeError|Error|NaN" \
logs
```

## 查看所有已完成运行

```bash
find work/artifacts \
  -path "*/reports/metrics.json" \
  -printf '%h\n' \
  | sort
```

---

# 十七、你现在立即执行的内容

当前只需要先做下面五件事：

```text
1. 运行最终 DRAGEN seed0/1/2
2. 检查主模型三个正式目录
3. 重新计算 Valid-best MCC 阈值
4. 检查 Baseline 统一入口和六个配置是否存在
5. 先跑 LEN-GNN Smoke Test
```

然后：

```text
LEN-GNN 闭环成功
→ 六个 Baseline Seed1
→ 冻结 Baseline
→ Baseline 三种子
→ 核心消融 Seed1
→ 关键消融三种子
```

最重要的是：**先统一训练输出和阈值评估流程，再批量运行。**否则后面即使跑完几十个实验，结果也可能因为阈值、目录或输出格式不一致而无法公平比较。