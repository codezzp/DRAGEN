# 实验流程

本文档记录 DRAGEN 实验的流程约定。当前预处理链路已经够用，后续不再扩展树和窗口设计，重点转入特征、弱标签、pack、训练评估和结果表。

## 1. 选择 run

从数据配置中选择一个 run：

```text
configs/data/run_0001.yaml
configs/data/run_0002.yaml
configs/data/run_0003.yaml
```

每个配置对应一个实验数据单元：

```text
work/runs/<run_id>/
```

## 2. 检查 processed 数据

确认该 run 下至少包含：

```text
processed/
processed/events/
processed/mapping/
processed/text/
processed/user/
```

不要修改或移动已有 `processed/` 数据。

## 3. 构建或复用组织化任务表

入口脚本：

```text
scripts/01_build_org_tables.py
```

输出位置：

```text
work/runs/<run_id>/org_task/
```

## 4. 分析转发时间分布

入口脚本：

```text
scripts/02_analyze_time_distribution.py
```

该阶段用于观察不同观测期下的可用级联数量，为窗口配置和 dev cascade list 提供依据。

## 5. 构建窗口

窗口构建已经有正式入口。实现位置：

```text
src/dragen/windowing/window_builder.py
src/dragen/windowing/node_window_builder.py
src/dragen/windowing/edge_window_builder.py
src/dragen/windowing/text_window_builder.py
```

窗口参数来自：

```text
configs/window/
```

当前可运行命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode star
```

输出位置：

```text
work/runs/<run_id>/windows/obs_<obs>_win<window>_step<step>/
```

每次正式构建后至少检查：

- `window_diagnostics.json` 中 `retweet_text_early_violations` 是否为 0。
- `root_text_window_rows` 是否等于 `num_cascades * windows_per_cascade`。
- 四张 CSV 是否均存在。

## 6. 冻结正式输入

当前正式输入固定为三套：

```text
Fixed-5m Star:
  work/runs/run_0002/windows/obs_1800_win300_step300/
Fixed-5m HybridTree:
  work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree/
MultiScale HybridTree:
  work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
```

HybridTree Light 使用轻量多证据方法，不扫描关注图：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --max-observation-seconds 1800 --out-dir work/runs/run_0002/edges/hybrid_tree_light
```

验收项：

- `tree_valid_ratio = 1.0`
- `invalid_time_edges = 0`
- `root_child_ratio` 不接近 1
- `text_sim_lift > 0`

Fixed-5m HybridTree:

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv --out-dir work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree
```

MultiScale HybridTree:

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv --out-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree
```

窗口验收项：

- `num_window_rows = 85263 * 6`
- `retweet_text_early_violations = 0`
- `edge_window_rows > 0`
- MultiScale 的 `edge_window_table.csv` 同时存在 `current` 和 `context`
- MultiScale 的 `context_edges >= current_edges`

后续不要继续做关注图全量扫描、RoBERTa、复杂情绪模型、1h/2h 多观测期或扩展结构消融，除非主实验闭环已经完成。

## 7. 特征构建

正式入口：

```powershell
python scripts/11_build_features.py --run-id run_0002 --tree-edges work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv
```

输出：

```text
work/runs/run_0002/features/obs_1800_win300_step300_star/
work/runs/run_0002/features/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
```

每个目录包含：

```text
window_features.csv
node_window_features.csv
feature_diagnostics.json
```

验收项：

- `window_features` 行数等于对应 `window_table`
- `node_window_features` 行数等于对应 `node_window_table`
- `nan_count = 0`
- `inf_count = 0`

## 8. 弱监督标签

正式入口：

```powershell
python scripts/12_build_weak_labels.py --run-id run_0002 --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree
```

输出：

```text
work/runs/run_0002/labels/weak_event_labels.csv
work/runs/run_0002/labels/label_diagnostics.json
```

标签规则：

- top 20% `weak_score` 为 positive。
- bottom 50% `weak_score` 为 negative。
- middle 30% 为 ignore。
- split 按 `cascade_idx` hash 划分 train/valid/test，不按窗口划分。

## 9. Pack 构建

正式入口：

```powershell
python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels/weak_event_labels.csv --out-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree
```

输出：

```text
work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
  train.pt
  valid.pt
  test.pt
  meta.json
  pack_diagnostics.json
```

当前 `.pt` 为 pickle stream，读取方式见 `meta.json`。验收项：

- `T = 6`
- train/valid/test 都有正负样本
- `node_mask` 非空
- `edge_alignment_errors = 0`

## 10. 训练和评估

训练和评估时必须记录：

- data config
- window config
- model config
- train config
- random seed
- checkpoint path
- metrics

没有记录这些配置的结果，不应该拿来做正式对比。

最小闭环顺序：

```text
CAC-Stat
Campaign-GNN
Temporal-GNN
DRAGEN-Full
w/o Tree
w/o MultiScale
w/o Role
w/o Gate
```

## 11. 结果表

最终结果表写入 `work/artifacts/reports/`：

```text
data_statistics.csv
tree_compare_table.csv
window_compare_table.csv
main_results.csv
risk_retrieval_results.csv
ablation_results.csv
```

`work/` 不提交。需要提交的是 `docs/run_notes.md`、本协议和必要的结果摘要文档。

### 11.1 公平对比指标

主实验、风险预警和消融实验只使用所有模型都能产生的事件级输出：

```text
predictions/event_predictions.csv
```

统一字段：

```text
cascade_idx
split
y_true
y_prob
y_pred
```

主实验表只放事件级公平指标：

```text
model,input_variant,accuracy,balanced_accuracy,precision,recall,specificity,f1,macro_f1,auc,ap,mcc,brier,ece
```

风险预警表只放事件级排序指标：

```text
model,input_variant,precision_at_100,precision_at_500,recall_at_500,precision_at_1pct,recall_at_1pct,precision_at_5pct,recall_at_5pct
```

消融表使用同一套事件级指标，并以 DRAGEN-Full 为基准计算：

```text
delta_auc
delta_ap
delta_f1
delta_mcc
```

### 11.2 DRAGEN 解释性指标

以下指标只用于 DRAGEN-Full 或具有对应解释输出的消融模型，不与普通 baseline 强行对比：

```text
prob_jump_mean
role_transition_rate
shock_weighted_jump
mean_gate_obs
mean_gate_prior
uncertainty_wrong_mean
attention_entropy
top5_attention_mass_mean
```

解释性分析入口：

```bash
python scripts/19_analyze_predictions.py --artifact-dir work/artifacts/dragen_full_run0002
```

输出：

```text
reports/temporal_stability_metrics.json
reports/interpretability_metrics.json
reports/diagnostic_summary.csv
```

角色集合固定为：

```text
producer
amplifier
suppressor
reframer
ordinary
```

禁止在角色输出中使用 `bridge`。

## 12. 记录实验

每次重要实验或结构调整都记录到 [run_notes.md](run_notes.md)。
