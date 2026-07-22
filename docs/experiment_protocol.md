<!-- DOC_STATUS -->
> 历史参考：早期实验协议。当前 run_0002 边界见 `docs/experiment_guide.md`。

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

训练和评估必须优先使用配置文件：

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_run0002.yaml
```

临时实验允许用 CLI 覆盖配置：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_full_run0002_seed1
```

参数优先级：

```text
脚本默认值 < YAML 配置 < CLI 覆盖
```

训练和评估时必须记录并随结果落盘：

- data config
- window config
- model config
- train config
- random seed
- checkpoint path
- metrics

每次训练开始会自动写：

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
```

没有这些配置快照的结果，不应该拿来做正式对比。

训练过程中每个 epoch 会写：

```text
reports/epoch_metrics.csv
reports/loss_breakdown.json
checkpoints/last.pt
```

valid 指标刷新历史最优时写：

```text
checkpoints/best.pt
```

断点续训：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --resume work/artifacts/dragen_full_run0002_seed0/checkpoints/last.pt
```

TensorBoard 可选开启，使用 PyTorch `SummaryWriter`，不引入 TensorFlow 训练框架：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --tensorboard
```

查看曲线：

```bash
tensorboard --logdir work/artifacts --host 0.0.0.0 --port 6006
```

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

当前已提供的正式配置：

```text
configs/train/dragen_full_debug.yaml
configs/train/dragen_full_run0002.yaml
configs/train/ablation_no_tree.yaml
configs/train/ablation_no_multiscale.yaml
configs/train/ablation_no_role.yaml
configs/train/ablation_no_memory.yaml
configs/train/ablation_no_global_prior.yaml
configs/train/ablation_no_adaptive_sampling.yaml
configs/train/ablation_no_gate.yaml
configs/train/ablation_no_uncertainty.yaml
configs/train/result_tables_run0002.yaml
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
python scripts/19_analyze_predictions.py --artifact-dir work/artifacts/dragen_full_run0002_seed0
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


## 2026-07-02 Update: Labels, Global Candidates, and Packs

The existing weak labels under `work/runs/run_0002/labels/` are Label-v1 debug labels. They use global `weak_score` quantiles (`top 20%` positive, `bottom 50%` negative, middle ignored). This remains valid for pipeline checks, but formal thesis experiments should move to Label-v2: stratified multi-rule weak labels. The Label-v2 design is documented in `docs/label_design.md` and should write to `work/runs/run_0002/labels_v2_stratified_score/` without overwriting Label-v1.

A real follow-graph candidate pool is now part of the formal pack path. The offline step only builds candidate edges; it does not precompute sampled neighbors:

```powershell
python scripts/10_build_global_candidate_edges.py --run-id run_0002 --follow-edges graph/follow_edges.tsv
```

Output:

```text
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_diagnostics.json
```

Current run_0002 diagnostics:

```text
follow_edges_scanned      = 413,503,687
candidate_edges_written   = 1,399,062
cascades_with_candidates  = 65,041
```

`13_build_packs.py` automatically reads the default candidate table when it exists and writes `global_candidate_edge_index` and `global_candidate_edge_weight` into each sample. Current rebuilt pack diagnostics:

```text
train = 41,750
valid = 9,175
test  = 8,759
samples_with_global_candidates = 41,289
total_global_candidate_edges   = 890,037
global_candidate_alignment_errors = 0
```


## Implemented Multi-Version Label Pipeline

Current scripts:

```text
scripts/12a_export_weak_labels_v1_score_rank.py
scripts/12b_build_weak_labels_v2.py
scripts/12c_build_weak_labels_v3_lf_vote.py
scripts/12d_build_weak_labels_v4_coordination.py
scripts/12e_build_weak_labels_v5_ensemble.py
scripts/12f_compare_weak_labels.py
```

Current output directories:

```text
work/runs/run_0002/labels_v1_score_rank/
work/runs/run_0002/labels_v2_stratified_score/
work/runs/run_0002/labels_v3_lf_vote/
work/runs/run_0002/labels_v4_coordination_network/
work/runs/run_0002/labels_v5_ensemble_consensus/
work/runs/run_0002/label_comparison/label_version_comparison.csv
```

Build commands:

```powershell
python scripts/12a_export_weak_labels_v1_score_rank.py --run-id run_0002
python scripts/12b_build_weak_labels_v2.py --run-id run_0002
python scripts/12c_build_weak_labels_v3_lf_vote.py --run-id run_0002
python scripts/12d_build_weak_labels_v4_coordination.py --run-id run_0002
python scripts/12e_build_weak_labels_v5_ensemble.py --run-id run_0002
python scripts/12f_compare_weak_labels.py --run-id run_0002
```

Current label comparison summary:

```text
v1 score_rank:              pos=17,053 neg=42,631 ignore=25,579 corr_size=0.057
v2 stratified_score:        pos=4,177  neg=15,728 ignore=65,358 corr_size=0.240
v3 lf_vote:                 pos=3,179  neg=2,974  ignore=79,110 corr_size=0.128
v4 coordination_network:    pos=5,848  neg=10,974 ignore=68,441 corr_size=0.203
v5 ensemble_consensus:      pos=1,392  neg=3,911  ignore=79,960 corr_size=0.071
```

Independent packs have been built for Label-v2 through Label-v5:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v2/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v3/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v4/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5/
```

All these packs include `global_candidate_edge_index` and `global_candidate_edge_weight`.


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
