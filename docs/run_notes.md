# 实验记录

这个文件用于记录简短实验备注。每次重要实验、参数变更或结构调整都单独写一段。

## 模板

```text
日期：
Run ID：
Data config：
Window config：
Model config：
Train config：
命令：
输出：
备注：
``` 

```text
日期：2026-07-01
Run ID：run_0002
阶段：文档同步配置驱动实验流程
分支：experiment/run-0002-code
更新文档：
  README.md
  docs/server_experiment_guide.md
  docs/results_summary.md
  docs/experiment_protocol.md
  docs/run_notes.md
同步内容：
  --config 配置驱动训练命令
  CLI 覆盖优先级
  resolved_config.yaml / command.txt / git_info.json 保存规则
  epoch_metrics.csv / last.pt / best.pt 保存机制
  TensorBoard 启动与端口转发
  result_tables_run0002.yaml 结果表导出命令
备注：
  本次只更新文档，不修改模型结构、不重建数据。
```

```text
日期：2026-07-01
Run ID：run_0002
阶段：配置驱动实验入口
分支：experiment/run-0002-code
新增模块：
  src/dragen/config.py
新增配置：
  configs/train/dragen_full_run0002.yaml
  configs/train/dragen_full_debug.yaml
  configs/train/ablation_no_tree.yaml
  configs/train/ablation_no_multiscale.yaml
  configs/train/ablation_no_role.yaml
  configs/train/ablation_no_memory.yaml
  configs/train/ablation_no_global_prior.yaml
  configs/train/ablation_no_adaptive_sampling.yaml
  configs/train/ablation_no_gate.yaml
  configs/train/ablation_no_uncertainty.yaml
  configs/train/result_tables_run0002.yaml
支持入口：
  scripts/14_train_cac_stat.py --config
  scripts/15_train_gnn_baselines.py --config
  scripts/16_train_dragen_full.py --config
  scripts/17_run_ablation.py --config
  scripts/18_export_result_tables.py --config
  scripts/19_analyze_predictions.py --config
优先级：
  脚本默认值 < YAML 配置 < CLI 覆盖
训练产物：
  reports/resolved_config.yaml
  reports/command.txt
  reports/git_info.json
备注：
  不重构预处理配置，不重建窗口、特征、标签或 pack。
```

```text
日期：2026-07-01
Run ID：run_0002
阶段：TensorBoard 可视化日志
分支：experiment/run-0002-code
新增参数：
  --tensorboard
  --tb-log-dir
实现：
  使用 torch.utils.tensorboard.SummaryWriter，不改变 PyTorch 训练框架。
  --tensorboard 开启时默认写入 <out-dir>/tb。
  --tb-log-dir 提供时写入指定目录。
记录曲线：
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
备注：
  TensorBoard 只负责可视化日志，不引入 TensorFlow 训练框架。
```

```text
日期：2026-07-01
Run ID：run_0002
阶段：训练保存机制与断点续训
分支：experiment/run-0002-code
新增参数：
  --seed
  --resume
  --save-every-epoch
  --eval-every
保存机制：
  每个 epoch 结束后写 reports/epoch_metrics.csv
  每个 epoch 结束后更新 reports/loss_breakdown.json
  每个 epoch 结束后保存 checkpoints/last.pt
  valid AUC/F1 刷新历史最优时保存 checkpoints/best.pt
  开启 --save-every-epoch 时额外保存 checkpoints/epoch_{epoch}.pt
断点续训：
  python scripts/16_train_dragen_full.py ... --resume work/artifacts/dragen_full_run0002/checkpoints/last.pt
验证：
  python -m py_compile scripts/16_train_dragen_full.py src/dragen/training/trainer.py
  极小样本 1 epoch 生成 epoch_metrics.csv / last.pt / best.pt
  使用 --resume last.pt 成功从第 2 轮继续训练
备注：
  不修改模型结构，不修改 pack，不重建数据。
```

```text
日期：2026-07-01
Run ID：run_0002
阶段：实验闭环输入、特征、弱标签、pack
分支：experiment/run-0002-code
策略：
  不继续扩展结构设计；不扫描关注图；不上传 work/。
  使用 30m 观测期、5m 步长，完成 Fixed-5m Star / Fixed-5m HybridTree / MultiScale HybridTree 三套正式输入。
命令：
  python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --max-observation-seconds 1800 --out-dir work/runs/run_0002/edges/hybrid_tree_light
  python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv --out-dir work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree
  python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv --out-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree
  python scripts/11_build_features.py --run-id run_0002 --tree-edges work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv
  python scripts/12_build_weak_labels.py --run-id run_0002 --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree
  python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels/weak_event_labels.csv --out-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree
输出：
  work/runs/run_0002/edges/hybrid_tree_light/
  work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree/
  work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
  work/runs/run_0002/features/obs_1800_win300_step300_star/
  work/runs/run_0002/features/obs_1800_win300_step300_hybrid_tree/
  work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
  work/runs/run_0002/labels/
  work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
结果：
  HybridTree Light:
    cascades = 85263
    tree_edges = 1392078
    tree_valid_ratio = 1.0
    invalid_time_edges = 0
    avg_depth = 7.50
    max_depth = 32
    root_child_ratio = 0.101
    text_sim_lift = 0.00107
  Fixed-5m HybridTree window:
    window_rows = 511578
    node_window_rows = 5940259
    edge_window_rows = 1392078
    text_window_rows = 6032866
    retweet_text_early_violations = 0
  MultiScale HybridTree window:
    window_rows = 511578
    node_window_rows = 5940259
    edge_window_rows = 4019952
    text_window_rows = 6032866
    current_edges = 1392078
    context_edges = 2627874
    retweet_text_early_violations = 0
  Feature v1:
    三套输入均生成 window_features.csv / node_window_features.csv / feature_diagnostics.json
    每套 window_features = 511578
    每套 node_window_features = 5940259
    nan_count = 0
    inf_count = 0
  Weak labels:
    cascades = 85263
    positive = 17053
    negative = 42631
    ignore = 25579
    split = train 70% / valid 15% / test 15% by cascade_idx hash
  Pack:
    path = work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
    format = pickle_stream stored as train.pt / valid.pt / test.pt
    train = 41750
    valid = 9175
    test = 8759
    total_samples = 59684
    T = 6
    node_mask empty = 0
    edge_alignment_errors = 0
备注：
  当前环境未安装 torch，pack 使用 pickle stream 写入 .pt。读取方式见 meta.json。
  下一步进入 baseline 与 DRAGEN-Full 训练评估，不再扩展预处理结构。
```

```text
日期：2026-07-01
Run ID：run_0002
阶段：评估指标体系扩展
分支：experiment/run-0002-code
新增入口：
  scripts/19_analyze_predictions.py
更新入口：
  scripts/18_export_result_tables.py
口径：
  主实验、风险预警和消融实验只使用 predictions/event_predictions.csv 计算事件级公平指标。
  DRAGEN-Full 的角色、门控、不确定性、注意力和时序稳定性指标只用于解释性分析，不与普通 baseline 强行对比。
输出：
  reports/event_metrics_extended.json
  reports/risk_retrieval_metrics.json
  reports/temporal_stability_metrics.json
  reports/interpretability_metrics.json
  reports/diagnostic_summary.csv
结果表：
  work/artifacts/reports/main_results.csv
  work/artifacts/reports/risk_retrieval_results.csv
  work/artifacts/reports/ablation_results.csv
备注：
  角色集合固定为 producer, amplifier, suppressor, reframer, ordinary。
  输出中禁止 bridge。
```

```text
日期：2026-07-01
Run ID：run_0002
阶段：DRAGEN-Full 论文版模型与 debug 训练
分支：experiment/run-0002-code
新增入口：
  scripts/16_train_dragen_full.py
  scripts/17_run_ablation.py
  scripts/18_export_result_tables.py
核心模块：
  source_evidence_encoder / evidence_reader / local_role_encoder / adaptive_global_sampler
  global_prior_encoder / temporal_memory / manipulation_state / bayesian_gate / event_pooling / dragen_full
角色集合：
  producer, amplifier, suppressor, reframer, ordinary
命令：
  python scripts/16_train_dragen_full.py --pack-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree --out-dir work/artifacts/dragen_full_debug --epochs 1 --batch-size 2 --max-train-samples 200 --max-valid-samples 100 --max-test-samples 100 --hidden-dim 64 --role-num 5 --top-k-global 20 --lambda-jump 0.01 --lambda-struct 0.005 --lambda-align 0.001 --lambda-uncertainty 0.001 --lambda-role 0.0 --device cpu
输出：
  work/artifacts/dragen_full_debug/checkpoints/best.pt
  work/artifacts/dragen_full_debug/reports/metrics.json
  work/artifacts/dragen_full_debug/reports/loss_breakdown.json
  work/artifacts/dragen_full_debug/predictions/event_predictions.csv
  work/artifacts/dragen_full_debug/predictions/node_window_predictions.csv
  work/artifacts/dragen_full_debug/predictions/role_distribution.csv
  work/artifacts/dragen_full_debug/predictions/gate_weights.csv
  work/artifacts/dragen_full_debug/predictions/uncertainty.csv
  work/artifacts/dragen_full_debug/predictions/event_attention.csv
  work/artifacts/dragen_full_debug/predictions/sampled_global_neighbors.csv
结果：
  debug epoch = 1
  train_loss = 0.3926
  valid_auc = 0.9113
  test_auc = 0.9005
  forward 输出包含论文核心变量：event/node logits, source_evidence, local_role_repr, global_prior, history_state, manip_state, role_prob, shock, gate weights, uncertainty, event_attention。
  role_distribution.csv 只使用固定五类角色。
备注：
  已安装 CPU 版 PyTorch 用于本地 debug。
  特征进入模型前在 pack_reader.collate_fn 中做 signed log1p 稳定化，避免原始计数和时间尺度导致训练溢出。
  下一步可以启动正式训练或先实现 baseline 主实验。
```

## 记录

- 已整理工程结构，`scripts/` 只保留入口脚本。
- 已有数据继续保留在 `work/runs/`，不要移动或重命名。
- 已实现窗口划分模块和入口 `scripts/05_build_windows.py`。

```text
日期：2026-06-29
Run ID：run_0002
Window config：configs/window/obs_30m_step5m.yaml
命令：python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml
输出：work/runs/run_0002/windows/obs_1800_win300_step300/
结果：
  cascades = 85263
  windows_per_cascade = 6
  window_rows = 511578
  node_window_rows = 5940259
  edge_window_rows = 1392078
  text_window_rows = 6032866
  root_text_window_rows = 511578
  retweet_text_early_violations = 0
验证：
  python -m py_compile 通过
  直接调用 tests/test_window_builder.py 中的窗口测试函数通过
备注：
  当前环境没有安装 pytest，因此未运行 python -m pytest。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：Tree-v1 调试
命令：python scripts/06_build_inferred_tree.py --run-id run_0002 --method time_only --max-cascades 10 --out-dir work/runs/run_0002/edges/_debug_tree_time
输出：work/runs/run_0002/edges/_debug_tree_time/
结果：
  cascades = 10
  tree_edges = 5685
  follow_parent_ratio = 0.000000
  root_fallback_ratio = 0.000000
备注：
  该调试版本未扫描 7.3GB follow_edges.tsv，只验证 time_only 代理树字段和结构。
```

```text
日期：2026-06-29
Run ID：run_0002
Window config：configs/window/obs_30m_step5m.yaml
Edge mode：inferred_tree
命令：python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/_debug_tree_time/inferred_tree_edge_table.csv --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_win300_step300_tree
输出：work/runs/run_0002/windows/_debug_obs_1800_win300_step300_tree/
结果：
  cascades = 10
  windows = 60
  node_windows = 214
  edge_windows = 43
  text_windows = 224
备注：
  树形边已经可以进入窗口构建链路，后续可生成正式 tree 窗口目录。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：避免线性树
结论：
  time_only 会将大级联构造成接近线性的深链，不适合作为正式树结构。
  branching_time 已设为默认推荐方法。
对比结果：
  time_only 调试样本：
    avg_depth = 2592.58
    max_depth = 5405
  branching_time 调试样本：
    avg_depth = 7.62
    max_depth = 18
备注：
  time_only 保留为结构实验中的线性化对照或反例。
```

```text
日期：2026-06-29
Run ID：run_0002
结构可视化：cascade_idx = 78857
命令：
  python scripts/06_build_inferred_tree.py --run-id run_0002 --cascade-id 78857 --max-observation-seconds 1800 --method branching_time --max-parent-gap 1800 --out-dir work/runs/run_0002/edges/_viz_cascade_78857_branching
  python scripts/07_visualize_tree.py --run-id run_0002 --cascade-id 78857 --tree-edges work/runs/run_0002/edges/_viz_cascade_78857_branching/inferred_tree_edge_table.csv --out work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg --max-nodes 180
输出：
  work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg
结果：
  observed_edges = 7471
  avg_depth = 8.82
  max_depth = 16
  num_branching_parents = 258
  time_gap_mean = 60.55s
  invalid_time_edges = 0
  tree_valid_ratio = 1.0
备注：
  该可视化证明当前 branching_time 结构不是线性链。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：轻量 HybridTree 小样本
命令：
  python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --max-cascades 10 --max-observation-seconds 1800 --out-dir work/runs/run_0002/edges/_debug_tree_hybrid_light_obs1800
输出：
  work/runs/run_0002/edges/_debug_tree_hybrid_light_obs1800/
结果：
  num_tree_edges = 43
  avg_depth = 5.67
  depth_p90 = 10
  max_depth = 11
  root_child_ratio = 0.163
  parent_child_text_sim_mean = 0.258
  random_pair_text_sim_mean = 0.085
  text_sim_lift = 0.173
  same_or_adjacent_window_edge_ratio = 0.953
  time_gap_mean = 188.02s
  time_gap_p90 = 482s
  top1_parent_child_ratio = 0.116
  top5_parent_child_ratio = 0.372
  branch_entropy = 3.086
  invalid_time_edges = 0
  tree_valid_ratio = 1.0
备注：
  这是轻量 HybridTree 的可运行诊断版本，默认未加载关注边，因此 follow_supported_edge_ratio = 0。
```

```text
日期：2026-06-29
Run ID：run_0002
结构实验：同批 10 cascade / 30m 观测期结构对照
输出：
  work/runs/run_0002/edges/_debug_tree_compare_obs1800.csv
对照设置：
  time_only
  branching_time
  hybrid_no_text
  hybrid
结果摘要：
  time_only:
    avg_depth = 9.19
    max_depth = 22
    text_sim_lift = 0.019
    top1_parent_child_ratio = 0.023
    branch_entropy = 3.761
  branching_time:
    avg_depth = 6.51
    max_depth = 12
    text_sim_lift = 0.080
    top1_parent_child_ratio = 0.116
    branch_entropy = 3.195
  hybrid_no_text:
    avg_depth = 6.51
    max_depth = 12
    text_sim_lift = 0.080
    top1_parent_child_ratio = 0.116
    branch_entropy = 3.195
  hybrid:
    avg_depth = 5.67
    max_depth = 11
    text_sim_lift = 0.173
    top1_parent_child_ratio = 0.116
    branch_entropy = 3.086
结论：
  在当前未加载关注边的小样本中，hybrid 与 hybrid_no_text 的主要差异来自文本相似项。
  hybrid 的 text_sim_lift 高于 hybrid_no_text，说明文本项在该小样本中提升了父子边文本一致性。
  该样本只有 43 条边，不能作为正式结论，只作为链路和指标验证。
```

```text
日期：2026-06-29
Run ID：run_0002
窗口实验：Causal MultiScale 小样本
配置：configs/window/obs_30m_step5m_multiscale.yaml
命令：
  python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode star --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_step300_multiscale_star
输出：
  work/runs/run_0002/windows/_debug_obs_1800_step300_multiscale_star/
结果：
  cascades = 10
  windows = 60
  node_windows = 214
  edge_windows = 124
  text_windows = 224
  current_edges = 43
  context_edges = 81
  retweet_text_early_violations = 0
备注：
  MultiScale 保持 6 个推断端点；window/node 表用 cur/ctx/cum 多尺度字段，edge 表用 current/context 两种 window_scope。
```


## 2026-07-02 Global Candidate Pack and Label Design Note

Global follow candidate extraction completed for run_0002:

```text
follow_edges_scanned    = 413,503,687
candidate_edges_written = 1,399,062
```

Packs were rebuilt at:

```text
work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
```

Pack diagnostics after rebuild:

```text
train = 41,750
valid = 9,175
test  = 8,759
samples_with_global_candidates = 41,289
total_global_candidate_edges = 890,037
global_candidate_alignment_errors = 0
```

The current Label-v1 weak labels remain useful for pipeline closure. Formal experiments should introduce Label-v2 stratified multi-rule weak labels before final reporting.


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
