# Results Summary

## Current Stage

As of 2026-07-01, preprocessing is frozen for `run_0002`. The project has moved from window and structure validation into the experiment-closure stage.

Current branch:

```text
experiment/run-0002-code
```

Repository policy:

```text
work/ is ignored and must not be committed.
graph/follow_edges.tsv is excluded from the pushed experiment branch.
```

The current experiment execution is config-driven. Training, ablation, analysis, and result-table scripts support:

```bash
--config configs/train/<name>.yaml
```

Priority:

```text
script defaults < YAML config < CLI overrides
```

Each training run writes reproducibility metadata:

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
```

## Completed Inputs

### Fixed-5m Star

```text
work/runs/run_0002/windows/obs_1800_win300_step300/
```

Summary:

```text
cascades = 85263
window_rows = 511578
node_window_rows = 5940259
edge_window_rows = 1392078
text_window_rows = 6032866
retweet_text_early_violations = 0
```

### HybridTree Light

```text
work/runs/run_0002/edges/hybrid_tree_light/
```

Summary:

```text
cascades = 85263
tree_edges = 1392078
tree_valid_ratio = 1.0
invalid_time_edges = 0
avg_depth = 7.50
max_depth = 32
root_child_ratio = 0.101
text_sim_lift = 0.00107
```

### Fixed-5m HybridTree

```text
work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree/
```

Summary:

```text
window_rows = 511578
node_window_rows = 5940259
edge_window_rows = 1392078
text_window_rows = 6032866
retweet_text_early_violations = 0
```

### MultiScale HybridTree

```text
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
```

Summary:

```text
window_rows = 511578
node_window_rows = 5940259
edge_window_rows = 4019952
current_edges = 1392078
context_edges = 2627874
text_window_rows = 6032866
retweet_text_early_violations = 0
```

## Feature, Label, And Pack Outputs

Feature directories:

```text
work/runs/run_0002/features/obs_1800_win300_step300_star/
work/runs/run_0002/features/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
```

Feature validation:

```text
window_features rows = 511578 for each variant
node_window_features rows = 5940259 for each variant
nan_count = 0
inf_count = 0
```

Weak labels:

```text
work/runs/run_0002/labels/weak_event_labels.csv
positive = 17053
negative = 42631
ignore = 25579
```

Pack:

```text
work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree/
train = 41750
valid = 9175
test = 8759
T = 6
edge_alignment_errors = 0
```

The current `.pt` files are pickle streams because the local environment does not have `torch` installed.

## Next Work

## DRAGEN-Full Debug

DRAGEN-Full has replaced the previous light-model plan. The implementation keeps the thesis Chapter 4 modules:

```text
source evidence encoding
selective evidence reading
local role encoding
adaptive global sampling
global prior encoding
temporal memory
manipulation state accumulation
evidence shock
prior-observation Bayesian gate
uncertainty head
event attention pooling
```

Debug command completed for one epoch:

```text
out_dir = work/artifacts/dragen_full_debug
train_loss = 0.3926
valid_auc = 0.9113
test_auc = 0.9005
```

Exported files:

```text
reports/metrics.json
reports/loss_breakdown.json
predictions/event_predictions.csv
predictions/node_window_predictions.csv
predictions/role_distribution.csv
predictions/gate_weights.csv
predictions/uncertainty.csv
predictions/event_attention.csv
predictions/sampled_global_neighbors.csv
checkpoints/best.pt
checkpoints/last.pt
reports/epoch_metrics.csv
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
```

Roles are fixed to:

```text
producer
amplifier
suppressor
reframer
ordinary
```

No role outside this fixed set is used in DRAGEN-Full outputs.

## Training Controls

DRAGEN-Full training now supports per-epoch persistence, resume, and TensorBoard:

```text
reports/epoch_metrics.csv
reports/loss_breakdown.json
checkpoints/last.pt
checkpoints/best.pt
optional checkpoints/epoch_{epoch}.pt
optional <out-dir>/tb
```

Recommended config commands:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_debug.yaml
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_run0002.yaml
```

Resume:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --resume work/artifacts/dragen_full_run0002_seed0/checkpoints/last.pt
```

TensorBoard:

```bash
tensorboard --logdir work/artifacts --host 0.0.0.0 --port 6006
```

## Configured Experiments

Main DRAGEN-Full configs:

```text
configs/train/dragen_full_debug.yaml
configs/train/dragen_full_run0002.yaml
```

Ablation configs:

```text
configs/train/ablation_no_tree.yaml
configs/train/ablation_no_multiscale.yaml
configs/train/ablation_no_role.yaml
configs/train/ablation_no_memory.yaml
configs/train/ablation_no_global_prior.yaml
configs/train/ablation_no_adaptive_sampling.yaml
configs/train/ablation_no_gate.yaml
configs/train/ablation_no_uncertainty.yaml
```

Result table config:

```text
configs/train/result_tables_run0002.yaml
```

## Next Work

Do not expand preprocessing before main model results are complete. The next tasks are:

```text
1. CAC-Stat baseline.
2. Campaign-GNN baseline.
3. Temporal-GNN baseline.
4. DRAGEN-Full formal run.
5. Ablations: w/o Tree, w/o MultiScale, w/o Role, w/o Memory, w/o Global Prior, w/o Adaptive Sampling, w/o Gate, w/o Uncertainty.
6. Export main_results.csv, risk_retrieval_results.csv, and ablation_results.csv under work/artifacts/reports/.
```

## Evaluation Metric Update

Evaluation is now split into two categories.

Fair event-level metrics are used for main experiments, risk-retrieval tables, and ablations. They are computed only from:

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

Fair metrics:

```text
accuracy
balanced_accuracy
precision
recall
specificity
f1
macro_f1
auc
ap
mcc
brier
ece
precision_at_100
precision_at_500
recall_at_500
precision_at_1pct
recall_at_1pct
precision_at_5pct
recall_at_5pct
```

DRAGEN-specific interpretability metrics are separate and must not be compared against baselines that do not export node-window explanations. They read:

```text
predictions/node_window_predictions.csv
predictions/role_distribution.csv
predictions/gate_weights.csv
predictions/uncertainty.csv
predictions/event_attention.csv
```

Post-training analysis entry:

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_full_run0002_seed0
```

Outputs:

```text
reports/event_metrics_extended.json
reports/risk_retrieval_metrics.json
reports/temporal_stability_metrics.json
reports/interpretability_metrics.json
reports/diagnostic_summary.csv
```

Result-table entry:

```bash
python scripts/18_export_result_tables.py \
  --config configs/train/result_tables_run0002.yaml
```
