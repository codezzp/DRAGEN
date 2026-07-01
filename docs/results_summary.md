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

## Next Work

Do not expand preprocessing before main model results are complete. The next tasks are:

```text
1. CAC-Stat baseline.
2. Campaign-GNN baseline.
3. Temporal-GNN baseline.
4. DRAGEN-Full formal run.
5. Ablations: w/o Tree, w/o MultiScale, w/o Role, w/o Memory, w/o Global Prior, w/o Adaptive Sampling, w/o Gate, w/o Uncertainty.
6. Export main_results.csv and ablation_results.csv under work/artifacts/reports/.
```
