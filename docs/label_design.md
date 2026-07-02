# Weak Label Design

This document separates the current debug label set from the recommended formal label design.

## Label-v1: Current Debug Labels

Current implementation:

```text
scripts/12_build_weak_labels.py
src/dragen/labels/weak_label_builder.py
```

Output:

```text
work/runs/run_0002/labels/weak_event_labels.csv
work/runs/run_0002/labels/label_diagnostics.json
```

Label-v1 builds one event-level `weak_score` from feature-derived components and applies global quantile thresholds:

```text
bottom 50% weak_score -> weak_label = 0
middle 30% weak_score -> weak_label = -1
top 20% weak_score    -> weak_label = 1
```

The split is still hash-based by `cascade_idx`, so windows from the same cascade do not cross train/valid/test.

Current run_0002 counts:

```text
num_cascades = 85,263
positive     = 17,053
negative     = 42,631
ignore       = 25,579
```

Label-v1 is useful for pipeline debugging and initial closure, but it should not be treated as the final thesis label set. Its components overlap with model inputs such as heat, degree, node activity, text count, and structural statistics. Very high validation AUC under Label-v1 may indicate that the model has learned the weak-label scoring rule rather than a robust manipulation signal.

## Label-v2: Recommended Formal Labels

Formal experiments should use a separate output directory and must not overwrite Label-v1:

```text
work/runs/run_0002/labels_v2_stratified_score/weak_event_labels.csv
work/runs/run_0002/labels_v2_stratified_score/label_diagnostics.json
```

Recommended builder entry:

```text
scripts/12b_build_weak_labels_v2.py
```

Label-v2 should be a stratified multi-rule weak label set. It should compute four event-level evidence scores:

```text
burst_score          propagation burst evidence
coordination_score   synchronized behavior evidence
structure_score      organized structural evidence
text_score           repeated narrative / text similarity evidence
```

Each score should be percentile-rank normalized to `[0, 1]`. The aggregate score should downweight pure popularity:

```text
weak_score = 0.25 * burst_score
           + 0.30 * coordination_score
           + 0.25 * structure_score
           + 0.20 * text_score
```

Evidence hits:

```text
evidence_hit_count = I(burst_score >= 0.8)
                   + I(coordination_score >= 0.8)
                   + I(structure_score >= 0.8)
                   + I(text_score >= 0.8)
```

Size buckets by observed retweet count:

```text
[8, 20), [20, 50), [50, 100), [100, 300), [300, +inf)
```

Within each size bucket:

```text
positive: weak_score in bucket top 20% and evidence_hit_count >= 2
negative: weak_score in bucket bottom 50% and evidence_hit_count <= 1
ignore:   all remaining cascades
```

Recommended `weak_event_labels.csv` fields:

```text
cascade_idx
weak_score
burst_score
coordination_score
structure_score
text_score
evidence_hit_count
size_bucket
label
split
```

`13_build_packs.py` accepts either `weak_label` or `label`. Label-v1 uses `weak_label`; Label-v2 may use `label` as the public field name.

The purpose is to compare active cascades against active cascades, instead of letting the classifier separate large events from small events.

## Required Diagnostics

`label_diagnostics.json` for Label-v2 should include at least:

```text
num_cascades
positive
negative
ignore
label_ratio
label_by_size_bucket
score_quantiles
score_by_label
observed_retweet_count_by_label
feature_correlation_with_label
split_distribution
warnings
```

Warnings should be emitted when:

```text
observed_retweet_count has high correlation with label
any size bucket has too few positive or negative samples
positive or negative samples are concentrated in one size bucket
```

## Pack Usage

`13_build_packs.py` currently defaults to `work/runs/<run_id>/labels/weak_event_labels.csv`. To use Label-v2, pass the label file explicitly:

```powershell
python scripts/13_build_packs.py `
  --run-id run_0002 `
  --labels work/runs/run_0002/labels_v2_stratified_score/weak_event_labels.csv `
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv `
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v2
```

Do not mix Label-v1 and Label-v2 results in the same result table without naming the label version.


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
