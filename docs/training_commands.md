# DRAGEN RoBERTa-only 训练命令

本分支定位为：`Feature-v2 + RoBERTaText + Adaptive Global Sampling`。不维护 StatText，不使用无后缀旧 pack。

## 1. 特征构建

```bash
python scripts/11_build_features_v2.py \
  --run-id run_0002 \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --tree-edges work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv \
  --out-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree
```

## 2. RoBERTa 文本预处理

根文本和转发文本都要离线编码。训练阶段只读取 pack，不重新调用 RoBERTa。

```bash
python scripts/10_encode_text_roberta.py \
  --run-id run_0002 \
  --model-name hfl/chinese-roberta-wwm-ext \
  --max-length 128 \
  --batch-size 64 \
  --device auto \
  --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext

python scripts/10b_reduce_text_embeddings.py \
  --in-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext \
  --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64 \
  --dim 64

python scripts/11b_build_text_semantic_features.py \
  --run-id run_0002 \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --text-emb-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64 \
  --out-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree \
  --dim 64
```

## 3. Pack 命名

本分支只使用以下命名，不再使用旧的 `...global_follow_label_v*` 无 `feature_v2` 后缀 pack。

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text_evidence_v2
```

## 4. 构建各 Label 的 RoBERTaText Pack

```bash
python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels_v2_stratified_score/weak_event_labels.csv --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv --text-semantic-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text

python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels_v3_lf_vote/weak_event_labels.csv --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv --text-semantic-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text

python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels_v4_coordination_network/weak_event_labels.csv --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv --text-semantic-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text

python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels_v5_ensemble_consensus/weak_event_labels.csv --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv --text-semantic-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

## 5. Evidence-v2 Pack

```bash
python scripts/11c_build_non_text_evidence_v2.py \
  --run-id run_0002 \
  --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --out-dir work/runs/run_0002/evidence/obs_1800_step300_multiscale_hybrid_tree_global_follow

python scripts/13_build_packs.py \
  --run-id run_0002 \
  --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --labels work/runs/run_0002/labels_v4_coordination_network/weak_event_labels.csv \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --text-semantic-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree \
  --non-text-evidence-dir work/runs/run_0002/evidence/obs_1800_step300_multiscale_hybrid_tree_global_follow \
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text_evidence_v2
```

## 6. 训练

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v2_roberta_text.yaml
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v3_roberta_text.yaml
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4_roberta_text.yaml
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5_roberta_text.yaml
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4_roberta_text_evidence_v2.yaml
```

训练可视化输出在：

```text
work/artifacts/<run>/reports/epoch_metrics.csv
work/artifacts/<run>/reports/loss_breakdown.json
work/artifacts/<run>/reports/training_curves.png
work/artifacts/<run>/tb/
```
