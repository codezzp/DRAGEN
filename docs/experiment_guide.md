# RoBERTa-only 实验指南

本分支只做一条路线：`Feature-v2 + RoBERTaText + Adaptive Global Sampling`。旧 StatText、旧无后缀 pack 和兼容入口不放在这个分支里维护。

## 分支边界

保留：

```text
scripts/11_build_features_v2.py
scripts/10_encode_text_roberta.py
scripts/10b_reduce_text_embeddings.py
scripts/11b_build_text_semantic_features.py
scripts/11c_build_non_text_evidence_v2.py
configs/train/dragen_full_label_v*_roberta_text.yaml
configs/train/dragen_full_label_v4_roberta_text_evidence_v2.yaml
```

不保留：

```text
scripts/bert.py
scripts/scripts_compat.py
configs/train/*stat_text*.yaml
configs/train/dragen_full_label_v2.yaml
configs/train/dragen_full_label_v3.yaml
configs/train/dragen_full_label_v4.yaml
configs/train/dragen_full_label_v5.yaml
```

## 实验顺序

1. 构建 `features_v2`。
2. 构建 RoBERTa 文本嵌入和窗口语义特征。
3. 为 Label-v2/v3/v4/v5 分别构建 `_feature_v2_..._roberta_text` pack。
4. 每套 label 先跑 1 epoch，检查 AUC、AP、F1、MCC 和 loss 是否稳定。
5. 以 Label-v4 为主线构建 `roberta_text_evidence_v2` pack，比较 Evidence-v2 的增益。
6. 正式训练时每个实验目录必须包含 label、feature_v2、roberta_text、seed 信息。

## Pack 检查

正确 pack 的 `meta.json` 至少应包含：

```json
{
  "text_semantic_dim": 64,
  "sample_keys": ["node_text_x", "window_text_x"],
  "node_feature_columns": [],
  "window_feature_columns": []
}
```

`node_feature_columns` 和 `window_feature_columns` 会写入 Feature-v2 的完整列名。模型缺少 `node_text_x` 会直接报错，不会回退到旧文本路线。
