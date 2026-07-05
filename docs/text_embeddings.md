# RoBERTa 文本语义特征

本文档记录当前 RoBERTa text 产物和重建命令。当前分支：

```text
experiment/run-0002-roberta-only
```

训练阶段不重新调用 RoBERTa，只读取已经写入 pack 的：

```text
node_text_x
window_text_x
```

## 当前产物

原始 768 维 RoBERTa embedding：

```text
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext/root_text_emb.npy
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext/retweet_text_emb.npy
```

规模：

```text
root_text_emb.npy    = (85263, 768)
retweet_text_emb.npy = (193331, 768)
```

64 维降维：

```text
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64/root_text_emb64.npy
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64/retweet_text_emb64.npy
```

规模：

```text
root_text_emb64.npy    = (85263, 64)
retweet_text_emb64.npy = (193331, 64)
```

按窗口可见性聚合后的语义特征：

```text
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/node_text_features.npy
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/window_text_features.npy
```

规模：

```text
node_text_features.npy   = (567718, 64)
window_text_features.npy = (511578, 64)
```

元数据：

```text
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/text_semantic_feature_meta.json
```

## 覆盖情况说明

`text_window_table.csv` 中所有 root 文本都有 embedding。很多 retweet 行没有原始文本，因此这些节点没有 retweet 语义，pack 中对应 `node_text_x` 为零向量。

窗口级 `window_text_x` 仍然覆盖所有窗口，因为 root 文本在每个窗口可见。

## 重建命令

### 1. 编码 RoBERTa

```bash
python scripts/10_encode_text_roberta.py \
  --run-id run_0002 \
  --model-name hfl/chinese-roberta-wwm-ext \
  --max-length 128 \
  --batch-size 32 \
  --device cuda \
  --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext
```

如果显存不足，把 `--batch-size 32` 改为 `16`。

### 2. 降维到 64

```bash
python scripts/10b_reduce_text_embeddings.py \
  --in-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext \
  --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64 \
  --dim 64 \
  --seed 0
```

### 3. 聚合到窗口语义

```bash
python scripts/11b_build_text_semantic_features.py \
  --run-id run_0002 \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --text-emb-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_reduced64 \
  --out-dir work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64 \
  --dim 64
```

### 4. 写入 pack

```bash
python scripts/13_build_packs.py \
  --run-id run_0002 \
  --feature-dir work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree \
  --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree \
  --labels work/runs/run_0002/labels_v2_stratified_score/weak_event_labels.csv \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --text-semantic-dir work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64 \
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
```

## 服务器训练注意

服务器只训练时，不需要传 `text_embeddings/` 或 `text_semantic/`，只需要传已经构建好的 pack：

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```
