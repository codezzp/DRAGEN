# 文本语义嵌入流程

当前实验拆成两条文本路线：

```text
DRAGEN-Full-FollowAdaptive-StatText
DRAGEN-Full-FollowAdaptive-RoBERTaText
```

`StatText` 是稳定主线，只使用文本统计特征和 HybridTree 轻量文本相似度。`RoBERTaText` 是语义文本增强版，不覆盖当前统计文本版本。

## 设计原则

- 根文本和转发文本都要编码：根文本按 `cascade_idx` 对齐，转发文本按 `tweet_idx` 对齐。
- RoBERTa 只对原始文本编码一次，不按窗口重复编码。
- 降维只对缓存 embedding 做一次；不同窗口设置复用同一套 64 维 embedding。
- 窗口相关步骤只做 embedding 聚合，不重新调用 RoBERTa。
- 训练脚本和 pack 构建脚本只读取离线产物，不动态计算文本标签或文本 embedding。
- 旧 pack 不包含 `node_text_x/window_text_x` 时，模型自动退回统计文本特征。

## 泄漏边界

当前 `scripts/10b_reduce_text_embeddings.py` 使用固定种子的 Gaussian random projection：

```text
reducer_type = fixed_random_projection
reducer_fit_scope = none_no_fit
```

它不是 PCA、SVD、IncrementalPCA 或 AutoEncoder，不会在 train/valid/test embedding 上拟合降维参数，因此不存在 test 分布参与 reducer fit 的 transductive leakage。meta 中会记录 `reducer_fit_scope: none_no_fit` 和对应的 `leakage_note`。

如果以后改成 PCA/SVD/AutoEncoder，必须新增 `--fit-split train`，只用 train split 对应文本拟合 reducer，再 transform train/valid/test。

## 1. 安装可选依赖

只在需要跑 RoBERTa 编码的机器上安装：

```bash
python -m pip install -r requirements-text.txt
```

普通训练环境只需要 `requirements.txt`。

## 2. 原始文本编码，只跑一次

```bash
python scripts/10_encode_text_roberta.py   --run-id run_0002   --model-name hfl/chinese-roberta-wwm-ext   --max-length 128   --batch-size 64   --device auto   --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext
```

输入：

```text
work/runs/run_0002/processed/text/root_text.jsonl
work/runs/run_0002/processed/text/retweet_text.jsonl
```

输出包含根文本和转发文本两套 embedding：

```text
root_text_emb.npy
root_text_emb.idx.json
root_text_emb.meta.json
retweet_text_emb.npy
retweet_text_emb.idx.json
retweet_text_emb.meta.json
```

缓存判断字段：`run_id`、`model_name`、`max_length`、`pooling`、`normalize`。一致时默认跳过；需要重算时加 `--force`。

编码 meta 记录：

```text
model_name
tokenizer_name
max_length
pooling
normalize
embedding_dim
source_file_hash
num_samples
created_at
```

## 3. 降维，只跑一次

```bash
python scripts/10b_reduce_text_embeddings.py   --in-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext   --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64   --dim 64
```

输出 64 维 embedding：

```text
root_text_emb64.npy
root_text_emb64.idx.json
root_text_emb64.meta.json
retweet_text_emb64.npy
retweet_text_emb64.idx.json
retweet_text_emb64.meta.json
```

降维 meta 记录：

```text
reducer_type
reducer_fit_scope
source_embedding_dim
reduced_dim
source_file_hash
source_idx_hash
seed
created_at
```

## 4. 按窗口聚合，每套窗口各跑一次

主窗口示例：

```bash
python scripts/11b_build_text_semantic_features.py   --run-id run_0002   --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree   --text-emb-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64   --out-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree   --dim 64
```

输出：

```text
node_text_features.npy
node_text_feature_index.json
window_text_features.npy
window_text_feature_index.json
text_semantic_feature_meta.json
```

聚合使用 `text_window_table.csv` 的可见性结果：root 文本从第一个窗口起可见，retweet 文本只有在对应 `post_offset < end_offset` 后才可见，因此不会把未来文本提前喂给模型。

## 5. 构建 RoBERTaText Pack

Label-v4 RoBERTaText：

```bash
python scripts/13_build_packs.py   --run-id run_0002   --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree   --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree   --labels work/runs/run_0002/labels_v4_coordination_network/weak_event_labels.csv   --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv   --text-semantic-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree   --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v4_roberta_text
```

Label-v4 StatText 对照 pack 不传 `--text-semantic-dir`：

```bash
python scripts/13_build_packs.py   --run-id run_0002   --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree   --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree   --labels work/runs/run_0002/labels_v4_coordination_network/weak_event_labels.csv   --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv   --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v4_stat_text
```

## 6. 最小对照实验

优先只比较 Label-v4 的两条路线：

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4_stat_text.yaml
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4_roberta_text.yaml
```

二者应保持同一 label、同一 split、同一模型参数、同一 seed、同一 epoch，只改变文本输入。先各跑 1 epoch debug；如果指标和 loss 正常，再跑正式 10 epoch。

## 7. 论文表述

可以写成：

```text
为进一步刻画节点在传播过程中的语义表达特征，本文在统计文本特征之外引入 RoBERTa 语义表示。具体而言，首先离线使用中文 RoBERTa 对根文本和转发文本进行编码，并将文本表示按级联编号和转发编号保存。随后，通过固定随机投影将高维语义向量压缩到低维空间，并在滑动窗口内按照文本可见性约束聚合为节点级和窗口级语义特征。训练阶段仅读取预处理得到的语义特征，不重新编码文本，从而保证文本表示构建与模型训练过程相互分离。
```
