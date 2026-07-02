# RoBERTa 文本嵌入说明

本分支是 RoBERTa-only。根文本和转发文本都必须离线编码，并通过 pack 字段 `node_text_x` / `window_text_x` 进入模型。

## 边界

- RoBERTa 编码、降维、窗口聚合都属于预处理阶段。
- 训练阶段只读取 pack，不重新编码文本。
- `scripts/bert.py` 和 `scripts/scripts_compat.py` 不在本分支维护。
- 旧 pack 缺少 `node_text_x` 时模型会直接报错。

## 输出目录

```text
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext/
work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64/
work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree/
```

## 论文表述

本文离线使用中文 RoBERTa 对根文本和转发文本进行编码，并将文本表示按级联编号和转发编号保存。随后，通过降维模块将高维语义向量压缩到低维空间，并在滑动窗口内按照文本可见性约束聚合为节点级和窗口级语义特征。训练阶段仅读取预处理得到的语义特征，不重新编码文本。
