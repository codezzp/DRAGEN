"""Compatibility note for semantic text features.

旧版本在这里直接初始化 RoBERTa，并写死 RUN_ID。现在文本语义嵌入已经移到离线脚本：

1. scripts/10_encode_text_roberta.py 只对原始文本编码一次。
2. scripts/10b_reduce_text_embeddings.py 将 embedding 降到 64 维。
3. scripts/11b_build_text_semantic_features.py 按窗口聚合可见文本 embedding。

训练阶段和 pack 构建阶段不会触发 RoBERTa 编码。
"""

from __future__ import annotations

TEXT_EMBEDDING_PIPELINE = [
    "scripts/10_encode_text_roberta.py",
    "scripts/10b_reduce_text_embeddings.py",
    "scripts/11b_build_text_semantic_features.py",
]
