import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ========== 1. 全局配置 ==========
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LENGTH = 128
BATCH_SIZE = 64
NORMALIZE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ID = "run_0001"  # ⚠️ 请改成你实际的 run_id

print(f"🚀 正在初始化模型: {MODEL_NAME} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# ========== 2. 核心编码函数 ==========
def encode_jsonl(input_jsonl, out_npy, out_idx, out_meta, key_field="cascade_idx"):
	"""
	统一的文本编码函数
	:param key_field: 用于对齐的主键（root 文本用 'cascade_idx'，retweet 文本用 'tweet_idx'）
	"""
	print(f"\n📖 开始处理: {input_jsonl} (按主键 {key_field} 对齐)")

	rows = []
	with open(input_jsonl, "r", encoding="utf-8") as f:
		for line in f:
			if not line.strip(): continue
			obj = json.loads(line)

			# 提取主键和文本
			key_id = int(obj[key_field])
			txt = obj.get("text", "") or ""  # 防止 None
			rows.append((key_id, txt))

	if not rows:
		print(f"⚠️ 文件 {input_jsonl} 为空或不存在，跳过。")
		return

	# 🚨 强制按主键升序排列，彻底杜绝 PyTorch Dataset 加载时的乱序问题！
	rows.sort(key=lambda x: x[0])
	key_list = [k for k, _ in rows]
	texts = [t for _, t in rows]
	N = len(texts)
	print(f"✅ 成功读取 {N} 条记录，已按 {key_field} 排序。")

	embs = []
	with torch.no_grad():
		for i in tqdm(range(0, N, BATCH_SIZE), desc=f"编码 {key_field}"):
			batch_text = texts[i:i + BATCH_SIZE]
			inputs = tokenizer(
				batch_text,
				padding="max_length",
				truncation=True,
				max_length=MAX_LENGTH,
				return_tensors="pt"
			).to(DEVICE)

			out = model(**inputs)
			# CLS pooling: 最后一层，第0个 token
			cls_emb = out.last_hidden_state[:, 0, :]

			if NORMALIZE:
				cls_emb = F.normalize(cls_emb, p=2, dim=1)

			embs.append(cls_emb.cpu().numpy().astype(np.float32))

	# 合并并落盘
	final_emb = np.concatenate(embs, axis=0)
	np.save(out_npy, final_emb)

	with open(out_idx, "w", encoding="utf-8") as f:
		json.dump(key_list, f, ensure_ascii=False)

	# 记录元数据
	meta_info = {
		"model": MODEL_NAME,
		"max_length": MAX_LENGTH,
		"pooling": "CLS",
		"normalize": NORMALIZE,
		"num_samples": N,
		"align_key": key_field,
		"d_t": final_emb.shape[1]
	}
	with open(out_meta, "w", encoding="utf-8") as f:
		json.dump(meta_info, f, ensure_ascii=False, indent=2)

	print(f"🎉 产物已保存 -> {out_npy} (Shape: {final_emb.shape})")


# ========== 3. 执行任务 ==========
if __name__ == "__main__":
	base_dir = f"work/runs/{RUN_ID}/processed/text"

	# 任务 A：生成 Root Text Embeddings (图级特征，对齐 cascade_idx)
	encode_jsonl(
		input_jsonl=f"{base_dir}/root_text.jsonl",
		out_npy=f"{base_dir}/root_text_emb.npy",
		out_idx=f"{base_dir}/root_text_emb.idx.json",
		out_meta=f"{base_dir}/root_text_emb.meta.json",
		key_field="cascade_idx"
	)

	# 任务 B：生成 Retweet Text Embeddings (节点级特征，对齐 tweet_idx)
	# 确保你已经运行过 `weibo2casicff export retweet_text_from_cascades`
	encode_jsonl(
		input_jsonl=f"{base_dir}/retweet_text.jsonl",
		out_npy=f"{base_dir}/retweet_text_emb.npy",
		out_idx=f"{base_dir}/retweet_text_emb.idx.json",
		out_meta=f"{base_dir}/retweet_text_emb.meta.json",
		key_field="tweet_idx"
	)

	print("\n✅ 所有文本编码任务全部完成！")