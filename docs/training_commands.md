# Training Commands

Use this document as the compact command index. Detailed config behavior is documented in `docs/configuration.md`; server transfer details are in `docs/server_experiment_guide.md`.

## Branch

Current follow-up branch:

```bash
git checkout experiment/run-0002-next
```

## Label-Version Training

Label-v2 stratified score:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v2.yaml
```

Label-v3 labeling-function vote:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v3.yaml
```

Label-v4 coordination network:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v4.yaml
```

Label-v5 ensemble consensus:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5.yaml
```

## Smoke Test

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_label_v5
```

## Seeded Repeats

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v5_seed1
```

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --seed 2 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v5_seed2
```

## Existing Ablations

```bash
python scripts/17_run_ablation.py --config configs/train/ablation_no_adaptive_sampling.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_global_prior.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_memory.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_gate.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_uncertainty.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_role.yaml
```

`w/o Tree` and `w/o MultiScale` are input-pack ablations:

```bash
python scripts/16_train_dragen_full.py --config configs/train/ablation_no_tree.yaml
python scripts/16_train_dragen_full.py --config configs/train/ablation_no_multiscale.yaml
```

## Resume

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --resume work/artifacts/dragen_follow_adaptive_label_v5_seed0/checkpoints/last.pt
```

## TensorBoard

```bash
tensorboard --logdir work/artifacts --host 0.0.0.0 --port 6006
```

## 训练曲线

配置里启用 `plot_every_epoch` 后，训练会写出曲线汇总文件：

```text
<out-dir>/reports/training_curves.png
```

如果当前环境没有安装 `matplotlib`，会自动写出 HTML 版本：

```text
<out-dir>/reports/training_curves.html
```

训练结束后也可以从已有 report 文件重新生成：

```bash
python scripts/20_plot_training_curves.py --artifact-dir work/artifacts/dragen_follow_adaptive_label_v5_seed0
```

## DataLoader 参数覆盖

GPU 服务器上，当前标签配置默认从 `num_workers: 4` 起步。只有在调吞吐时，才需要用 CLI 临时覆盖：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5.yaml \
  --num-workers 8 \
  --prefetch-factor 2
```

## Result Tables

```bash
python scripts/18_export_result_tables.py --config configs/train/result_tables_run0002.yaml
```

## Prediction Analysis

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_follow_adaptive_label_v5_seed0
```

## 文本语义增强

RoBERTa 文本嵌入是离线预处理，不在训练阶段运行；根文本和转发文本都会编码。完整流程见 `docs/text_embeddings.md`。

```bash
python scripts/10_encode_text_roberta.py --run-id run_0002 --device auto
python scripts/10b_reduce_text_embeddings.py   --in-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext   --out-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64   --dim 64
python scripts/11b_build_text_semantic_features.py   --run-id run_0002   --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree   --text-emb-dir work/runs/run_0002/text_embeddings/chinese_roberta_wwm_ext_dim64   --out-dir work/runs/run_0002/text_semantic_features/obs_1800_step300_multiscale_hybrid_tree
```

## Pack Rebuild Example

```bash
python scripts/13_build_packs.py \
  --run-id run_0002 \
  --labels work/runs/run_0002/labels_v5_ensemble_consensus/weak_event_labels.csv \
  --global-candidate-edges work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv \
  --out-dir packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5
```
