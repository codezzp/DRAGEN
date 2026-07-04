# DRAGEN

DRAGEN is the experiment repository for organized cascade prediction. The current active line is:

```text
Feature-v2 + RoBERTa Text + Adaptive Global Sampling + Global Follow candidates
```

Current branch:

```text
experiment/run-0002-roberta-only
```

Large experiment artifacts are intentionally ignored by Git:

```text
work/
packs/
graph/follow_edges.tsv
*.zip
```

Keep code, configs, and documentation in Git. Transfer packs and artifacts with `rsync`, `scp`, or another file transfer tool.

## Current Status

The workstation has already built the formal RoBERTa-text packs for `run_0002`:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

Use `v2` as the main label version. Use `v5` as the strict-label robustness check. `v3` and `v4` are available for later comparison, but they are not the first training priority.

Pack input dimensions verified locally:

```text
window_x      = (B, 6, 24)
node_x        = (B, 6, N, 47)
node_text_x   = (B, 6, N, 64)
window_text_x = (B, 6, 64)
```

Text semantic notes:

```text
RoBERTa root embedding    = (85263, 768)
RoBERTa retweet embedding = (193331, 768)
Reduced dimension         = 64
node_text_features        = (567718, 64)
window_text_features      = (511578, 64)
```

Many retweet rows in `text_window_table.csv` have no raw text. Those nodes receive zero `node_text_x`; every window still has root text semantics in `window_text_x`.

## Server Setup

Clone and checkout the active branch:

```bash
git clone git@github.com:codezzp/DRAGEN.git
cd DRAGEN
git checkout experiment/run-0002-roberta-only
```

If the repository already exists:

```bash
cd DRAGEN
git fetch codezzp
git checkout experiment/run-0002-roberta-only
git pull
```

Recommended Python environment:

```text
Python >= 3.10
PyTorch with CUDA
numpy pandas scipy scikit-learn tqdm pyyaml matplotlib networkx
transformers tokenizers safetensors huggingface_hub
```

Install the non-PyTorch dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install numpy pandas scipy scikit-learn tqdm pyyaml matplotlib networkx
python -m pip install transformers accelerate datasets sentencepiece tokenizers safetensors huggingface_hub
python -m pip install tensorboard
```

Install PyTorch according to the server CUDA version. Example for CUDA 12.8 wheels:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## Data To Transfer To Server

For training only, transfer these two directories first:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text
```

Each pack must contain:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

Optional, for later label comparison:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v3_roberta_text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v4_roberta_text
```

Optional audit files:

```text
work/runs/run_0002/label_comparison/label_version_comparison.csv
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/text_semantic_feature_meta.json
work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree/feature_diagnostics.json
```

You do not need to transfer `graph/follow_edges.tsv` for normal training, because the global candidate edges are already packed into the `.pt` files.

Example transfer from Windows PowerShell:

```powershell
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text user@server:/path/to/DRAGEN/packs/
scp -r packs\obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text user@server:/path/to/DRAGEN/packs/
```

Example transfer with `rsync`:

```bash
rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/

rsync -av --progress packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text/ \
  user@server:/path/to/DRAGEN/packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v5_roberta_text/
```

## Stage 0: Smoke Test

First verify that the server can read the v2 pack and collate text fields:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dragen.data.pack_reader import PickleStreamDataset, collate_fn
p = 'packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text/train.pt'
ds = PickleStreamDataset(p, max_samples=2, split='train-smoke')
b = collate_fn([ds[0], ds[1]])
print('window_x', tuple(b['window_x'].shape))
print('node_x', tuple(b['node_x'].shape))
print('node_text_x', tuple(b['node_text_x'].shape))
print('window_text_x', tuple(b['window_text_x'].shape))
print('global edges', [tuple(x.shape) for x in b['global_candidate_edge_index']])
PY
```

Then run a one-epoch small DRAGEN smoke test:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-valid-samples 128 \
  --max-test-samples 128 \
  --out-dir work/artifacts/_smoke_dragen_v2_roberta_text
```

Check outputs:

```bash
ls work/artifacts/_smoke_dragen_v2_roberta_text/reports
ls work/artifacts/_smoke_dragen_v2_roberta_text/predictions
```

## Main Training Plan

Run order:

```text
1. Label-v2 DRAGEN-Full seed 0
2. Label-v2 DRAGEN-Full seed 1 and seed 2 if time allows
3. Label-v2 core ablations
4. Label-v5 strict-label robustness run
5. Export tables and analyze predictions
```

Current baseline entry points are placeholders in this branch:

```text
src/dragen/baselines/cac_stat.py
src/dragen/baselines/campaign_gnn.py
src/dragen/baselines/temporal_gnn.py
```

Do not expect `scripts/14_train_cac_stat.py` or `scripts/15_train_gnn_baselines.py` to produce formal baseline results until those implementations are added. For now, the server-ready path is DRAGEN-Full and DRAGEN ablations.

## Label-v2 Main DRAGEN-Full

Seed 0:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml
```

Seed 1:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed1
```

Seed 2:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --seed 2 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed2
```

Resume from `last.pt`:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v2_roberta_text.yaml \
  --resume work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed0/checkpoints/last.pt
```

## Label-v5 Strict Robustness

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5_roberta_text.yaml
```

Seed override example:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_label_v5_roberta_text.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_follow_adaptive_label_v5_roberta_text_feature_v2_seed1
```

## Ablation Runs

The existing ablation YAML files currently point to Label-v4 packs. If the thesis main ablation must use Label-v2, override `--pack-dir` and `--out-dir` from the CLI.

Common v2 pack:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text
```

Core module ablations on Label-v2:

```bash
python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_global_prior.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_global_prior

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_role.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_role

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_memory.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_memory

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_gate.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_gate

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_uncertainty.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_uncertainty

python scripts/17_run_ablation.py \
  --config configs/train/ablation_no_adaptive_sampling.yaml \
  --pack-dir packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text \
  --out-dir work/artifacts/label_v2_ablation_no_adaptive_sampling
```

`w/o RoBERTa Text`, `w/o MultiScale Context`, and `w/o HybridTree` need compatible non-text or alternate-structure packs. Do not run those ablations with the current RoBERTa-only pack unless the code/config is updated for that input.

## Metrics And Thresholds

Current metric code still computes classification metrics at `threshold = 0.5`. For formal thesis reporting, update or post-process predictions with this policy:

```text
Select the threshold that maximizes F1 on the validation set, then apply that fixed threshold to the test set.
```

Prediction files are written under:

```text
work/artifacts/<run>/predictions/
```

Important files:

```text
valid_event_predictions.csv
test_event_predictions.csv
event_predictions.csv
node_window_predictions.csv
role_distribution.csv
gate_weights.csv
uncertainty.csv
event_attention.csv
sampled_global_neighbors.csv
```

Run post-training analysis:

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed0
```

Export result tables after runs finish:

```bash
python scripts/18_export_result_tables.py \
  --run-dirs \
    work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed0 \
    work/artifacts/dragen_follow_adaptive_label_v5_roberta_text_feature_v2_seed0 \
  --ablation-run-dirs \
    work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed0 \
    work/artifacts/label_v2_ablation_no_global_prior \
    work/artifacts/label_v2_ablation_no_role \
    work/artifacts/label_v2_ablation_no_memory \
    work/artifacts/label_v2_ablation_no_gate \
    work/artifacts/label_v2_ablation_no_uncertainty \
    work/artifacts/label_v2_ablation_no_adaptive_sampling \
  --full-run-dir work/artifacts/dragen_follow_adaptive_label_v2_roberta_text_feature_v2_seed0 \
  --out-dir work/artifacts/reports
```

Expected final tables:

```text
work/artifacts/reports/main_results.csv
work/artifacts/reports/risk_retrieval_results.csv
work/artifacts/reports/ablation_results.csv
```

## Output Files To Copy Back

For each server run, copy back:

```text
work/artifacts/<run>/reports/
work/artifacts/<run>/predictions/
work/artifacts/<run>/checkpoints/best.pt
```

If checkpoints are too large, copy at least:

```text
reports/metrics.json
reports/loss_breakdown.json
reports/epoch_metrics.csv
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
predictions/*.csv
```

## Rebuilding Packs On Server

Normal server training should not rebuild packs. If rebuilding is necessary, transfer these inputs first:

```text
work/runs/run_0002/features_v2/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
work/runs/run_0002/global_graph/obs_1800_step300_multiscale_hybrid_tree/global_candidate_edge_table.csv
work/runs/run_0002/text_semantic/obs_1800_step300_multiscale_hybrid_tree_roberta64/
work/runs/run_0002/labels_v2_stratified_score/
work/runs/run_0002/labels_v5_ensemble_consensus/
```

Example rebuild for Label-v2:

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

Do not rebuild `graph/follow_edges.tsv` on the server unless absolutely necessary; that scan is large and is already reflected in the packed global candidate fields.

## More Docs

```text
docs/training_commands.md       RoBERTa preprocessing and pack build commands
docs/results_summary.md         Current experiment status
docs/label_design.md            Weak label versions
docs/configuration.md           YAML config rules
docs/server_experiment_guide.md Historical server migration notes
```
