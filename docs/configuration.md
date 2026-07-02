# Configuration Guide

This project should be run from YAML configs whenever possible. CLI flags are for temporary overrides, debugging, and one-off path changes.

## Config Directory Layout

```text
configs/
  data/      run-level data/source settings
  window/    observation/window definitions
  model/     model and ablation module switches
  train/     executable training/evaluation configs
```

`configs/train/*.yaml` are the main entry points for experiments because they combine data, model, train, loss, logging, checkpoint, and output settings in one file.

## Priority Rules

All scripts using `dragen.config.apply_config()` follow this priority:

```text
script defaults < YAML config < explicit CLI flags
```

Example:

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_full_run0002_seed1
```

Here the YAML supplies the full experiment definition, while `--seed` and `--out-dir` override only those two fields.

## Supported YAML Sections

The parser currently recognizes these sections:

```text
data
model
train
loss
output
logging
checkpoint
tables
analysis
```

Top-level keys outside those sections are preserved when they match script arguments, e.g. `run_id`, `experiment_name`, and `ablation`.

## Field Mapping

### `data`

```yaml
data:
  pack_dir: packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5
  input_variant: MultiScale-HybridTree-GlobalFollow-LabelV5
  artifact_dir: work/artifacts/dragen_follow_adaptive_label_v5_seed0
```

Maps to:

```text
--pack-dir
--input-variant
--artifact-dir
```

`pack_dir` is the most important field for label-version experiments. A pack path fixes both the feature/window variant and the weak-label version.

### `model`

```yaml
model:
  hidden_dim: 64
  role_num: 5
  top_k_global: 20
  dropout: 0.1
  use_global_prior: true
  use_adaptive_sampling: true
  use_memory: true
  use_gate: true
  use_uncertainty: true
  use_role: true
```

Supported aliases:

```text
use_adaptive_sampling -> use_adaptive_sampler
use_temporal_memory   -> use_memory
use_prior_observation_gate -> use_gate
```

`w/o Adaptive Sampling` should keep `use_global_prior: true` but set `use_adaptive_sampling: false`. This preserves the global candidate pool and disables only learnable dynamic scoring.

### `train`

```yaml
train:
  epochs: 10
  batch_size: 8
  lr: 0.001
  weight_decay: 0.00001
  seed: 0
  device: auto
  eval_every: 1
  max_train_samples:
  max_valid_samples:
  max_test_samples:
```

`max_*_samples` should be used only for debug configs.

### `loss`

```yaml
loss:
  lambda_jump: 0.01
  lambda_struct: 0.005
  lambda_align: 0.001
  lambda_uncertainty: 0.001
  lambda_role: 0.0
  lambda_sampler_edge: 0.005
  lambda_sampler_hub: 0.001
  lambda_sampler_temp: 0.005
```

Sampler losses are used by the learnable adaptive global sampler:

```text
L_sampler_edge  structural evidence BCE over candidate pairs
L_sampler_hub   penalty for repeatedly selecting high-degree nodes
L_sampler_temp  evidence-shock-weighted temporal smoothness
```

### `output`, `logging`, `checkpoint`

```yaml
output:
  out_dir: work/artifacts/dragen_follow_adaptive_label_v5_seed0

logging:
  tensorboard: true
  tb_log_dir:

checkpoint:
  resume:
  save_every_epoch: false
```

Every formal run should have a unique `output.out_dir`. Do not reuse an artifact directory across different label versions or seeds.

### `tables`

Used by result-table export:

```yaml
tables:
  run_dirs:
    - work/artifacts/cac_stat_run0002
    - work/artifacts/dragen_follow_adaptive_label_v5_seed0
  ablation_run_dirs:
    - work/artifacts/dragen_follow_adaptive_label_v5_seed0
    - work/artifacts/ablation_no_adaptive_sampling_label_v5
  full_run_dir: work/artifacts/dragen_follow_adaptive_label_v5_seed0
  out_dir: work/artifacts/reports
```

### `analysis`

Used by prediction-analysis scripts:

```yaml
analysis:
  artifact_dir: work/artifacts/dragen_follow_adaptive_label_v5_seed0
  out_dir: work/artifacts/analysis/dragen_follow_adaptive_label_v5_seed0
```

## Recommended Configs for Label-Version Experiments

Use one train config per formal label version. For example:

```text
configs/train/dragen_full_label_v2.yaml
configs/train/dragen_full_label_v3.yaml
configs/train/dragen_full_label_v4.yaml
configs/train/dragen_full_label_v5.yaml
```

A Label-v5 config should look like:

```yaml
run_id: run_0002
experiment_name: dragen_follow_adaptive_label_v5_seed0

data:
  pack_dir: packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5
  input_variant: MultiScale-HybridTree-GlobalFollow-LabelV5

model:
  name: DRAGEN-Full
  hidden_dim: 64
  role_num: 5
  top_k_global: 20
  use_global_prior: true
  use_adaptive_sampling: true
  use_memory: true
  use_gate: true
  use_uncertainty: true
  use_role: true
  dropout: 0.1

train:
  epochs: 10
  batch_size: 8
  lr: 0.001
  weight_decay: 0.00001
  seed: 0
  device: auto
  eval_every: 1

loss:
  lambda_jump: 0.01
  lambda_struct: 0.005
  lambda_align: 0.001
  lambda_uncertainty: 0.001
  lambda_role: 0.0
  lambda_sampler_edge: 0.005
  lambda_sampler_hub: 0.001
  lambda_sampler_temp: 0.005

logging:
  tensorboard: true
  tb_log_dir:

checkpoint:
  resume:
  save_every_epoch: false

output:
  out_dir: work/artifacts/dragen_follow_adaptive_label_v5_seed0
```

Run it with:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5.yaml
```


## Label-Comparison Outputs

The multi-version label pipeline writes:

```text
work/runs/run_0002/label_comparison/label_version_comparison.csv
```

Use this table before training to decide which label version is suitable for a formal run. The current label-specific training configs point at the already-built label-version packs:

```text
configs/train/dragen_full_label_v2.yaml -> packs/..._label_v2
configs/train/dragen_full_label_v3.yaml -> packs/..._label_v3
configs/train/dragen_full_label_v4.yaml -> packs/..._label_v4
configs/train/dragen_full_label_v5.yaml -> packs/..._label_v5
```

## Ablation Config Rules

Ablation configs should change one thing at a time.

Examples:

```yaml
# w/o Adaptive Sampling
model:
  use_global_prior: true
  use_adaptive_sampling: false
```

```yaml
# w/o Global Prior
model:
  use_global_prior: false
```

```yaml
# w/o Memory
model:
  use_memory: false
```

For `w/o Tree` and `w/o MultiScale`, change `data.pack_dir` to the corresponding pack variant instead of toggling a model flag.

## Reproducibility Metadata

Every `scripts/16_train_dragen_full.py` run writes:

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
```

Use `reports/resolved_config.yaml` as the authoritative record for a completed run. It includes both the source YAML and the final resolved CLI arguments.

## Common Commands

Main DRAGEN-Full run:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_run0002.yaml
```

Debug run:

```bash
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_debug.yaml
```

Ablation run:

```bash
python scripts/17_run_ablation.py --config configs/train/ablation_no_adaptive_sampling.yaml
```

Result tables:

```bash
python scripts/18_export_result_tables.py --config configs/train/result_tables_run0002.yaml
```

Prediction analysis:

```bash
python scripts/19_analyze_predictions.py --config configs/train/<analysis_config>.yaml
```

## Maintenance Rules

- Add new formal runs as new YAML files under `configs/train/`.
- Do not encode label versions only in artifact names; make them visible in `data.pack_dir`, `data.input_variant`, and `experiment_name`.
- Keep CLI overrides out of final experiment records unless the override is intentional and documented.
- Keep debug configs separate from formal configs.
- When adding a new YAML field, update `src/dragen/config.py::flatten_config` if a script needs to consume it.


## Server Usage With Configs

Server runs should not rely on long hand-written CLI commands. Use these config files after transferring the corresponding `packs/` directories:

```text
configs/train/dragen_full_label_v2.yaml
configs/train/dragen_full_label_v3.yaml
configs/train/dragen_full_label_v4.yaml
configs/train/dragen_full_label_v5.yaml
```

The pack paths expected by those configs are:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v2/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v3/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v4/
packs/obs_1800_step300_multiscale_hybrid_tree_global_follow_label_v5/
```

A minimal server workflow is:

```bash
git checkout experiment/run-0002-next
python -m pip install -r requirements.txt
python scripts/16_train_dragen_full.py --config configs/train/dragen_full_label_v5.yaml
```

See `docs/server_experiment_guide.md` for file transfer and `docs/training_commands.md` for the command index.
