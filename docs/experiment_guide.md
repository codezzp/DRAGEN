# Key-user Pool Experiment Guide

This guide describes the current active branch:

```text
feature/key-user-pool-global-prior
```

Current experiment line:

```text
Feature-v2 + RoBERTa Text + Key-user Pool Global Prior
```

This branch keeps the old edge-list global branch for compatibility, but the recommended formal v2 run uses the fixed window-level `key_user_pool` branch.

## Branch Boundary

Keep and maintain:

```text
scripts/11_build_features_v2.py
scripts/10_encode_text_roberta.py
scripts/10b_reduce_text_embeddings.py
scripts/11b_build_text_semantic_features.py
scripts/11c_build_non_text_evidence_v2.py
scripts/13b_build_key_user_pool_packs.py
configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml
configs/train/dragen_full_label_v*_roberta_text.yaml
src/dragen/models/key_user_global_prior.py
```

Legacy compatibility files may remain in the repository, but they are not the first formal server-training path for this branch.

## Required Pack

Formal v2 key-user training reads:

```text
packs/obs_1800_step300_multiscale_hybrid_tree_feature_v2_global_follow_label_v2_roberta_text_key_user_pool
```

The pack must contain:

```text
train.pt
valid.pt
test.pt
meta.json
pack_diagnostics.json
```

The key-user fields added to each sample are:

```text
key_user_idx    [T, R]
key_user_weight [T, R]
key_user_hop    [T, R]
key_user_mask   [T, R]
```

Default values are `T=6`, `R=32`, `max_hops=4`.

## Training Order

1. Checkout `feature/key-user-pool-global-prior`.
2. Upload or build the v2 key-user pool pack.
3. Run key-user pack shape smoke.
4. Run 64/32/32 end-to-end smoke.
5. Run 512/256/256 speed test.
6. Run formal v2 key-user seed0.
7. Sync back `reports/` and `predictions/`.
8. Add seed1/seed2 only after seed0 is stable.

## Formal Config

Use:

```text
configs/train/dragen_full_label_v2_roberta_text_key_user_pool.yaml
```

The expected output directory is:

```text
work/artifacts/dragen_follow_key_user_pool_label_v2_roberta_text_feature_v2_seed0
```

## Notes

Training does not run RoBERTa. RoBERTa encoding, reduction, semantic aggregation, and key-user pool construction are offline preprocessing steps. The training stage only reads the pack.

Do not prioritize the old edge-list `dragen_full_label_v2_roberta_text.yaml` Full run for formal v2 training on this branch. The speed diagnosis showed the global edge-list branch is the main bottleneck.
