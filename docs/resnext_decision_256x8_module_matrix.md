# ResNeXt Decision 256x8 Module Matrix

This matrix defines the canonical `256x8` learned-weighting mainline.
The goal is to improve the learned branch itself through matched ablations and low-capacity repairs, not to replace the mainline with `equal-weight`.
The shared anchor is [configs/autoresearch_formal_resnext_decision_256x8.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal_resnext_decision_256x8.yaml).

## Anchor

- Mainline role: the single canonical learned recipe for all subsequent module analysis
- Geometry: `image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`
- Budget: `epochs=15`, `batch_size=6`, `num_workers=12`
- Scalars: `freeze_layers=3`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`
- Seeds: `42`, `123`, `456`

## Matrix

| ID | Purpose | Config / Override | Runtime Env | Expected Comparison |
|---|---|---|---|---|
| `L0-equal` | Matched control only | base config + `model.equal_weight_fusion=true` | none | fixed reporting/control reference, not the optimization target |
| `L1-learned` | Canonical mainline anchor | base config | none | baseline learned-weighting recipe to improve |
| `L2-minimal` | Remove richer reliability path | base config + `model.minimal_fusion_baseline=true` | none | tests whether a simpler learned path improves absolute performance |
| `L3-no-mixer` | Keep calibrator, drop cross-view token mixing | base config | `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` | isolates mixer contribution against canonical learned |
| `L4-no-calibrator` | Keep mixer, drop shared residual calibrator | base config | `ANKLE_DISABLE_FUSION_CALIBRATOR=1` | isolates calibrator contribution against canonical learned |
| `L5-temp1p5` | Low-capacity shrinkage probe | base config + strongest learned branch runtime path | `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`, `ANKLE_LEARNED_FUSION_TEMPERATURE=1.5` | tests lighter softening on the strongest learned branch |
| `L5-temp2p0` | Low-capacity shrinkage probe | base config + strongest learned branch runtime path | `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1`, `ANKLE_LEARNED_FUSION_TEMPERATURE=2.0` | tests stronger softening on the strongest learned branch |

## Recommended Order

1. Lock `L1-learned` as the only canonical mainline recipe across `42/123/456`
2. Run `L2-minimal`, `L3-no-mixer`, and `L4-no-calibrator` as single-module ablations against `L1`
3. Promote the strongest learned branch from step 2 into `L5-temp1p5` / `L5-temp2p0`
4. Keep `L0-equal` only as a matched control snapshot for reporting and final discussion

## Materialized Configs

- The full matrix has been materialized under [configs/generated_resnext_decision_256x8_matrix](/dataset/HH/ankle-ct/configs/generated_resnext_decision_256x8_matrix).
- Current inventory: `42` runnable YAMLs = `7 lanes × 3 seeds × 2 phases`.
- The launch manifest lives at [configs/generated_resnext_decision_256x8_matrix/manifest.json](/dataset/HH/ankle-ct/configs/generated_resnext_decision_256x8_matrix/manifest.json).
- The generator is [scripts/prepare_resnext_decision_256x8_matrix.py](/dataset/HH/ankle-ct/scripts/prepare_resnext_decision_256x8_matrix.py).

## Commands

```bash
.venv/bin/python scripts/prepare_resnext_decision_256x8_matrix.py --phase formal proxy
.venv/bin/python scripts/run_train_with_config_env.py --config configs/generated_resnext_decision_256x8_matrix/formal/cmp_resnext_decision_256x8_l1_learned_formal_s42.yaml
```

## Notes

- `equal-weight` remains mandatory as a matched control, but it is not the optimization target for this mainline.
- A learned ablation can remain on the mainline even if it is still below `equal-weight`, as long as it improves the canonical learned branch in `val_acc`, `val_auc`, or seed stability.
- As of `2026-04-22`, the strongest learned branch from the matched single-module sweep is `L3-no-mixer`, so `L5-temp*` should be interpreted as `L3-no-mixer + temperature`, not `L1 + temperature`.
- `L3/L4/L5` can now be serialized into `config.runtime_env` or Optuna `study.env`; trial artifacts will keep the explicit env map instead of relying on shell history alone.
- Direct single-run launches that depend on `runtime_env` should go through [scripts/run_train_with_config_env.py](/dataset/HH/ankle-ct/scripts/run_train_with_config_env.py), not raw `train.py`.
- The matrix has already been materialized; current execution status and results should be read from [backlog.md](/dataset/HH/ankle-ct/backlog.md), not inferred from this note alone.
