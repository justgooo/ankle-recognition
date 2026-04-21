# ResNeXt Decision 256x8 Module Matrix

This matrix defines the next `256x8` learned-weighting campaign without starting training.
The shared anchor is [configs/autoresearch_formal_resnext_decision_256x8.yaml](/dataset/HH/ankle-ct/configs/autoresearch_formal_resnext_decision_256x8.yaml).

## Anchor

- Geometry: `image_size=256`, `num_slices_per_view=8`, `trim_edge_slices=2`
- Budget: `epochs=15`, `batch_size=6`, `num_workers=12`
- Scalars: `freeze_layers=3`, `lr=1e-4`, `weight_decay=5e-4`, `dropout=0.25`, `gradient_clip_norm=2.0`
- Seeds: `42`, `123`, `456`

## Matrix

| ID | Purpose | Config / Override | Runtime Env | Expected Comparison |
|---|---|---|---|---|
| `L0-equal` | Strong control | base config + `model.equal_weight_fusion=true` | none | control upper bound for current legacy geometry |
| `L1-learned` | Learned anchor | base config | none | current learned-weighting reference |
| `L2-minimal` | Remove richer reliability path | base config + `model.minimal_fusion_baseline=true` | none | tests whether raw per-view confidence is enough |
| `L3-no-mixer` | Keep calibrator, drop cross-view token mixing | base config | `ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1` | isolates mixer contribution |
| `L4-no-calibrator` | Keep mixer, drop shared residual calibrator | base config | `ANKLE_DISABLE_FUSION_CALIBRATOR=1` | isolates calibrator contribution |
| `L5-temp1p5` | Low-capacity shrinkage probe | base config | `ANKLE_LEARNED_FUSION_TEMPERATURE=1.5` | tests lighter softening |
| `L5-temp2p0` | Low-capacity shrinkage probe | base config | `ANKLE_LEARNED_FUSION_TEMPERATURE=2.0` | tests stronger softening |

## Recommended Order

1. `L0-equal` vs `L1-learned` on matched seeds
2. `L2-minimal` on the same seeds
3. `L3-no-mixer` and `L4-no-calibrator`
4. `L5-temp1p5` and `L5-temp2p0` only on the strongest learned branch from steps 2-3

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

- `equal-weight` remains mandatory as a matched control even though it is not the desired final method.
- `L3/L4/L5` can now be serialized into `config.runtime_env` or Optuna `study.env`; trial artifacts will keep the explicit env map instead of relying on shell history alone.
- Direct single-run launches that depend on `runtime_env` should go through [scripts/run_train_with_config_env.py](/dataset/HH/ankle-ct/scripts/run_train_with_config_env.py), not raw `train.py`.
- No training has been started for this matrix yet; only configs and manifests have been materialized.
