# Fair Backbone Compare Protocol

This protocol turns the local backbone screen into a repeatable and more fair comparison for:

- `resunet`
- `resnext`
- `senet`
- `cspnet`

It is intentionally separate from the canonical mainline winner-tracking workflow. The goal here is not "which config wins under the current ResUNet-tuned pipeline", but "which backbone family is strongest after the same amount of protocol attention."

## Why the previous quick compare was not fair enough

- It used a single `1-epoch` run, which is too shallow for larger timm backbones.
- It inherited one shared recipe from the existing local workflow instead of giving each backbone an equal tuning budget.
- `cspnet` previously did not honor `freeze_layers=3` correctly because its timm backbone exposes `stages[...]` rather than `stages_0 / stages_1 / stages_2`.
- `resunet` and timm backbones do not expose identical internal structure, so we should compare them in stages rather than over-trusting one fixed recipe.

## Protocol summary

### Stage 0: Preflight (required before training)

Verify the effective freezing and trainable parameter counts from the actual config, not from the YAML intent alone.

```powershell
.\.venv\Scripts\python.exe scripts\backbone_preflight.py --config `
  configs\cmp_fair_local8g_stage1_resunet.yaml `
  configs\cmp_fair_local8g_stage1_resnext.yaml `
  configs\cmp_fair_local8g_stage1_senet.yaml `
  configs\cmp_fair_local8g_stage1_cspnet.yaml
```

Pass conditions:

- `dummy_output_shape` is valid for every config.
- `freeze_layers=3` actually freezes the expected early stages.
- No backbone is accidentally left almost fully trainable because of a wrapper mismatch.

### Stage 1: Matched-recipe proxy screen

Run the new `cmp_fair_local8g_stage1_*` configs.

Fixed choices:

- same local data geometry: `256 x 8`
- same fusion path: `feature`
- same pooling path: mean pooling (`use_attention_pooling=false`)
- same view setting: `share_backbone=false`
- same optimizer family: `Adam`
- same regularization defaults: `dropout=0.3`, `weight_decay=1e-4`, `augmentation=true`
- same freeze depth request: `freeze_layers=3`
- same proxy budget: `4 epochs`
- same seed for the first-pass screen: `42`

This stage answers one narrow question:

> Under one neutral plug-in recipe, which backbone integrates best with the current feature-fusion local pipeline?

### Stage 2: Equal-budget micro-tuning

Do not give extra manual love to only one backbone. Give every backbone the same tiny search budget:

- `freeze_layers in {2, 3}`
- `lr in {5e-5, 1e-4}`

That is exactly 4 runs per backbone. Keep all other settings equal to Stage 1.

This stage answers a different question:

> If every backbone gets the same small chance to adapt, which one reaches the best local proxy score?

Scoring rule:

- primary: `val_acc`
- tie-break: `val_auc`
- if still tied: prefer the simpler / lower-VRAM option

### Stage 3: Stability confirmation

Take the top 2 backbones from Stage 2 and re-run their best Stage 2 config on:

- `seed=42`
- `seed=123`

If one backbone only wins on one seed but collapses on the second, do not promote it as the backbone winner yet.

This stage answers:

> Which candidate is not only good once, but stable enough to deserve longer proxy or formal promotion?

## Interpretation rules

- Stage 1 winner = best plug-in under one matched recipe.
- Stage 2 winner = best backbone after equal micro-tuning budget.
- Stage 3 winner = stable candidate worth promotion.

Do not mix these conclusions together in the ledger.

## Configs added for Stage 1

- `configs/cmp_fair_local8g_stage1_resunet.yaml`
- `configs/cmp_fair_local8g_stage1_resnext.yaml`
- `configs/cmp_fair_local8g_stage1_senet.yaml`
- `configs/cmp_fair_local8g_stage1_cspnet.yaml`

## Notes

- This protocol is still a local `8 GB` campaign. It should not be compared directly with the canonical `24 GB` mainline records.
- `resunet` still has a structurally richer encoder path than the timm backbones, so fairness here means "equal protocol treatment", not "identical internal architecture."
- The first mandatory check is always the preflight script, because wrapper bugs can otherwise invalidate the whole comparison.
