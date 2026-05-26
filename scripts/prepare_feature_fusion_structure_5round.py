from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from script_runtime import REPO_ROOT, repo_relative


SEEDS = (42, 123, 456)


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    rationale: str
    expected: str
    fusion_hidden_dim: int = 256
    runtime_env: dict[str, str] = field(default_factory=dict)


VARIANTS = (
    Variant(
        key="ff_struct01_glu_hidden384",
        title="FF-STRUCT-01-GLU-HIDDEN384-3SEED-FORMAL",
        rationale=(
            "Increase only the GLU fusion-head hidden width from 256 to 384 while keeping "
            "ResNeXt 256x8 geometry, mean pooling, recalibration, xview mixer, and scalars fixed."
        ),
        expected=(
            "A wider gated fused-token head may model higher-order cross-view feature interactions "
            "that the existing 256 bottleneck compresses, improving val_acc if current head capacity "
            "is the limiting feature-fusion path."
        ),
        fusion_hidden_dim=384,
    ),
    Variant(
        key="ff_struct02_glu_hidden128",
        title="FF-STRUCT-02-GLU-HIDDEN128-3SEED-FORMAL",
        rationale=(
            "Reduce only the GLU fusion-head hidden width from 256 to 128 while keeping all "
            "feature-fusion modules and training scalars fixed."
        ),
        expected=(
            "A narrower gated head may regularize fused-token gating and reduce seed-level "
            "variance, especially the weaker seed456 anchor, while preserving the GLU mechanism "
            "that beat the plain MLP ablation."
        ),
        fusion_hidden_dim=128,
    ),
    Variant(
        key="ff_struct03_xview_residual00625",
        title="FF-STRUCT-03-XVIEW-RESIDUAL00625-3SEED-FORMAL",
        rationale=(
            "Change only the CrossViewAttention residual scale from 0.125 to 0.0625 while "
            "keeping the mixer architecture, GLU head, and scalars fixed."
        ),
        expected=(
            "A weaker cross-view residual may retain the positive interaction signal while "
            "reducing noisy view-token overwrite, improving stability when a view-specific "
            "feature is already discriminative."
        ),
        runtime_env={"ANKLE_FEATURE_XVIEW_RESIDUAL_SCALE": "0.0625"},
    ),
    Variant(
        key="ff_struct04_xview_residual025",
        title="FF-STRUCT-04-XVIEW-RESIDUAL025-3SEED-FORMAL",
        rationale=(
            "Change only the CrossViewAttention residual scale from 0.125 to 0.25 while "
            "keeping the mixer architecture, GLU head, and scalars fixed."
        ),
        expected=(
            "A stronger cross-view residual may help the fused representation use non-dominant "
            "view context before concatenation, improving val_acc if the current mixer under-shares "
            "complementary evidence."
        ),
        runtime_env={"ANKLE_FEATURE_XVIEW_RESIDUAL_SCALE": "0.25"},
    ),
    Variant(
        key="ff_struct05_xview_dim512",
        title="FF-STRUCT-05-XVIEW-DIM512-3SEED-FORMAL",
        rationale=(
            "Change only the CrossViewAttention internal attention dimension from 256 to 512 "
            "while keeping one layer, four heads, residual scale 0.125, and all training scalars fixed."
        ),
        expected=(
            "Full-dimensional cross-view attention may preserve channel detail that the 256-dim "
            "mixer compresses, improving feature-level multi-view interaction without changing "
            "input geometry or the final GLU head."
        ),
        runtime_env={"ANKLE_FEATURE_XVIEW_ATTENTION_DIM": "512"},
    ),
)


def yaml_quote(value: str) -> str:
    return json.dumps(value)


def config_text(variant: Variant, seed: int) -> str:
    experiment_name = f"paper_feature_fusion_structure_5round_{variant.key}_s{seed}"
    output_dir = f"runs/paper_feature_fusion/structure_5round/{variant.key}_s{seed}"
    lines = [
        f"# {variant.title}: {variant.rationale}",
        f"experiment_name: {experiment_name}",
        f"output_dir: {output_dir}",
        f"seed: {seed}",
        "",
        "data:",
        "  csv_path: data/realdata/metadata.csv",
        "  base_dir: .",
        "  image_size: 256",
        "  num_slices_per_view: 8",
        "  trim_edge_slices: 2",
        "  batch_size: 6",
        "  num_workers: 12",
        "  val_ratio: 0.25",
        "  hu_min: -1000",
        "  hu_max: 1000",
        "",
        "model:",
        "  backbone: resnext",
        "  fusion_type: feature",
        "  share_backbone: false",
        "  use_pretrained: true",
        "  freeze_layers: 3",
        f"  fusion_hidden_dim: {variant.fusion_hidden_dim}",
        "  dropout: 0.25",
        "  use_attention_pooling: false",
        "  minimal_fusion_baseline: false",
        "",
        "train:",
        "  epochs: 15",
        "  lr: 1.0e-4",
        "  weight_decay: 1.0e-4",
        "  class_weight: true",
        "  device: auto",
        "  augmentation: true",
        "  scheduler: none",
        "  label_smoothing: 0.0",
        "  gradient_clip_norm: 1.0",
    ]
    if variant.runtime_env:
        lines.extend(["", "runtime_env:"])
        for key, value in sorted(variant.runtime_env.items()):
            lines.append(f"  {key}: {yaml_quote(value)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    config_dir = REPO_ROOT / "configs" / "feature_fusion_structure_5round"
    config_dir.mkdir(parents=True, exist_ok=True)
    configs: list[str] = []
    manifest: dict[str, object] = {
        "description": "Five single-variable 3-seed formal feature-fusion structure refinements.",
        "anchor": "PAPER-FEATURE-FUSION-GATED-HEAD-3SEED-FORMAL",
        "seeds": list(SEEDS),
        "variants": [],
    }
    for variant in VARIANTS:
        variant_configs: list[str] = []
        for seed in SEEDS:
            path = config_dir / f"{variant.key}_formal_s{seed}.yaml"
            path.write_text(config_text(variant, seed), encoding="utf-8")
            configs.append(repo_relative(path))
            variant_configs.append(repo_relative(path))
        manifest["variants"].append(
            {
                "key": variant.key,
                "title": variant.title,
                "rationale": variant.rationale,
                "expected": variant.expected,
                "fusion_hidden_dim": variant.fusion_hidden_dim,
                "runtime_env": dict(variant.runtime_env),
                "configs": variant_configs,
            }
        )
    manifest_path = config_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for config in configs:
        print(config)
    print(repo_relative(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
