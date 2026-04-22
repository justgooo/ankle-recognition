from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge three single-view decision-fusion expert checkpoints into one "
            "multi-view initialization checkpoint."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--axial-checkpoint", required=True)
    parser.add_argument("--coronal-checkpoint", required=True)
    parser.add_argument("--sagittal-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    return parser.parse_args()


def load_checkpoint(path_like: str) -> dict:
    path = Path(path_like).resolve()
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"checkpoint is missing model_state_dict: {path}")
    checkpoint["_resolved_path"] = str(path)
    return checkpoint


def transplant_view_modules(
    target_state_dict: dict[str, torch.Tensor],
    source_state_dict: dict[str, torch.Tensor],
    view_index: int,
) -> int:
    copied = 0
    prefixes = (
        f"view_encoders.{view_index}.",
        f"view_poolings.{view_index}.",
        f"view_classifiers.{view_index}.",
        f"classifier_view_recalibrators.{view_index}.",
    )
    for key, value in source_state_dict.items():
        if key.startswith(prefixes) and key in target_state_dict:
            target_state_dict[key] = value.clone()
            copied += 1
    return copied


def main() -> int:
    args = parse_args()
    base_checkpoint = load_checkpoint(args.base_checkpoint)
    axial_checkpoint = load_checkpoint(args.axial_checkpoint)
    coronal_checkpoint = load_checkpoint(args.coronal_checkpoint)
    sagittal_checkpoint = load_checkpoint(args.sagittal_checkpoint)

    merged_state_dict = {
        key: value.clone()
        for key, value in base_checkpoint["model_state_dict"].items()
    }
    copied_counts = {
        "axial": transplant_view_modules(
            merged_state_dict,
            axial_checkpoint["model_state_dict"],
            view_index=0,
        ),
        "coronal": transplant_view_modules(
            merged_state_dict,
            coronal_checkpoint["model_state_dict"],
            view_index=1,
        ),
        "sagittal": transplant_view_modules(
            merged_state_dict,
            sagittal_checkpoint["model_state_dict"],
            view_index=2,
        ),
    }

    output_path = Path(args.output_checkpoint).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": merged_state_dict,
        "config": base_checkpoint.get("config"),
        "merged_single_view_experts": {
            "base_checkpoint": base_checkpoint["_resolved_path"],
            "axial_checkpoint": axial_checkpoint["_resolved_path"],
            "coronal_checkpoint": coronal_checkpoint["_resolved_path"],
            "sagittal_checkpoint": sagittal_checkpoint["_resolved_path"],
            "copied_key_counts": copied_counts,
        },
    }
    torch.save(payload, output_path)
    print(f"saved_merged_checkpoint={output_path}")
    print(f"copied_key_counts={copied_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
