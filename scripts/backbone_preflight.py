#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed
from train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect effective backbone freezing and trainable parameter counts for compare configs."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML config paths to inspect.",
    )
    parser.add_argument(
        "--show-trainable",
        type=int,
        default=12,
        help="How many trainable encoder parameter names to print.",
    )
    return parser.parse_args()


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def module_status(module: torch.nn.Module) -> str:
    total, trainable = count_parameters(module)
    if total == 0:
        return "n/a"
    if trainable == 0:
        return f"frozen ({trainable}/{total})"
    if trainable == total:
        return f"trainable ({trainable}/{total})"
    return f"mixed ({trainable}/{total})"


def resolve_encoder(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "shared_encoder"):
        return model.shared_encoder
    if hasattr(model, "view_encoders"):
        return model.view_encoders[0]
    raise RuntimeError("Unable to resolve encoder from model instance.")


def append_module_if_present(
    sections: list[tuple[str, torch.nn.Module]],
    owner: object,
    name: str,
    label: str | None = None,
) -> None:
    if hasattr(owner, name):
        sections.append((label or name, getattr(owner, name)))


def collect_encoder_sections(encoder: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    sections: list[tuple[str, torch.nn.Module]] = []

    for name in ("enc1", "enc2", "enc3", "enc4", "bottleneck", "ag2", "ag3", "ag4", "skip_proj"):
        append_module_if_present(sections, encoder, name)

    backbone = getattr(encoder, "backbone", None)
    if backbone is not None:
        for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "stem", "head"):
            append_module_if_present(sections, backbone, name, label=f"backbone.{name}")

        stages = getattr(backbone, "stages", None)
        if isinstance(stages, (torch.nn.Sequential, torch.nn.ModuleList, list, tuple)):
            for index, stage in enumerate(list(stages)):
                sections.append((f"backbone.stages[{index}]", stage))

    append_module_if_present(sections, encoder, "proj")
    return sections


def run_dummy_forward(model: torch.nn.Module, config: dict) -> tuple[int, ...]:
    data_cfg = config["data"]
    dummy = torch.zeros(
        1,
        3,
        int(data_cfg["num_slices_per_view"]),
        int(data_cfg["image_size"]),
        int(data_cfg["image_size"]),
    )
    model.eval()
    with torch.no_grad():
        output = model(dummy)
    return tuple(output.shape)


def inspect_config(config_path: Path, show_trainable: int) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))

    model = build_model(config)
    encoder = resolve_encoder(model)

    model_total, model_trainable = count_parameters(model)
    encoder_total, encoder_trainable = count_parameters(encoder)
    share_backbone = bool(config["model"]["share_backbone"])
    encoder_copies = 1 if share_backbone else 3
    trainable_names = [name for name, param in encoder.named_parameters() if param.requires_grad]
    output_shape = run_dummy_forward(model, config)

    print(f"=== {config_path} ===")
    print(f"backbone={config['model'].get('backbone')} fusion={config['model'].get('fusion_type')} share_backbone={share_backbone}")
    print(
        "model_params="
        f"{model_trainable:,}/{model_total:,} "
        f"({model_trainable / model_total:.2%} trainable)"
    )
    print(
        "single_encoder_params="
        f"{encoder_trainable:,}/{encoder_total:,} "
        f"({encoder_trainable / encoder_total:.2%} trainable)"
    )
    print(f"effective_encoder_copies={encoder_copies} effective_encoder_trainable={encoder_trainable * encoder_copies:,}")
    print(f"dummy_output_shape={output_shape}")
    print("encoder_section_status:")
    for section_name, module in collect_encoder_sections(encoder):
        print(f"  - {section_name}: {module_status(module)}")

    print("trainable_encoder_tensors:")
    for name in trainable_names[:show_trainable]:
        print(f"  - {name}")
    if len(trainable_names) > show_trainable:
        remaining = len(trainable_names) - show_trainable
        print(f"  - ... ({remaining} more)")
    print()


def main() -> None:
    args = parse_args()
    for config_path in args.config:
        inspect_config(Path(config_path), show_trainable=args.show_trainable)


if __name__ == "__main__":
    main()
