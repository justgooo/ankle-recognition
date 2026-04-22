from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import choose_device, compute_class_weights, load_config, save_json, set_seed
from train import (
    build_dataloaders,
    build_model,
    build_scheduler,
    evaluate,
    get_current_lr,
    score_for_model_selection,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a stage-wise decision-fusion experiment by initializing from an existing "
            "checkpoint and fine-tuning only a selected parameter subset."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the stage-wise YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data/model, load the init checkpoint, print trainable parameters, then exit.",
    )
    return parser.parse_args()


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_runtime_env(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"config.runtime_env must be a mapping, got {type(raw).__name__}.")
    return {str(key): str(value) for key, value in raw.items()}


def normalize_trainable_prefixes(raw: Any) -> list[str]:
    if raw is None:
        raise SystemExit("config.stagewise.trainable_prefixes is required.")
    if not isinstance(raw, list) or not raw:
        raise SystemExit("config.stagewise.trainable_prefixes must be a non-empty list.")
    return [str(item) for item in raw]


def save_manifest(
    output_dir: Path,
    config_path: Path,
    runtime_env: dict[str, str],
    init_checkpoint: Path,
    trainable_prefixes: list[str],
) -> None:
    payload = {
        "config_path": str(config_path),
        "runtime_env": runtime_env,
        "init_checkpoint": str(init_checkpoint),
        "trainable_prefixes": trainable_prefixes,
        "launched_at": time.time(),
    }
    with (output_dir / "stagewise_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def apply_runtime_env(runtime_env: dict[str, str]) -> None:
    os.environ.update(runtime_env)


def freeze_to_prefixes(
    model: torch.nn.Module,
    trainable_prefixes: list[str],
) -> tuple[list[str], list[str], int, int]:
    matched_prefixes: list[str] = []
    trainable_names: list[str] = []
    frozen_names: list[str] = []

    for name, parameter in model.named_parameters():
        should_train = any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in trainable_prefixes
        )
        parameter.requires_grad = should_train
        if should_train:
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    for prefix in trainable_prefixes:
        if any(name == prefix or name.startswith(prefix + ".") for name, _ in model.named_parameters()):
            matched_prefixes.append(prefix)

    if not trainable_names:
        raise SystemExit(
            "No parameters matched config.stagewise.trainable_prefixes; refusing to launch a no-op run."
        )

    trainable_param_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_param_count = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    return matched_prefixes, trainable_names, trainable_param_count, frozen_param_count


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_config(str(config_path))
    runtime_env = normalize_runtime_env(config.get("runtime_env"))
    apply_runtime_env(runtime_env)

    stagewise_cfg = config.get("stagewise")
    if not isinstance(stagewise_cfg, dict):
        raise SystemExit("config.stagewise must be present for stage-wise training.")
    init_checkpoint = resolve_repo_path(str(stagewise_cfg.get("init_checkpoint", "")))
    if not init_checkpoint.is_file():
        raise SystemExit(f"init checkpoint not found: {init_checkpoint}")
    strict_load = bool(stagewise_cfg.get("strict_load", True))
    trainable_prefixes = normalize_trainable_prefixes(stagewise_cfg.get("trainable_prefixes"))

    set_seed(config["seed"])
    output_dir = resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("best.pt", "history.json", "summary.json", "stagewise_manifest.json"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    device = choose_device(config["train"]["device"])
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"config={config_path.relative_to(REPO_ROOT)}")
    print(f"output_dir={output_dir.relative_to(REPO_ROOT)}")
    print(f"runtime_env={json.dumps(runtime_env, ensure_ascii=False, sort_keys=True)}")
    print(f"init_checkpoint={init_checkpoint.relative_to(REPO_ROOT)}")

    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)
    model = build_model(config).to(device)

    checkpoint = torch.load(init_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict_load)
    matched_prefixes, trainable_names, trainable_param_count, frozen_param_count = freeze_to_prefixes(
        model,
        trainable_prefixes,
    )
    print(f"matched_trainable_prefixes={matched_prefixes}")
    print(f"trainable_param_count={trainable_param_count}")
    print(f"frozen_param_count={frozen_param_count}")
    preview_names = trainable_names[:12]
    if preview_names:
        print("trainable_parameter_preview=" + json.dumps(preview_names, ensure_ascii=False))

    save_manifest(
        output_dir=output_dir,
        config_path=config_path,
        runtime_env=runtime_env,
        init_checkpoint=init_checkpoint,
        trainable_prefixes=trainable_prefixes,
    )

    if args.dry_run:
        return 0

    run_start_time = time.perf_counter()

    label_smoothing = float(config["train"].get("label_smoothing", 0.0))
    if config["train"]["class_weight"]:
        class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Label smoothing enabled: {label_smoothing}")

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = build_scheduler(config, optimizer)

    gradient_clip_norm = config["train"].get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        print(f"Gradient clipping enabled: max_norm={gradient_clip_norm}")
    augmentation_enabled = bool(config["train"].get("augmentation", False))
    if augmentation_enabled:
        print("Data augmentation enabled (random flip + rotation)")
    early_stopping_patience = config["train"].get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        print(f"Early stopping enabled: patience={early_stopping_patience}")
    epochs_without_improvement = 0

    history: list[dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    init_metrics = None
    if val_loader is not None:
        init_val_loss, init_metrics = evaluate(model, val_loader, criterion, device)
        best_score: tuple[float, float] | float = score_for_model_selection(init_metrics)
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "val_loss": init_val_loss,
                **{f"val_{key}": value for key, value in init_metrics.items()},
                "note": "init_checkpoint",
            }
        )
        print(
            f"init_val_loss={init_val_loss:.4f} | "
            f"init_val_auc={init_metrics['auc']:.4f} | init_val_acc={init_metrics['accuracy']:.4f}"
        )
    else:
        best_score = float("-inf")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stagewise": {
                "init_checkpoint": str(init_checkpoint),
                "trainable_prefixes": trainable_prefixes,
            },
        },
        best_path,
    )

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")
        current_lr = get_current_lr(optimizer)
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip_norm=gradient_clip_norm,
            augmentation=augmentation_enabled,
        )
        record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "lr": current_lr}

        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            record.update(
                {
                    "val_loss": val_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
            )
            print(
                f"lr={current_lr:.6g} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )
            current_score = score_for_model_selection(val_metrics)
        else:
            print(f"lr={current_lr:.6g} | train_loss={train_loss:.4f}")
            current_score = -train_loss

        history.append(record)
        if current_score > best_score:
            best_score = current_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stagewise": {
                        "init_checkpoint": str(init_checkpoint),
                        "trainable_prefixes": trainable_prefixes,
                    },
                },
                best_path,
            )
            print(f"Saved best model to: {best_path}")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {early_stopping_patience} epochs)"
            )
            break

        if scheduler is not None:
            scheduler.step()

    save_json({"history": history}, output_dir / "history.json")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary: dict[str, Any] = {
        "model_selection": "best_val_accuracy_then_auc",
        "stagewise": {
            "init_checkpoint": str(init_checkpoint),
            "trainable_prefixes": trainable_prefixes,
            "matched_trainable_prefixes": matched_prefixes,
            "trainable_param_count": trainable_param_count,
            "frozen_param_count": frozen_param_count,
        },
    }
    if init_metrics is not None:
        summary["init_val"] = init_metrics
    if val_loader is not None:
        _, final_val_metrics = evaluate(model, val_loader, criterion, device)
        summary["best_val"] = final_val_metrics
    if test_loader is not None:
        _, final_test_metrics = evaluate(model, test_loader, criterion, device)
        summary["test"] = final_test_metrics

    total_seconds = time.perf_counter() - run_start_time
    peak_vram_mb = float("nan")
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    summary["runtime"] = {
        "total_seconds": total_seconds,
        "peak_vram_mb": peak_vram_mb,
    }

    save_json(summary, output_dir / "summary.json")
    print("\nTraining finished.")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    if "init_val" in summary:
        print(f"init_val_auc={summary['init_val'].get('auc', float('nan')):.6f}")
        print(f"init_val_f1={summary['init_val'].get('f1', float('nan')):.6f}")
        print(f"init_val_accuracy={summary['init_val'].get('accuracy', float('nan')):.6f}")
    if "best_val" in summary:
        print(f"best_val_auc={summary['best_val'].get('auc', float('nan')):.6f}")
        print(f"best_val_f1={summary['best_val'].get('f1', float('nan')):.6f}")
        print(f"best_val_accuracy={summary['best_val'].get('accuracy', float('nan')):.6f}")
    print(f"peak_vram_mb={peak_vram_mb:.1f}")
    print(f"total_seconds={total_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
