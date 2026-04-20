from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from paper_repro.common import (
    build_dataloaders,
    build_loss,
    build_optimizer,
    build_scheduler,
    detect_device,
    ensure_output_dir,
    format_float,
    load_config_with_inheritance,
    metric_score,
    save_history,
    save_json,
    set_seed,
)
from paper_repro.models import build_c3_expert_model, build_paper_model
from src.utils import compute_metrics


VIEW_NAMES = ("axial", "coronal", "sagittal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper reproduction model for ankle CT.")
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def compute_supervised_loss(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mixup_targets: tuple[torch.Tensor, torch.Tensor, float] | None = None,
) -> torch.Tensor:
    if mixup_targets is None:
        return criterion(logits, labels)
    labels_a, labels_b, lam = mixup_targets
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
    gradient_clip_norm: float | None,
    max_batches: int | None,
    mixup_alpha: float | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    total_loss = 0.0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp and is_train and device.type == "cuda")

    iterator = tqdm(loader, leave=False)
    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        batch_size = labels.shape[0]

        mixup_targets = None
        if is_train and mixup_alpha is not None and mixup_alpha > 0 and batch_size > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(batch_size, device=device)
            images = lam * images + (1.0 - lam) * images[permutation]
            mixup_targets = (labels, labels[permutation], lam)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(images)
            loss = compute_supervised_loss(criterion, logits, labels, mixup_targets=mixup_targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = probs.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        iterator.set_description(f"{'train' if is_train else 'eval'} loss={loss.item():.4f}")

    metrics = compute_metrics(all_labels, all_preds, all_probs) if all_labels else {
        "accuracy": 0.0,
        "auc": 0.0,
        "f1": 0.0,
        "specificity": 0.0,
        "sensitivity": 0.0,
    }
    average_loss = total_loss / max(1, total_samples)
    return average_loss, metrics


def collect_predictions(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    labels_all: list[int] = []
    preds_all: list[int] = []
    probs_all: list[list[float]] = []
    iterator = tqdm(loader, leave=False)
    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                probs = torch.softmax(model(images), dim=1)
        preds = probs.argmax(dim=1)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
    metrics = compute_metrics(labels_all, preds_all, probs_all) if labels_all else {
        "accuracy": 0.0,
        "auc": 0.0,
        "f1": 0.0,
        "specificity": 0.0,
        "sensitivity": 0.0,
    }
    return {
        "labels": labels_all,
        "preds": preds_all,
        "probs": probs_all,
        "metrics": metrics,
    }


def maybe_apply_training_stage(
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    epoch: int,
) -> str | None:
    family = str(config["model"]["family"]).lower()
    if family != "d4_hybrid_25d_3d" or not hasattr(model, "set_training_stage"):
        return None

    stages = list(config["train"].get("stages", []))
    if not stages:
        return None

    stage_cfg = stages[-1]
    for candidate in stages:
        if epoch <= int(candidate.get("until_epoch", epoch)):
            stage_cfg = candidate
            break
    stage_name = str(stage_cfg.get("mode", "full"))
    model.set_training_stage(
        stage_name,
        unfreeze_last_blocks=int(stage_cfg.get("unfreeze_last_blocks", 2)),
    )
    return stage_name


def fit_model(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    output_dir: Path,
    train_loader,
    val_loader,
    test_loader,
    train_df,
    device: torch.device,
    summary_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = build_loss(config, train_df, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    train_cfg = config["train"]
    epochs = int(train_cfg["epochs"])
    amp = bool(train_cfg.get("amp", True))
    patience = int(train_cfg.get("early_stopping_patience", 3))
    gradient_clip_norm = train_cfg.get("gradient_clip_norm")
    gradient_clip_norm = float(gradient_clip_norm) if gradient_clip_norm is not None else None
    max_train_batches = train_cfg.get("max_train_batches")
    max_val_batches = train_cfg.get("max_val_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    mixup_alpha = train_cfg.get("mixup_alpha")
    mixup_alpha = float(mixup_alpha) if mixup_alpha is not None else None

    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_loss = float("inf")
    bad_epochs = 0
    peak_vram_mb = 0.0
    started = time.time()
    previous_stage_name: str | None = None

    for epoch in range(1, epochs + 1):
        stage_name = maybe_apply_training_stage(config=config, model=model, epoch=epoch)
        if stage_name is not None and stage_name != previous_stage_name:
            bad_epochs = 0
            previous_stage_name = stage_name
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=amp,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=max_train_batches,
            mixup_alpha=mixup_alpha,
        )
        val_loss = None
        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    amp=amp,
                    gradient_clip_norm=None,
                    max_batches=max_val_batches,
                )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss if val_loss is not None else train_loss)
        else:
            scheduler.step()

        if device.type == "cuda":
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated(device) / (1024 * 1024))

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
        }
        if stage_name is not None:
            record["stage"] = stage_name
        if val_metrics is not None and val_loss is not None:
            record["val_loss"] = val_loss
            record["val"] = val_metrics
        history.append(record)
        save_history(history, output_dir)

        improved = False
        if val_metrics is not None:
            score = metric_score(val_metrics)
            if score > best_score:
                improved = True
                best_score = score
                best_epoch = epoch
                best_loss = float(val_loss)
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            if train_loss < best_loss:
                improved = True
                best_epoch = epoch
                best_loss = train_loss
                best_train_metrics = train_metrics
                torch.save(model.state_dict(), output_dir / "best.pt")

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={format_float(train_metrics.get('accuracy', 0.0))} "
            f"val_acc={format_float((val_metrics or {}).get('accuracy', 0.0))}"
        )

        if improved:
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} after {bad_epochs} non-improving epochs.")
                break

    best_path = output_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    if val_loader is not None:
        best_val_metrics = collect_predictions(
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            max_batches=max_val_batches,
        )["metrics"]
    if test_loader is not None:
        best_test_metrics = collect_predictions(
            model=model,
            loader=test_loader,
            device=device,
            amp=amp,
            max_batches=max_val_batches,
        )["metrics"]

    summary = {
        "paper": config["paper"],
        "config_path": None,
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "best_train": best_train_metrics,
        "test": best_test_metrics,
        "runtime": {
            "total_seconds": time.time() - started,
            "peak_vram_mb": peak_vram_mb,
            "device": str(device),
        },
        "notes": config.get("notes", []),
        "model_family": config["model"]["family"],
    }
    if summary_overrides:
        for key, value in summary_overrides.items():
            summary[key] = value

    save_json(summary, output_dir / "summary.json")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    if best_val_metrics is not None:
        print(f"best_val_accuracy={best_val_metrics.get('accuracy', float('nan')):.6f}")
        print(f"best_val_auc={best_val_metrics.get('auc', float('nan')):.6f}")
        print(f"best_val_f1={best_val_metrics.get('f1', float('nan')):.6f}")
    return summary


def derive_c3_weights(branch_summaries: list[dict[str, Any]], metric_name: str) -> list[float]:
    raw_weights = []
    for summary in branch_summaries:
        metrics = summary.get("best_val") or {}
        score = float(metrics.get(metric_name, 0.0))
        if not np.isfinite(score) or score <= 0:
            score = float(metrics.get("accuracy", 0.0))
        if not np.isfinite(score) or score <= 0:
            score = 1e-6
        raw_weights.append(score)
    total = sum(raw_weights)
    return [value / total for value in raw_weights]


def evaluate_fused_branches(
    *,
    models: list[torch.nn.Module],
    weights: list[float],
    loader,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    branch_predictions = [
        collect_predictions(
            model=model,
            loader=loader,
            device=device,
            amp=amp,
            max_batches=max_batches,
        )
        for model in models
    ]
    labels = branch_predictions[0]["labels"] if branch_predictions else []
    positive_probs = []
    for sample_index in range(len(labels)):
        fused_positive = 0.0
        for weight, prediction in zip(weights, branch_predictions, strict=True):
            fused_positive += weight * float(prediction["probs"][sample_index][1])
        positive_probs.append(fused_positive)
    fused_probs = [[1.0 - score, score] for score in positive_probs]
    fused_preds = [int(score >= 0.5) for score in positive_probs]
    metrics = compute_metrics(labels, fused_preds, fused_probs) if labels else {
        "accuracy": 0.0,
        "auc": 0.0,
        "f1": 0.0,
        "specificity": 0.0,
        "sensitivity": 0.0,
    }
    return {
        "labels": labels,
        "preds": fused_preds,
        "probs": fused_probs,
        "metrics": metrics,
    }


def run_c3_offline_decision_fusion(
    *,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)
    train_cfg = config["train"]
    amp = bool(train_cfg.get("amp", True))
    max_val_batches = train_cfg.get("max_val_batches")
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None

    started = time.time()
    branch_summaries: list[dict[str, Any]] = []
    branch_models: list[torch.nn.Module] = []
    peak_vram_mb = 0.0
    branches_dir = output_dir / "branches"
    branches_dir.mkdir(parents=True, exist_ok=True)

    for view_index, view_name in enumerate(VIEW_NAMES):
        branch_model = build_c3_expert_model(config, view_index).to(device)
        branch_output_dir = branches_dir / view_name
        branch_paper = dict(config["paper"])
        branch_paper["id"] = f"{config['paper']['id']}-{view_name.upper()}"
        branch_paper["title"] = f"{config['paper']['title']} ({view_name})"
        branch_summary = fit_model(
            model=branch_model,
            config=config,
            output_dir=branch_output_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_df=train_df,
            device=device,
            summary_overrides={
                "paper": branch_paper,
                "config_path": str(Path(config["config_path"]).resolve()),
                "model_family": "c3_single_expert",
                "notes": [
                    *config.get("notes", []),
                    f"Single-view expert for offline decision fusion ({view_name}).",
                ],
            },
        )
        branch_summaries.append(branch_summary)
        branch_models.append(branch_model)
        peak_vram_mb = max(peak_vram_mb, float(branch_summary["runtime"]["peak_vram_mb"]))

    metric_name = str(config["model"].get("fusion_weight_metric", "auc"))
    fusion_weights = derive_c3_weights(branch_summaries, metric_name=metric_name)
    fused_val = None
    fused_test = None
    if val_loader is not None:
        fused_val = evaluate_fused_branches(
            models=branch_models,
            weights=fusion_weights,
            loader=val_loader,
            device=device,
            amp=amp,
            max_batches=max_val_batches,
        )["metrics"]
    if test_loader is not None:
        fused_test = evaluate_fused_branches(
            models=branch_models,
            weights=fusion_weights,
            loader=test_loader,
            device=device,
            amp=amp,
            max_batches=max_val_batches,
        )["metrics"]

    summary = {
        "paper": config["paper"],
        "config_path": str(Path(config["config_path"]).resolve()),
        "best_epoch": None,
        "best_val": fused_val,
        "best_train": None,
        "test": fused_test,
        "runtime": {
            "total_seconds": time.time() - started,
            "peak_vram_mb": peak_vram_mb,
            "device": str(device),
        },
        "notes": [
            *config.get("notes", []),
            "Experts are trained independently and fused offline using validation-derived heuristic weights.",
        ],
        "model_family": config["model"]["family"],
        "workflow": "offline_decision_fusion",
        "fusion_weight_metric": metric_name,
        "fusion_weights": {
            view_name: weight for view_name, weight in zip(VIEW_NAMES, fusion_weights, strict=True)
        },
        "branch_summaries": [
            {
                "view": view_name,
                "output_dir": str((branches_dir / view_name).resolve()),
                "best_val": branch_summary.get("best_val"),
                "test": branch_summary.get("test"),
            }
            for view_name, branch_summary in zip(VIEW_NAMES, branch_summaries, strict=True)
        ],
    }
    save_json(summary, output_dir / "summary.json")
    save_json(
        {
            "workflow": "offline_decision_fusion",
            "fusion_weight_metric": metric_name,
            "fusion_weights": summary["fusion_weights"],
            "branch_summaries": summary["branch_summaries"],
        },
        output_dir / "history.json",
    )
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    if fused_val is not None:
        print(f"best_val_accuracy={fused_val.get('accuracy', float('nan')):.6f}")
        print(f"best_val_auc={fused_val.get('auc', float('nan')):.6f}")
        print(f"best_val_f1={fused_val.get('f1', float('nan')):.6f}")
    return summary


def main() -> None:
    args = parse_args()
    config = load_config_with_inheritance(args.config)
    config["config_path"] = str(Path(args.config).resolve())
    set_seed(int(config["seed"]))
    output_dir = ensure_output_dir(config)
    device = detect_device(config)

    family = str(config["model"]["family"]).lower()
    workflow = str(config["model"].get("workflow", "standard")).lower()
    if family == "c3_decision_fusion" and workflow == "offline_decision_fusion":
        run_c3_offline_decision_fusion(config=config, output_dir=output_dir, device=device)
        return

    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)
    model = build_paper_model(config).to(device)
    fit_model(
        model=model,
        config=config,
        output_dir=output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_df=train_df,
        device=device,
        summary_overrides={
            "config_path": str(Path(args.config).resolve()),
        },
    )


if __name__ == "__main__":
    main()
