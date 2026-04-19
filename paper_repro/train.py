from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

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
from paper_repro.models import build_paper_model
from src.utils import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper reproduction model for ankle CT.")
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


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

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

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


def main() -> None:
    args = parse_args()
    config = load_config_with_inheritance(args.config)
    set_seed(int(config["seed"]))
    output_dir = ensure_output_dir(config)
    device = detect_device(config)

    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)
    model = build_paper_model(config).to(device)
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

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=amp,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=max_train_batches,
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
        with torch.no_grad():
            _, best_val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                amp=amp,
                gradient_clip_norm=None,
                max_batches=max_val_batches,
            )
    if test_loader is not None:
        with torch.no_grad():
            _, best_test_metrics = run_epoch(
                model=model,
                loader=test_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                amp=amp,
                gradient_clip_norm=None,
                max_batches=max_val_batches,
            )

    summary = {
        "paper": config["paper"],
        "config_path": str(Path(args.config).resolve()),
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
    save_json(summary, output_dir / "summary.json")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    if best_val_metrics is not None:
        print(f"best_val_accuracy={best_val_metrics.get('accuracy', float('nan')):.6f}")
        print(f"best_val_auc={best_val_metrics.get('auc', float('nan')):.6f}")
        print(f"best_val_f1={best_val_metrics.get('f1', float('nan')):.6f}")


if __name__ == "__main__":
    main()

