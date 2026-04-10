"""
kfold_ensemble.py — 5-Fold 交叉验证 + 3-Seed 模型集成
======================================================
一键运行完整的 K-Fold CV + Ensemble 评估：
  - 5-Fold 分层交叉验证（每个样本都会被测试到）
  - 每个 Fold 内用 3 个不同随机种子训练模型
  - 3 个模型的预测概率取平均（集成）
  - 在集成概率上做零漏诊阈值评估
  - 最终报告 5 个 Fold 的均值 ± 标准差

用法:
    .venv/Scripts/python.exe kfold_ensemble.py --config configs/kfold_ensemble.yaml

输出:
    runs/kfold_ensemble/
    ├── fold_0/seed_42/   (best.pt, summary.json)
    ├── fold_0/seed_123/
    ├── ...
    ├── fold_results.json
    └── final_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VIEW_COLUMNS, PatientCTDataset, canonicalize_view_columns
from src.utils import (
    choose_device,
    compute_class_weights,
    compute_metrics,
    load_config,
    save_json,
    set_seed,
)
from train import (
    apply_augmentation,
    build_model,
    build_scheduler,
    evaluate,
    get_current_lr,
    score_for_model_selection,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold CV + 3-Seed Ensemble evaluation.")
    parser.add_argument("--config", type=str, default="configs/kfold_ensemble.yaml")
    parser.add_argument("--resume-from-fold", type=int, default=0,
                        help="Resume from this fold index (0-based). Skips completed folds.")
    return parser.parse_args()


def build_fold_dataloaders(train_df, val_df, test_df, data_cfg):
    """Build DataLoaders for a single fold."""
    common_kwargs = {
        "base_dir": data_cfg["base_dir"],
        "image_size": data_cfg["image_size"],
        "num_slices_per_view": data_cfg["num_slices_per_view"],
        "trim_edge_slices": data_cfg.get("trim_edge_slices", 0),
        "hu_min": data_cfg["hu_min"],
        "hu_max": data_cfg["hu_max"],
    }

    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory = torch.cuda.is_available()

    train_dataset = PatientCTDataset(train_df, **common_kwargs)
    val_dataset = PatientCTDataset(val_df, **common_kwargs)
    test_dataset = PatientCTDataset(test_df, **common_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect (labels, positive_class_probs) from a dataloader."""
    model.eval()
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="inference", leave=False):
        images = batch["images"].to(device)
        labels = batch["label"]

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


def find_no_miss_threshold(labels, probs):
    """Find threshold = min positive probability (ensures FN=0 on val)."""
    positive_probs = probs[labels == 1]
    if len(positive_probs) == 0:
        return 0.5
    return float(positive_probs.min())


def evaluate_at_threshold(labels, probs, threshold):
    """Evaluate binary metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n = len(labels)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def train_single_model(config, train_df, val_df, seed, output_dir, device):
    """Train a single model and return validation and test predictions."""
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best.pt"

    # Check if already trained (for resume support)
    if best_path.exists() and (output_dir / "summary.json").exists():
        print(f"    [skip] Already trained: {output_dir}")
        # Load and return the model
        model = build_model(config).to(device)
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    # Build dataloaders
    data_cfg = config["data"]
    common_kwargs = {
        "base_dir": data_cfg["base_dir"],
        "image_size": data_cfg["image_size"],
        "num_slices_per_view": data_cfg["num_slices_per_view"],
        "trim_edge_slices": data_cfg.get("trim_edge_slices", 0),
        "hu_min": data_cfg["hu_min"],
        "hu_max": data_cfg["hu_max"],
    }
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory = torch.cuda.is_available()

    train_dataset = PatientCTDataset(train_df, **common_kwargs)
    val_dataset = PatientCTDataset(val_df, **common_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    # Build model
    model = build_model(config).to(device)

    # Loss & optimizer
    train_cfg = config["train"]
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if train_cfg["class_weight"]:
        class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_scheduler(config, optimizer)

    gradient_clip_norm = train_cfg.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    augmentation_enabled = bool(train_cfg.get("augmentation", False))

    # Training loop
    best_score = -1.0
    start_time = time.perf_counter()

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip_norm=gradient_clip_norm,
            augmentation=augmentation_enabled,
        )

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        current_score = score_for_model_selection(val_metrics)
        current_lr = get_current_lr(optimizer)

        print(f"      epoch {epoch}/{train_cfg['epochs']} | "
              f"lr={current_lr:.6g} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_auc={val_metrics['auc']:.4f}")

        if current_score > best_score:
            best_score = current_score
            torch.save({"model_state_dict": model.state_dict(), "config": config}, best_path)

        if scheduler is not None:
            scheduler.step()

    elapsed = time.perf_counter() - start_time
    peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0

    # Load best model
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save summary
    _, final_val = evaluate(model, val_loader, criterion, device)
    save_json({
        "best_val": final_val,
        "runtime": {"total_seconds": elapsed, "peak_vram_mb": peak_vram},
    }, output_dir / "summary.json")

    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    kfold_cfg = config.get("kfold", {})
    n_splits = kfold_cfg.get("n_splits", 5)
    seeds = kfold_cfg.get("seeds", [42, 123, 456])
    inner_val_ratio = kfold_cfg.get("inner_val_ratio", 0.2)
    resume_fold = args.resume_from_fold

    output_base = Path(config["output_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    device = choose_device(config["train"]["device"])
    print(f"Using device: {device}")
    print(f"K-Fold config: {n_splits} folds × {len(seeds)} seeds = {n_splits * len(seeds)} models total")
    print(f"Seeds: {seeds}")
    print(f"Inner validation ratio: {inner_val_ratio}")
    print()

    # Load full dataset (ignore existing split column)
    data_cfg = config["data"]
    df = canonicalize_view_columns(pd.read_csv(data_cfg["csv_path"]))
    df["label"] = df["label"].astype(int)
    labels = df["label"].values
    print(f"Total samples: {len(df)} (positive: {(labels == 1).sum()}, negative: {(labels == 0).sum()})")
    print()

    # Setup K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_fold_results = []
    grand_start = time.perf_counter()

    for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(df, labels)):
        if fold_idx < resume_fold:
            print(f"[Fold {fold_idx}] skipped (resume_from_fold={resume_fold})")
            # Try to load existing results
            fold_result_path = output_base / f"fold_{fold_idx}" / "fold_result.json"
            if fold_result_path.exists():
                with open(fold_result_path) as f:
                    all_fold_results.append(json.load(f))
            continue

        fold_start = time.perf_counter()
        print(f"{'='*70}")
        print(f"FOLD {fold_idx}/{n_splits - 1}")
        print(f"{'='*70}")

        # Split train_val into train and val
        train_val_df = df.iloc[train_val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=inner_val_ratio,
            stratify=train_val_df["label"],
            random_state=42,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        print(f"  train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
        print(f"  train labels: 0={int((train_df['label']==0).sum())} 1={int((train_df['label']==1).sum())}")
        print(f"  test  labels: 0={int((test_df['label']==0).sum())} 1={int((test_df['label']==1).sum())}")
        print()

        # Build test dataloader (shared across seeds)
        test_dataset = PatientCTDataset(
            test_df,
            base_dir=data_cfg["base_dir"],
            image_size=data_cfg["image_size"],
            num_slices_per_view=data_cfg["num_slices_per_view"],
            trim_edge_slices=data_cfg.get("trim_edge_slices", 0),
            hu_min=data_cfg["hu_min"],
            hu_max=data_cfg["hu_max"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        # Train models with different seeds and collect predictions
        seed_val_probs = {}
        seed_test_probs = {}
        seed_results = {}

        for seed in seeds:
            seed_dir = output_base / f"fold_{fold_idx}" / f"seed_{seed}"
            print(f"  [Fold {fold_idx} / Seed {seed}]")

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Train
            model = train_single_model(
                config, train_df, val_df, seed, seed_dir, device,
            )

            # Collect val predictions (for threshold)
            val_dataset = PatientCTDataset(
                val_df,
                base_dir=data_cfg["base_dir"],
                image_size=data_cfg["image_size"],
                num_slices_per_view=data_cfg["num_slices_per_view"],
                trim_edge_slices=data_cfg.get("trim_edge_slices", 0),
                hu_min=data_cfg["hu_min"],
                hu_max=data_cfg["hu_max"],
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=data_cfg["batch_size"],
                shuffle=False,
                num_workers=data_cfg["num_workers"],
                pin_memory=torch.cuda.is_available(),
            )

            val_labels, val_probs = collect_predictions(model, val_loader, device)
            test_labels, test_probs = collect_predictions(model, test_loader, device)

            seed_val_probs[seed] = val_probs
            seed_test_probs[seed] = test_probs

            # Individual seed evaluation
            threshold = find_no_miss_threshold(val_labels, val_probs)
            val_eval = evaluate_at_threshold(val_labels, val_probs, threshold)
            test_eval = evaluate_at_threshold(test_labels, test_probs, threshold)

            seed_results[seed] = {
                "no_miss_threshold": threshold,
                "val": val_eval,
                "test": test_eval,
            }

            print(f"    single-model: val_acc={val_eval['accuracy']:.3f} "
                  f"val_spe={val_eval['specificity']:.3f} | "
                  f"test_acc={test_eval['accuracy']:.3f} "
                  f"test_spe={test_eval['specificity']:.3f}")

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # ---- Ensemble evaluation ----
        print(f"\n  [Fold {fold_idx} / Ensemble ({len(seeds)} models)]")

        # Average val probs across seeds
        ensemble_val_probs = np.mean([seed_val_probs[s] for s in seeds], axis=0)
        # Average test probs across seeds
        ensemble_test_probs = np.mean([seed_test_probs[s] for s in seeds], axis=0)

        # Threshold on ensemble val probs
        ensemble_threshold = find_no_miss_threshold(val_labels, ensemble_val_probs)
        ensemble_val_eval = evaluate_at_threshold(val_labels, ensemble_val_probs, ensemble_threshold)
        ensemble_test_eval = evaluate_at_threshold(test_labels, ensemble_test_probs, ensemble_threshold)

        # Also compute AUC on test set for the ensemble
        from sklearn.metrics import roc_auc_score
        test_unique = np.unique(test_labels)
        if len(test_unique) == 2:
            ensemble_test_auc = float(roc_auc_score(test_labels, ensemble_test_probs))
            ensemble_val_auc = float(roc_auc_score(val_labels, ensemble_val_probs))
        else:
            ensemble_test_auc = float("nan")
            ensemble_val_auc = float("nan")

        fold_elapsed = time.perf_counter() - fold_start

        print(f"    ensemble: val_acc={ensemble_val_eval['accuracy']:.3f} "
              f"val_spe={ensemble_val_eval['specificity']:.3f} "
              f"val_auc={ensemble_val_auc:.3f}")
        print(f"    ensemble: test_acc={ensemble_test_eval['accuracy']:.3f} "
              f"test_spe={ensemble_test_eval['specificity']:.3f} "
              f"test_sen={ensemble_test_eval['sensitivity']:.3f} "
              f"test_auc={ensemble_test_auc:.3f}")
        print(f"    fold time: {fold_elapsed/60:.1f} min")
        print()

        fold_result = {
            "fold": fold_idx,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "test_positive": int((test_df["label"] == 1).sum()),
            "test_negative": int((test_df["label"] == 0).sum()),
            "individual_seeds": {str(s): seed_results[s] for s in seeds},
            "ensemble": {
                "no_miss_threshold": ensemble_threshold,
                "val": {**ensemble_val_eval, "auc": ensemble_val_auc},
                "test": {**ensemble_test_eval, "auc": ensemble_test_auc},
            },
            "elapsed_seconds": fold_elapsed,
        }
        all_fold_results.append(fold_result)

        # Save intermediate fold result
        save_json(fold_result, output_base / f"fold_{fold_idx}" / "fold_result.json")

    # ======================================================================
    # Final aggregation
    # ======================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    # Ensemble metrics across folds
    ens_val_accs = [r["ensemble"]["val"]["accuracy"] for r in all_fold_results]
    ens_val_spes = [r["ensemble"]["val"]["specificity"] for r in all_fold_results]
    ens_val_aucs = [r["ensemble"]["val"].get("auc", float("nan")) for r in all_fold_results]
    ens_test_accs = [r["ensemble"]["test"]["accuracy"] for r in all_fold_results]
    ens_test_spes = [r["ensemble"]["test"]["specificity"] for r in all_fold_results]
    ens_test_sens = [r["ensemble"]["test"]["sensitivity"] for r in all_fold_results]
    ens_test_aucs = [r["ensemble"]["test"].get("auc", float("nan")) for r in all_fold_results]
    ens_test_fns = [r["ensemble"]["test"]["fn"] for r in all_fold_results]

    # Individual model metrics (average of all single models across folds)
    ind_val_accs = []
    ind_test_accs = []
    ind_test_sens = []
    for r in all_fold_results:
        for s, sr in r["individual_seeds"].items():
            ind_val_accs.append(sr["val"]["accuracy"])
            ind_test_accs.append(sr["test"]["accuracy"])
            ind_test_sens.append(sr["test"]["sensitivity"])

    def fmt_mean_std(values):
        return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

    print(f"\n--- Ensemble ({len(seeds)}-model average) ---")
    print(f"  Val  no_miss_acc : {fmt_mean_std(ens_val_accs)}")
    print(f"  Val  specificity : {fmt_mean_std(ens_val_spes)}")
    print(f"  Val  AUC         : {fmt_mean_std(ens_val_aucs)}")
    print(f"  Test accuracy    : {fmt_mean_std(ens_test_accs)}")
    print(f"  Test sensitivity : {fmt_mean_std(ens_test_sens)}")
    print(f"  Test specificity : {fmt_mean_std(ens_test_spes)}")
    print(f"  Test AUC         : {fmt_mean_std(ens_test_aucs)}")
    print(f"  Test total FN    : {sum(ens_test_fns)} across {n_splits} folds")

    print(f"\n--- Individual models (all {n_splits * len(seeds)} single-seed runs) ---")
    print(f"  Val  no_miss_acc : {fmt_mean_std(ind_val_accs)}")
    print(f"  Test accuracy    : {fmt_mean_std(ind_test_accs)}")
    print(f"  Test sensitivity : {fmt_mean_std(ind_test_sens)}")

    print(f"\n--- Per-fold detail ---")
    print(f"  {'Fold':>4} | {'Ens Val Acc':>11} | {'Ens Test Acc':>12} | {'Test Sen':>8} | {'Test Spe':>8} | {'Test AUC':>8} | {'FN':>2}")
    for r in all_fold_results:
        e = r["ensemble"]
        print(f"  {r['fold']:>4} | "
              f"{e['val']['accuracy']:>11.3f} | "
              f"{e['test']['accuracy']:>12.3f} | "
              f"{e['test']['sensitivity']:>8.3f} | "
              f"{e['test']['specificity']:>8.3f} | "
              f"{e['test'].get('auc', float('nan')):>8.3f} | "
              f"{e['test']['fn']:>2}")

    total_elapsed = time.perf_counter() - grand_start

    # Save final summary
    final_summary = {
        "n_folds": n_splits,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "total_samples": len(df),
        "ensemble_summary": {
            "val_no_miss_acc": {"mean": float(np.mean(ens_val_accs)), "std": float(np.std(ens_val_accs))},
            "val_specificity": {"mean": float(np.mean(ens_val_spes)), "std": float(np.std(ens_val_spes))},
            "val_auc": {"mean": float(np.nanmean(ens_val_aucs)), "std": float(np.nanstd(ens_val_aucs))},
            "test_accuracy": {"mean": float(np.mean(ens_test_accs)), "std": float(np.std(ens_test_accs))},
            "test_sensitivity": {"mean": float(np.mean(ens_test_sens)), "std": float(np.std(ens_test_sens))},
            "test_specificity": {"mean": float(np.mean(ens_test_spes)), "std": float(np.std(ens_test_spes))},
            "test_auc": {"mean": float(np.nanmean(ens_test_aucs)), "std": float(np.nanstd(ens_test_aucs))},
            "test_total_fn": int(sum(ens_test_fns)),
        },
        "individual_summary": {
            "val_no_miss_acc": {"mean": float(np.mean(ind_val_accs)), "std": float(np.std(ind_val_accs))},
            "test_accuracy": {"mean": float(np.mean(ind_test_accs)), "std": float(np.std(ind_test_accs))},
            "test_sensitivity": {"mean": float(np.mean(ind_test_sens)), "std": float(np.std(ind_test_sens))},
        },
        "fold_results": all_fold_results,
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_hours": total_elapsed / 3600,
    }
    save_json(final_summary, output_base / "final_summary.json")
    save_json(all_fold_results, output_base / "fold_results.json")

    print(f"\nTotal time: {total_elapsed/3600:.1f} hours")
    print(f"Results saved to: {output_base / 'final_summary.json'}")


if __name__ == "__main__":
    main()
