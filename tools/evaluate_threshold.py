"""
evaluate_threshold.py — 阈值评估工具
====================================
对已训练好的 best.pt 做阈值后处理评估，找到验证集零漏诊阈值，
并在该阈值下评估验证集和测试集的表现。

用法：
    python tools/evaluate_threshold.py --run_dir runs/autoresearch_proxy --config configs/autoresearch_proxy.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import VIEW_COLUMNS, PatientCTDataset, canonicalize_view_columns
from src.model import (
    MultiViewAttentionClassifier,
    MultiViewCTClassifier,
    MultiViewDecisionFusionClassifier,
)
from src.utils import choose_device, load_config
from train import build_dataloaders, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate threshold for zero-miss diagnosis.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiment run directory (contains best.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML used for training")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect all predictions (labels, probabilities) from a data loader."""
    model.eval()
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="inference", leave=False):
        images = batch["images"].to(device)
        labels = batch["label"]

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(positive)

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


def find_no_miss_threshold(labels, probs):
    """Find the highest threshold where FN=0 on the given data.

    This equals the minimum positive-class probability among all positive samples.
    """
    positive_probs = probs[labels == 1]
    if len(positive_probs) == 0:
        return 0.5  # No positives, use default
    return float(positive_probs.min())


def evaluate_at_threshold(labels, probs, threshold):
    """Evaluate binary classification metrics at a given threshold."""
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    best_path = run_dir / "best.pt"

    if not best_path.exists():
        print(f"ERROR: {best_path} does not exist. Train a model first.")
        sys.exit(1)

    # Load config and build components
    config = load_config(args.config)
    device = choose_device(config["train"]["device"])
    print(f"Using device: {device}")

    # Build data loaders
    _, val_loader, test_loader, _ = build_dataloaders(config)

    # Build model and load weights
    model = build_model(config).to(device)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights from: {best_path}")

    results = {
        "run_dir": str(run_dir),
        "config": args.config,
        "fusion_type": config["model"].get("fusion_type", "feature"),
    }

    # ---- Validation set ----
    if val_loader is not None:
        val_labels, val_probs = collect_predictions(model, val_loader, device)
        threshold = find_no_miss_threshold(val_labels, val_probs)
        val_metrics = evaluate_at_threshold(val_labels, val_probs, threshold)

        results["no_miss_threshold"] = threshold
        results["val"] = val_metrics
        results["val_positive_prob_summary"] = {
            "min": float(val_probs[val_labels == 1].min()) if (val_labels == 1).any() else None,
            "median": float(np.median(val_probs[val_labels == 1])) if (val_labels == 1).any() else None,
            "max": float(val_probs[val_labels == 1].max()) if (val_labels == 1).any() else None,
        }

        print(f"\n=== Validation Set (threshold={threshold:.6f}) ===")
        print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
        print(f"  F1:          {val_metrics['f1']:.4f}")
        print(f"  Sensitivity: {val_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {val_metrics['specificity']:.4f}")
        print(f"  FN:          {val_metrics['fn']}")
    else:
        print("No validation set available.")

    # ---- Test set ----
    if test_loader is not None:
        test_labels, test_probs = collect_predictions(model, test_loader, device)
        test_metrics = evaluate_at_threshold(test_labels, test_probs, threshold)

        results["test"] = test_metrics

        print(f"\n=== Test Set (threshold={threshold:.6f}) ===")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  F1:          {test_metrics['f1']:.4f}")
        print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
        print(f"  FN:          {test_metrics['fn']}")
    else:
        print("No test set available.")

    # Save results
    output_path = run_dir / "threshold_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary line for easy parsing
    if "no_miss_threshold" in results:
        val = results.get("val", {})
        print(f"\nno_miss_threshold={results['no_miss_threshold']:.6f}")
        print(f"no_miss_val_acc={val.get('accuracy', 0):.6f}")
        print(f"no_miss_val_spe={val.get('specificity', 0):.6f}")


if __name__ == "__main__":
    main()
