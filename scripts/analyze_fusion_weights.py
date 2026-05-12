from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import VIEW_COLUMNS
from src.utils import compute_metrics, load_config, save_json, set_seed
from train import build_dataloaders, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect sample-level decision-fusion telemetry for a checkpoint, "
            "including fusion weights, per-view margins, and correctness alignment."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the training config or Optuna trial config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to output_dir/best.pt.")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path. Defaults to <output_dir>/fusion_weight_analysis.json.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional small-batch smoke limit for debugging.",
    )
    return parser.parse_args()


def apply_runtime_env(runtime_env: dict[str, str]) -> None:
    for key, value in runtime_env.items():
        os.environ[str(key)] = str(value)


def to_float_dict(view_names: list[str], values: list[float]) -> dict[str, float]:
    return {
        view_name: float(value)
        for view_name, value in zip(view_names, values)
    }


def to_int_dict(view_names: list[str], values: list[int]) -> dict[str, int]:
    return {
        view_name: int(value)
        for view_name, value in zip(view_names, values)
    }


def to_bool_dict(view_names: list[str], values: list[bool]) -> dict[str, bool]:
    return {
        view_name: bool(value)
        for view_name, value in zip(view_names, values)
    }


def top_indices(values: list[float], atol: float = 1e-8) -> list[int]:
    max_value = max(values)
    return [index for index, value in enumerate(values) if abs(value - max_value) <= atol]


def hit_rate(matches: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return float(matches / total)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return float("nan")
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if np.isclose(x_array.std(), 0.0) or np.isclose(y_array.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def apply_active_view_mask(
    fusion_weights: torch.Tensor,
    active_view_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if active_view_mask is None:
        return fusion_weights, None
    if active_view_mask.ndim == 1:
        active_view_mask = active_view_mask.unsqueeze(0)
    if active_view_mask.shape[0] == 1 and fusion_weights.shape[0] != 1:
        active_view_mask = active_view_mask.expand(fusion_weights.shape[0], -1)
    if active_view_mask.shape != fusion_weights.shape:
        raise ValueError(
            "active_view_mask must have shape (3,) or match fusion_weights batch shape."
        )
    mask = active_view_mask.to(device=fusion_weights.device, dtype=fusion_weights.dtype)
    masked_weights = fusion_weights * mask
    normalizer = masked_weights.sum(dim=1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError("active_view_mask must keep at least one view per sample.")
    return masked_weights / normalizer, mask


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    runtime_env = {
        str(key): str(value)
        for key, value in dict(config.get("runtime_env", {})).items()
    }
    apply_runtime_env(runtime_env)

    set_seed(int(config["seed"]))
    train_loader, val_loader, test_loader, _ = build_dataloaders(config)
    del train_loader

    if args.split == "val":
        loader = val_loader
    else:
        loader = test_loader
    if loader is None:
        raise SystemExit(f"Requested split `{args.split}` is not available for {config_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    if not hasattr(model, "forward_with_decision_info"):
        raise SystemExit("This script only supports decision-fusion models exposing forward_with_decision_info().")

    output_dir = Path(config["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    view_names = [name.removesuffix("_dir") for name in VIEW_COLUMNS]
    full_labels: list[int] = []
    full_preds: list[int] = []
    full_probs: list[list[float]] = []
    per_view_labels: list[list[int]] = [[] for _ in view_names]
    per_view_preds: list[list[int]] = [[] for _ in view_names]
    per_view_probs: list[list[list[float]]] = [[] for _ in view_names]

    all_weights: list[float] = []
    all_pred_margins: list[float] = []
    all_true_margins: list[float] = []
    all_correctness: list[float] = []
    all_confidences: list[float] = []
    all_scaled_confidences: list[float] = []
    correct_view_weights: list[float] = []
    incorrect_view_weights: list[float] = []
    raw_weight_sums = [0.0 for _ in view_names]
    active_mask_records: list[list[float]] = []

    top_weight_hits_pred_margin = 0
    top_weight_hits_true_margin = 0
    top_weight_correct = 0
    mixed_correctness_top_weight_correct = 0
    mixed_correctness_samples = 0
    fusion_correct = 0

    top_weight_counts = [0 for _ in view_names]
    top_pred_margin_counts = [0 for _ in view_names]
    top_true_margin_counts = [0 for _ in view_names]
    weight_sums = [0.0 for _ in view_names]
    confidence_sums = [0.0 for _ in view_names]
    scaled_confidence_sums = [0.0 for _ in view_names]
    pred_margin_sums = [0.0 for _ in view_names]
    true_margin_sums = [0.0 for _ in view_names]

    sample_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"fusion-telemetry:{args.split}", leave=False), start=1):
            images = batch["images"].to(device)
            labels = batch["label"].to(device)
            patient_ids = batch["patient_id"]

            fused_logits, decision_info = model.forward_with_decision_info(images)
            view_logits = decision_info["view_logits"]  # (B, 3, 2)
            raw_fusion_weights = decision_info["fusion_weights"].squeeze(-1)  # (B, 3)
            fusion_weights, active_view_mask = apply_active_view_mask(
                raw_fusion_weights,
                decision_info.get("active_view_mask"),
            )
            if active_view_mask is not None:
                active_mask_records.extend(active_view_mask.cpu().tolist())
            confidences = decision_info["confidences"].squeeze(-1)  # (B, 3)
            scaled_confidences = decision_info["scaled_confidences"].squeeze(-1)  # (B, 3)

            fused_probs = torch.softmax(fused_logits, dim=1)
            fused_preds = torch.argmax(fused_probs, dim=1)
            fusion_correct_mask = fused_preds.eq(labels)

            view_probs = torch.softmax(view_logits, dim=-1)
            view_preds = torch.argmax(view_probs, dim=-1)  # (B, 3)
            view_correct = view_preds.eq(labels.unsqueeze(1))  # (B, 3)
            pred_margins = view_logits.max(dim=-1).values - view_logits.min(dim=-1).values

            label_indices = labels.view(-1, 1, 1).expand(-1, view_logits.shape[1], 1)
            wrong_label_indices = (1 - labels).view(-1, 1, 1).expand(-1, view_logits.shape[1], 1)
            true_class_logits = view_logits.gather(dim=-1, index=label_indices).squeeze(-1)
            other_class_logits = view_logits.gather(dim=-1, index=wrong_label_indices).squeeze(-1)
            true_margins = true_class_logits - other_class_logits

            full_labels.extend(labels.cpu().tolist())
            full_preds.extend(fused_preds.cpu().tolist())
            full_probs.extend(fused_probs.cpu().tolist())
            fusion_correct += int(fusion_correct_mask.sum().item())

            for view_index in range(len(view_names)):
                per_view_labels[view_index].extend(labels.cpu().tolist())
                per_view_preds[view_index].extend(view_preds[:, view_index].cpu().tolist())
                per_view_probs[view_index].extend(view_probs[:, view_index, :].cpu().tolist())

            batch_size = labels.shape[0]
            for sample_index in range(batch_size):
                label_value = int(labels[sample_index].item())
                fused_pred_value = int(fused_preds[sample_index].item())
                fused_prob_values = fused_probs[sample_index].cpu().tolist()
                fused_correct_value = bool(fusion_correct_mask[sample_index].item())

                weight_values = fusion_weights[sample_index].cpu().tolist()
                raw_weight_values = raw_fusion_weights[sample_index].cpu().tolist()
                active_mask_values = (
                    active_view_mask[sample_index].cpu().tolist()
                    if active_view_mask is not None
                    else None
                )
                confidence_values = confidences[sample_index].cpu().tolist()
                scaled_confidence_values = scaled_confidences[sample_index].cpu().tolist()
                pred_margin_values = pred_margins[sample_index].cpu().tolist()
                true_margin_values = true_margins[sample_index].cpu().tolist()
                view_pred_values = view_preds[sample_index].cpu().tolist()
                view_correct_values = [bool(value) for value in view_correct[sample_index].cpu().tolist()]
                view_abnormal_probs = view_probs[sample_index, :, 1].cpu().tolist()

                weight_argmax = int(np.argmax(np.asarray(weight_values, dtype=np.float64)))
                pred_margin_argmax = int(np.argmax(np.asarray(pred_margin_values, dtype=np.float64)))
                true_margin_argmax = int(np.argmax(np.asarray(true_margin_values, dtype=np.float64)))
                top_weight_candidates = top_indices(weight_values)
                top_pred_margin_candidates = top_indices(pred_margin_values)
                top_true_margin_candidates = top_indices(true_margin_values)
                correct_view_indices = [index for index, value in enumerate(view_correct_values) if value]

                weight_hit_pred_margin = bool(set(top_weight_candidates) & set(top_pred_margin_candidates))
                weight_hit_true_margin = bool(set(top_weight_candidates) & set(top_true_margin_candidates))
                top_weight_is_correct = bool(set(top_weight_candidates) & set(correct_view_indices))
                mixed_correctness = any(view_correct_values) and not all(view_correct_values)

                if weight_hit_pred_margin:
                    top_weight_hits_pred_margin += 1
                if weight_hit_true_margin:
                    top_weight_hits_true_margin += 1
                if top_weight_is_correct:
                    top_weight_correct += 1
                if mixed_correctness:
                    mixed_correctness_samples += 1
                    if top_weight_is_correct:
                        mixed_correctness_top_weight_correct += 1

                top_weight_counts[weight_argmax] += 1
                top_pred_margin_counts[pred_margin_argmax] += 1
                top_true_margin_counts[true_margin_argmax] += 1

                for view_index, _view_name in enumerate(view_names):
                    weight_value = float(weight_values[view_index])
                    confidence_value = float(confidence_values[view_index])
                    scaled_confidence_value = float(scaled_confidence_values[view_index])
                    pred_margin_value = float(pred_margin_values[view_index])
                    true_margin_value = float(true_margin_values[view_index])
                    is_correct = bool(view_correct_values[view_index])

                    all_weights.append(weight_value)
                    all_pred_margins.append(pred_margin_value)
                    all_true_margins.append(true_margin_value)
                    all_correctness.append(float(is_correct))
                    all_confidences.append(confidence_value)
                    all_scaled_confidences.append(scaled_confidence_value)

                    weight_sums[view_index] += weight_value
                    confidence_sums[view_index] += confidence_value
                    scaled_confidence_sums[view_index] += scaled_confidence_value
                    pred_margin_sums[view_index] += pred_margin_value
                    true_margin_sums[view_index] += true_margin_value
                    raw_weight_sums[view_index] += float(raw_weight_values[view_index])

                    if is_correct:
                        correct_view_weights.append(weight_value)
                    else:
                        incorrect_view_weights.append(weight_value)

                sample_records.append(
                    {
                        "patient_id": str(patient_ids[sample_index]),
                        "label": label_value,
                        "fusion_prediction": {
                            "pred": fused_pred_value,
                            "abnormal_prob": float(fused_prob_values[1]),
                            "correct": fused_correct_value,
                        },
                        "top_weight_views": [view_names[index] for index in top_weight_candidates],
                        "top_pred_margin_views": [view_names[index] for index in top_pred_margin_candidates],
                        "top_true_margin_views": [view_names[index] for index in top_true_margin_candidates],
                        "top_weight_hits_pred_margin": weight_hit_pred_margin,
                        "top_weight_hits_true_margin": weight_hit_true_margin,
                        "top_weight_is_correct_view": top_weight_is_correct,
                        "mixed_view_correctness": mixed_correctness,
                        "views": {
                            view_name: {
                                "fusion_weight": float(weight_values[view_index]),
                                "raw_fusion_weight": float(raw_weight_values[view_index]),
                                "active_view_mask": (
                                    float(active_mask_values[view_index])
                                    if active_mask_values is not None
                                    else None
                                ),
                                "confidence_logit": float(confidence_values[view_index]),
                                "scaled_confidence_logit": float(scaled_confidence_values[view_index]),
                                "pred": int(view_pred_values[view_index]),
                                "abnormal_prob": float(view_abnormal_probs[view_index]),
                                "correct": bool(view_correct_values[view_index]),
                                "pred_margin": float(pred_margin_values[view_index]),
                                "true_margin": float(true_margin_values[view_index]),
                            }
                            for view_index, view_name in enumerate(view_names)
                        },
                    }
                )

            if args.limit_batches is not None and batch_index >= args.limit_batches:
                break

    total_samples = len(full_labels)
    active_mask_unique = sorted(
        {
            tuple(float(value) for value in mask_values)
            for mask_values in active_mask_records
        }
    )
    per_view_summaries: list[dict[str, Any]] = []
    for view_index, view_name in enumerate(view_names):
        metrics = compute_metrics(
            per_view_labels[view_index],
            per_view_preds[view_index],
            per_view_probs[view_index],
        )
        per_view_summaries.append(
            {
                "view": view_name,
                "metrics": metrics,
                "mean_fusion_weight": float(weight_sums[view_index] / total_samples),
                "mean_raw_fusion_weight": float(raw_weight_sums[view_index] / total_samples),
                "mean_confidence_logit": float(confidence_sums[view_index] / total_samples),
                "mean_scaled_confidence_logit": float(scaled_confidence_sums[view_index] / total_samples),
                "mean_pred_margin": float(pred_margin_sums[view_index] / total_samples),
                "mean_true_margin": float(true_margin_sums[view_index] / total_samples),
                "top_weight_count": int(top_weight_counts[view_index]),
                "top_weight_rate": float(top_weight_counts[view_index] / total_samples),
                "top_pred_margin_count": int(top_pred_margin_counts[view_index]),
                "top_pred_margin_rate": float(top_pred_margin_counts[view_index] / total_samples),
                "top_true_margin_count": int(top_true_margin_counts[view_index]),
                "top_true_margin_rate": float(top_true_margin_counts[view_index] / total_samples),
            }
        )

    output_json = Path(args.output_json) if args.output_json else output_dir / "fusion_weight_analysis.json"
    report = {
        "analysis": "fusion_weight_telemetry",
        "split": args.split,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_env": runtime_env,
        "total_samples": total_samples,
        "summary": {
            "full_fusion_metrics": compute_metrics(full_labels, full_preds, full_probs),
            "fusion_weight_semantics": (
                "active_mask_renormalized"
                if active_mask_unique
                else "raw_model_fusion_weights"
            ),
            "active_view_masks": [
                to_float_dict(view_names, list(mask_values))
                for mask_values in active_mask_unique
            ],
            "fusion_correct_rate": float(fusion_correct / total_samples),
            "top_weight_hit_rate": {
                "pred_margin": hit_rate(top_weight_hits_pred_margin, total_samples),
                "true_margin": hit_rate(top_weight_hits_true_margin, total_samples),
            },
            "top_weight_correct_rate": hit_rate(top_weight_correct, total_samples),
            "mixed_correctness_samples": int(mixed_correctness_samples),
            "mixed_correctness_top_weight_correct_rate": hit_rate(
                mixed_correctness_top_weight_correct,
                mixed_correctness_samples,
            ),
            "mean_weight_by_view_prediction_correctness": {
                "correct": safe_mean(correct_view_weights),
                "incorrect": safe_mean(incorrect_view_weights),
                "delta_correct_minus_incorrect": float(
                    safe_mean(correct_view_weights) - safe_mean(incorrect_view_weights)
                ),
            },
            "pearson_correlation": {
                "weight_vs_confidence_logit": safe_pearson(all_weights, all_confidences),
                "weight_vs_scaled_confidence_logit": safe_pearson(all_weights, all_scaled_confidences),
                "weight_vs_pred_margin": safe_pearson(all_weights, all_pred_margins),
                "weight_vs_true_margin": safe_pearson(all_weights, all_true_margins),
                "weight_vs_view_correctness": safe_pearson(all_weights, all_correctness),
            },
            "top_weight_view_distribution": to_int_dict(view_names, top_weight_counts),
            "top_pred_margin_view_distribution": to_int_dict(view_names, top_pred_margin_counts),
            "top_true_margin_view_distribution": to_int_dict(view_names, top_true_margin_counts),
        },
        "per_view": per_view_summaries,
        "samples": sample_records,
    }
    save_json(report, output_json)
    print(f"Saved fusion-weight telemetry to: {output_json}")


if __name__ == "__main__":
    main()
