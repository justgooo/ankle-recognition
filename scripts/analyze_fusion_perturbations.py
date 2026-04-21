from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
            "Run a matched perturbation analysis for a decision-fusion checkpoint and "
            "measure how fusion weights migrate when one view is degraded."
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
        "--perturb-view",
        choices=("axial", "coronal", "sagittal"),
        default="axial",
        help="Which view to perturb.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path. Defaults to <output_dir>/perturbation_summary.json.",
    )
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=9,
        help="Odd kernel size for the deterministic Gaussian blur perturbation.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=2.0,
        help="Sigma for the deterministic Gaussian blur perturbation.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional small-batch smoke limit for debugging.",
    )
    parser.add_argument(
        "--max-flip-examples",
        type=int,
        default=5,
        help="How many changed-prediction examples to keep.",
    )
    return parser.parse_args()


def apply_runtime_env(runtime_env: dict[str, str]) -> None:
    for key, value in runtime_env.items():
        os.environ[str(key)] = str(value)


def top_indices(values: list[float], atol: float = 1e-8) -> list[int]:
    max_value = max(values)
    return [index for index, value in enumerate(values) if abs(value - max_value) <= atol]


def to_int_dict(view_names: list[str], values: list[int]) -> dict[str, int]:
    return {
        view_name: int(value)
        for view_name, value in zip(view_names, values)
    }


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def safe_pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return float("nan")
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if np.isclose(x_array.std(), 0.0) or np.isclose(y_array.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def build_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("blur-kernel-size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("blur-sigma must be > 0.")
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def apply_view_gaussian_blur(
    images: torch.Tensor,
    view_index: int,
    kernel: torch.Tensor,
) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    perturbed = images.clone()
    selected = perturbed[:, view_index]
    batch_size, num_slices, height, width = selected.shape
    flat = selected.reshape(batch_size * num_slices, 1, height, width)
    flat = F.pad(flat, (pad, pad, pad, pad), mode="reflect")
    flat = F.conv2d(flat, kernel)
    perturbed[:, view_index] = flat.reshape(batch_size, num_slices, height, width)
    return perturbed


def compute_true_margins(view_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_indices = labels.view(-1, 1, 1).expand(-1, view_logits.shape[1], 1)
    wrong_label_indices = (1 - labels).view(-1, 1, 1).expand(-1, view_logits.shape[1], 1)
    true_class_logits = view_logits.gather(dim=-1, index=label_indices).squeeze(-1)
    other_class_logits = view_logits.gather(dim=-1, index=wrong_label_indices).squeeze(-1)
    return true_class_logits - other_class_logits


def summarize_example_deltas(
    sample: dict[str, Any],
    view_names: list[str],
    target_view_name: str,
) -> dict[str, Any]:
    return {
        "patient_id": sample["patient_id"],
        "label": sample["label"],
        "baseline_pred": sample["baseline_fusion_prediction"]["pred"],
        "perturbed_pred": sample["perturbed_fusion_prediction"]["pred"],
        "baseline_correct": sample["baseline_fusion_prediction"]["correct"],
        "perturbed_correct": sample["perturbed_fusion_prediction"]["correct"],
        "baseline_top_weight_views": sample["baseline_top_weight_views"],
        "perturbed_top_weight_views": sample["perturbed_top_weight_views"],
        "target_view_weight_delta": sample["views"][target_view_name]["weight_delta"],
        "views": {
            view_name: {
                "baseline_weight": sample["views"][view_name]["baseline_weight"],
                "perturbed_weight": sample["views"][view_name]["perturbed_weight"],
                "weight_delta": sample["views"][view_name]["weight_delta"],
                "baseline_pred_margin": sample["views"][view_name]["baseline_pred_margin"],
                "perturbed_pred_margin": sample["views"][view_name]["perturbed_pred_margin"],
                "pred_margin_delta": sample["views"][view_name]["pred_margin_delta"],
            }
            for view_name in view_names
        },
    }


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
    perturb_view_index = view_names.index(args.perturb_view)
    kernel = build_gaussian_kernel(
        kernel_size=args.blur_kernel_size,
        sigma=args.blur_sigma,
        device=device,
        dtype=torch.float32,
    )

    baseline_labels: list[int] = []
    baseline_preds: list[int] = []
    baseline_probs: list[list[float]] = []
    perturbed_preds: list[int] = []
    perturbed_probs: list[list[float]] = []
    per_view_baseline_preds: list[list[int]] = [[] for _ in view_names]
    per_view_baseline_probs: list[list[list[float]]] = [[] for _ in view_names]
    per_view_perturbed_preds: list[list[int]] = [[] for _ in view_names]
    per_view_perturbed_probs: list[list[list[float]]] = [[] for _ in view_names]

    baseline_weight_sums = [0.0 for _ in view_names]
    perturbed_weight_sums = [0.0 for _ in view_names]
    baseline_confidence_sums = [0.0 for _ in view_names]
    perturbed_confidence_sums = [0.0 for _ in view_names]
    baseline_scaled_confidence_sums = [0.0 for _ in view_names]
    perturbed_scaled_confidence_sums = [0.0 for _ in view_names]
    baseline_pred_margin_sums = [0.0 for _ in view_names]
    perturbed_pred_margin_sums = [0.0 for _ in view_names]
    baseline_true_margin_sums = [0.0 for _ in view_names]
    perturbed_true_margin_sums = [0.0 for _ in view_names]
    baseline_top_weight_counts = [0 for _ in view_names]
    perturbed_top_weight_counts = [0 for _ in view_names]
    switch_destination_counts = [0 for _ in view_names]

    target_weight_deltas: list[float] = []
    target_pred_margin_deltas: list[float] = []
    target_true_margin_deltas: list[float] = []
    top_weight_switch_count = 0
    changed_prediction_count = 0
    improved_over_baseline_count = 0
    regressed_from_baseline_count = 0
    target_migrated_away_count = 0
    target_weight_drop_count = 0
    target_margin_drop_weight_drop_count = 0
    changed_examples: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(loader, desc=f"fusion-perturbation:{args.split}", leave=False),
            start=1,
        ):
            images = batch["images"].to(device)
            labels = batch["label"].to(device)
            patient_ids = batch["patient_id"]

            baseline_logits, baseline_info = model.forward_with_decision_info(images)
            perturbed_images = apply_view_gaussian_blur(images, perturb_view_index, kernel)
            perturbed_logits, perturbed_info = model.forward_with_decision_info(perturbed_images)

            baseline_view_logits = baseline_info["view_logits"]
            perturbed_view_logits = perturbed_info["view_logits"]
            baseline_weights = baseline_info["fusion_weights"].squeeze(-1)
            perturbed_weights = perturbed_info["fusion_weights"].squeeze(-1)
            baseline_confidences = baseline_info["confidences"].squeeze(-1)
            perturbed_confidences = perturbed_info["confidences"].squeeze(-1)
            baseline_scaled_confidences = baseline_info["scaled_confidences"].squeeze(-1)
            perturbed_scaled_confidences = perturbed_info["scaled_confidences"].squeeze(-1)

            baseline_probs_tensor = torch.softmax(baseline_logits, dim=1)
            baseline_preds_tensor = torch.argmax(baseline_probs_tensor, dim=1)
            perturbed_probs_tensor = torch.softmax(perturbed_logits, dim=1)
            perturbed_preds_tensor = torch.argmax(perturbed_probs_tensor, dim=1)

            baseline_view_probs = torch.softmax(baseline_view_logits, dim=-1)
            perturbed_view_probs = torch.softmax(perturbed_view_logits, dim=-1)
            baseline_view_preds = torch.argmax(baseline_view_probs, dim=-1)
            perturbed_view_preds = torch.argmax(perturbed_view_probs, dim=-1)
            baseline_pred_margins = baseline_view_logits.max(dim=-1).values - baseline_view_logits.min(dim=-1).values
            perturbed_pred_margins = (
                perturbed_view_logits.max(dim=-1).values - perturbed_view_logits.min(dim=-1).values
            )
            baseline_true_margins = compute_true_margins(baseline_view_logits, labels)
            perturbed_true_margins = compute_true_margins(perturbed_view_logits, labels)

            baseline_labels.extend(labels.cpu().tolist())
            baseline_preds.extend(baseline_preds_tensor.cpu().tolist())
            baseline_probs.extend(baseline_probs_tensor.cpu().tolist())
            perturbed_preds.extend(perturbed_preds_tensor.cpu().tolist())
            perturbed_probs.extend(perturbed_probs_tensor.cpu().tolist())

            for view_index in range(len(view_names)):
                per_view_baseline_preds[view_index].extend(baseline_view_preds[:, view_index].cpu().tolist())
                per_view_baseline_probs[view_index].extend(baseline_view_probs[:, view_index, :].cpu().tolist())
                per_view_perturbed_preds[view_index].extend(perturbed_view_preds[:, view_index].cpu().tolist())
                per_view_perturbed_probs[view_index].extend(perturbed_view_probs[:, view_index, :].cpu().tolist())

            batch_size = labels.shape[0]
            for sample_index in range(batch_size):
                label_value = int(labels[sample_index].item())
                baseline_pred_value = int(baseline_preds_tensor[sample_index].item())
                perturbed_pred_value = int(perturbed_preds_tensor[sample_index].item())
                baseline_correct = baseline_pred_value == label_value
                perturbed_correct = perturbed_pred_value == label_value

                baseline_weight_values = baseline_weights[sample_index].cpu().tolist()
                perturbed_weight_values = perturbed_weights[sample_index].cpu().tolist()
                baseline_confidence_values = baseline_confidences[sample_index].cpu().tolist()
                perturbed_confidence_values = perturbed_confidences[sample_index].cpu().tolist()
                baseline_scaled_confidence_values = baseline_scaled_confidences[sample_index].cpu().tolist()
                perturbed_scaled_confidence_values = perturbed_scaled_confidences[sample_index].cpu().tolist()
                baseline_pred_margin_values = baseline_pred_margins[sample_index].cpu().tolist()
                perturbed_pred_margin_values = perturbed_pred_margins[sample_index].cpu().tolist()
                baseline_true_margin_values = baseline_true_margins[sample_index].cpu().tolist()
                perturbed_true_margin_values = perturbed_true_margins[sample_index].cpu().tolist()

                baseline_top_weight = int(np.argmax(np.asarray(baseline_weight_values, dtype=np.float64)))
                perturbed_top_weight = int(np.argmax(np.asarray(perturbed_weight_values, dtype=np.float64)))
                baseline_top_weight_views = [view_names[index] for index in top_indices(baseline_weight_values)]
                perturbed_top_weight_views = [view_names[index] for index in top_indices(perturbed_weight_values)]

                baseline_top_weight_counts[baseline_top_weight] += 1
                perturbed_top_weight_counts[perturbed_top_weight] += 1
                if baseline_top_weight != perturbed_top_weight:
                    top_weight_switch_count += 1
                if baseline_top_weight == perturb_view_index and perturbed_top_weight != perturb_view_index:
                    target_migrated_away_count += 1
                    switch_destination_counts[perturbed_top_weight] += 1

                weight_delta = (
                    float(perturbed_weight_values[perturb_view_index])
                    - float(baseline_weight_values[perturb_view_index])
                )
                pred_margin_delta = (
                    float(perturbed_pred_margin_values[perturb_view_index])
                    - float(baseline_pred_margin_values[perturb_view_index])
                )
                true_margin_delta = (
                    float(perturbed_true_margin_values[perturb_view_index])
                    - float(baseline_true_margin_values[perturb_view_index])
                )
                target_weight_deltas.append(weight_delta)
                target_pred_margin_deltas.append(pred_margin_delta)
                target_true_margin_deltas.append(true_margin_delta)
                if weight_delta < 0:
                    target_weight_drop_count += 1
                if pred_margin_delta < 0 and weight_delta < 0:
                    target_margin_drop_weight_drop_count += 1

                if baseline_pred_value != perturbed_pred_value:
                    changed_prediction_count += 1
                if perturbed_correct and not baseline_correct:
                    improved_over_baseline_count += 1
                if baseline_correct and not perturbed_correct:
                    regressed_from_baseline_count += 1

                for view_index in range(len(view_names)):
                    baseline_weight_sums[view_index] += float(baseline_weight_values[view_index])
                    perturbed_weight_sums[view_index] += float(perturbed_weight_values[view_index])
                    baseline_confidence_sums[view_index] += float(baseline_confidence_values[view_index])
                    perturbed_confidence_sums[view_index] += float(perturbed_confidence_values[view_index])
                    baseline_scaled_confidence_sums[view_index] += float(
                        baseline_scaled_confidence_values[view_index]
                    )
                    perturbed_scaled_confidence_sums[view_index] += float(
                        perturbed_scaled_confidence_values[view_index]
                    )
                    baseline_pred_margin_sums[view_index] += float(baseline_pred_margin_values[view_index])
                    perturbed_pred_margin_sums[view_index] += float(perturbed_pred_margin_values[view_index])
                    baseline_true_margin_sums[view_index] += float(baseline_true_margin_values[view_index])
                    perturbed_true_margin_sums[view_index] += float(perturbed_true_margin_values[view_index])

                sample_record = {
                    "patient_id": str(patient_ids[sample_index]),
                    "label": label_value,
                    "baseline_fusion_prediction": {
                        "pred": baseline_pred_value,
                        "abnormal_prob": float(baseline_probs_tensor[sample_index, 1].item()),
                        "correct": bool(baseline_correct),
                    },
                    "perturbed_fusion_prediction": {
                        "pred": perturbed_pred_value,
                        "abnormal_prob": float(perturbed_probs_tensor[sample_index, 1].item()),
                        "correct": bool(perturbed_correct),
                    },
                    "baseline_top_weight_views": baseline_top_weight_views,
                    "perturbed_top_weight_views": perturbed_top_weight_views,
                    "top_weight_switched": bool(baseline_top_weight != perturbed_top_weight),
                    "views": {
                        view_name: {
                            "baseline_weight": float(baseline_weight_values[view_index]),
                            "perturbed_weight": float(perturbed_weight_values[view_index]),
                            "weight_delta": float(
                                perturbed_weight_values[view_index] - baseline_weight_values[view_index]
                            ),
                            "baseline_confidence_logit": float(baseline_confidence_values[view_index]),
                            "perturbed_confidence_logit": float(perturbed_confidence_values[view_index]),
                            "confidence_logit_delta": float(
                                perturbed_confidence_values[view_index] - baseline_confidence_values[view_index]
                            ),
                            "baseline_scaled_confidence_logit": float(
                                baseline_scaled_confidence_values[view_index]
                            ),
                            "perturbed_scaled_confidence_logit": float(
                                perturbed_scaled_confidence_values[view_index]
                            ),
                            "scaled_confidence_logit_delta": float(
                                perturbed_scaled_confidence_values[view_index]
                                - baseline_scaled_confidence_values[view_index]
                            ),
                            "baseline_pred_margin": float(baseline_pred_margin_values[view_index]),
                            "perturbed_pred_margin": float(perturbed_pred_margin_values[view_index]),
                            "pred_margin_delta": float(
                                perturbed_pred_margin_values[view_index] - baseline_pred_margin_values[view_index]
                            ),
                            "baseline_true_margin": float(baseline_true_margin_values[view_index]),
                            "perturbed_true_margin": float(perturbed_true_margin_values[view_index]),
                            "true_margin_delta": float(
                                perturbed_true_margin_values[view_index] - baseline_true_margin_values[view_index]
                            ),
                            "baseline_pred": int(baseline_view_preds[sample_index, view_index].item()),
                            "perturbed_pred": int(perturbed_view_preds[sample_index, view_index].item()),
                            "baseline_abnormal_prob": float(baseline_view_probs[sample_index, view_index, 1].item()),
                            "perturbed_abnormal_prob": float(
                                perturbed_view_probs[sample_index, view_index, 1].item()
                            ),
                        }
                        for view_index, view_name in enumerate(view_names)
                    },
                }
                sample_records.append(sample_record)

                remaining_slots = args.max_flip_examples - len(changed_examples)
                if remaining_slots > 0 and baseline_pred_value != perturbed_pred_value:
                    changed_examples.append(
                        {
                            "patient_id": str(patient_ids[sample_index]),
                            "label": label_value,
                            "baseline_pred": baseline_pred_value,
                            "perturbed_pred": perturbed_pred_value,
                            "baseline_abnormal_prob": float(baseline_probs_tensor[sample_index, 1].item()),
                            "perturbed_abnormal_prob": float(perturbed_probs_tensor[sample_index, 1].item()),
                            "baseline_correct": bool(baseline_correct),
                            "perturbed_correct": bool(perturbed_correct),
                            "baseline_top_weight_views": baseline_top_weight_views,
                            "perturbed_top_weight_views": perturbed_top_weight_views,
                            "target_view_weight_delta": float(weight_delta),
                        }
                    )

            if args.limit_batches is not None and batch_index >= args.limit_batches:
                break

    total_samples = len(baseline_labels)
    baseline_metrics = compute_metrics(baseline_labels, baseline_preds, baseline_probs)
    perturbed_metrics = compute_metrics(baseline_labels, perturbed_preds, perturbed_probs)

    per_view_summaries: list[dict[str, Any]] = []
    for view_index, view_name in enumerate(view_names):
        baseline_view_metrics = compute_metrics(
            baseline_labels,
            per_view_baseline_preds[view_index],
            per_view_baseline_probs[view_index],
        )
        perturbed_view_metrics = compute_metrics(
            baseline_labels,
            per_view_perturbed_preds[view_index],
            per_view_perturbed_probs[view_index],
        )
        per_view_summaries.append(
            {
                "view": view_name,
                "baseline": {
                    "metrics": baseline_view_metrics,
                    "mean_fusion_weight": float(baseline_weight_sums[view_index] / total_samples),
                    "mean_confidence_logit": float(baseline_confidence_sums[view_index] / total_samples),
                    "mean_scaled_confidence_logit": float(
                        baseline_scaled_confidence_sums[view_index] / total_samples
                    ),
                    "mean_pred_margin": float(baseline_pred_margin_sums[view_index] / total_samples),
                    "mean_true_margin": float(baseline_true_margin_sums[view_index] / total_samples),
                    "top_weight_count": int(baseline_top_weight_counts[view_index]),
                    "top_weight_rate": float(baseline_top_weight_counts[view_index] / total_samples),
                },
                "perturbed": {
                    "metrics": perturbed_view_metrics,
                    "mean_fusion_weight": float(perturbed_weight_sums[view_index] / total_samples),
                    "mean_confidence_logit": float(perturbed_confidence_sums[view_index] / total_samples),
                    "mean_scaled_confidence_logit": float(
                        perturbed_scaled_confidence_sums[view_index] / total_samples
                    ),
                    "mean_pred_margin": float(perturbed_pred_margin_sums[view_index] / total_samples),
                    "mean_true_margin": float(perturbed_true_margin_sums[view_index] / total_samples),
                    "top_weight_count": int(perturbed_top_weight_counts[view_index]),
                    "top_weight_rate": float(perturbed_top_weight_counts[view_index] / total_samples),
                },
                "delta": {
                    "accuracy": float(perturbed_view_metrics["accuracy"] - baseline_view_metrics["accuracy"]),
                    "auc": float(perturbed_view_metrics["auc"] - baseline_view_metrics["auc"]),
                    "f1": float(perturbed_view_metrics["f1"] - baseline_view_metrics["f1"]),
                    "mean_fusion_weight": float(
                        (perturbed_weight_sums[view_index] - baseline_weight_sums[view_index]) / total_samples
                    ),
                    "mean_confidence_logit": float(
                        (perturbed_confidence_sums[view_index] - baseline_confidence_sums[view_index])
                        / total_samples
                    ),
                    "mean_scaled_confidence_logit": float(
                        (
                            perturbed_scaled_confidence_sums[view_index]
                            - baseline_scaled_confidence_sums[view_index]
                        )
                        / total_samples
                    ),
                    "mean_pred_margin": float(
                        (perturbed_pred_margin_sums[view_index] - baseline_pred_margin_sums[view_index])
                        / total_samples
                    ),
                    "mean_true_margin": float(
                        (perturbed_true_margin_sums[view_index] - baseline_true_margin_sums[view_index])
                        / total_samples
                    ),
                    "top_weight_count": int(perturbed_top_weight_counts[view_index] - baseline_top_weight_counts[view_index]),
                    "top_weight_rate": float(
                        (perturbed_top_weight_counts[view_index] - baseline_top_weight_counts[view_index])
                        / total_samples
                    ),
                },
            }
        )

    largest_weight_drop_samples = sorted(
        sample_records,
        key=lambda item: item["views"][args.perturb_view]["weight_delta"],
    )[: args.max_flip_examples]
    largest_weight_gain_samples = sorted(
        sample_records,
        key=lambda item: item["views"][args.perturb_view]["weight_delta"],
        reverse=True,
    )[: args.max_flip_examples]

    output_json = Path(args.output_json) if args.output_json else output_dir / "perturbation_summary.json"
    report = {
        "analysis": "fusion_weight_perturbation",
        "split": args.split,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_env": runtime_env,
        "perturbation": {
            "type": "gaussian_blur",
            "view": args.perturb_view,
            "blur_kernel_size": int(args.blur_kernel_size),
            "blur_sigma": float(args.blur_sigma),
        },
        "total_samples": total_samples,
        "summary": {
            "baseline_metrics": baseline_metrics,
            "perturbed_metrics": perturbed_metrics,
            "delta_vs_baseline": {
                "accuracy": float(perturbed_metrics["accuracy"] - baseline_metrics["accuracy"]),
                "auc": float(perturbed_metrics["auc"] - baseline_metrics["auc"]),
                "f1": float(perturbed_metrics["f1"] - baseline_metrics["f1"]),
            },
            "changed_prediction_count": int(changed_prediction_count),
            "changed_prediction_rate": float(changed_prediction_count / total_samples),
            "improved_over_baseline_count": int(improved_over_baseline_count),
            "regressed_from_baseline_count": int(regressed_from_baseline_count),
            "top_weight_switch_count": int(top_weight_switch_count),
            "top_weight_switch_rate": float(top_weight_switch_count / total_samples),
            "target_view": {
                "name": args.perturb_view,
                "mean_weight_delta": safe_mean(target_weight_deltas),
                "median_weight_delta": safe_median(target_weight_deltas),
                "mean_pred_margin_delta": safe_mean(target_pred_margin_deltas),
                "median_pred_margin_delta": safe_median(target_pred_margin_deltas),
                "mean_true_margin_delta": safe_mean(target_true_margin_deltas),
                "median_true_margin_delta": safe_median(target_true_margin_deltas),
                "weight_drop_count": int(target_weight_drop_count),
                "weight_drop_rate": float(target_weight_drop_count / total_samples),
                "top_weight_count_before": int(baseline_top_weight_counts[perturb_view_index]),
                "top_weight_rate_before": float(baseline_top_weight_counts[perturb_view_index] / total_samples),
                "top_weight_count_after": int(perturbed_top_weight_counts[perturb_view_index]),
                "top_weight_rate_after": float(perturbed_top_weight_counts[perturb_view_index] / total_samples),
                "migrated_away_count": int(target_migrated_away_count),
                "migrated_away_rate": float(target_migrated_away_count / total_samples),
                "margin_drop_and_weight_drop_count": int(target_margin_drop_weight_drop_count),
                "margin_drop_and_weight_drop_rate": float(
                    target_margin_drop_weight_drop_count / total_samples
                ),
                "pearson": {
                    "weight_delta_vs_pred_margin_delta": safe_pearson(
                        target_weight_deltas,
                        target_pred_margin_deltas,
                    ),
                    "weight_delta_vs_true_margin_delta": safe_pearson(
                        target_weight_deltas,
                        target_true_margin_deltas,
                    ),
                },
            },
            "top_weight_view_distribution": {
                "before": to_int_dict(view_names, baseline_top_weight_counts),
                "after": to_int_dict(view_names, perturbed_top_weight_counts),
                "migration_destinations_after_leaving_target": to_int_dict(view_names, switch_destination_counts),
            },
            "changed_examples": changed_examples,
            "largest_target_weight_drops": [
                summarize_example_deltas(item, view_names, args.perturb_view)
                for item in largest_weight_drop_samples
            ],
            "largest_target_weight_gains": [
                summarize_example_deltas(item, view_names, args.perturb_view)
                for item in largest_weight_gain_samples
            ],
        },
        "per_view": per_view_summaries,
        "samples": sample_records,
    }
    save_json(report, output_json)
    print(f"Saved perturbation analysis to: {output_json}")


if __name__ == "__main__":
    main()
