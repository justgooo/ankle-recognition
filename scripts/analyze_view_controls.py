from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

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
        description="Run matched single-view / leave-one-view-out controls for a decision-fusion checkpoint."
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
        help="Optional output path. Defaults to <output_dir>/view_ablation_summary.json.",
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
        help="How many changed-prediction examples to keep per control.",
    )
    return parser.parse_args()


def apply_runtime_env(runtime_env: dict[str, str]) -> None:
    for key, value in runtime_env.items():
        os.environ[str(key)] = str(value)


def accuracy_then_auc(item: dict[str, Any]) -> tuple[float, float]:
    metrics = item["metrics"]
    auc = metrics.get("auc")
    if auc is None or auc != auc:
        auc = float("-inf")
    return float(metrics["accuracy"]), float(auc)


def tensor_mask(values: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=torch.float32)


def summarize_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return examples[:limit]


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
    control_specs = [
        ("full_fusion", None, "full"),
        ("single_view_axial", [1.0, 0.0, 0.0], "single_view"),
        ("single_view_coronal", [0.0, 1.0, 0.0], "single_view"),
        ("single_view_sagittal", [0.0, 0.0, 1.0], "single_view"),
        ("leave_out_axial", [0.0, 1.0, 1.0], "leave_one_out"),
        ("leave_out_coronal", [1.0, 0.0, 1.0], "leave_one_out"),
        ("leave_out_sagittal", [1.0, 1.0, 0.0], "leave_one_out"),
    ]
    scenario_data: dict[str, dict[str, Any]] = {
        name: {
            "mode": mode,
            "active_views": (
                view_names
                if mask is None
                else [view_names[index] for index, keep in enumerate(mask) if keep > 0]
            ),
            "labels": [],
            "preds": [],
            "probs": [],
            "changed_examples": [],
            "changed_prediction_count": 0,
            "improved_over_full_count": 0,
            "regressed_from_full_count": 0,
        }
        for name, mask, mode in control_specs
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"view-controls:{args.split}", leave=False), start=1):
            images = batch["images"].to(device)
            labels = batch["label"].to(device)
            patient_ids = batch["patient_id"]

            full_logits, decision_info = model.forward_with_decision_info(images)
            scenario_logits: dict[str, torch.Tensor] = {"full_fusion": full_logits}
            view_logits = decision_info["view_logits"]
            fusion_weights = decision_info["fusion_weights"]
            for name, mask_values, _mode in control_specs[1:]:
                mask = tensor_mask(mask_values, device=device)
                scenario_logits[name] = model.fuse_decisions(
                    view_logits=view_logits,
                    fusion_weights=fusion_weights,
                    active_view_mask=mask,
                )

            full_probs = torch.softmax(full_logits, dim=1)
            full_preds = torch.argmax(full_probs, dim=1)
            full_correct = full_preds.eq(labels)

            for name, _mask_values, _mode in control_specs:
                logits = scenario_logits[name]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                data = scenario_data[name]
                data["labels"].extend(labels.cpu().tolist())
                data["preds"].extend(preds.cpu().tolist())
                data["probs"].extend(probs.cpu().tolist())

                if name == "full_fusion":
                    continue

                changed = preds.ne(full_preds)
                improved = preds.eq(labels) & (~full_correct)
                regressed = (~preds.eq(labels)) & full_correct
                data["changed_prediction_count"] += int(changed.sum().item())
                data["improved_over_full_count"] += int(improved.sum().item())
                data["regressed_from_full_count"] += int(regressed.sum().item())

                remaining_slots = args.max_flip_examples - len(data["changed_examples"])
                if remaining_slots > 0:
                    changed_indices = torch.nonzero(changed, as_tuple=False).flatten().tolist()
                    for offset in changed_indices[:remaining_slots]:
                        data["changed_examples"].append(
                            {
                                "patient_id": str(patient_ids[offset]),
                                "label": int(labels[offset].item()),
                                "full_pred": int(full_preds[offset].item()),
                                "scenario_pred": int(preds[offset].item()),
                                "full_abnormal_prob": float(full_probs[offset, 1].item()),
                                "scenario_abnormal_prob": float(probs[offset, 1].item()),
                                "full_correct": bool(full_correct[offset].item()),
                                "scenario_correct": bool(preds[offset].eq(labels[offset]).item()),
                            }
                        )

            if args.limit_batches is not None and batch_index >= args.limit_batches:
                break

    scenario_summaries: list[dict[str, Any]] = []
    full_metrics = None
    full_record = scenario_data["full_fusion"]
    total_samples = len(full_record["labels"])
    for name, _mask_values, mode in control_specs:
        data = scenario_data[name]
        metrics = compute_metrics(data["labels"], data["preds"], data["probs"])
        scenario_summary = {
            "name": name,
            "mode": mode,
            "active_views": data["active_views"],
            "metrics": metrics,
        }
        if name == "full_fusion":
            full_metrics = metrics
        else:
            scenario_summary.update(
                {
                    "delta_vs_full": {
                        "accuracy": float(metrics["accuracy"] - full_metrics["accuracy"]),
                        "auc": float(metrics["auc"] - full_metrics["auc"]),
                        "f1": float(metrics["f1"] - full_metrics["f1"]),
                    },
                    "changed_prediction_count": int(data["changed_prediction_count"]),
                    "changed_prediction_rate": float(data["changed_prediction_count"] / total_samples),
                    "improved_over_full_count": int(data["improved_over_full_count"]),
                    "regressed_from_full_count": int(data["regressed_from_full_count"]),
                    "changed_examples": summarize_examples(data["changed_examples"], args.max_flip_examples),
                }
            )
        scenario_summaries.append(scenario_summary)

    single_view_controls = [
        item for item in scenario_summaries if item["mode"] == "single_view"
    ]
    leave_one_out_controls = [
        item for item in scenario_summaries if item["mode"] == "leave_one_out"
    ]
    best_single_view = max(single_view_controls, key=accuracy_then_auc)
    best_leave_one_out = max(leave_one_out_controls, key=accuracy_then_auc)

    output_json = Path(args.output_json) if args.output_json else output_dir / "view_ablation_summary.json"
    report = {
        "analysis": "matched_view_controls",
        "split": args.split,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "runtime_env": runtime_env,
        "total_samples": total_samples,
        "controls": scenario_summaries,
        "summary": {
            "full_fusion": next(item for item in scenario_summaries if item["name"] == "full_fusion"),
            "best_single_view": best_single_view,
            "best_leave_one_out": best_leave_one_out,
            "full_minus_best_single_view": {
                "accuracy": float(full_metrics["accuracy"] - best_single_view["metrics"]["accuracy"]),
                "auc": float(full_metrics["auc"] - best_single_view["metrics"]["auc"]),
                "f1": float(full_metrics["f1"] - best_single_view["metrics"]["f1"]),
            },
            "full_minus_best_leave_one_out": {
                "accuracy": float(full_metrics["accuracy"] - best_leave_one_out["metrics"]["accuracy"]),
                "auc": float(full_metrics["auc"] - best_leave_one_out["metrics"]["auc"]),
                "f1": float(full_metrics["f1"] - best_leave_one_out["metrics"]["f1"]),
            },
        },
    }
    save_json(report, output_json)
    print(f"Saved matched control summary to: {output_json}")


if __name__ == "__main__":
    main()
