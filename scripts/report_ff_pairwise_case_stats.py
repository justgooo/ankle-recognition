#!/usr/bin/env python3
"""Paired validation case analysis for feature-fusion gated head vs baseline.

This is a read-only validation analysis. It rebuilds the validation dataloader,
loads existing checkpoints, and compares per-case predictions across matched
seeds. It does not evaluate or report test-set metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import compute_metrics, load_config, save_json, set_seed
from train import build_dataloaders, build_model


SEEDS = (42, 123, 456)
RUNS = {
    "gated_head": {
        42: {
            "config": "configs/paper_feature_fusion_gated_head_formal_s42.yaml",
            "checkpoint": "runs/paper_feature_fusion/gated_head_s42/best.pt",
        },
        123: {
            "config": "configs/paper_feature_fusion_gated_head_formal_s123.yaml",
            "checkpoint": "runs/paper_feature_fusion/gated_head_s123/best.pt",
        },
        456: {
            "config": "configs/paper_feature_fusion_gated_head_formal_s456.yaml",
            "checkpoint": "runs/paper_feature_fusion/gated_head_s456/best.pt",
        },
    },
    "minimal_baseline": {
        42: {
            "config": "configs/paper_feature_fusion_minimal_baseline_formal_s42.yaml",
            "checkpoint": "runs/paper_feature_fusion/minimal_baseline_s42/best.pt",
        },
        123: {
            "config": "configs/paper_feature_fusion_minimal_baseline_formal_s123.yaml",
            "checkpoint": "runs/paper_feature_fusion/minimal_baseline_s123/best.pt",
        },
        456: {
            "config": "configs/paper_feature_fusion_minimal_baseline_formal_s456.yaml",
            "checkpoint": "runs/paper_feature_fusion/minimal_baseline_s456/best.pt",
        },
    },
}

FEATURE_ENV_KEYS = (
    "ANKLE_FEATURE_DISABLE_VIEW_RECALIBRATION",
    "ANKLE_FEATURE_DISABLE_CROSS_VIEW_MIXER",
    "ANKLE_FEATURE_DISABLE_GLU_HEAD",
    "ANKLE_FEATURE_XVIEW_ATTENTION_DIM",
    "ANKLE_FEATURE_XVIEW_NUM_HEADS",
    "ANKLE_FEATURE_XVIEW_NUM_LAYERS",
    "ANKLE_FEATURE_XVIEW_DROPOUT",
    "ANKLE_FEATURE_XVIEW_RESIDUAL_SCALE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired per-case validation statistics for the feature-fusion "
            "gated-head model against the matched minimal baseline."
        )
    )
    parser.add_argument(
        "--output-json",
        default="autoresearch_logs/ff_pairwise_case_stats.json",
        help="Path for the full JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default="autoresearch_logs/ff_pairwise_case_stats.md",
        help="Path for the compact Markdown report.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Override dataloader workers for this analysis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional validation batch-size override.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for paired CI.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260527,
        help="RNG seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional smoke limit. Leave unset for the real report.",
    )
    parser.add_argument(
        "--keep-pretrained-init",
        action="store_true",
        help="Keep config use_pretrained=true before loading checkpoints.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def reset_feature_runtime_env(runtime_env: dict[str, str]) -> None:
    for key in FEATURE_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in runtime_env.items():
        os.environ[str(key)] = str(value)


def load_best_val_metrics(config_path: Path) -> dict[str, float] | None:
    config = load_config(config_path)
    summary_path = REPO_ROOT / config["output_dir"] / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    best_val = summary.get("best_val")
    if not isinstance(best_val, dict):
        return None
    return {
        "accuracy": float(best_val["accuracy"]),
        "auc": float(best_val["auc"]),
        "f1": float(best_val["f1"]),
        "specificity": float(best_val["specificity"]),
        "sensitivity": float(best_val["sensitivity"]),
    }


def round_float(value: float, digits: int = 6) -> float:
    if math.isnan(float(value)):
        return float("nan")
    return round(float(value), digits)


def rounded_metrics(metrics: dict[str, float], digits: int = 6) -> dict[str, float]:
    return {key: round_float(float(value), digits) for key, value in metrics.items()}


def error_type(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 1 and pred == 0:
        return "FN"
    if label == 0 and pred == 1:
        return "FP"
    return "ERR"


def load_run_predictions(
    method: str,
    seed: int,
    run_info: dict[str, str],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config_path = REPO_ROOT / run_info["config"]
    checkpoint_path = REPO_ROOT / run_info["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = load_config(config_path)
    runtime_env = {
        str(key): str(value)
        for key, value in dict(config.get("runtime_env", {})).items()
    }
    reset_feature_runtime_env(runtime_env)
    set_seed(int(config["seed"]))

    if args.num_workers is not None:
        config["data"]["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        config["data"]["batch_size"] = int(args.batch_size)
    if not args.keep_pretrained_init:
        config["model"]["use_pretrained"] = False

    train_loader, val_loader, test_loader, _ = build_dataloaders(config)
    del train_loader, test_loader
    if val_loader is None:
        raise RuntimeError(f"No validation loader for {config_path}")

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels: list[int] = []
    preds: list[int] = []
    probs: list[list[float]] = []
    samples: list[dict[str, Any]] = []

    with torch.no_grad():
        iterator = tqdm(
            val_loader,
            desc=f"ff-pairwise:{method}:s{seed}",
            leave=False,
        )
        for batch_index, batch in enumerate(iterator, start=1):
            images = batch["images"].to(device, non_blocking=True)
            label_tensor = batch["label"].to(device, non_blocking=True)
            patient_ids = batch["patient_id"]

            logits = model(images)
            prob_tensor = torch.softmax(logits, dim=1)
            pred_tensor = torch.argmax(prob_tensor, dim=1)

            batch_labels = label_tensor.cpu().numpy().astype(int).tolist()
            batch_preds = pred_tensor.cpu().numpy().astype(int).tolist()
            batch_probs = prob_tensor.cpu().numpy().astype(float).tolist()

            labels.extend(batch_labels)
            preds.extend(batch_preds)
            probs.extend(batch_probs)

            for patient_id, label, pred, prob in zip(
                patient_ids,
                batch_labels,
                batch_preds,
                batch_probs,
            ):
                samples.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "patient_id": str(patient_id),
                        "label": int(label),
                        "pred": int(pred),
                        "correct": bool(pred == label),
                        "error_type": error_type(int(label), int(pred)),
                        "normal_prob": float(prob[0]),
                        "abnormal_prob": float(prob[1]),
                    }
                )

            if args.limit_batches is not None and batch_index >= args.limit_batches:
                break

    metrics = compute_metrics(labels, preds, probs)
    expected_best_val = load_best_val_metrics(config_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    del model

    return {
        "method": method,
        "seed": int(seed),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "num_samples": len(samples),
        "metrics": rounded_metrics(metrics),
        "expected_best_val": rounded_metrics(expected_best_val) if expected_best_val else None,
        "samples": samples,
    }


def index_samples(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {sample["patient_id"]: sample for sample in run["samples"]}


def transition_name(minimal_correct: bool, gated_correct: bool) -> str:
    if minimal_correct and gated_correct:
        return "both_correct"
    if not minimal_correct and not gated_correct:
        return "both_wrong"
    if not minimal_correct and gated_correct:
        return "fixed_by_gated"
    return "broken_by_gated"


def compare_runs(predictions: dict[str, dict[int, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}

    for seed in SEEDS:
        gated = predictions["gated_head"][seed]
        minimal = predictions["minimal_baseline"][seed]
        gated_by_id = index_samples(gated)
        minimal_by_id = index_samples(minimal)
        patient_ids = sorted(set(gated_by_id) & set(minimal_by_id))
        if len(patient_ids) != len(gated_by_id) or len(patient_ids) != len(minimal_by_id):
            raise RuntimeError(f"Seed {seed} validation patient IDs do not match.")

        seed_rows: list[dict[str, Any]] = []
        for patient_id in patient_ids:
            gated_sample = gated_by_id[patient_id]
            minimal_sample = minimal_by_id[patient_id]
            if int(gated_sample["label"]) != int(minimal_sample["label"]):
                raise RuntimeError(f"Label mismatch for seed {seed} patient {patient_id}")
            minimal_correct = bool(minimal_sample["correct"])
            gated_correct = bool(gated_sample["correct"])
            transition = transition_name(minimal_correct, gated_correct)
            row = {
                "seed": int(seed),
                "patient_id": patient_id,
                "label": int(gated_sample["label"]),
                "transition": transition,
                "minimal_pred": int(minimal_sample["pred"]),
                "minimal_correct": minimal_correct,
                "minimal_error_type": str(minimal_sample["error_type"]),
                "minimal_normal_prob": float(minimal_sample["normal_prob"]),
                "minimal_abnormal_prob": float(minimal_sample["abnormal_prob"]),
                "gated_pred": int(gated_sample["pred"]),
                "gated_correct": gated_correct,
                "gated_error_type": str(gated_sample["error_type"]),
                "gated_normal_prob": float(gated_sample["normal_prob"]),
                "gated_abnormal_prob": float(gated_sample["abnormal_prob"]),
                "correct_delta": int(gated_correct) - int(minimal_correct),
                "abnormal_prob_delta": float(gated_sample["abnormal_prob"])
                - float(minimal_sample["abnormal_prob"]),
            }
            rows.append(row)
            seed_rows.append(row)

        transitions = Counter(row["transition"] for row in seed_rows)
        per_seed[str(seed)] = {
            "num_cases": len(seed_rows),
            "metrics": {
                "minimal_baseline": minimal["metrics"],
                "gated_head": gated["metrics"],
                "delta_gated_minus_minimal": metric_delta(gated["metrics"], minimal["metrics"]),
            },
            "transitions": counter_dict(transitions),
            "net_correct_delta": int(
                transitions.get("fixed_by_gated", 0) - transitions.get("broken_by_gated", 0)
            ),
            "fixed_cases": case_briefs(seed_rows, "fixed_by_gated"),
            "broken_cases": case_briefs(seed_rows, "broken_by_gated"),
        }

    return rows, per_seed


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(a) & set(b))
    return {key: round_float(float(a[key]) - float(b[key])) for key in keys}


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def case_briefs(rows: list[dict[str, Any]], transition: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["transition"] == transition]
    return [
        {
            "patient_id": row["patient_id"],
            "label": int(row["label"]),
            "minimal": {
                "pred": int(row["minimal_pred"]),
                "error_type": row["minimal_error_type"],
                "abnormal_prob": round_float(row["minimal_abnormal_prob"], 4),
            },
            "gated": {
                "pred": int(row["gated_pred"]),
                "error_type": row["gated_error_type"],
                "abnormal_prob": round_float(row["gated_abnormal_prob"], 4),
            },
            "abnormal_prob_delta": round_float(row["abnormal_prob_delta"], 4),
        }
        for row in selected
    ]


def metrics_from_arrays(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    return compute_metrics(
        labels[indices].tolist(),
        preds[indices].tolist(),
        probs[indices].tolist(),
    )


def summarize_values(values: list[float]) -> dict[str, float | int]:
    valid = np.asarray([value for value in values if not math.isnan(float(value))], dtype=np.float64)
    if valid.size == 0:
        return {
            "mean": float("nan"),
            "p2_5": float("nan"),
            "p50": float("nan"),
            "p97_5": float("nan"),
            "n_valid": 0,
        }
    return {
        "mean": round_float(float(np.mean(valid))),
        "p2_5": round_float(float(np.percentile(valid, 2.5))),
        "p50": round_float(float(np.percentile(valid, 50.0))),
        "p97_5": round_float(float(np.percentile(valid, 97.5))),
        "n_valid": int(valid.size),
    }


def build_bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    minimal_preds = np.asarray([row["minimal_pred"] for row in rows], dtype=np.int64)
    gated_preds = np.asarray([row["gated_pred"] for row in rows], dtype=np.int64)
    minimal_probs = np.asarray(
        [[row["minimal_normal_prob"], row["minimal_abnormal_prob"]] for row in rows],
        dtype=np.float64,
    )
    gated_probs = np.asarray(
        [[row["gated_normal_prob"], row["gated_abnormal_prob"]] for row in rows],
        dtype=np.float64,
    )
    fixed_flags = np.asarray([row["transition"] == "fixed_by_gated" for row in rows], dtype=np.int64)
    broken_flags = np.asarray([row["transition"] == "broken_by_gated" for row in rows], dtype=np.int64)

    patient_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        patient_to_indices[row["patient_id"]].append(index)
    patient_ids = np.asarray(sorted(patient_to_indices), dtype=object)
    cluster_indices = [np.asarray(patient_to_indices[patient_id], dtype=np.int64) for patient_id in patient_ids]

    rng = np.random.default_rng(seed)
    accumulators: dict[str, list[float]] = {
        "accuracy_delta": [],
        "auc_delta": [],
        "f1_delta": [],
        "fixed_count": [],
        "broken_count": [],
        "net_correct_delta": [],
        "fixed_rate": [],
        "broken_rate": [],
    }

    if samples <= 0:
        return {"mode": "cluster_by_patient", "samples": 0, "metrics": {}}

    for _ in range(samples):
        chosen_cluster_positions = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        indices = np.concatenate([cluster_indices[position] for position in chosen_cluster_positions])
        minimal_metrics = metrics_from_arrays(labels, minimal_preds, minimal_probs, indices)
        gated_metrics = metrics_from_arrays(labels, gated_preds, gated_probs, indices)
        fixed_count = float(np.sum(fixed_flags[indices]))
        broken_count = float(np.sum(broken_flags[indices]))
        total = float(indices.size)

        accumulators["accuracy_delta"].append(
            float(gated_metrics["accuracy"] - minimal_metrics["accuracy"])
        )
        accumulators["auc_delta"].append(float(gated_metrics["auc"] - minimal_metrics["auc"]))
        accumulators["f1_delta"].append(float(gated_metrics["f1"] - minimal_metrics["f1"]))
        accumulators["fixed_count"].append(fixed_count)
        accumulators["broken_count"].append(broken_count)
        accumulators["net_correct_delta"].append(fixed_count - broken_count)
        accumulators["fixed_rate"].append(fixed_count / total)
        accumulators["broken_rate"].append(broken_count / total)

    return {
        "mode": "cluster_by_patient",
        "samples": int(samples),
        "seed": int(seed),
        "cluster_count": int(len(patient_ids)),
        "row_count_per_resample": int(len(rows)),
        "metrics": {
            name: summarize_values(values)
            for name, values in accumulators.items()
        },
    }


def exact_mcnemar_p(fixed: int, broken: int) -> float:
    total = int(fixed + broken)
    if total <= 0:
        return float("nan")
    smaller = min(int(fixed), int(broken))
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2 ** total)
    return float(min(1.0, 2.0 * tail))


def aggregate_report(
    predictions: dict[str, dict[int, dict[str, Any]]],
    rows: list[dict[str, Any]],
    per_seed: dict[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    transitions = Counter(row["transition"] for row in rows)
    fixed = int(transitions.get("fixed_by_gated", 0))
    broken = int(transitions.get("broken_by_gated", 0))
    method_means = {}
    for method in ("minimal_baseline", "gated_head"):
        metrics_by_seed = [predictions[method][seed]["metrics"] for seed in SEEDS]
        method_means[method] = {
            key: round_float(float(np.mean([metrics[key] for metrics in metrics_by_seed])))
            for key in ("accuracy", "auc", "f1", "specificity", "sensitivity")
        }

    fixed_counter = Counter(row["patient_id"] for row in rows if row["transition"] == "fixed_by_gated")
    broken_counter = Counter(row["patient_id"] for row in rows if row["transition"] == "broken_by_gated")
    fixed_label_counter = Counter(row["label"] for row in rows if row["transition"] == "fixed_by_gated")
    broken_label_counter = Counter(row["label"] for row in rows if row["transition"] == "broken_by_gated")

    return {
        "scope": {
            "split": "validation",
            "uses_test_metrics": False,
            "methods": ["minimal_baseline", "gated_head"],
            "seeds": list(SEEDS),
            "row_count": len(rows),
            "unique_patient_count": len({row["patient_id"] for row in rows}),
        },
        "method_mean_metrics": method_means,
        "mean_delta_gated_minus_minimal": metric_delta(
            method_means["gated_head"],
            method_means["minimal_baseline"],
        ),
        "paired_transitions": {
            **counter_dict(transitions),
            "fixed_by_gated": fixed,
            "broken_by_gated": broken,
            "net_correct_delta": int(fixed - broken),
            "mcnemar_exact_p_two_sided_seed_case": round_float(exact_mcnemar_p(fixed, broken)),
        },
        "fixed_labels": counter_dict(fixed_label_counter),
        "broken_labels": counter_dict(broken_label_counter),
        "patient_recurrence": {
            "fixed_by_gated": counter_dict(fixed_counter),
            "broken_by_gated": counter_dict(broken_counter),
        },
        "per_seed": per_seed,
        "bootstrap": build_bootstrap(rows, bootstrap_samples, bootstrap_seed),
        "paired_rows": rows,
        "run_metrics": {
            method: {str(seed): predictions[method][seed]["metrics"] for seed in SEEDS}
            for method in ("minimal_baseline", "gated_head")
        },
        "run_paths": {
            method: {
                str(seed): {
                    "config": predictions[method][seed]["config"],
                    "checkpoint": predictions[method][seed]["checkpoint"],
                }
                for seed in SEEDS
            }
            for method in ("minimal_baseline", "gated_head")
        },
    }


def format_metric_triplet(metrics: dict[str, float]) -> str:
    return f"{metrics['accuracy']:.6f}/{metrics['auc']:.6f}/{metrics['f1']:.6f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    delta = report["mean_delta_gated_minus_minimal"]
    transitions = report["paired_transitions"]
    bootstrap = report["bootstrap"]["metrics"]
    lines = [
        "# FF Pairwise Case Stats",
        "",
        "Scope: validation split only. No test metrics are used or reported.",
        "",
        "## 3-seed means",
        "",
        f"- minimal_baseline acc/auc/f1: {format_metric_triplet(report['method_mean_metrics']['minimal_baseline'])}",
        f"- gated_head acc/auc/f1: {format_metric_triplet(report['method_mean_metrics']['gated_head'])}",
        f"- delta gated-minimal acc/auc/f1: {delta['accuracy']:.6f}/{delta['auc']:.6f}/{delta['f1']:.6f}",
        "",
        "## Paired transitions",
        "",
        f"- seed-case rows: {report['scope']['row_count']} ({report['scope']['unique_patient_count']} unique patients x 3 seeds)",
        f"- fixed_by_gated: {transitions['fixed_by_gated']}",
        f"- broken_by_gated: {transitions['broken_by_gated']}",
        f"- net_correct_delta: {transitions['net_correct_delta']}",
        f"- McNemar exact p over seed-case discordants: {transitions['mcnemar_exact_p_two_sided_seed_case']:.6f}",
        "",
        "## Bootstrap CI",
        "",
        f"- mode: {report['bootstrap']['mode']}, resamples: {report['bootstrap']['samples']}",
        (
            "- accuracy_delta CI: "
            f"{bootstrap['accuracy_delta']['p2_5']:.6f} to {bootstrap['accuracy_delta']['p97_5']:.6f} "
            f"(median {bootstrap['accuracy_delta']['p50']:.6f})"
        ),
        (
            "- net_correct_delta CI: "
            f"{bootstrap['net_correct_delta']['p2_5']:.6f} to {bootstrap['net_correct_delta']['p97_5']:.6f} "
            f"(median {bootstrap['net_correct_delta']['p50']:.6f})"
        ),
        "",
        "## Per seed",
        "",
    ]
    for seed in map(str, SEEDS):
        seed_report = report["per_seed"][seed]
        seed_delta = seed_report["metrics"]["delta_gated_minus_minimal"]
        seed_transitions = seed_report["transitions"]
        lines.append(
            "- seed "
            f"{seed}: delta acc/auc/f1 {seed_delta['accuracy']:.6f}/"
            f"{seed_delta['auc']:.6f}/{seed_delta['f1']:.6f}; "
            f"fixed {seed_transitions.get('fixed_by_gated', 0)}, "
            f"broken {seed_transitions.get('broken_by_gated', 0)}, "
            f"net {seed_report['net_correct_delta']}"
        )

    lines.extend(["", "## Recurring cases", ""])
    fixed_recurrence = report["patient_recurrence"]["fixed_by_gated"]
    broken_recurrence = report["patient_recurrence"]["broken_by_gated"]
    fixed_top = sorted(fixed_recurrence.items(), key=lambda item: (-item[1], item[0]))[:10]
    broken_top = sorted(broken_recurrence.items(), key=lambda item: (-item[1], item[0]))[:10]
    lines.append(
        "- fixed_by_gated top: "
        + (", ".join(f"{pid}({count})" for pid, count in fixed_top) if fixed_top else "none")
    )
    lines.append(
        "- broken_by_gated top: "
        + (", ".join(f"{pid}({count})" for pid, count in broken_top) if broken_top else "none")
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    predictions: dict[str, dict[int, dict[str, Any]]] = {
        "minimal_baseline": {},
        "gated_head": {},
    }
    for method in ("minimal_baseline", "gated_head"):
        for seed in SEEDS:
            predictions[method][seed] = load_run_predictions(
                method=method,
                seed=seed,
                run_info=RUNS[method][seed],
                device=device,
                args=args,
            )

    rows, per_seed = compare_runs(predictions)
    report = aggregate_report(
        predictions=predictions,
        rows=rows,
        per_seed=per_seed,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    if device.type == "cuda":
        report["runtime"] = {
            "device": str(device),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        }
    else:
        report["runtime"] = {"device": str(device), "peak_vram_mb": float("nan")}

    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    save_json(report, output_json)
    write_markdown(report, output_md)

    delta = report["mean_delta_gated_minus_minimal"]
    transitions = report["paired_transitions"]
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    print(
        "mean_delta_gated_minus_minimal="
        f"{delta['accuracy']:.6f}/{delta['auc']:.6f}/{delta['f1']:.6f}"
    )
    print(
        "fixed/broken/net="
        f"{transitions['fixed_by_gated']}/"
        f"{transitions['broken_by_gated']}/"
        f"{transitions['net_correct_delta']}"
    )


if __name__ == "__main__":
    main()
