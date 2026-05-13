#!/usr/bin/env python3
"""Multi-seed validation for the DFR-111 post-hoc sagittal rescue rule.

This analysis does not train or modify checkpoints.  It applies the best
label-free DFR-111 fp-risk sagittal gate residual to the existing DFR-25
seed42/123/456 fusion telemetry, then optionally ranks the same observable rule
family by 3-seed aggregate accuracy.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_dfr111_confirmed_target_posthoc import (  # noqa: E402
    VIEWS,
    VIEW_INDEX,
    base_error_type,
    counter_dict,
    load_json,
    metrics_from_abnormal,
    round_float,
    save_json,
    softmax,
    summarize_weights,
)


DFR25_TELEMETRY = {
    "42": (
        "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
        "trials/trial_0001/run/fusion_weight_analysis.json"
    ),
    "123": (
        "runs/optuna_main_autoloop/iter_0002_20260423_031222/"
        "trials/trial_0000/run/fusion_weight_analysis.json"
    ),
    "456": (
        "runs/optuna_main_autoloop/iter_0003_20260423_033048/"
        "trials/trial_0000/run/fusion_weight_analysis.json"
    ),
}
DEFAULT_OUTPUT = "autoresearch_logs/dfr112_multiseed_posthoc_fp_risk.json"
FIXED_DFR111_FP_RISK = {
    "rule": "fp_risk",
    "residual": 5.0,
    "params": {
        "axial_min": 0.4,
        "axial_max": 0.9,
        "fused_abnormal_min": 0.8,
        "sagittal_normal_min": 0.55,
        "coronal_abnormal_max": 0.525,
        "confidence_gap_max": 5.0,
    },
}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def telemetry_arrays_model_equivalent(telemetry: dict[str, Any]) -> dict[str, Any]:
    samples = telemetry["samples"]
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    base_preds = np.asarray(
        [int(sample["fusion_prediction"]["pred"]) for sample in samples],
        dtype=np.int64,
    )
    base_abnormal = np.asarray(
        [float(sample["fusion_prediction"]["abnormal_prob"]) for sample in samples],
        dtype=np.float64,
    )
    scaled_confidence_logits = np.asarray(
        [
            [
                float(
                    sample["views"][view].get(
                        "scaled_confidence_logit",
                        sample["views"][view]["confidence_logit"],
                    )
                )
                for view in VIEWS
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )
    abnormal_probs = np.asarray(
        [
            [float(sample["views"][view]["abnormal_prob"]) for view in VIEWS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    return {
        "samples": samples,
        "patient_ids": [str(sample["patient_id"]) for sample in samples],
        "labels": labels,
        "base_preds": base_preds,
        "base_abnormal": base_abnormal,
        "scaled_confidence_logits": scaled_confidence_logits,
        "abnormal_probs": abnormal_probs,
        "view_margins": probability_logit(abnormal_probs),
    }


def model_equivalent_abnormal(
    arrays: dict[str, Any],
    scaled_confidence_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = softmax(scaled_confidence_logits)
    fused_margin = (weights * arrays["view_margins"]).sum(axis=1)
    return sigmoid(fused_margin), weights


def summarize_base_model_equivalent(
    telemetry: dict[str, Any],
    arrays: dict[str, Any],
) -> dict[str, Any]:
    labels = arrays["labels"]
    model_abnormal, model_weights = model_equivalent_abnormal(
        arrays,
        arrays["scaled_confidence_logits"],
    )
    stored_weights = np.asarray(
        [
            [float(sample["views"][view]["fusion_weight"]) for view in VIEWS]
            for sample in telemetry["samples"]
        ],
        dtype=np.float64,
    )
    return {
        "stored_fusion_metrics": {
            key: round_float(value)
            for key, value in telemetry["summary"]["full_fusion_metrics"].items()
        },
        "stored_weight_summary": summarize_weights(stored_weights),
        "model_equivalent_metrics": metrics_from_abnormal(labels, model_abnormal),
        "model_equivalent_weight_summary": summarize_weights(model_weights),
        "max_abs_abnormal_delta_vs_stored": round_float(
            float(np.max(np.abs(model_abnormal - arrays["base_abnormal"]))),
            digits=10,
        ),
        "error_types": counter_dict(
            Counter(
                base_error_type(int(label), int(pred))
                for label, pred in zip(labels, arrays["base_preds"])
            )
        ),
    }


def observable_mask_model_equivalent(
    arrays: dict[str, Any],
    *,
    axial_min: float,
    axial_max: float,
    sagittal_normal_min: float,
    coronal_abnormal_max: float,
    fused_abnormal_min: float,
    confidence_gap_max: float,
) -> np.ndarray:
    logits = arrays["scaled_confidence_logits"]
    abnormal = arrays["abnormal_probs"]
    fused, _weights = model_equivalent_abnormal(arrays, logits)
    top = np.argmax(logits, axis=1)
    axial_abnormal = abnormal[:, VIEW_INDEX["axial"]]
    sagittal_normal = 1.0 - abnormal[:, VIEW_INDEX["sagittal"]]
    coronal_abnormal = abnormal[:, VIEW_INDEX["coronal"]]
    confidence_gap = logits[:, VIEW_INDEX["axial"]] - logits[:, VIEW_INDEX["sagittal"]]
    return (
        (top == VIEW_INDEX["axial"])
        & (fused >= fused_abnormal_min)
        & (axial_abnormal >= axial_min)
        & (axial_abnormal <= axial_max)
        & (sagittal_normal >= sagittal_normal_min)
        & (coronal_abnormal <= coronal_abnormal_max)
        & (confidence_gap <= confidence_gap_max)
    )


def evaluate_fp_risk_candidate(
    arrays: dict[str, Any],
    residual: float,
    params: dict[str, float],
) -> dict[str, Any]:
    mask = observable_mask_model_equivalent(arrays, **params)
    logits = arrays["scaled_confidence_logits"].copy()
    logits[mask, VIEW_INDEX["sagittal"]] += residual
    abnormal, weights = model_equivalent_abnormal(arrays, logits)
    labels = arrays["labels"]
    preds = (abnormal >= 0.5).astype(np.int64)
    base_preds = arrays["base_preds"]
    patient_ids = arrays["patient_ids"]
    top = np.argmax(weights, axis=1)

    fixed = []
    broken = []
    triggered = []
    sagittal_top = []
    for index, patient_id in enumerate(patient_ids):
        patient_id = str(patient_id)
        if mask[index]:
            triggered.append(patient_id)
        if top[index] == VIEW_INDEX["sagittal"]:
            sagittal_top.append(patient_id)
        if base_preds[index] != labels[index] and preds[index] == labels[index]:
            fixed.append(patient_id)
        elif base_preds[index] == labels[index] and preds[index] != labels[index]:
            broken.append(patient_id)

    return {
        "rule": "fp_risk",
        "residual": round_float(residual),
        "params": {key: round_float(value) for key, value in params.items()},
        "metrics": metrics_from_abnormal(labels, abnormal),
        "triggered_count": len(triggered),
        "triggered_target_count": 0,
        "sagittal_top_count": len(sagittal_top),
        "sagittal_top_target_count": 0,
        "fixed_dfr25_errors": len(fixed),
        "broken_dfr25_correct": len(broken),
        "net_delta_vs_dfr25": len(fixed) - len(broken),
        "fixed_ids": sorted(fixed),
        "broken_ids": sorted(broken),
        "triggered_ids": sorted(triggered),
        "triggered_target_ids": [],
        "sagittal_top_ids": sorted(sagittal_top),
        "sagittal_top_target_ids": [],
        "weight_summary": summarize_weights(weights),
    }


def summarize_candidate_across_seeds(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in per_seed.values()]
    weight_summaries = [item["weight_summary"] for item in per_seed.values()]
    total_triggered = sum(int(item["triggered_count"]) for item in per_seed.values())
    total_sagittal_top = sum(int(item["sagittal_top_count"]) for item in per_seed.values())
    total_fixed = sum(int(item["fixed_dfr25_errors"]) for item in per_seed.values())
    total_broken = sum(int(item["broken_dfr25_correct"]) for item in per_seed.values())
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    for summary in weight_summaries:
        for view in VIEWS:
            top_counts[view] += int(summary["top_weight_count"][view])
            mean_weights[view].append(float(summary["mean_fusion_weight"][view]))
    return {
        "mean_metrics": {
            "accuracy": round_float(mean([float(item["accuracy"]) for item in metrics])),
            "auc": round_float(mean([float(item["auc"]) for item in metrics])),
            "f1": round_float(mean([float(item["f1"]) for item in metrics])),
        },
        "total_triggered": int(total_triggered),
        "total_sagittal_top": int(total_sagittal_top),
        "total_fixed_dfr25_errors": int(total_fixed),
        "total_broken_dfr25_correct": int(total_broken),
        "total_net_delta_vs_dfr25": int(total_fixed - total_broken),
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
    }


def baseline_aggregate(seed_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        report["baseline"]["stored_fusion_metrics"]
        for report in seed_reports.values()
    ]
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    error_types = Counter()
    for report in seed_reports.values():
        weight_summary = report["baseline"]["stored_weight_summary"]
        for view in VIEWS:
            top_counts[view] += int(weight_summary["top_weight_count"][view])
            mean_weights[view].append(float(weight_summary["mean_fusion_weight"][view]))
        error_types.update(report["baseline"]["error_types"])
    return {
        "mean_metrics": {
            "accuracy": round_float(mean([float(item["accuracy"]) for item in metrics])),
            "auc": round_float(mean([float(item["auc"]) for item in metrics])),
            "f1": round_float(mean([float(item["f1"]) for item in metrics])),
        },
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
        "error_types": counter_dict(error_types),
    }


def fixed_candidate_for_seed(arrays: dict[str, Any]) -> dict[str, Any]:
    return evaluate_fp_risk_candidate(
        arrays,
        float(FIXED_DFR111_FP_RISK["residual"]),
        dict(FIXED_DFR111_FP_RISK["params"]),
    )


def grid_candidates_for_seed(arrays: dict[str, Any]) -> list[dict[str, Any]]:
    residuals = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    axial_mins = (0.40, 0.50, 0.60, 0.70, 0.80, 0.84)
    axial_maxes = (0.88, 0.90, 0.93, 0.95, 0.98)
    sagittal_normal_mins = (0.55, 0.60, 0.64, 0.70, 0.75)
    coronal_abnormal_maxes = (0.48, 0.50, 0.525, 0.55, 0.60)
    fused_abnormal_mins = (0.60, 0.70, 0.75, 0.80, 0.85)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for residual in residuals:
        for axial_min in axial_mins:
            for axial_max in axial_maxes:
                if axial_min > axial_max:
                    continue
                for sagittal_normal_min in sagittal_normal_mins:
                    for coronal_abnormal_max in coronal_abnormal_maxes:
                        for fused_abnormal_min in fused_abnormal_mins:
                            params = {
                                "axial_min": axial_min,
                                "axial_max": axial_max,
                                "sagittal_normal_min": sagittal_normal_min,
                                "coronal_abnormal_max": coronal_abnormal_max,
                                "fused_abnormal_min": fused_abnormal_min,
                                "confidence_gap_max": 5.0,
                            }
                            candidate = evaluate_fp_risk_candidate(
                                arrays,
                                residual,
                                params,
                            )
                            key = (
                                residual,
                                tuple(candidate["triggered_ids"]),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            candidates.append(candidate)
    return candidates


def candidate_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    params = candidate["params"]
    return (
        candidate["rule"],
        float(candidate["residual"]),
        float(params["axial_min"]),
        float(params["axial_max"]),
        float(params["sagittal_normal_min"]),
        float(params["coronal_abnormal_max"]),
        float(params["fused_abnormal_min"]),
        float(params["confidence_gap_max"]),
    )


def candidate_from_identity(arrays: dict[str, Any], identity: tuple[Any, ...]) -> dict[str, Any]:
    rule, residual, axial_min, axial_max, sagittal_normal_min, coronal_abnormal_max, fused_abnormal_min, confidence_gap_max = identity
    return evaluate_fp_risk_candidate(
        arrays,
        float(residual),
        {
            "axial_min": float(axial_min),
            "axial_max": float(axial_max),
            "sagittal_normal_min": float(sagittal_normal_min),
            "coronal_abnormal_max": float(coronal_abnormal_max),
            "fused_abnormal_min": float(fused_abnormal_min),
            "confidence_gap_max": float(confidence_gap_max),
        },
    )


def aggregate_grid(seed_arrays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    identities: set[tuple[Any, ...]] = set()
    for arrays in seed_arrays.values():
        for candidate in grid_candidates_for_seed(arrays):
            identities.add(candidate_identity(candidate))

    aggregated: list[dict[str, Any]] = []
    for identity in identities:
        per_seed = {
            seed: candidate_from_identity(arrays, identity)
            for seed, arrays in seed_arrays.items()
        }
        summary = summarize_candidate_across_seeds(per_seed)
        aggregated.append(
            {
                "identity": {
                    "rule": identity[0],
                    "residual": round_float(float(identity[1])),
                    "params": {
                        "axial_min": round_float(float(identity[2])),
                        "axial_max": round_float(float(identity[3])),
                        "sagittal_normal_min": round_float(float(identity[4])),
                        "coronal_abnormal_max": round_float(float(identity[5])),
                        "fused_abnormal_min": round_float(float(identity[6])),
                        "confidence_gap_max": round_float(float(identity[7])),
                    },
                },
                "aggregate": summary,
                "per_seed": per_seed,
            }
        )
    return aggregated


def aggregate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    per_seed = item["per_seed"]
    min_acc = min(float(seed["metrics"]["accuracy"]) for seed in per_seed.values())
    max_broken = max(int(seed["broken_dfr25_correct"]) for seed in per_seed.values())
    return (
        metrics["accuracy"],
        metrics["auc"],
        min_acc,
        aggregate["total_net_delta_vs_dfr25"],
        aggregate["total_sagittal_top"],
        -aggregate["total_broken_dfr25_correct"],
        -max_broken,
        -aggregate["total_triggered"],
    )


def no_harm_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    per_seed = item["per_seed"]
    min_acc = min(float(seed["metrics"]["accuracy"]) for seed in per_seed.values())
    return (
        aggregate["total_broken_dfr25_correct"] == 0,
        metrics["accuracy"],
        metrics["auc"],
        min_acc,
        aggregate["total_fixed_dfr25_errors"],
        aggregate["total_sagittal_top"],
        -aggregate["total_triggered"],
    )


def triggered_patient_details(
    arrays: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    samples = arrays["samples"]
    sample_by_id = {str(sample["patient_id"]): sample for sample in samples}
    details: list[dict[str, Any]] = []
    for patient_id in candidate["triggered_ids"]:
        sample = sample_by_id[str(patient_id)]
        details.append(
            {
                "patient_id": str(patient_id),
                "label": int(sample["label"]),
                "base_pred": int(sample["fusion_prediction"]["pred"]),
                "base_abnormal": round_float(float(sample["fusion_prediction"]["abnormal_prob"])),
                "view_abnormal": {
                    view: round_float(float(sample["views"][view]["abnormal_prob"]))
                    for view in VIEWS
                },
                "confidence_logits": {
                    view: round_float(float(sample["views"][view]["confidence_logit"]))
                    for view in VIEWS
                },
                "base_weights": {
                    view: round_float(float(sample["views"][view]["fusion_weight"]))
                    for view in VIEWS
                },
            }
        )
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Only evaluate the fixed DFR-111 fp-risk rule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    seed_arrays: dict[str, dict[str, Any]] = {}
    seed_reports: dict[str, dict[str, Any]] = {}
    for seed, relative_path in DFR25_TELEMETRY.items():
        telemetry = load_json(repo_root / relative_path)
        arrays = telemetry_arrays_model_equivalent(telemetry)
        seed_arrays[seed] = arrays
        seed_reports[seed] = {
            "input_telemetry": relative_path,
            "baseline": summarize_base_model_equivalent(telemetry, arrays),
        }

    fixed_per_seed = {
        seed: fixed_candidate_for_seed(arrays)
        for seed, arrays in seed_arrays.items()
    }
    for seed, candidate in fixed_per_seed.items():
        seed_reports[seed]["fixed_dfr111_fp_risk"] = candidate
        seed_reports[seed]["fixed_triggered_patient_details"] = triggered_patient_details(
            seed_arrays[seed],
            candidate,
        )

    grid_ranked: list[dict[str, Any]] = []
    best_no_harm: dict[str, Any] | None = None
    if not args.skip_grid:
        grid = aggregate_grid(seed_arrays)
        grid_ranked = sorted(grid, key=aggregate_sort_key, reverse=True)
        no_harm_ranked = sorted(grid, key=no_harm_sort_key, reverse=True)
        best_no_harm = no_harm_ranked[0] if no_harm_ranked else None

    fixed_summary = summarize_candidate_across_seeds(fixed_per_seed)
    baseline_summary = baseline_aggregate(seed_reports)
    report = {
        "analysis": "dfr112_multiseed_posthoc_fp_risk",
        "description": (
            "Analysis-only 3-seed validation of the DFR-111 label-free fp-risk "
            "post-hoc sagittal gate residual on DFR-25 telemetry.  No training, "
            "checkpoint mutation, or test-set metric is used."
        ),
        "fixed_rule": FIXED_DFR111_FP_RISK,
        "baseline_dfr25_aggregate": baseline_summary,
        "fixed_rule_aggregate": fixed_summary,
        "per_seed": seed_reports,
        "best_grid_by_mean_accuracy": grid_ranked[0] if grid_ranked else None,
        "best_grid_no_harm_preferred": best_no_harm,
        "top_20_grid_by_mean_accuracy": grid_ranked[:20],
    }

    save_json(repo_root / args.output, report)
    print(f"Saved DFR-112 multi-seed post-hoc report to: {repo_root / args.output}")
    print("baseline", baseline_summary["mean_metrics"], baseline_summary["aggregate_top_weight_count"])
    print("fixed", fixed_summary["mean_metrics"], fixed_summary["aggregate_top_weight_count"])
    print(
        "fixed_delta",
        "net=",
        fixed_summary["total_net_delta_vs_dfr25"],
        "fixed=",
        fixed_summary["total_fixed_dfr25_errors"],
        "broken=",
        fixed_summary["total_broken_dfr25_correct"],
        "triggered=",
        fixed_summary["total_triggered"],
    )
    if grid_ranked:
        best = grid_ranked[0]
        print("best_grid", best["identity"], best["aggregate"]["mean_metrics"], best["aggregate"]["aggregate_top_weight_count"])
    if best_no_harm:
        print(
            "best_no_harm",
            best_no_harm["identity"],
            best_no_harm["aggregate"]["mean_metrics"],
            best_no_harm["aggregate"]["aggregate_top_weight_count"],
        )


if __name__ == "__main__":
    main()
