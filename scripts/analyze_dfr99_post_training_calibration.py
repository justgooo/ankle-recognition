#!/usr/bin/env python3
"""Audit whether post-training confidence calibration can hit DFR-96 targets.

This analysis is intentionally label-aware on the validation telemetry: it is a
diagnostic for whether an observable, low-coverage gate-calibration rule exists.
It does not train a model or write checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import compute_metrics


VIEWS = ("axial", "coronal", "sagittal")
VIEW_INDEX = {view: index for index, view in enumerate(VIEWS)}
DFR25_SEED42_TELEMETRY = (
    "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR98_BEST_TELEMETRY = (
    "runs/optuna_main_autoloop/iter_0003_20260512_210836/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR96_REPORT = "autoresearch_logs/dfr96_gate_target_calibration_audit/report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path(DFR25_SEED42_TELEMETRY),
        help="Validation fusion_weight_analysis.json to calibrate.",
    )
    parser.add_argument(
        "--dfr98-telemetry",
        type=Path,
        default=Path(DFR98_BEST_TELEMETRY),
        help="Optional DFR-98 telemetry used only as a reference summary.",
    )
    parser.add_argument(
        "--dfr96-report",
        type=Path,
        default=Path(DFR96_REPORT),
        help="DFR-96 calibration report with target/protect sample categories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autoresearch_logs/dfr99_post_training_calibration_audit/report.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def round_float(value: float, digits: int = 6) -> float:
    if isinstance(value, float) and not np.isfinite(value):
        return value
    return round(float(value), digits)


def sample_top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str:
    views = sample.get(key) or []
    return str(views[0]) if views else "none"


def error_type(label: int, pred: int) -> str:
    if label == pred:
        return "TP" if label == 1 else "TN"
    return "FN" if label == 1 else "FP"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def telemetry_arrays(telemetry: dict[str, Any]) -> dict[str, Any]:
    samples = telemetry["samples"]
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    base_preds = np.asarray(
        [int(sample["fusion_prediction"]["pred"]) for sample in samples],
        dtype=np.int64,
    )
    confidence_logits = np.asarray(
        [
            [float(sample["views"][view]["confidence_logit"]) for view in VIEWS]
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
    view_preds = np.asarray(
        [
            [int(sample["views"][view]["pred"]) for view in VIEWS]
            for sample in samples
        ],
        dtype=np.int64,
    )
    true_margins = np.asarray(
        [
            [float(sample["views"][view]["true_margin"]) for view in VIEWS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    pred_margins = np.asarray(
        [
            [float(sample["views"][view]["pred_margin"]) for view in VIEWS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    return {
        "samples": samples,
        "patient_ids": [str(sample["patient_id"]) for sample in samples],
        "labels": labels,
        "base_preds": base_preds,
        "confidence_logits": confidence_logits,
        "abnormal_probs": abnormal_probs,
        "view_preds": view_preds,
        "true_margins": true_margins,
        "pred_margins": pred_margins,
    }


def metrics_from_abnormal_probs(labels: np.ndarray, abnormal_probs: np.ndarray) -> dict[str, float]:
    abnormal_probs = np.clip(abnormal_probs, 0.0, 1.0)
    preds = (abnormal_probs >= 0.5).astype(np.int64)
    probs = np.stack([1.0 - abnormal_probs, abnormal_probs], axis=1)
    return {
        key: round_float(value)
        for key, value in compute_metrics(labels, preds, probs).items()
    }


def summarize_weights(weights: np.ndarray) -> dict[str, Any]:
    top = np.argmax(weights, axis=1)
    counts = Counter(VIEWS[index] for index in top)
    total = int(weights.shape[0])
    return {
        "mean_fusion_weight": {
            view: round_float(weights[:, index].mean())
            for view, index in VIEW_INDEX.items()
        },
        "top_weight_count": {view: int(counts.get(view, 0)) for view in VIEWS},
        "top_weight_rate": {
            view: round_float(counts.get(view, 0) / total)
            for view in VIEWS
        },
    }


def target_sets(report: dict[str, Any]) -> dict[str, set[str]]:
    records = report["sample_level_calibration"]["records"]
    validated = {
        str(record["patient_id"])
        for record in records
        if record.get("validated_promote")
    }
    target_sagittal = {
        str(record["patient_id"])
        for record in records
        if record.get("recommended_target_view") == "sagittal"
    }
    missed = {
        str(record["patient_id"])
        for record in records
        if record.get("needs_nonaxial") and not record.get("validated_promote")
    }
    fragile = {
        str(record["patient_id"])
        for record in records
        if record.get("fragile_protect")
    }
    protect_axial = {
        str(record["patient_id"])
        for record in records
        if record.get("protect_axial")
    }
    return {
        "validated_promote": validated,
        "target_sagittal": target_sagittal,
        "missed_or_unvalidated_need": missed,
        "fragile_protect": fragile,
        "protect_axial": protect_axial,
    }


def triggered_mask(
    arrays: dict[str, Any],
    rule: str,
    axial_abnormal_threshold: float,
    sagittal_normal_threshold: float,
    fused_abnormal_threshold: float,
    confidence_gap_max: float,
) -> np.ndarray:
    confidence_logits = arrays["confidence_logits"]
    abnormal_probs = arrays["abnormal_probs"]
    view_preds = arrays["view_preds"]
    base_weights = softmax(confidence_logits)
    base_fused_abnormal = (base_weights * abnormal_probs).sum(axis=1)
    top_view = np.argmax(confidence_logits, axis=1)

    axial_abnormal = abnormal_probs[:, VIEW_INDEX["axial"]] >= axial_abnormal_threshold
    sagittal_normal = (
        1.0 - abnormal_probs[:, VIEW_INDEX["sagittal"]]
    ) >= sagittal_normal_threshold
    fused_abnormal = base_fused_abnormal >= fused_abnormal_threshold
    axial_top = top_view == VIEW_INDEX["axial"]
    sagittal_under_gap = (
        confidence_logits[:, VIEW_INDEX["axial"]]
        - confidence_logits[:, VIEW_INDEX["sagittal"]]
    ) <= confidence_gap_max
    sagittal_pred_normal = view_preds[:, VIEW_INDEX["sagittal"]] == 0
    coronal_not_abnormal = view_preds[:, VIEW_INDEX["coronal"]] == 0

    if rule == "sagittal_normal":
        return axial_top & axial_abnormal & sagittal_normal & fused_abnormal
    if rule == "sagittal_normal_gap":
        return axial_top & axial_abnormal & sagittal_normal & fused_abnormal & sagittal_under_gap
    if rule == "sagittal_normal_coronal_guard":
        return (
            axial_top
            & axial_abnormal
            & sagittal_normal
            & fused_abnormal
            & coronal_not_abnormal
        )
    if rule == "sagittal_pred_normal":
        return axial_top & fused_abnormal & sagittal_pred_normal
    raise ValueError(f"Unknown rule: {rule}")


def evaluate_candidate(
    arrays: dict[str, Any],
    sets: dict[str, set[str]],
    rule: str,
    residual: float,
    axial_abnormal_threshold: float,
    sagittal_normal_threshold: float,
    fused_abnormal_threshold: float,
    confidence_gap_max: float,
) -> dict[str, Any]:
    logits = arrays["confidence_logits"].copy()
    mask = triggered_mask(
        arrays,
        rule,
        axial_abnormal_threshold,
        sagittal_normal_threshold,
        fused_abnormal_threshold,
        confidence_gap_max,
    )
    logits[mask, VIEW_INDEX["sagittal"]] += residual
    weights = softmax(logits)
    abnormal = (weights * arrays["abnormal_probs"]).sum(axis=1)
    preds = (abnormal >= 0.5).astype(np.int64)
    labels = arrays["labels"]
    base_preds = arrays["base_preds"]
    patient_ids = arrays["patient_ids"]
    top = np.argmax(weights, axis=1)

    fixed = []
    broken = []
    triggered_ids = []
    sagittal_top_ids = []
    for index, patient_id in enumerate(patient_ids):
        if mask[index]:
            triggered_ids.append(patient_id)
        if top[index] == VIEW_INDEX["sagittal"]:
            sagittal_top_ids.append(patient_id)
        if base_preds[index] != labels[index] and preds[index] == labels[index]:
            fixed.append(patient_id)
        if base_preds[index] == labels[index] and preds[index] != labels[index]:
            broken.append(patient_id)

    triggered_set = set(triggered_ids)
    sagittal_top_set = set(sagittal_top_ids)
    fragile = sets["fragile_protect"]
    target = sets["target_sagittal"]
    validated = sets["validated_promote"]
    metrics = metrics_from_abnormal_probs(labels, abnormal)
    weight_summary = summarize_weights(weights)
    return {
        "rule": rule,
        "residual": round_float(residual),
        "axial_abnormal_threshold": round_float(axial_abnormal_threshold),
        "sagittal_normal_threshold": round_float(sagittal_normal_threshold),
        "fused_abnormal_threshold": round_float(fused_abnormal_threshold),
        "confidence_gap_max": round_float(confidence_gap_max),
        "metrics": metrics,
        "triggered_count": len(triggered_ids),
        "triggered_validated_target_count": len(triggered_set & validated),
        "triggered_target_sagittal_count": len(triggered_set & target),
        "triggered_fragile_protect_count": len(triggered_set & fragile),
        "sagittal_top_count": len(sagittal_top_ids),
        "sagittal_top_validated_target_count": len(sagittal_top_set & validated),
        "sagittal_top_target_sagittal_count": len(sagittal_top_set & target),
        "sagittal_top_fragile_protect_count": len(sagittal_top_set & fragile),
        "fixed_dfr25_errors": len(fixed),
        "broken_dfr25_correct": len(broken),
        "net_delta_vs_dfr25": len(fixed) - len(broken),
        "fixed_case_ids": sorted(fixed),
        "broken_case_ids": sorted(broken),
        "triggered_case_ids": sorted(triggered_ids),
        "sagittal_top_case_ids": sorted(sagittal_top_ids),
        "weight_summary": weight_summary,
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["metrics"]["accuracy"],
        candidate["metrics"]["auc"],
        candidate["net_delta_vs_dfr25"],
        candidate["sagittal_top_validated_target_count"],
        -candidate["broken_dfr25_correct"],
        -candidate["triggered_fragile_protect_count"],
        -candidate["triggered_count"],
    )


def grid_search(arrays: dict[str, Any], sets: dict[str, set[str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rules = (
        "sagittal_normal",
        "sagittal_normal_gap",
        "sagittal_normal_coronal_guard",
        "sagittal_pred_normal",
    )
    residuals = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    axial_thresholds = (0.55, 0.6, 0.65, 0.7, 0.75)
    sagittal_thresholds = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
    fused_thresholds = (0.5, 0.55, 0.6, 0.65)
    gap_thresholds = (2.0, 2.5, 3.0, 3.5, 4.0)

    for rule in rules:
        for residual in residuals:
            for axial_threshold in axial_thresholds:
                for sagittal_threshold in sagittal_thresholds:
                    for fused_threshold in fused_thresholds:
                        for gap_threshold in gap_thresholds:
                            candidates.append(
                                evaluate_candidate(
                                    arrays,
                                    sets,
                                    rule,
                                    residual,
                                    axial_threshold,
                                    sagittal_threshold,
                                    fused_threshold,
                                    gap_threshold,
                                )
                            )
    return candidates


def summarize_existing_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    samples = telemetry["samples"]
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    abnormal = np.asarray(
        [float(sample["fusion_prediction"]["abnormal_prob"]) for sample in samples],
        dtype=np.float64,
    )
    weights = np.asarray(
        [
            [float(sample["views"][view]["fusion_weight"]) for view in VIEWS]
            for sample in samples
        ],
        dtype=np.float64,
    )
    preds = (abnormal >= 0.5).astype(np.int64)
    return {
        "metrics": metrics_from_abnormal_probs(labels, abnormal),
        "weight_summary": summarize_weights(weights),
        "error_types": {
            key: int(value)
            for key, value in Counter(
                error_type(int(label), int(pred))
                for label, pred in zip(labels, preds)
            ).items()
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    telemetry_path = (repo_root / args.telemetry).resolve()
    dfr98_path = (repo_root / args.dfr98_telemetry).resolve()
    dfr96_path = (repo_root / args.dfr96_report).resolve()

    telemetry = load_json(telemetry_path)
    dfr98 = load_json(dfr98_path)
    dfr96 = load_json(dfr96_path)
    arrays = telemetry_arrays(telemetry)
    sets = target_sets(dfr96)
    base_weights = softmax(arrays["confidence_logits"])
    base_abnormal = (base_weights * arrays["abnormal_probs"]).sum(axis=1)
    base_summary = {
        "metrics": metrics_from_abnormal_probs(arrays["labels"], base_abnormal),
        "weight_summary": summarize_weights(base_weights),
    }
    candidates = grid_search(arrays, sets)
    candidates_sorted = sorted(candidates, key=candidate_sort_key, reverse=True)
    best = candidates_sorted[0]
    target_sorted = sorted(
        candidates,
        key=lambda item: (
            item["sagittal_top_validated_target_count"],
            item["metrics"]["accuracy"],
            item["metrics"]["auc"],
            -item["broken_dfr25_correct"],
            -item["sagittal_top_fragile_protect_count"],
        ),
        reverse=True,
    )
    target_best = target_sorted[0]
    no_harm_candidates = [
        item
        for item in candidates
        if item["broken_dfr25_correct"] == 0
        and item["triggered_fragile_protect_count"] == 0
    ]
    no_harm_best = (
        sorted(no_harm_candidates, key=candidate_sort_key, reverse=True)[0]
        if no_harm_candidates
        else None
    )
    report = {
        "analysis": "dfr99_post_training_calibration_audit",
        "description": (
            "Simulates label-free post-training sagittal confidence-logit "
            "calibration rules on DFR-25 seed42 validation telemetry.  The goal "
            "is to test whether observable triggers can cover the DFR-96 tiny "
            "sagittal target without harming fragile axial-protect samples."
        ),
        "input_telemetry": str(args.telemetry),
        "dfr98_reference_telemetry": str(args.dfr98_telemetry),
        "dfr96_report": str(args.dfr96_report),
        "target_sets": {key: sorted(value) for key, value in sets.items()},
        "baseline_from_confidence_logits": base_summary,
        "dfr98_reference": summarize_existing_telemetry(dfr98),
        "candidate_count": len(candidates),
        "best_by_accuracy_auc": best,
        "best_by_sagittal_target_coverage": target_best,
        "best_no_harm_candidate": no_harm_best,
        "top_20_by_accuracy_auc": candidates_sorted[:20],
        "top_20_by_sagittal_target_coverage": target_sorted[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Saved DFR-99 post-training calibration audit to: {args.output}")
    print(
        "baseline: "
        f"acc={base_summary['metrics']['accuracy']:.6f} "
        f"auc={base_summary['metrics']['auc']:.6f} "
        f"f1={base_summary['metrics']['f1']:.6f} "
        f"top={base_summary['weight_summary']['top_weight_count']} "
        f"mean={base_summary['weight_summary']['mean_fusion_weight']}"
    )
    print(
        "best_by_accuracy_auc: "
        f"rule={best['rule']} residual={best['residual']} "
        f"acc={best['metrics']['accuracy']:.6f} "
        f"auc={best['metrics']['auc']:.6f} "
        f"f1={best['metrics']['f1']:.6f} "
        f"net={best['net_delta_vs_dfr25']} "
        f"triggered={best['triggered_count']} "
        f"sagittal_top={best['sagittal_top_count']} "
        f"validated_top={best['sagittal_top_validated_target_count']} "
        f"fragile_top={best['sagittal_top_fragile_protect_count']} "
        f"top={best['weight_summary']['top_weight_count']}"
    )
    print(
        "best_by_sagittal_target_coverage: "
        f"rule={target_best['rule']} residual={target_best['residual']} "
        f"acc={target_best['metrics']['accuracy']:.6f} "
        f"auc={target_best['metrics']['auc']:.6f} "
        f"net={target_best['net_delta_vs_dfr25']} "
        f"sagittal_top={target_best['sagittal_top_count']} "
        f"validated_top={target_best['sagittal_top_validated_target_count']} "
        f"fragile_top={target_best['sagittal_top_fragile_protect_count']} "
        f"top={target_best['weight_summary']['top_weight_count']}"
    )
    if no_harm_best is None:
        print("best_no_harm_candidate: none")
    else:
        print(
            "best_no_harm_candidate: "
            f"rule={no_harm_best['rule']} residual={no_harm_best['residual']} "
            f"acc={no_harm_best['metrics']['accuracy']:.6f} "
            f"auc={no_harm_best['metrics']['auc']:.6f} "
            f"triggered={no_harm_best['triggered_count']} "
            f"sagittal_top={no_harm_best['sagittal_top_count']} "
            f"validated_top={no_harm_best['sagittal_top_validated_target_count']}"
        )


if __name__ == "__main__":
    main()
