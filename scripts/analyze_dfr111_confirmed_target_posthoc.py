#!/usr/bin/env python3
"""Post-hoc upper-bound scan for DFR-108 confirmed sagittal targets.

This analysis does not train a model.  It reweights DFR-25 seed42 validation
telemetry by adding small residuals to the sagittal gate confidence logit under
observable rules, then recomputes the probability-mixture decision.
"""

from __future__ import annotations

import argparse
import json
import math
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
NONAXIAL = ("coronal", "sagittal")
DFR25 = (
    "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DEFAULT_OUTPUT = "autoresearch_logs/dfr111_confirmed_target_posthoc_scan.json"
EPS = 1e-8


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def round_float(value: float, digits: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        return float(value)
    return round(float(value), digits)


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def class_prob(abnormal: float, class_index: int) -> float:
    return abnormal if class_index == 1 else 1.0 - abnormal


def class_log_prob(abnormal: float, class_index: int) -> float:
    return math.log(max(class_prob(abnormal, class_index), EPS))


def class_margin(abnormal: float, class_index: int) -> float:
    return class_log_prob(abnormal, class_index) - class_log_prob(abnormal, 1 - class_index)


def telemetry_arrays(telemetry: dict[str, Any]) -> dict[str, Any]:
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
    true_margins = np.asarray(
        [
            [float(sample["views"][view]["true_margin"]) for view in VIEWS]
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
        "confidence_logits": confidence_logits,
        "abnormal_probs": abnormal_probs,
        "true_margins": true_margins,
    }


def metrics_from_abnormal(labels: np.ndarray, abnormal: np.ndarray) -> dict[str, float]:
    abnormal = np.clip(abnormal, 0.0, 1.0)
    preds = (abnormal >= 0.5).astype(np.int64)
    probs = np.stack([1.0 - abnormal, abnormal], axis=1)
    return {key: round_float(value) for key, value in compute_metrics(labels, preds, probs).items()}


def summarize_weights(weights: np.ndarray) -> dict[str, Any]:
    top = np.argmax(weights, axis=1)
    counts = Counter(VIEWS[int(index)] for index in top)
    total = int(weights.shape[0])
    return {
        "mean_fusion_weight": {
            view: round_float(float(weights[:, index].mean()))
            for view, index in VIEW_INDEX.items()
        },
        "top_weight_count": {view: int(counts.get(view, 0)) for view in VIEWS},
        "top_weight_rate": {
            view: round_float(counts.get(view, 0) / total)
            for view in VIEWS
        },
    }


def base_error_type(label: int, pred: int) -> str:
    if label == pred:
        return "TP" if label == 1 else "TN"
    return "FN" if label == 1 else "FP"


def confirmed_true_target_ids(arrays: dict[str, Any], gap: float) -> set[str]:
    ids: set[str] = set()
    labels = arrays["labels"]
    abnormal = arrays["abnormal_probs"]
    patient_ids = arrays["patient_ids"]
    for index, patient_id in enumerate(patient_ids):
        class_index = int(labels[index])
        axial_log_prob = class_log_prob(float(abnormal[index, VIEW_INDEX["axial"]]), class_index)
        best_advantage = -math.inf
        best_view = "none"
        for view in NONAXIAL:
            view_abnormal = float(abnormal[index, VIEW_INDEX[view]])
            if class_margin(view_abnormal, class_index) < 0.0:
                advantage = -math.inf
            else:
                advantage = class_log_prob(view_abnormal, class_index) - axial_log_prob
            if advantage > best_advantage:
                best_advantage = advantage
                best_view = view
        if best_advantage >= gap and best_view == "sagittal":
            ids.add(str(patient_id))
    return ids


def observable_mask(
    arrays: dict[str, Any],
    rule: str,
    *,
    axial_min: float,
    axial_max: float,
    sagittal_normal_min: float,
    coronal_abnormal_max: float,
    fused_abnormal_min: float,
    confidence_gap_max: float,
) -> np.ndarray:
    logits = arrays["confidence_logits"]
    weights = softmax(logits)
    abnormal = arrays["abnormal_probs"]
    fused = (weights * abnormal).sum(axis=1)
    top = np.argmax(logits, axis=1)
    axial_abnormal = abnormal[:, VIEW_INDEX["axial"]]
    sagittal_normal = 1.0 - abnormal[:, VIEW_INDEX["sagittal"]]
    coronal_abnormal = abnormal[:, VIEW_INDEX["coronal"]]
    confidence_gap = logits[:, VIEW_INDEX["axial"]] - logits[:, VIEW_INDEX["sagittal"]]

    base = (
        (top == VIEW_INDEX["axial"])
        & (fused >= fused_abnormal_min)
        & (axial_abnormal >= axial_min)
        & (axial_abnormal <= axial_max)
        & (sagittal_normal >= sagittal_normal_min)
    )
    if rule == "fp_risk":
        return base & (coronal_abnormal <= coronal_abnormal_max)
    if rule == "fp_risk_gap":
        return base & (coronal_abnormal <= coronal_abnormal_max) & (confidence_gap <= confidence_gap_max)
    if rule == "fp_risk_no_coronal_guard":
        return base
    if rule == "target_like_true_margin_proxy":
        return (
            base
            & (coronal_abnormal <= coronal_abnormal_max)
            & ((1.0 - abnormal[:, VIEW_INDEX["sagittal"]]) > (1.0 - axial_abnormal))
        )
    raise ValueError(f"Unknown rule: {rule}")


def candidate_mask(
    arrays: dict[str, Any],
    rule: str,
    target_ids: set[str],
    params: dict[str, float],
) -> np.ndarray:
    if rule == "oracle_true_targets":
        return np.asarray([pid in target_ids for pid in arrays["patient_ids"]], dtype=bool)
    return observable_mask(arrays, rule, **params)


def evaluate_candidate(
    arrays: dict[str, Any],
    target_ids: set[str],
    rule: str,
    residual: float,
    params: dict[str, float],
) -> dict[str, Any]:
    mask = candidate_mask(arrays, rule, target_ids, params)
    logits = arrays["confidence_logits"].copy()
    logits[mask, VIEW_INDEX["sagittal"]] += residual
    weights = softmax(logits)
    abnormal = (weights * arrays["abnormal_probs"]).sum(axis=1)
    labels = arrays["labels"]
    preds = (abnormal >= 0.5).astype(np.int64)
    base_preds = arrays["base_preds"]
    patient_ids = arrays["patient_ids"]
    top = np.argmax(weights, axis=1)

    fixed = []
    broken = []
    triggered = []
    target_triggered = []
    sagittal_top = []
    sagittal_top_targets = []
    for index, patient_id in enumerate(patient_ids):
        patient_id = str(patient_id)
        if mask[index]:
            triggered.append(patient_id)
            if patient_id in target_ids:
                target_triggered.append(patient_id)
        if top[index] == VIEW_INDEX["sagittal"]:
            sagittal_top.append(patient_id)
            if patient_id in target_ids:
                sagittal_top_targets.append(patient_id)
        if base_preds[index] != labels[index] and preds[index] == labels[index]:
            fixed.append(patient_id)
        elif base_preds[index] == labels[index] and preds[index] != labels[index]:
            broken.append(patient_id)

    metrics = metrics_from_abnormal(labels, abnormal)
    return {
        "rule": rule,
        "residual": round_float(residual),
        "params": {key: round_float(value) for key, value in params.items()},
        "metrics": metrics,
        "triggered_count": len(triggered),
        "triggered_target_count": len(target_triggered),
        "sagittal_top_count": len(sagittal_top),
        "sagittal_top_target_count": len(sagittal_top_targets),
        "fixed_dfr25_errors": len(fixed),
        "broken_dfr25_correct": len(broken),
        "net_delta_vs_dfr25": len(fixed) - len(broken),
        "fixed_ids": sorted(fixed),
        "broken_ids": sorted(broken),
        "triggered_ids": sorted(triggered),
        "triggered_target_ids": sorted(target_triggered),
        "sagittal_top_ids": sorted(sagittal_top),
        "sagittal_top_target_ids": sorted(sagittal_top_targets),
        "weight_summary": summarize_weights(weights),
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["metrics"]["accuracy"],
        candidate["metrics"]["auc"],
        candidate["net_delta_vs_dfr25"],
        candidate["sagittal_top_target_count"],
        -candidate["broken_dfr25_correct"],
        -candidate["triggered_count"],
    )


def target_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["sagittal_top_target_count"],
        candidate["triggered_target_count"],
        candidate["metrics"]["accuracy"],
        candidate["net_delta_vs_dfr25"],
        -candidate["broken_dfr25_correct"],
        -candidate["triggered_count"],
    )


def build_grid(arrays: dict[str, Any], target_ids: set[str]) -> list[dict[str, Any]]:
    rules = (
        "oracle_true_targets",
        "fp_risk",
        "fp_risk_gap",
        "fp_risk_no_coronal_guard",
        "target_like_true_margin_proxy",
    )
    residuals = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
    axial_mins = (0.40, 0.50, 0.60, 0.70, 0.80, 0.84)
    axial_maxes = (0.90, 0.93, 0.95, 0.98, 1.0)
    sagittal_normal_mins = (0.55, 0.60, 0.64, 0.70, 0.75)
    coronal_abnormal_maxes = (0.48, 0.50, 0.525, 0.55, 0.60)
    fused_abnormal_mins = (0.45, 0.50, 0.60, 0.70, 0.75, 0.80)
    confidence_gap_maxes = (2.0, 3.0, 4.0, 5.0)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for rule in rules:
        for residual in residuals:
            for axial_min in axial_mins:
                for axial_max in axial_maxes:
                    if axial_min > axial_max:
                        continue
                    for sagittal_normal_min in sagittal_normal_mins:
                        for coronal_abnormal_max in coronal_abnormal_maxes:
                            for fused_abnormal_min in fused_abnormal_mins:
                                for confidence_gap_max in confidence_gap_maxes:
                                    params = {
                                        "axial_min": axial_min,
                                        "axial_max": axial_max,
                                        "sagittal_normal_min": sagittal_normal_min,
                                        "coronal_abnormal_max": coronal_abnormal_max,
                                        "fused_abnormal_min": fused_abnormal_min,
                                        "confidence_gap_max": confidence_gap_max,
                                    }
                                    mask = candidate_mask(arrays, rule, target_ids, params)
                                    key = (
                                        rule,
                                        residual,
                                        tuple(np.flatnonzero(mask).tolist()),
                                    )
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    candidates.append(
                                        evaluate_candidate(
                                            arrays,
                                            target_ids,
                                            rule,
                                            residual,
                                            params,
                                        )
                                    )
    return candidates


def summarize_base(telemetry: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    labels = arrays["labels"]
    base_abnormal = arrays["base_abnormal"]
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
        "probability_mixture_metrics": metrics_from_abnormal(labels, base_abnormal),
        "error_types": counter_dict(
            Counter(
                base_error_type(int(label), int(pred))
                for label, pred in zip(labels, arrays["base_preds"])
            )
        ),
    }


def target_records(arrays: dict[str, Any], target_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, patient_id in enumerate(arrays["patient_ids"]):
        if patient_id not in target_ids:
            continue
        label = int(arrays["labels"][index])
        pred = int(arrays["base_preds"][index])
        records.append(
            {
                "patient_id": patient_id,
                "label": label,
                "base_pred": pred,
                "base_error_type": base_error_type(label, pred),
                "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                "abnormal_probs": {
                    view: round_float(float(arrays["abnormal_probs"][index, view_index]))
                    for view, view_index in VIEW_INDEX.items()
                },
                "confidence_logits": {
                    view: round_float(float(arrays["confidence_logits"][index, view_index]))
                    for view, view_index in VIEW_INDEX.items()
                },
                "true_margins": {
                    view: round_float(float(arrays["true_margins"][index, view_index]))
                    for view, view_index in VIEW_INDEX.items()
                },
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--telemetry", type=Path, default=Path(DFR25))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--gap", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    telemetry_path = repo_root / args.telemetry
    telemetry = load_json(telemetry_path)
    arrays = telemetry_arrays(telemetry)
    target_ids = confirmed_true_target_ids(arrays, args.gap)
    candidates = build_grid(arrays, target_ids)
    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    target_ranked = sorted(candidates, key=target_sort_key, reverse=True)
    no_harm = [item for item in candidates if item["broken_dfr25_correct"] == 0]
    no_harm_ranked = sorted(no_harm, key=candidate_sort_key, reverse=True)
    oracle = [item for item in candidates if item["rule"] == "oracle_true_targets"]
    oracle_ranked = sorted(oracle, key=candidate_sort_key, reverse=True)
    report = {
        "analysis": "dfr111_confirmed_target_posthoc_scan",
        "description": (
            "Analysis-only DFR-25 seed42 post-hoc sagittal gate scan over the "
            "DFR-108 confirmed true target set.  This checks whether any "
            "observable eval-side rule can beat DFR-25 before adding more "
            "train-time auxiliary mechanisms."
        ),
        "input_telemetry": str(args.telemetry),
        "parameters": {"confirmed_gap": args.gap},
        "target_ids": sorted(target_ids),
        "target_records": target_records(arrays, target_ids),
        "baseline": summarize_base(telemetry, arrays),
        "candidate_count": len(candidates),
        "best_by_accuracy_auc": ranked[0],
        "best_by_target_coverage": target_ranked[0],
        "best_no_harm": no_harm_ranked[0] if no_harm_ranked else None,
        "best_oracle_target_only": oracle_ranked[0] if oracle_ranked else None,
        "top_20_by_accuracy_auc": ranked[:20],
        "top_20_by_target_coverage": target_ranked[:20],
    }
    save_json(repo_root / args.output, report)
    print(f"Saved DFR-111 post-hoc scan to: {repo_root / args.output}")
    print(
        "targets=",
        len(target_ids),
        "candidate_count=",
        len(candidates),
    )
    for label, item in (
        ("best_by_accuracy_auc", report["best_by_accuracy_auc"]),
        ("best_by_target_coverage", report["best_by_target_coverage"]),
        ("best_no_harm", report["best_no_harm"]),
        ("best_oracle_target_only", report["best_oracle_target_only"]),
    ):
        if item is None:
            print(label, "none")
            continue
        print(
            label,
            "rule=",
            item["rule"],
            "residual=",
            item["residual"],
            "acc=",
            item["metrics"]["accuracy"],
            "auc=",
            item["metrics"]["auc"],
            "f1=",
            item["metrics"]["f1"],
            "net=",
            item["net_delta_vs_dfr25"],
            "triggered=",
            item["triggered_count"],
            "target_triggered=",
            item["triggered_target_count"],
            "sag_top=",
            item["sagittal_top_count"],
            "target_sag_top=",
            item["sagittal_top_target_count"],
            "top=",
            item["weight_summary"]["top_weight_count"],
        )


if __name__ == "__main__":
    main()
