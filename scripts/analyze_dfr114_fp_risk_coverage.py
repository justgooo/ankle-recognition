#!/usr/bin/env python3
"""Safety/coverage audit for DFR-113 fp-risk gate candidates.

This script is analysis-only: it reads existing DFR-25 validation telemetry,
replays label-free eval-side gate residual candidates, and reports the no-harm
frontier plus the remaining DFR-25 errors that are not safely covered.
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
    round_float,
    save_json,
    softmax,
    summarize_weights,
)
from scripts.analyze_dfr112_multiseed_posthoc import (  # noqa: E402
    DFR25_TELEMETRY,
    model_equivalent_abnormal,
    telemetry_arrays_model_equivalent,
)


DEFAULT_OUTPUT = "autoresearch_logs/dfr114_fp_risk_safety_coverage_audit.json"
TARGET_MODES = ("sagittal", "coronal", "best_nonaxial_normal")
BASELINE_IDENTITY = {
    "target_mode": "sagittal",
    "residual": 5.0,
    "params": {
        "axial_min": 0.4,
        "axial_max": 0.88,
        "target_normal_min": 0.55,
        "other_nonaxial_abnormal_max": 0.525,
        "fused_abnormal_min": 0.8,
        "confidence_gap_max": 5.0,
    },
}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = int(labels.shape[0] - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # One-based average rank for tied scores.
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def metrics_from_abnormal_fast(labels: np.ndarray, abnormal: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    abnormal = np.clip(np.asarray(abnormal, dtype=np.float64), 0.0, 1.0)
    preds = (abnormal >= 0.5).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = int(labels.shape[0])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = recall
    return {
        "accuracy": round_float((tp + tn) / total if total else float("nan")),
        "f1": round_float(f1),
        "auc": round_float(auc_from_scores(labels, abnormal)),
        "specificity": round_float(specificity),
        "sensitivity": round_float(sensitivity),
    }


def trigger_and_targets(
    arrays: dict[str, Any],
    *,
    target_mode: str,
    axial_min: float,
    axial_max: float,
    target_normal_min: float,
    other_nonaxial_abnormal_max: float,
    fused_abnormal_min: float,
    confidence_gap_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits = arrays["scaled_confidence_logits"]
    abnormal = arrays["abnormal_probs"]
    fused, _weights = model_equivalent_abnormal(arrays, logits)
    top = np.argmax(logits, axis=1)
    axial_index = VIEW_INDEX["axial"]
    coronal_index = VIEW_INDEX["coronal"]
    sagittal_index = VIEW_INDEX["sagittal"]
    axial_abnormal = abnormal[:, axial_index]
    confidence_gap = logits[:, axial_index] - logits[:, sagittal_index]

    if target_mode == "sagittal":
        target_indices = np.full(abnormal.shape[0], sagittal_index, dtype=np.int64)
        target_abnormal = abnormal[:, sagittal_index]
        other_abnormal = abnormal[:, coronal_index]
    elif target_mode == "coronal":
        target_indices = np.full(abnormal.shape[0], coronal_index, dtype=np.int64)
        target_abnormal = abnormal[:, coronal_index]
        other_abnormal = abnormal[:, sagittal_index]
    elif target_mode == "best_nonaxial_normal":
        choose_coronal = abnormal[:, coronal_index] <= abnormal[:, sagittal_index]
        target_indices = np.where(choose_coronal, coronal_index, sagittal_index)
        target_abnormal = np.minimum(
            abnormal[:, coronal_index],
            abnormal[:, sagittal_index],
        )
        other_abnormal = np.maximum(
            abnormal[:, coronal_index],
            abnormal[:, sagittal_index],
        )
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    mask = (
        (top == axial_index)
        & (fused >= fused_abnormal_min)
        & (axial_abnormal >= axial_min)
        & (axial_abnormal <= axial_max)
        & ((1.0 - target_abnormal) >= target_normal_min)
        & (other_abnormal <= other_nonaxial_abnormal_max)
        & (confidence_gap <= confidence_gap_max)
    )
    return mask, target_indices


def evaluate_candidate(
    arrays: dict[str, Any],
    *,
    target_mode: str,
    residual: float,
    params: dict[str, float],
) -> dict[str, Any]:
    mask, target_indices = trigger_and_targets(
        arrays,
        target_mode=target_mode,
        **params,
    )
    logits = arrays["scaled_confidence_logits"].copy()
    rows = np.flatnonzero(mask)
    logits[rows, target_indices[rows]] += residual
    abnormal, weights = model_equivalent_abnormal(arrays, logits)
    labels = arrays["labels"]
    preds = (abnormal >= 0.5).astype(np.int64)
    base_preds = arrays["base_preds"]
    top = np.argmax(weights, axis=1)

    fixed: list[str] = []
    broken: list[str] = []
    triggered: list[str] = []
    target_top_counts = Counter()
    target_trigger_counts = Counter()
    for index, patient_id in enumerate(arrays["patient_ids"]):
        patient_id = str(patient_id)
        if mask[index]:
            triggered.append(patient_id)
            target_trigger_counts[VIEWS[int(target_indices[index])]] += 1
        if base_preds[index] != labels[index] and preds[index] == labels[index]:
            fixed.append(patient_id)
        elif base_preds[index] == labels[index] and preds[index] != labels[index]:
            broken.append(patient_id)
        if mask[index] and top[index] == target_indices[index]:
            target_top_counts[VIEWS[int(target_indices[index])]] += 1

    return {
        "target_mode": target_mode,
        "residual": round_float(residual),
        "params": {key: round_float(value) for key, value in params.items()},
        "metrics": metrics_from_abnormal_fast(labels, abnormal),
        "triggered_count": int(mask.sum()),
        "triggered_ids": sorted(triggered),
        "target_trigger_count": {view: int(target_trigger_counts.get(view, 0)) for view in VIEWS},
        "target_top_count": {view: int(target_top_counts.get(view, 0)) for view in VIEWS},
        "fixed_dfr25_errors": len(fixed),
        "broken_dfr25_correct": len(broken),
        "net_delta_vs_dfr25": len(fixed) - len(broken),
        "fixed_ids": sorted(fixed),
        "broken_ids": sorted(broken),
        "weight_summary": summarize_weights(weights),
    }


def identity_tuple(item: dict[str, Any]) -> tuple[Any, ...]:
    params = item["params"]
    return (
        item["target_mode"],
        float(item["residual"]),
        float(params["axial_min"]),
        float(params["axial_max"]),
        float(params["target_normal_min"]),
        float(params["other_nonaxial_abnormal_max"]),
        float(params["fused_abnormal_min"]),
        float(params["confidence_gap_max"]),
    )


def identity_payload(identity: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "target_mode": str(identity[0]),
        "residual": round_float(float(identity[1])),
        "params": {
            "axial_min": round_float(float(identity[2])),
            "axial_max": round_float(float(identity[3])),
            "target_normal_min": round_float(float(identity[4])),
            "other_nonaxial_abnormal_max": round_float(float(identity[5])),
            "fused_abnormal_min": round_float(float(identity[6])),
            "confidence_gap_max": round_float(float(identity[7])),
        },
    }


def evaluate_identity(arrays: dict[str, Any], identity: tuple[Any, ...]) -> dict[str, Any]:
    payload = identity_payload(identity)
    return evaluate_candidate(
        arrays,
        target_mode=payload["target_mode"],
        residual=float(payload["residual"]),
        params={key: float(value) for key, value in payload["params"].items()},
    )


def aggregate_candidate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in per_seed.values()]
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    total_fixed = 0
    total_broken = 0
    total_triggered = 0
    trigger_counts = Counter()
    target_top_counts = Counter()
    for item in per_seed.values():
        total_fixed += int(item["fixed_dfr25_errors"])
        total_broken += int(item["broken_dfr25_correct"])
        total_triggered += int(item["triggered_count"])
        trigger_counts.update(item["target_trigger_count"])
        target_top_counts.update(item["target_top_count"])
        summary = item["weight_summary"]
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
        "target_trigger_count": {view: int(trigger_counts.get(view, 0)) for view in VIEWS},
        "target_top_count": {view: int(target_top_counts.get(view, 0)) for view in VIEWS},
        "total_fixed_dfr25_errors": int(total_fixed),
        "total_broken_dfr25_correct": int(total_broken),
        "total_net_delta_vs_dfr25": int(total_fixed - total_broken),
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
    }


def aggregate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    per_seed = item["per_seed"]
    min_acc = min(float(seed_item["metrics"]["accuracy"]) for seed_item in per_seed.values())
    max_broken = max(int(seed_item["broken_dfr25_correct"]) for seed_item in per_seed.values())
    return (
        -aggregate["total_broken_dfr25_correct"],
        aggregate["total_fixed_dfr25_errors"],
        metrics["accuracy"],
        metrics["auc"],
        min_acc,
        aggregate["total_net_delta_vs_dfr25"],
        -max_broken,
        -aggregate["total_triggered"],
    )


def frontier_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    return (
        aggregate["total_fixed_dfr25_errors"],
        -aggregate["total_broken_dfr25_correct"],
        aggregate["total_net_delta_vs_dfr25"],
        metrics["accuracy"],
        metrics["auc"],
        -aggregate["total_triggered"],
    )


def build_identities(seed_arrays: dict[str, dict[str, Any]]) -> set[tuple[Any, ...]]:
    # Bounded DFR-113 neighborhood audit.  The goal is not to re-open an
    # unconstrained threshold search; it is to test whether the validated rule
    # can be safely tightened/relaxed around its actual decision boundary.
    residuals = (4.0, 5.0, 6.0, 8.0)
    axial_mins = (0.40, 0.60, 0.75)
    axial_maxes = (0.88, 0.90, 0.93, 0.95)
    target_normal_mins = (0.55, 0.60, 0.70)
    other_maxes = (0.45, 0.50, 0.525, 0.55)
    fused_mins = (0.75, 0.80, 0.85, 0.90)
    gap_maxes = (3.0, 5.0, 7.5)

    identities: set[tuple[Any, ...]] = set()
    signatures: set[tuple[Any, ...]] = set()
    baseline = {
        "target_mode": BASELINE_IDENTITY["target_mode"],
        "residual": BASELINE_IDENTITY["residual"],
        "params": dict(BASELINE_IDENTITY["params"]),
    }
    identities.add(identity_tuple(baseline))
    for target_mode in TARGET_MODES:
        for residual in residuals:
            for axial_min in axial_mins:
                for axial_max in axial_maxes:
                    if axial_min > axial_max:
                        continue
                    for target_normal_min in target_normal_mins:
                        for other_nonaxial_abnormal_max in other_maxes:
                            for fused_abnormal_min in fused_mins:
                                for confidence_gap_max in gap_maxes:
                                    candidate = {
                                        "target_mode": target_mode,
                                        "residual": residual,
                                        "params": {
                                            "axial_min": axial_min,
                                            "axial_max": axial_max,
                                            "target_normal_min": target_normal_min,
                                            "other_nonaxial_abnormal_max": other_nonaxial_abnormal_max,
                                            "fused_abnormal_min": fused_abnormal_min,
                                            "confidence_gap_max": confidence_gap_max,
                                        },
                                    }
                                    signature_parts = []
                                    has_trigger = False
                                    for seed, arrays in seed_arrays.items():
                                        mask, target_indices = trigger_and_targets(
                                            arrays,
                                            target_mode=target_mode,
                                            **candidate["params"],
                                        )
                                        rows = tuple(int(row) for row in np.flatnonzero(mask))
                                        targets = tuple(int(target_indices[row]) for row in rows)
                                        has_trigger = has_trigger or bool(rows)
                                        signature_parts.append((seed, rows, targets))
                                    signature = (target_mode, residual, tuple(signature_parts))
                                    # Many nearby threshold combinations trigger exactly the same
                                    # samples; evaluate each distinct observable action once.
                                    if has_trigger and signature not in signatures:
                                        signatures.add(signature)
                                        identities.add(identity_tuple(candidate))
    return identities


def baseline_summary(seed_arrays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = []
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    errors = Counter()
    error_records: list[dict[str, Any]] = []
    for seed, arrays in seed_arrays.items():
        abnormal, weights = model_equivalent_abnormal(
            arrays,
            arrays["scaled_confidence_logits"],
        )
        metrics.append(metrics_from_abnormal_fast(arrays["labels"], abnormal))
        summary = summarize_weights(weights)
        for view in VIEWS:
            top_counts[view] += int(summary["top_weight_count"][view])
            mean_weights[view].append(float(summary["mean_fusion_weight"][view]))
        for index, patient_id in enumerate(arrays["patient_ids"]):
            label = int(arrays["labels"][index])
            pred = int(arrays["base_preds"][index])
            error_type = base_error_type(label, pred)
            errors.update([error_type])
            if label != pred:
                views = arrays["samples"][index]["views"]
                error_records.append(
                    {
                        "seed": seed,
                        "patient_id": str(patient_id),
                        "label": label,
                        "base_pred": pred,
                        "error_type": error_type,
                        "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                        "top_weight_views": arrays["samples"][index]["top_weight_views"],
                        "view_abnormal": {
                            view: round_float(float(views[view]["abnormal_prob"]))
                            for view in VIEWS
                        },
                        "scaled_confidence_logit": {
                            view: round_float(
                                float(
                                    views[view].get(
                                        "scaled_confidence_logit",
                                        views[view]["confidence_logit"],
                                    )
                                )
                            )
                            for view in VIEWS
                        },
                        "true_margin": {
                            view: round_float(float(views[view]["true_margin"]))
                            for view in VIEWS
                        },
                    }
                )
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
        "error_types": counter_dict(errors),
        "error_records": error_records,
    }


def remaining_error_audit(
    seed_arrays: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    fixed_by_key = {
        (seed, patient_id)
        for seed, seed_candidate in candidate["per_seed"].items()
        for patient_id in seed_candidate["fixed_ids"]
    }
    records: list[dict[str, Any]] = []
    for seed, arrays in seed_arrays.items():
        identity = tuple(candidate["identity_tuple"])
        seed_candidate = evaluate_identity(arrays, identity)
        mask, target_indices = trigger_and_targets(
            arrays,
            target_mode=str(identity[0]),
            axial_min=float(identity[2]),
            axial_max=float(identity[3]),
            target_normal_min=float(identity[4]),
            other_nonaxial_abnormal_max=float(identity[5]),
            fused_abnormal_min=float(identity[6]),
            confidence_gap_max=float(identity[7]),
        )
        triggered_ids = set(seed_candidate["triggered_ids"])
        for index, patient_id in enumerate(arrays["patient_ids"]):
            patient_id = str(patient_id)
            label = int(arrays["labels"][index])
            pred = int(arrays["base_preds"][index])
            if label == pred or (seed, patient_id) in fixed_by_key:
                continue
            abnormal = arrays["abnormal_probs"][index]
            logits = arrays["scaled_confidence_logits"][index]
            fused, _weights = model_equivalent_abnormal(
                arrays,
                arrays["scaled_confidence_logits"],
            )
            target_view = VIEWS[int(target_indices[index])]
            records.append(
                {
                    "seed": seed,
                    "patient_id": patient_id,
                    "label": label,
                    "base_pred": pred,
                    "error_type": base_error_type(label, pred),
                    "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                    "model_equivalent_abnormal": round_float(float(fused[index])),
                    "triggered_by_best": patient_id in triggered_ids,
                    "target_view_if_triggered": target_view,
                    "reason": explain_miss(
                        label=label,
                        pred=pred,
                        triggered=bool(mask[index]),
                        target_view=target_view,
                        abnormal=abnormal,
                        logits=logits,
                        params=identity_payload(identity)["params"],
                    ),
                    "view_abnormal": {
                        view: round_float(float(abnormal[VIEW_INDEX[view]]))
                        for view in VIEWS
                    },
                    "scaled_confidence_logit": {
                        view: round_float(float(logits[VIEW_INDEX[view]]))
                        for view in VIEWS
                    },
                }
            )
    return records


def explain_miss(
    *,
    label: int,
    pred: int,
    triggered: bool,
    target_view: str,
    abnormal: np.ndarray,
    logits: np.ndarray,
    params: dict[str, Any],
) -> str:
    if label == 1 and pred == 0:
        return "positive FN; fp-risk normal-rescue gate is intentionally not designed to boost abnormal evidence"
    if triggered:
        if float(abnormal[VIEW_INDEX[target_view]]) > 0.45:
            return "triggered but target view is not normal enough to pull fused abnormal below threshold"
        return "triggered but residual/weight shift is insufficient"
    axial_abnormal = float(abnormal[VIEW_INDEX["axial"]])
    target_abnormal = float(abnormal[VIEW_INDEX[target_view]])
    top = int(np.argmax(logits))
    if top != VIEW_INDEX["axial"]:
        return "base top confidence is not axial"
    if axial_abnormal > float(params["axial_max"]):
        return "axial abnormal exceeds selected safety cap"
    if axial_abnormal < float(params["axial_min"]):
        return "axial abnormal below selected fp-risk floor"
    if (1.0 - target_abnormal) < float(params["target_normal_min"]):
        return "target non-axial view is not normal enough"
    return "missed by combined fused-risk/confidence-gap/non-target guard"


def pareto_frontier(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_fixed: dict[int, dict[str, Any]] = {}
    for item in items:
        fixed = int(item["aggregate"]["total_fixed_dfr25_errors"])
        current = best_by_fixed.get(fixed)
        if current is None or frontier_sort_key(item) > frontier_sort_key(current):
            best_by_fixed[fixed] = item
    return [best_by_fixed[key] for key in sorted(best_by_fixed)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    seed_arrays = {
        seed: telemetry_arrays_model_equivalent(load_json(repo_root / relative_path))
        for seed, relative_path in DFR25_TELEMETRY.items()
    }
    identities = build_identities(seed_arrays)
    aggregated: list[dict[str, Any]] = []
    for identity in identities:
        per_seed = {
            seed: evaluate_identity(arrays, identity)
            for seed, arrays in seed_arrays.items()
        }
        aggregate = aggregate_candidate(per_seed)
        aggregated.append(
            {
                "identity": identity_payload(identity),
                "identity_tuple": list(identity),
                "aggregate": aggregate,
                "per_seed": per_seed,
            }
        )

    ranked = sorted(aggregated, key=aggregate_sort_key, reverse=True)
    no_harm = [
        item
        for item in ranked
        if int(item["aggregate"]["total_broken_dfr25_correct"]) == 0
    ]
    best_no_harm = no_harm[0] if no_harm else None
    baseline_candidate = next(
        item
        for item in aggregated
        if tuple(item["identity_tuple"]) == identity_tuple(BASELINE_IDENTITY)
    )
    coverage_frontier = sorted(aggregated, key=frontier_sort_key, reverse=True)[:50]
    report = {
        "analysis": "dfr114_fp_risk_safety_coverage_audit",
        "description": (
            "Analysis-only label-free safety/coverage audit after DFR-113. "
            "It scans eval-side gate residual candidates on DFR-25 validation "
            "telemetry only; no training, checkpoint mutation, or test metric is used."
        ),
        "baseline_dfr25": baseline_summary(seed_arrays),
        "dfr113_candidate": baseline_candidate,
        "candidate_count": len(aggregated),
        "no_harm_candidate_count": len(no_harm),
        "best_no_harm": best_no_harm,
        "best_by_safety_then_coverage": ranked[0] if ranked else None,
        "top_20_no_harm": no_harm[:20],
        "top_20_by_safety_then_coverage": ranked[:20],
        "coverage_frontier_top_50": coverage_frontier,
        "remaining_error_audit_for_best_no_harm": (
            remaining_error_audit(seed_arrays, best_no_harm) if best_no_harm else []
        ),
    }
    save_json(repo_root / args.output, report)
    print(f"Saved DFR-114 safety/coverage audit to: {repo_root / args.output}")
    print("baseline", report["baseline_dfr25"]["mean_metrics"], report["baseline_dfr25"]["aggregate_top_weight_count"])
    print("candidate_count", len(aggregated), "no_harm", len(no_harm))
    for label, item in (
        ("dfr113", baseline_candidate),
        ("best_no_harm", best_no_harm),
        ("best_safety_coverage", ranked[0] if ranked else None),
    ):
        if item is None:
            print(label, "none")
            continue
        print(
            label,
            item["identity"],
            item["aggregate"]["mean_metrics"],
            "fixed",
            item["aggregate"]["total_fixed_dfr25_errors"],
            "broken",
            item["aggregate"]["total_broken_dfr25_correct"],
            "top",
            item["aggregate"]["aggregate_top_weight_count"],
            "triggered",
            item["aggregate"]["total_triggered"],
        )


if __name__ == "__main__":
    main()
