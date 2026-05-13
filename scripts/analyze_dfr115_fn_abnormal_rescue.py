#!/usr/bin/env python3
"""Analysis-only audit for positive-FN abnormal-evidence rescue candidates.

The DFR-113/114 fp-risk gate is a normal-rescue mechanism for false positives.
This script tests the separate question: whether DFR-25 false negatives expose
label-free abnormal evidence that can be rescued without breaking DFR-25-correct
validation samples.  It only reads validation telemetry.
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
    summarize_weights,
)
from scripts.analyze_dfr112_multiseed_posthoc import (  # noqa: E402
    DFR25_TELEMETRY,
    model_equivalent_abnormal,
    telemetry_arrays_model_equivalent,
)
from scripts.analyze_dfr114_fp_risk_coverage import (  # noqa: E402
    BASELINE_IDENTITY as DFR113_FP_IDENTITY,
    trigger_and_targets as fp_trigger_and_targets,
    metrics_from_abnormal_fast,
)


DEFAULT_OUTPUT = "autoresearch_logs/dfr115_fn_abnormal_rescue_audit.json"
TARGET_MODES = ("axial", "coronal", "sagittal", "max_abnormal", "second_max_abnormal")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def target_indices_and_values(
    abnormal: np.ndarray,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_mode in VIEW_INDEX:
        index = VIEW_INDEX[target_mode]
        indices = np.full(abnormal.shape[0], index, dtype=np.int64)
        values = abnormal[:, index]
    elif target_mode == "max_abnormal":
        indices = np.argmax(abnormal, axis=1).astype(np.int64)
        values = abnormal[np.arange(abnormal.shape[0]), indices]
    elif target_mode == "second_max_abnormal":
        order = np.argsort(abnormal, axis=1)
        indices = order[:, -2].astype(np.int64)
        values = abnormal[np.arange(abnormal.shape[0]), indices]
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    max_values = abnormal.max(axis=1)
    return indices, values, max_values


def trigger_mask(
    arrays: dict[str, Any],
    *,
    target_mode: str,
    fused_abnormal_max: float,
    target_abnormal_min: float,
    max_abnormal_min: float,
    second_abnormal_min: float,
    target_confidence_gap_max: float,
    axial_abnormal_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    abnormal = arrays["abnormal_probs"]
    logits = arrays["scaled_confidence_logits"]
    fused, _weights = model_equivalent_abnormal(arrays, logits)
    target_indices, target_values, max_values = target_indices_and_values(
        abnormal,
        target_mode,
    )
    sorted_abnormal = np.sort(abnormal, axis=1)
    second_values = sorted_abnormal[:, -2]
    target_logits = logits[np.arange(logits.shape[0]), target_indices]
    top_other_logits = np.max(
        np.where(
            np.eye(logits.shape[1], dtype=bool)[target_indices],
            -np.inf,
            logits,
        ),
        axis=1,
    )
    confidence_gap = top_other_logits - target_logits
    mask = (
        (fused <= fused_abnormal_max)
        & (target_values >= target_abnormal_min)
        & (max_values >= max_abnormal_min)
        & (second_values >= second_abnormal_min)
        & (confidence_gap <= target_confidence_gap_max)
        & (abnormal[:, VIEW_INDEX["axial"]] <= axial_abnormal_max)
    )
    return mask, target_indices


def evaluate_candidate(
    arrays: dict[str, Any],
    *,
    target_mode: str,
    residual: float,
    params: dict[str, float],
) -> dict[str, Any]:
    mask, target_indices = trigger_mask(arrays, target_mode=target_mode, **params)
    logits = arrays["scaled_confidence_logits"].copy()
    rows = np.flatnonzero(mask)
    logits[rows, target_indices[rows]] += residual
    abnormal, weights = model_equivalent_abnormal(arrays, logits)
    labels = arrays["labels"]
    preds = (abnormal >= 0.5).astype(np.int64)
    base_preds = arrays["base_preds"]

    fixed: list[str] = []
    broken: list[str] = []
    triggered: list[str] = []
    target_trigger_counts = Counter()
    target_top_counts = Counter()
    top = np.argmax(weights, axis=1)
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
        float(params["fused_abnormal_max"]),
        float(params["target_abnormal_min"]),
        float(params["max_abnormal_min"]),
        float(params["second_abnormal_min"]),
        float(params["target_confidence_gap_max"]),
        float(params["axial_abnormal_max"]),
    )


def identity_payload(identity: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "target_mode": str(identity[0]),
        "residual": round_float(float(identity[1])),
        "params": {
            "fused_abnormal_max": round_float(float(identity[2])),
            "target_abnormal_min": round_float(float(identity[3])),
            "max_abnormal_min": round_float(float(identity[4])),
            "second_abnormal_min": round_float(float(identity[5])),
            "target_confidence_gap_max": round_float(float(identity[6])),
            "axial_abnormal_max": round_float(float(identity[7])),
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
    trigger_counts = Counter()
    target_top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    fixed = 0
    broken = 0
    triggered = 0
    for item in per_seed.values():
        fixed += int(item["fixed_dfr25_errors"])
        broken += int(item["broken_dfr25_correct"])
        triggered += int(item["triggered_count"])
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
        "total_triggered": int(triggered),
        "target_trigger_count": {view: int(trigger_counts.get(view, 0)) for view in VIEWS},
        "target_top_count": {view: int(target_top_counts.get(view, 0)) for view in VIEWS},
        "total_fixed_dfr25_errors": int(fixed),
        "total_broken_dfr25_correct": int(broken),
        "total_net_delta_vs_dfr25": int(fixed - broken),
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
    }


def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    per_seed = item["per_seed"]
    min_acc = min(float(seed_item["metrics"]["accuracy"]) for seed_item in per_seed.values())
    return (
        -aggregate["total_broken_dfr25_correct"],
        aggregate["total_fixed_dfr25_errors"],
        metrics["accuracy"],
        metrics["auc"],
        min_acc,
        -aggregate["total_triggered"],
    )


def build_identities(seed_arrays: dict[str, dict[str, Any]]) -> set[tuple[Any, ...]]:
    residuals = (2.0, 3.0, 4.0, 5.0, 6.0)
    fused_maxes = (0.10, 0.20, 0.30, 0.40, 0.50)
    target_mins = (0.45, 0.48, 0.50, 0.52, 0.55)
    max_mins = (0.45, 0.50, 0.55)
    second_mins = (0.0, 0.20, 0.45, 0.50)
    gap_maxes = (1.0, 2.0, 3.0, 5.0, 8.0)
    axial_maxes = (0.35, 0.50, 0.65, 1.0)
    identities: set[tuple[Any, ...]] = set()
    signatures: set[tuple[Any, ...]] = set()
    for target_mode in TARGET_MODES:
        for residual in residuals:
            for fused_abnormal_max in fused_maxes:
                for target_abnormal_min in target_mins:
                    for max_abnormal_min in max_mins:
                        if target_abnormal_min < max_abnormal_min and target_mode == "max_abnormal":
                            continue
                        for second_abnormal_min in second_mins:
                            for target_confidence_gap_max in gap_maxes:
                                for axial_abnormal_max in axial_maxes:
                                    candidate = {
                                        "target_mode": target_mode,
                                        "residual": residual,
                                        "params": {
                                            "fused_abnormal_max": fused_abnormal_max,
                                            "target_abnormal_min": target_abnormal_min,
                                            "max_abnormal_min": max_abnormal_min,
                                            "second_abnormal_min": second_abnormal_min,
                                            "target_confidence_gap_max": target_confidence_gap_max,
                                            "axial_abnormal_max": axial_abnormal_max,
                                        },
                                    }
                                    parts = []
                                    has_trigger = False
                                    for seed, arrays in seed_arrays.items():
                                        mask, target_indices = trigger_mask(
                                            arrays,
                                            target_mode=target_mode,
                                            **candidate["params"],
                                        )
                                        rows = tuple(int(row) for row in np.flatnonzero(mask))
                                        targets = tuple(int(target_indices[row]) for row in rows)
                                        has_trigger = has_trigger or bool(rows)
                                        parts.append((seed, rows, targets))
                                    signature = (target_mode, residual, tuple(parts))
                                    if has_trigger and signature not in signatures:
                                        signatures.add(signature)
                                        identities.add(identity_tuple(candidate))
    return identities


def baseline_summary(seed_arrays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = []
    top_counts = Counter()
    errors = Counter()
    fn_records: list[dict[str, Any]] = []
    for seed, arrays in seed_arrays.items():
        abnormal, weights = model_equivalent_abnormal(
            arrays,
            arrays["scaled_confidence_logits"],
        )
        metrics.append(metrics_from_abnormal_fast(arrays["labels"], abnormal))
        summary = summarize_weights(weights)
        for view in VIEWS:
            top_counts[view] += int(summary["top_weight_count"][view])
        for index, patient_id in enumerate(arrays["patient_ids"]):
            label = int(arrays["labels"][index])
            pred = int(arrays["base_preds"][index])
            errors.update([base_error_type(label, pred)])
            if label == 1 and pred == 0:
                sample = arrays["samples"][index]
                fn_records.append(
                    {
                        "seed": seed,
                        "patient_id": str(patient_id),
                        "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                        "top_weight_views": sample["top_weight_views"],
                        "view_abnormal": {
                            view: round_float(float(sample["views"][view]["abnormal_prob"]))
                            for view in VIEWS
                        },
                        "scaled_confidence_logit": {
                            view: round_float(
                                float(
                                    sample["views"][view].get(
                                        "scaled_confidence_logit",
                                        sample["views"][view]["confidence_logit"],
                                    )
                                )
                            )
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
        "error_types": counter_dict(errors),
        "false_negative_records": fn_records,
    }


def remaining_fn_audit(
    seed_arrays: dict[str, dict[str, Any]],
    candidate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    fixed = set()
    if candidate is not None:
        fixed = {
            (seed, patient_id)
            for seed, item in candidate["per_seed"].items()
            for patient_id in item["fixed_ids"]
        }
    records: list[dict[str, Any]] = []
    for seed, arrays in seed_arrays.items():
        for index, patient_id in enumerate(arrays["patient_ids"]):
            patient_id = str(patient_id)
            label = int(arrays["labels"][index])
            pred = int(arrays["base_preds"][index])
            if label != 1 or pred != 0 or (seed, patient_id) in fixed:
                continue
            abnormal = arrays["abnormal_probs"][index]
            max_abnormal = float(abnormal.max())
            second_abnormal = float(np.sort(abnormal)[-2])
            records.append(
                {
                    "seed": seed,
                    "patient_id": patient_id,
                    "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                    "max_view_abnormal": round_float(max_abnormal),
                    "second_view_abnormal": round_float(second_abnormal),
                    "reason": explain_fn(max_abnormal, second_abnormal),
                    "view_abnormal": {
                        view: round_float(float(abnormal[VIEW_INDEX[view]]))
                        for view in VIEWS
                    },
                }
            )
    return records


def explain_fn(max_abnormal: float, second_abnormal: float) -> str:
    if max_abnormal < 0.45:
        return "no view reaches even weak abnormal evidence"
    if max_abnormal < 0.50:
        return "only sub-threshold abnormal evidence is present"
    if second_abnormal < 0.45:
        return "single weak positive view with no second-view support"
    return "candidate rescue would require accepting borderline abnormal evidence"


def combined_with_dfr113_fp_rescue(
    seed_arrays: dict[str, dict[str, Any]],
    fn_candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if fn_candidate is None:
        return None
    fn_identity = tuple(fn_candidate["identity_tuple"])
    fn_payload = identity_payload(fn_identity)
    per_seed: dict[str, dict[str, Any]] = {}
    for seed, arrays in seed_arrays.items():
        logits = arrays["scaled_confidence_logits"].copy()

        fp_mask, fp_targets = fp_trigger_and_targets(
            arrays,
            target_mode=str(DFR113_FP_IDENTITY["target_mode"]),
            **DFR113_FP_IDENTITY["params"],
        )
        fp_rows = np.flatnonzero(fp_mask)
        logits[fp_rows, fp_targets[fp_rows]] += float(DFR113_FP_IDENTITY["residual"])

        fn_mask, fn_targets = trigger_mask(
            arrays,
            target_mode=fn_payload["target_mode"],
            **fn_payload["params"],
        )
        fn_rows = np.flatnonzero(fn_mask)
        logits[fn_rows, fn_targets[fn_rows]] += float(fn_payload["residual"])

        abnormal, weights = model_equivalent_abnormal(arrays, logits)
        labels = arrays["labels"]
        preds = (abnormal >= 0.5).astype(np.int64)
        base_preds = arrays["base_preds"]
        fixed: list[str] = []
        broken: list[str] = []
        triggered: list[str] = []
        for index, patient_id in enumerate(arrays["patient_ids"]):
            patient_id = str(patient_id)
            if fp_mask[index] or fn_mask[index]:
                triggered.append(patient_id)
            if base_preds[index] != labels[index] and preds[index] == labels[index]:
                fixed.append(patient_id)
            elif base_preds[index] == labels[index] and preds[index] != labels[index]:
                broken.append(patient_id)
        per_seed[seed] = {
            "metrics": metrics_from_abnormal_fast(labels, abnormal),
            "triggered_count": len(triggered),
            "triggered_ids": sorted(triggered),
            "fixed_dfr25_errors": len(fixed),
            "broken_dfr25_correct": len(broken),
            "net_delta_vs_dfr25": len(fixed) - len(broken),
            "fixed_ids": sorted(fixed),
            "broken_ids": sorted(broken),
            "weight_summary": summarize_weights(weights),
        }

    metrics = [item["metrics"] for item in per_seed.values()]
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    for item in per_seed.values():
        summary = item["weight_summary"]
        for view in VIEWS:
            top_counts[view] += int(summary["top_weight_count"][view])
            mean_weights[view].append(float(summary["mean_fusion_weight"][view]))
    aggregate = {
        "mean_metrics": {
            "accuracy": round_float(mean([float(item["accuracy"]) for item in metrics])),
            "auc": round_float(mean([float(item["auc"]) for item in metrics])),
            "f1": round_float(mean([float(item["f1"]) for item in metrics])),
        },
        "total_triggered": sum(int(item["triggered_count"]) for item in per_seed.values()),
        "total_fixed_dfr25_errors": sum(
            int(item["fixed_dfr25_errors"]) for item in per_seed.values()
        ),
        "total_broken_dfr25_correct": sum(
            int(item["broken_dfr25_correct"]) for item in per_seed.values()
        ),
        "total_net_delta_vs_dfr25": sum(
            int(item["net_delta_vs_dfr25"]) for item in per_seed.values()
        ),
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
    }
    return {
        "fp_identity": DFR113_FP_IDENTITY,
        "fn_identity": fn_candidate["identity"],
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


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
        aggregated.append(
            {
                "identity": identity_payload(identity),
                "identity_tuple": list(identity),
                "aggregate": aggregate_candidate(per_seed),
                "per_seed": per_seed,
            }
        )
    ranked = sorted(aggregated, key=sort_key, reverse=True)
    no_harm = [
        item
        for item in ranked
        if int(item["aggregate"]["total_broken_dfr25_correct"]) == 0
    ]
    best_no_harm = no_harm[0] if no_harm else None
    report = {
        "analysis": "dfr115_fn_abnormal_rescue_audit",
        "description": (
            "Analysis-only scan for label-free positive-FN abnormal-evidence "
            "rescue on DFR-25 validation telemetry. No training or test metric."
        ),
        "baseline_dfr25": baseline_summary(seed_arrays),
        "candidate_count": len(aggregated),
        "no_harm_candidate_count": len(no_harm),
        "best_no_harm": best_no_harm,
        "best_by_safety_then_coverage": ranked[0] if ranked else None,
        "best_no_harm_combined_with_dfr113_fp_rescue": combined_with_dfr113_fp_rescue(
            seed_arrays,
            best_no_harm,
        ),
        "top_20_no_harm": no_harm[:20],
        "top_20_by_safety_then_coverage": ranked[:20],
        "remaining_fn_audit_for_best_no_harm": remaining_fn_audit(
            seed_arrays,
            best_no_harm,
        ),
    }
    save_json(repo_root / args.output, report)
    print(f"Saved DFR-115 FN abnormal-rescue audit to: {repo_root / args.output}")
    print("baseline", report["baseline_dfr25"]["mean_metrics"], report["baseline_dfr25"]["aggregate_top_weight_count"])
    print("candidate_count", len(aggregated), "no_harm", len(no_harm))
    if best_no_harm:
        print(
            "best_no_harm",
            best_no_harm["identity"],
            best_no_harm["aggregate"]["mean_metrics"],
            "fixed",
            best_no_harm["aggregate"]["total_fixed_dfr25_errors"],
            "broken",
            best_no_harm["aggregate"]["total_broken_dfr25_correct"],
            "top",
            best_no_harm["aggregate"]["aggregate_top_weight_count"],
            "triggered",
            best_no_harm["aggregate"]["total_triggered"],
        )
    if ranked:
        best = ranked[0]
        print(
            "best_safety_coverage",
            best["identity"],
            best["aggregate"]["mean_metrics"],
            "fixed",
            best["aggregate"]["total_fixed_dfr25_errors"],
            "broken",
            best["aggregate"]["total_broken_dfr25_correct"],
            "top",
            best["aggregate"]["aggregate_top_weight_count"],
            "triggered",
            best["aggregate"]["total_triggered"],
        )
    combined = report["best_no_harm_combined_with_dfr113_fp_rescue"]
    if combined:
        print(
            "combined_dfr113_fp_plus_best_fn",
            combined["aggregate"]["mean_metrics"],
            "fixed",
            combined["aggregate"]["total_fixed_dfr25_errors"],
            "broken",
            combined["aggregate"]["total_broken_dfr25_correct"],
            "top",
            combined["aggregate"]["aggregate_top_weight_count"],
            "triggered",
            combined["aggregate"]["total_triggered"],
        )


if __name__ == "__main__":
    main()
