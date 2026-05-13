#!/usr/bin/env python3
"""Safety/frontier audit for DFR-116 posthoc combo candidates.

This analysis keeps the DFR-116 combo fixed, then tests whether one additional
label-free eval-side residual from the existing FP-normal-rescue or
FN-abnormal-rescue families can safely cover more validation errors.
It only reads validation telemetry and does not train or use test metrics.
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
    build_identities as build_fp_identities,
    identity_payload as fp_identity_payload,
    metrics_from_abnormal_fast,
    trigger_and_targets as fp_trigger_and_targets,
)
from scripts.analyze_dfr115_fn_abnormal_rescue import (  # noqa: E402
    build_identities as build_fn_identities,
    explain_fn,
    identity_payload as fn_identity_payload,
    trigger_mask as fn_trigger_and_targets,
)


DEFAULT_OUTPUT = "autoresearch_logs/dfr117_combo_frontier_audit.json"
DFR115_FN_IDENTITY = {
    "target_mode": "axial",
    "residual": 6.0,
    "params": {
        "fused_abnormal_max": 0.2,
        "target_abnormal_min": 0.45,
        "max_abnormal_min": 0.45,
        "second_abnormal_min": 0.0,
        "target_confidence_gap_max": 1.0,
        "axial_abnormal_max": 0.65,
    },
}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def combo_action_payloads() -> list[dict[str, Any]]:
    return [
        {
            "family": "fp_normal_rescue",
            "identity": DFR113_FP_IDENTITY,
        },
        {
            "family": "fn_abnormal_rescue",
            "identity": DFR115_FN_IDENTITY,
        },
    ]


def candidate_payload(family: str, identity: tuple[Any, ...]) -> dict[str, Any]:
    if family == "fp_normal_rescue":
        return {
            "family": family,
            "identity": fp_identity_payload(identity),
            "identity_tuple": list(identity),
        }
    if family == "fn_abnormal_rescue":
        return {
            "family": family,
            "identity": fn_identity_payload(identity),
            "identity_tuple": list(identity),
        }
    raise ValueError(f"Unknown family: {family}")


def trigger_for_payload(
    arrays: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    identity = payload["identity"]
    if payload["family"] == "fp_normal_rescue":
        return fp_trigger_and_targets(
            arrays,
            target_mode=str(identity["target_mode"]),
            **{key: float(value) for key, value in identity["params"].items()},
        )
    if payload["family"] == "fn_abnormal_rescue":
        return fn_trigger_and_targets(
            arrays,
            target_mode=str(identity["target_mode"]),
            **{key: float(value) for key, value in identity["params"].items()},
        )
    raise ValueError(f"Unknown family: {payload['family']}")


def apply_payload(
    logits: np.ndarray,
    arrays: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask, targets = trigger_for_payload(arrays, payload)
    rows = np.flatnonzero(mask)
    logits[rows, targets[rows]] += float(payload["identity"]["residual"])
    return mask, targets, rows


def apply_combo_logits(arrays: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    logits = arrays["scaled_confidence_logits"].copy()
    action_records: dict[str, Any] = {}
    for payload in combo_action_payloads():
        mask, targets, rows = apply_payload(logits, arrays, payload)
        action_records[payload["family"]] = {
            "triggered_rows": [int(row) for row in rows],
            "triggered_ids": [str(arrays["patient_ids"][row]) for row in rows],
            "target_views": [VIEWS[int(targets[row])] for row in rows],
        }
    return logits, action_records


def evaluate_logits(
    arrays: dict[str, Any],
    logits: np.ndarray,
    combo_preds: np.ndarray | None = None,
) -> dict[str, Any]:
    abnormal, weights = model_equivalent_abnormal(arrays, logits)
    labels = arrays["labels"]
    preds = (abnormal >= 0.5).astype(np.int64)
    base_preds = arrays["base_preds"]

    fixed: list[str] = []
    broken: list[str] = []
    incremental_fixed: list[str] = []
    incremental_broken: list[str] = []
    changed_from_combo: list[str] = []
    for index, patient_id in enumerate(arrays["patient_ids"]):
        patient_id = str(patient_id)
        label = int(labels[index])
        pred = int(preds[index])
        base_pred = int(base_preds[index])
        if base_pred != label and pred == label:
            fixed.append(patient_id)
        elif base_pred == label and pred != label:
            broken.append(patient_id)
        if combo_preds is not None:
            combo_pred = int(combo_preds[index])
            if combo_pred != pred:
                changed_from_combo.append(patient_id)
            if combo_pred != label and pred == label:
                incremental_fixed.append(patient_id)
            elif combo_pred == label and pred != label:
                incremental_broken.append(patient_id)

    return {
        "metrics": metrics_from_abnormal_fast(labels, abnormal),
        "fixed_dfr25_errors": len(fixed),
        "broken_dfr25_correct": len(broken),
        "net_delta_vs_dfr25": len(fixed) - len(broken),
        "fixed_ids": sorted(fixed),
        "broken_ids": sorted(broken),
        "incremental_fixed_vs_combo": len(incremental_fixed),
        "incremental_broken_vs_combo": len(incremental_broken),
        "incremental_fixed_ids": sorted(incremental_fixed),
        "incremental_broken_ids": sorted(incremental_broken),
        "changed_from_combo_ids": sorted(changed_from_combo),
        "weight_summary": summarize_weights(weights),
        "preds": preds,
        "abnormal": abnormal,
    }


def aggregate_per_seed(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in per_seed.values()]
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    total_fixed = 0
    total_broken = 0
    incremental_fixed = 0
    incremental_broken = 0
    total_triggered = 0
    extra_trigger_counts = Counter()
    extra_top_counts = Counter()
    for item in per_seed.values():
        total_fixed += int(item["fixed_dfr25_errors"])
        total_broken += int(item["broken_dfr25_correct"])
        incremental_fixed += int(item["incremental_fixed_vs_combo"])
        incremental_broken += int(item["incremental_broken_vs_combo"])
        total_triggered += int(item.get("extra_triggered_count", 0))
        extra_trigger_counts.update(item.get("extra_target_trigger_count", {}))
        extra_top_counts.update(item.get("extra_target_top_count", {}))
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
        "total_extra_triggered": int(total_triggered),
        "extra_target_trigger_count": {
            view: int(extra_trigger_counts.get(view, 0)) for view in VIEWS
        },
        "extra_target_top_count": {
            view: int(extra_top_counts.get(view, 0)) for view in VIEWS
        },
        "total_fixed_dfr25_errors": int(total_fixed),
        "total_broken_dfr25_correct": int(total_broken),
        "total_net_delta_vs_dfr25": int(total_fixed - total_broken),
        "incremental_fixed_vs_combo": int(incremental_fixed),
        "incremental_broken_vs_combo": int(incremental_broken),
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
    }


def evaluate_combo(seed_arrays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    action_records: dict[str, Any] = {}
    for seed, arrays in seed_arrays.items():
        logits, records = apply_combo_logits(arrays)
        item = evaluate_logits(arrays, logits)
        item.pop("preds")
        item.pop("abnormal")
        per_seed[seed] = item
        action_records[seed] = records
    return {
        "actions": combo_action_payloads(),
        "aggregate": aggregate_per_seed(per_seed),
        "per_seed": per_seed,
        "action_records": action_records,
    }


def evaluate_extra_candidate(
    seed_arrays: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    combo_pred_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    for seed, arrays in seed_arrays.items():
        logits, _records = apply_combo_logits(arrays)
        extra_mask, extra_targets, extra_rows = apply_payload(logits, arrays, payload)
        item = evaluate_logits(arrays, logits, combo_preds=combo_pred_cache[seed])
        item.pop("preds")
        item.pop("abnormal")
        target_trigger_counts = Counter()
        target_top_counts = Counter()
        abnormal, weights = model_equivalent_abnormal(arrays, logits)
        top_indices = np.argmax(weights, axis=1)
        for row in extra_rows:
            view = VIEWS[int(extra_targets[row])]
            target_trigger_counts[view] += 1
            if int(top_indices[row]) == int(extra_targets[row]):
                target_top_counts[view] += 1
        item["extra_triggered_count"] = int(extra_mask.sum())
        item["extra_triggered_ids"] = sorted(
            str(arrays["patient_ids"][row]) for row in extra_rows
        )
        item["extra_target_trigger_count"] = {
            view: int(target_trigger_counts.get(view, 0)) for view in VIEWS
        }
        item["extra_target_top_count"] = {
            view: int(target_top_counts.get(view, 0)) for view in VIEWS
        }
        per_seed[seed] = item
    return {
        "extra_candidate": payload,
        "aggregate": aggregate_per_seed(per_seed),
        "per_seed": per_seed,
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    per_seed = item["per_seed"]
    min_acc = min(float(seed_item["metrics"]["accuracy"]) for seed_item in per_seed.values())
    return (
        -aggregate["total_broken_dfr25_correct"],
        -aggregate["incremental_broken_vs_combo"],
        aggregate["total_fixed_dfr25_errors"],
        aggregate["incremental_fixed_vs_combo"],
        metrics["accuracy"],
        metrics["auc"],
        min_acc,
        -aggregate["total_extra_triggered"],
    )


def build_extra_candidates(seed_arrays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for family, identities in (
        ("fp_normal_rescue", build_fp_identities(seed_arrays)),
        ("fn_abnormal_rescue", build_fn_identities(seed_arrays)),
    ):
        for identity in identities:
            payload = candidate_payload(family, tuple(identity))
            parts = []
            has_trigger = False
            for seed, arrays in seed_arrays.items():
                mask, targets = trigger_for_payload(arrays, payload)
                rows = tuple(int(row) for row in np.flatnonzero(mask))
                target_tuple = tuple(int(targets[row]) for row in rows)
                has_trigger = has_trigger or bool(rows)
                parts.append((seed, rows, target_tuple))
            signature = (
                family,
                float(payload["identity"]["residual"]),
                tuple(parts),
            )
            if has_trigger and signature not in seen:
                seen.add(signature)
                candidates.append(payload)
    return candidates


def combo_pred_cache(seed_arrays: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for seed, arrays in seed_arrays.items():
        logits, _records = apply_combo_logits(arrays)
        abnormal, _weights = model_equivalent_abnormal(arrays, logits)
        cache[seed] = (abnormal >= 0.5).astype(np.int64)
    return cache


def summarize_remaining_errors_after_combo(
    seed_arrays: dict[str, dict[str, Any]],
    cache: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed, arrays in seed_arrays.items():
        for index, patient_id in enumerate(arrays["patient_ids"]):
            patient_id = str(patient_id)
            label = int(arrays["labels"][index])
            pred = int(cache[seed][index])
            if label == pred:
                continue
            abnormal = arrays["abnormal_probs"][index]
            max_abnormal = float(abnormal.max())
            second_abnormal = float(np.sort(abnormal)[-2])
            base_pred = int(arrays["base_preds"][index])
            reason = explain_remaining_error(label, pred, abnormal, arrays, index)
            records.append(
                {
                    "seed": seed,
                    "patient_id": patient_id,
                    "label": label,
                    "base_pred": base_pred,
                    "combo_pred": pred,
                    "error_type": base_error_type(label, pred),
                    "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                    "max_view_abnormal": round_float(max_abnormal),
                    "second_view_abnormal": round_float(second_abnormal),
                    "reason": reason,
                    "view_abnormal": {
                        view: round_float(float(abnormal[VIEW_INDEX[view]]))
                        for view in VIEWS
                    },
                }
            )
    return records


def explain_remaining_error(
    label: int,
    pred: int,
    abnormal: np.ndarray,
    arrays: dict[str, Any],
    index: int,
) -> str:
    if label == 1 and pred == 0:
        return explain_fn(float(abnormal.max()), float(np.sort(abnormal)[-2]))

    axial = float(abnormal[VIEW_INDEX["axial"]])
    coronal = float(abnormal[VIEW_INDEX["coronal"]])
    sagittal = float(abnormal[VIEW_INDEX["sagittal"]])
    fused = float(arrays["base_abnormal"][index])
    if axial > 0.88:
        return "FP with axial abnormal above DFR113 safety cap"
    if fused < 0.8:
        return "FP lacks high fused-abnormal risk trigger"
    if 1.0 - sagittal < 0.55:
        return "FP lacks sagittal normal evidence"
    if coronal > 0.525:
        return "FP has coronal abnormal above safety cap"
    return "FP near boundary but unsafe in frontier audit"


def baseline_dfr25_summary(seed_arrays: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = []
    top_counts = Counter()
    errors = Counter()
    for arrays in seed_arrays.values():
        abnormal, weights = model_equivalent_abnormal(
            arrays,
            arrays["scaled_confidence_logits"],
        )
        metrics.append(metrics_from_abnormal_fast(arrays["labels"], abnormal))
        summary = summarize_weights(weights)
        for view in VIEWS:
            top_counts[view] += int(summary["top_weight_count"][view])
        for label, pred in zip(arrays["labels"], arrays["base_preds"]):
            errors.update([base_error_type(int(label), int(pred))])
    return {
        "mean_metrics": {
            "accuracy": round_float(mean([float(item["accuracy"]) for item in metrics])),
            "auc": round_float(mean([float(item["auc"]) for item in metrics])),
            "f1": round_float(mean([float(item["f1"]) for item in metrics])),
        },
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "error_types": counter_dict(errors),
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
    combo_cache = combo_pred_cache(seed_arrays)
    combo = evaluate_combo(seed_arrays)
    candidates = build_extra_candidates(seed_arrays)
    evaluated = [
        evaluate_extra_candidate(seed_arrays, payload, combo_cache)
        for payload in candidates
    ]
    ranked = sorted(evaluated, key=candidate_sort_key, reverse=True)
    no_harm = [
        item
        for item in ranked
        if int(item["aggregate"]["total_broken_dfr25_correct"]) == 0
        and int(item["aggregate"]["incremental_broken_vs_combo"]) == 0
    ]
    improved_no_harm = [
        item
        for item in no_harm
        if int(item["aggregate"]["total_fixed_dfr25_errors"])
        > int(combo["aggregate"]["total_fixed_dfr25_errors"])
    ]
    fixed_gt_combo = [
        item
        for item in ranked
        if int(item["aggregate"]["total_fixed_dfr25_errors"])
        > int(combo["aggregate"]["total_fixed_dfr25_errors"])
    ]
    report = {
        "analysis": "dfr117_combo_frontier_audit",
        "description": (
            "Analysis-only DFR-116 combo frontier audit.  Fixed DFR-116 combo "
            "plus one additional label-free FP/FN residual candidate; no test metrics."
        ),
        "baseline_dfr25": baseline_dfr25_summary(seed_arrays),
        "baseline_dfr116_combo": combo,
        "candidate_count": len(evaluated),
        "no_harm_candidate_count": len(no_harm),
        "improved_no_harm_candidate_count": len(improved_no_harm),
        "best_no_harm": no_harm[0] if no_harm else None,
        "best_improved_no_harm": improved_no_harm[0] if improved_no_harm else None,
        "best_safety_then_coverage": ranked[0] if ranked else None,
        "frontier_fixed_gt_combo": fixed_gt_combo[:20],
        "top_20_no_harm": no_harm[:20],
        "top_20_by_safety_then_coverage": ranked[:20],
        "remaining_errors_after_combo": summarize_remaining_errors_after_combo(
            seed_arrays,
            combo_cache,
        ),
    }
    save_json(repo_root / args.output, report)
    print(f"Saved DFR-117 combo frontier audit to: {repo_root / args.output}")
    print("dfr25", report["baseline_dfr25"]["mean_metrics"], report["baseline_dfr25"]["aggregate_top_weight_count"])
    print("dfr116_combo", combo["aggregate"]["mean_metrics"], combo["aggregate"]["aggregate_top_weight_count"], "fixed", combo["aggregate"]["total_fixed_dfr25_errors"], "broken", combo["aggregate"]["total_broken_dfr25_correct"])
    print("candidate_count", len(evaluated), "no_harm", len(no_harm), "improved_no_harm", len(improved_no_harm))
    if no_harm:
        best = no_harm[0]
        print("best_no_harm", best["extra_candidate"], best["aggregate"]["mean_metrics"], "fixed", best["aggregate"]["total_fixed_dfr25_errors"], "broken", best["aggregate"]["total_broken_dfr25_correct"], "incremental", best["aggregate"]["incremental_fixed_vs_combo"], best["aggregate"]["incremental_broken_vs_combo"])
    if improved_no_harm:
        best = improved_no_harm[0]
        print("best_improved_no_harm", best["extra_candidate"], best["aggregate"]["mean_metrics"], "fixed", best["aggregate"]["total_fixed_dfr25_errors"], "broken", best["aggregate"]["total_broken_dfr25_correct"])
    if fixed_gt_combo:
        best = fixed_gt_combo[0]
        print("best_fixed_gt_combo", best["extra_candidate"], best["aggregate"]["mean_metrics"], "fixed", best["aggregate"]["total_fixed_dfr25_errors"], "broken", best["aggregate"]["total_broken_dfr25_correct"])


if __name__ == "__main__":
    main()
