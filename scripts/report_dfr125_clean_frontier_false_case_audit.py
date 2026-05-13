#!/usr/bin/env python3
"""Combined-clean DFR-116 false-case/frontier audit for DFR-125.

This read-only report applies the DFR-124 combined clean validation policy:
exclude validation patients duplicated in train and keep the first member of
val-val duplicate groups.  It then re-lists DFR-116 remaining errors and reruns
the DFR-117 one-extra-posthoc frontier under that clean policy.  No test metric
or data file is read or modified.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
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
from scripts.analyze_dfr114_fp_risk_coverage import metrics_from_abnormal_fast  # noqa: E402
from scripts.analyze_dfr117_combo_frontier_audit import (  # noqa: E402
    apply_combo_logits,
    apply_payload,
    build_extra_candidates,
    combo_action_payloads,
    explain_remaining_error,
)
from scripts.report_dfr120_dfr116_false_cases import evidence_bucket  # noqa: E402
from scripts.report_dfr124_duplicate_leakage_metric_sensitivity import (  # noqa: E402
    exclude_patients_mask,
    keep_first_val_val_mask,
    validation_patient_groups,
)


DEFAULT_DUPLICATE_REPORT = "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr125_clean_frontier_false_case_audit.json"
SEEDS = ("42", "123", "456")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def clean_mask(arrays: dict[str, Any], groups: dict[str, Any]) -> np.ndarray:
    patient_ids = np.asarray(arrays["patient_ids"], dtype=object)
    return keep_first_val_val_mask(patient_ids, groups["val_val_groups"]) & exclude_patients_mask(
        patient_ids,
        groups["train_val_validation_patients"],
    )


def clean_policy_summary(seed_arrays: dict[str, dict[str, Any]], groups: dict[str, Any]) -> dict[str, Any]:
    per_seed = {}
    for seed, arrays in seed_arrays.items():
        mask = clean_mask(arrays, groups)
        patient_ids = np.asarray(arrays["patient_ids"], dtype=object)
        per_seed[seed] = {
            "included_count": int(mask.sum()),
            "excluded_count": int((~mask).sum()),
            "excluded_patient_ids": sorted(str(patient_id) for patient_id in patient_ids[~mask]),
        }
    return {
        "groups": groups,
        "per_seed": per_seed,
    }


def evaluate_logits_clean(
    arrays: dict[str, Any],
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    combo_preds: np.ndarray | None = None,
) -> dict[str, Any]:
    abnormal, weights = model_equivalent_abnormal(arrays, logits)
    labels = arrays["labels"]
    preds = (abnormal >= 0.5).astype(np.int64)
    base_preds = arrays["base_preds"]

    fixed = []
    broken = []
    incremental_fixed = []
    incremental_broken = []
    changed_from_combo = []
    for index, patient_id in enumerate(arrays["patient_ids"]):
        if not bool(mask[index]):
            continue
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
        "metrics": metrics_from_abnormal_fast(labels[mask], abnormal[mask]),
        "effective_n": int(mask.sum()),
        "fixed_dfr25_errors": int(len(fixed)),
        "broken_dfr25_correct": int(len(broken)),
        "net_delta_vs_dfr25": int(len(fixed) - len(broken)),
        "fixed_ids": sorted(fixed),
        "broken_ids": sorted(broken),
        "incremental_fixed_vs_combo": int(len(incremental_fixed)),
        "incremental_broken_vs_combo": int(len(incremental_broken)),
        "incremental_fixed_ids": sorted(incremental_fixed),
        "incremental_broken_ids": sorted(incremental_broken),
        "changed_from_combo_ids": sorted(changed_from_combo),
        "weight_summary": summarize_weights(weights[mask]),
        "preds": preds,
        "abnormal": abnormal,
        "weights": weights,
    }


def aggregate_clean(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in per_seed.values()]
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    total_fixed = 0
    total_broken = 0
    incremental_fixed = 0
    incremental_broken = 0
    total_triggered_clean = 0
    total_triggered_full = 0
    effective_n = 0
    extra_trigger_counts = Counter()
    extra_top_counts = Counter()
    for item in per_seed.values():
        effective_n += int(item.get("effective_n", 0))
        total_fixed += int(item["fixed_dfr25_errors"])
        total_broken += int(item["broken_dfr25_correct"])
        incremental_fixed += int(item["incremental_fixed_vs_combo"])
        incremental_broken += int(item["incremental_broken_vs_combo"])
        total_triggered_clean += int(item.get("extra_triggered_count_clean", 0))
        total_triggered_full += int(item.get("extra_triggered_count_full", 0))
        extra_trigger_counts.update(item.get("extra_target_trigger_count_clean", {}))
        extra_top_counts.update(item.get("extra_target_top_count_clean", {}))
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
        "effective_n_total": int(effective_n),
        "total_extra_triggered_clean": int(total_triggered_clean),
        "total_extra_triggered_full": int(total_triggered_full),
        "extra_target_trigger_count_clean": {
            view: int(extra_trigger_counts.get(view, 0)) for view in VIEWS
        },
        "extra_target_top_count_clean": {
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


def baseline_dfr25_clean(seed_arrays: dict[str, dict[str, Any]], groups: dict[str, Any]) -> dict[str, Any]:
    per_seed = {}
    error_types = Counter()
    for seed, arrays in seed_arrays.items():
        mask = clean_mask(arrays, groups)
        item = evaluate_logits_clean(arrays, arrays["scaled_confidence_logits"], mask)
        item.pop("preds")
        item.pop("abnormal")
        item.pop("weights")
        per_seed[seed] = item
        for label, pred in zip(arrays["labels"][mask], arrays["base_preds"][mask]):
            error_types.update([base_error_type(int(label), int(pred))])
    aggregate = aggregate_clean(per_seed)
    aggregate["error_types"] = counter_dict(error_types)
    return {
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


def combo_pred_cache(seed_arrays: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    cache = {}
    for seed, arrays in seed_arrays.items():
        logits, _records = apply_combo_logits(arrays)
        abnormal, _weights = model_equivalent_abnormal(arrays, logits)
        cache[seed] = (abnormal >= 0.5).astype(np.int64)
    return cache


def combo_clean(seed_arrays: dict[str, dict[str, Any]], groups: dict[str, Any]) -> dict[str, Any]:
    per_seed = {}
    action_records = {}
    for seed, arrays in seed_arrays.items():
        mask = clean_mask(arrays, groups)
        logits, records = apply_combo_logits(arrays)
        item = evaluate_logits_clean(arrays, logits, mask)
        item.pop("preds")
        item.pop("abnormal")
        item.pop("weights")
        per_seed[seed] = item
        action_records[seed] = records
    return {
        "actions": combo_action_payloads(),
        "aggregate": aggregate_clean(per_seed),
        "per_seed": per_seed,
        "action_records": action_records,
    }


def evaluate_extra_candidate_clean(
    seed_arrays: dict[str, dict[str, Any]],
    groups: dict[str, Any],
    payload: dict[str, Any],
    combo_preds: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_seed = {}
    for seed, arrays in seed_arrays.items():
        mask = clean_mask(arrays, groups)
        logits, _records = apply_combo_logits(arrays)
        extra_mask, extra_targets, extra_rows = apply_payload(logits, arrays, payload)
        item = evaluate_logits_clean(arrays, logits, mask, combo_preds=combo_preds[seed])
        target_trigger_counts = Counter()
        target_top_counts = Counter()
        top_indices = np.argmax(item["weights"], axis=1)
        clean_rows = [int(row) for row in extra_rows if bool(mask[int(row)])]
        for row in clean_rows:
            view = VIEWS[int(extra_targets[row])]
            target_trigger_counts[view] += 1
            if int(top_indices[row]) == int(extra_targets[row]):
                target_top_counts[view] += 1
        item["extra_triggered_count_full"] = int(extra_mask.sum())
        item["extra_triggered_count_clean"] = int(len(clean_rows))
        item["extra_triggered_ids_clean"] = sorted(
            str(arrays["patient_ids"][row]) for row in clean_rows
        )
        item["extra_target_trigger_count_clean"] = {
            view: int(target_trigger_counts.get(view, 0)) for view in VIEWS
        }
        item["extra_target_top_count_clean"] = {
            view: int(target_top_counts.get(view, 0)) for view in VIEWS
        }
        item.pop("preds")
        item.pop("abnormal")
        item.pop("weights")
        per_seed[seed] = item
    return {
        "extra_candidate": payload,
        "aggregate": aggregate_clean(per_seed),
        "per_seed": per_seed,
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    metrics = aggregate["mean_metrics"]
    min_acc = min(float(seed_item["metrics"]["accuracy"]) for seed_item in item["per_seed"].values())
    return (
        -aggregate["total_broken_dfr25_correct"],
        -aggregate["incremental_broken_vs_combo"],
        aggregate["total_fixed_dfr25_errors"],
        aggregate["incremental_fixed_vs_combo"],
        metrics["accuracy"],
        metrics["auc"],
        min_acc,
        -aggregate["total_extra_triggered_clean"],
    )


def remaining_errors_after_combo_clean(
    seed_arrays: dict[str, dict[str, Any]],
    groups: dict[str, Any],
    combo_preds: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records = []
    for seed, arrays in seed_arrays.items():
        mask = clean_mask(arrays, groups)
        logits, _action_records = apply_combo_logits(arrays)
        abnormal, weights = model_equivalent_abnormal(arrays, logits)
        top_indices = np.argmax(weights, axis=1)
        for index, patient_id in enumerate(arrays["patient_ids"]):
            if not bool(mask[index]):
                continue
            label = int(arrays["labels"][index])
            pred = int(combo_preds[seed][index])
            if label == pred:
                continue
            view_abnormal = {
                view: round_float(float(arrays["abnormal_probs"][index, VIEW_INDEX[view]]))
                for view in VIEWS
            }
            error_type = base_error_type(label, pred)
            records.append(
                {
                    "seed": str(seed),
                    "patient_id": str(patient_id),
                    "case_id": f"{seed}:{patient_id}",
                    "label": label,
                    "base_pred": int(arrays["base_preds"][index]),
                    "combo_pred": pred,
                    "error_type": error_type,
                    "base_abnormal": round_float(float(arrays["base_abnormal"][index])),
                    "combo_abnormal": round_float(float(abnormal[index])),
                    "combo_top_weight": VIEWS[int(top_indices[index])],
                    "max_view_abnormal": round_float(float(arrays["abnormal_probs"][index].max())),
                    "second_view_abnormal": round_float(
                        float(np.sort(arrays["abnormal_probs"][index])[-2])
                    ),
                    "reason": explain_remaining_error(
                        label,
                        pred,
                        arrays["abnormal_probs"][index],
                        arrays,
                        index,
                    ),
                    "view_abnormal": view_abnormal,
                    "evidence_bucket": evidence_bucket(error_type, view_abnormal),
                }
            )
    return sorted(records, key=lambda item: (item["patient_id"], item["seed"]))


def remaining_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient = Counter(str(item["patient_id"]) for item in cases)
    return {
        "remaining_error_count": int(len(cases)),
        "unique_patient_count": int(len(by_patient)),
        "error_type_counts": counter_dict(Counter(item["error_type"] for item in cases)),
        "reason_counts": counter_dict(Counter(item["reason"] for item in cases)),
        "evidence_bucket_counts": counter_dict(Counter(item["evidence_bucket"] for item in cases)),
        "recurrence_counts": counter_dict(Counter(by_patient.values())),
    }


def patient_groups(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["patient_id"])].append(case)
    records = []
    for patient_id, items in grouped.items():
        records.append(
            {
                "patient_id": patient_id,
                "seed_count": int(len(items)),
                "seeds": sorted(str(item["seed"]) for item in items),
                "label": int(items[0]["label"]),
                "error_type_counts": counter_dict(Counter(item["error_type"] for item in items)),
                "evidence_bucket_counts": counter_dict(
                    Counter(item["evidence_bucket"] for item in items)
                ),
                "case_ids": sorted(str(item["case_id"]) for item in items),
            }
        )
    return sorted(records, key=lambda item: (-int(item["seed_count"]), item["patient_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--duplicate-report", type=Path, default=Path(DEFAULT_DUPLICATE_REPORT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    duplicate_report_path = args.duplicate_report
    if not duplicate_report_path.is_absolute():
        duplicate_report_path = repo_root / duplicate_report_path
    duplicate_report = load_json(duplicate_report_path)
    groups = validation_patient_groups(duplicate_report)
    seed_arrays = {
        seed: telemetry_arrays_model_equivalent(load_json(repo_root / DFR25_TELEMETRY[seed]))
        for seed in SEEDS
    }
    combo_preds = combo_pred_cache(seed_arrays)
    dfr25 = baseline_dfr25_clean(seed_arrays, groups)
    combo = combo_clean(seed_arrays, groups)
    remaining = remaining_errors_after_combo_clean(seed_arrays, groups, combo_preds)

    candidates = build_extra_candidates(seed_arrays)
    evaluated = [
        evaluate_extra_candidate_clean(seed_arrays, groups, payload, combo_preds)
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
        "analysis": "dfr125_clean_frontier_false_case_audit",
        "description": (
            "Combined-clean DFR-116 false-case and one-extra-posthoc frontier audit. "
            "Uses DFR-124 clean policy; no test metrics or data edits."
        ),
        "duplicate_report": str(duplicate_report_path),
        "clean_policy": clean_policy_summary(seed_arrays, groups),
        "baseline_dfr25_clean": dfr25,
        "baseline_dfr116_combo_clean": combo,
        "remaining_summary": remaining_summary(remaining),
        "remaining_patient_groups": patient_groups(remaining),
        "remaining_errors_after_combo_clean": remaining,
        "candidate_count": int(len(evaluated)),
        "no_harm_candidate_count": int(len(no_harm)),
        "improved_no_harm_candidate_count": int(len(improved_no_harm)),
        "best_no_harm": no_harm[0] if no_harm else None,
        "best_improved_no_harm": improved_no_harm[0] if improved_no_harm else None,
        "best_safety_then_coverage": ranked[0] if ranked else None,
        "frontier_fixed_gt_combo": fixed_gt_combo[:20],
        "top_20_no_harm": no_harm[:20],
        "top_20_by_safety_then_coverage": ranked[:20],
    }
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    save_json(output, report)

    print(f"Saved DFR-125 clean frontier false-case audit to: {output}")
    print(
        "dfr25_clean",
        dfr25["aggregate"]["mean_metrics"],
        dfr25["aggregate"]["aggregate_top_weight_count"],
    )
    print(
        "dfr116_combo_clean",
        combo["aggregate"]["mean_metrics"],
        combo["aggregate"]["aggregate_top_weight_count"],
        "fixed",
        combo["aggregate"]["total_fixed_dfr25_errors"],
        "broken",
        combo["aggregate"]["total_broken_dfr25_correct"],
    )
    print("remaining", report["remaining_summary"])
    print(
        "candidate_count",
        len(evaluated),
        "no_harm",
        len(no_harm),
        "improved_no_harm",
        len(improved_no_harm),
    )
    if no_harm:
        best = no_harm[0]
        print(
            "best_no_harm",
            best["extra_candidate"],
            best["aggregate"]["mean_metrics"],
            "fixed",
            best["aggregate"]["total_fixed_dfr25_errors"],
            "broken",
            best["aggregate"]["total_broken_dfr25_correct"],
            "incremental",
            best["aggregate"]["incremental_fixed_vs_combo"],
            best["aggregate"]["incremental_broken_vs_combo"],
        )
    if improved_no_harm:
        best = improved_no_harm[0]
        print(
            "best_improved_no_harm",
            best["extra_candidate"],
            best["aggregate"]["mean_metrics"],
            "fixed",
            best["aggregate"]["total_fixed_dfr25_errors"],
            "broken",
            best["aggregate"]["total_broken_dfr25_correct"],
        )


if __name__ == "__main__":
    main()
