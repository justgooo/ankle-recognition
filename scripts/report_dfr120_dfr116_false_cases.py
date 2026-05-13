#!/usr/bin/env python3
"""Branch-aware false-case report for the DFR-116 combo branch.

This report does not search thresholds.  It profiles the 14 remaining DFR-116
validation errors, groups them by patient recurrence and evidence pattern, and
checks whether historical 3-seed variants ever fixed each case without
acceptable collateral damage.  No test split is read.
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
    counter_dict,
    load_json,
    round_float,
    save_json,
)
from scripts.analyze_dfr112_multiseed_posthoc import (  # noqa: E402
    DFR25_TELEMETRY,
    telemetry_arrays_model_equivalent,
)
from scripts.analyze_dfr117_combo_frontier_audit import (  # noqa: E402
    combo_pred_cache,
    evaluate_combo,
    summarize_remaining_errors_after_combo,
)
from scripts.analyze_dfr118_variant_evidence_drift import (  # noqa: E402
    discover_complete_groups,
    reference_maps,
    summarize_combo_replay,
    summarize_stored_group,
)
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402


DEFAULT_OUTPUT = "autoresearch_logs/dfr120_dfr116_false_case_report.json"
SEEDS = ("42", "123", "456")
MAX_FIXER_EXAMPLES = 8


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def top_weight_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def view_abnormal_from_sample(sample: dict[str, Any]) -> dict[str, float]:
    return {
        view: float(sample["views"][view]["abnormal_prob"])
        for view in VIEWS
    }


def evidence_bucket(error_type: str, view_abnormal: dict[str, float]) -> str:
    axial = float(view_abnormal["axial"])
    coronal = float(view_abnormal["coronal"])
    sagittal = float(view_abnormal["sagittal"])
    values = sorted([axial, coronal, sagittal], reverse=True)
    max_value = values[0]
    second_value = values[1]
    if error_type == "FP":
        if axial >= 0.95:
            return "strong_axial_fp_above_safety_cap"
        if axial >= 0.88:
            return "axial_fp_above_safety_cap"
        if max(coronal, sagittal) >= 0.5:
            return "multi_view_fp_evidence"
        return "low_nonaxial_fp_evidence"
    if max_value < 0.45:
        return "fn_no_view_reaches_weak_abnormal"
    if max_value < 0.5:
        return "fn_only_subthreshold_abnormal"
    if second_value < 0.2:
        return "fn_single_weak_positive_view_no_support"
    if second_value < 0.45:
        return "fn_single_positive_view_weak_support"
    return "fn_multi_view_borderline_support"


def action_bucket(case: dict[str, Any]) -> str:
    error_type = str(case["error_type"])
    bucket = str(case["evidence_bucket"])
    recurrence = int(case["patient_seed_recurrence"])
    best_fixer_broken = case.get("best_historical_fixer_broken_combo")

    if best_fixer_broken == 0:
        return "candidate_for_safe_mechanism"
    if error_type == "FP" and "strong_axial" in bucket and recurrence >= 2:
        return "data_or_view_classifier_audit_first"
    if error_type == "FP" and "safety_cap" in bucket:
        return "not_gate_threshold_safe"
    if error_type == "FN" and bucket in {
        "fn_no_view_reaches_weak_abnormal",
        "fn_only_subthreshold_abnormal",
        "fn_single_weak_positive_view_no_support",
    }:
        return "slice_sampling_or_per_view_sensitivity_audit"
    if best_fixer_broken is not None and best_fixer_broken >= 5:
        return "historically_fixable_only_with_high_collateral"
    return "low_priority_no_safe_signal"


def build_historical_fix_index(
    repo_root: Path,
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    groups = discover_complete_groups(repo_root)
    fix_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summaries: list[tuple[str, str, dict[str, Any], str]] = []
    for name, paths in sorted(groups.items()):
        stored = summarize_stored_group(name, paths, references)
        if stored["group_type"] == "trained":
            summaries.append((name, "stored", stored, "stored"))
        replay = summarize_combo_replay(name, paths, references)
        if replay is not None:
            summaries.append((replay["name"], "combo_replay", replay, "combo_replay"))

    for name, mode, summary, metrics_key in summaries:
        comparison = summary["comparison_vs_dfr116_combo"]
        fixed_ids = set(comparison["fixed_combo_ids"])
        for case_id in fixed_ids:
            metrics = summary[metrics_key]["mean_metrics"]
            fix_index[case_id].append(
                {
                    "name": name,
                    "mode": mode,
                    "mean_metrics": metrics,
                    "fixed_combo_errors": int(comparison["fixed_dfr116_combo_errors"]),
                    "broken_combo_correct": int(comparison["broken_dfr116_combo_correct"]),
                    "protected_dfr116_fixed_broken": int(
                        comparison["protected_dfr116_fixed_broken"]
                    ),
                }
            )
    for case_id in list(fix_index):
        fix_index[case_id] = sorted(
            fix_index[case_id],
            key=lambda item: (
                int(item["protected_dfr116_fixed_broken"]),
                int(item["broken_combo_correct"]),
                -int(item["fixed_combo_errors"]),
                -float(item["mean_metrics"]["accuracy"]),
            ),
        )
    return fix_index


def enrich_cases(
    repo_root: Path,
    remaining_errors: list[dict[str, Any]],
    fix_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    dfr116_maps = {
        seed: sample_map(load_json(repo_root / DFR116_TELEMETRY[seed]))
        for seed in SEEDS
    }
    by_patient = Counter(str(item["patient_id"]) for item in remaining_errors)
    cases: list[dict[str, Any]] = []
    for item in remaining_errors:
        seed = str(item["seed"])
        patient_id = str(item["patient_id"])
        case_id = f"{seed}:{patient_id}"
        dfr116_sample = dfr116_maps[seed][patient_id]
        dfr116_view_abnormal = view_abnormal_from_sample(dfr116_sample)
        bucket = evidence_bucket(str(item["error_type"]), dfr116_view_abnormal)
        fixers = fix_index.get(case_id, [])
        best_fixer = fixers[0] if fixers else None
        case = {
            "case_id": case_id,
            "seed": seed,
            "patient_id": patient_id,
            "patient_seed_recurrence": int(by_patient[patient_id]),
            "label": int(item["label"]),
            "error_type": str(item["error_type"]),
            "reason_from_dfr117": str(item["reason"]),
            "dfr25_abnormal": round_float(float(item["base_abnormal"])),
            "dfr116_abnormal": round_float(float(dfr116_sample["fusion_prediction"]["abnormal_prob"])),
            "dfr25_top_weight": str(item.get("base_top_weight", "unknown")),
            "dfr116_top_weight": top_weight_view(dfr116_sample),
            "dfr25_view_abnormal": {
                view: round_float(float(item["view_abnormal"][view]))
                for view in VIEWS
            },
            "dfr116_view_abnormal": {
                view: round_float(value) for view, value in dfr116_view_abnormal.items()
            },
            "max_view_abnormal": round_float(max(dfr116_view_abnormal.values())),
            "second_view_abnormal": round_float(
                sorted(dfr116_view_abnormal.values(), reverse=True)[1]
            ),
            "evidence_bucket": bucket,
            "historical_fixer_count": len(fixers),
            "best_historical_fixer_broken_combo": (
                int(best_fixer["broken_combo_correct"]) if best_fixer else None
            ),
            "best_historical_fixer": best_fixer,
            "historical_fixer_examples": fixers[:MAX_FIXER_EXAMPLES],
        }
        case["action_bucket"] = action_bucket(case)
        cases.append(case)
    return sorted(
        cases,
        key=lambda case: (
            -int(case["patient_seed_recurrence"]),
            str(case["error_type"]),
            str(case["patient_id"]),
            str(case["seed"]),
        ),
    )


def patient_groups(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_patient[str(case["patient_id"])].append(case)
    groups = []
    for patient_id, items in by_patient.items():
        groups.append(
            {
                "patient_id": patient_id,
                "seed_count": len(items),
                "seeds": sorted(str(item["seed"]) for item in items),
                "label": int(items[0]["label"]),
                "error_type_counts": counter_dict(Counter(item["error_type"] for item in items)),
                "evidence_buckets": counter_dict(Counter(item["evidence_bucket"] for item in items)),
                "action_buckets": counter_dict(Counter(item["action_bucket"] for item in items)),
                "case_ids": sorted(str(item["case_id"]) for item in items),
            }
        )
    return sorted(groups, key=lambda item: (-int(item["seed_count"]), item["patient_id"]))


def report_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "remaining_error_count": len(cases),
        "error_type_counts": counter_dict(Counter(item["error_type"] for item in cases)),
        "reason_counts": counter_dict(Counter(item["reason_from_dfr117"] for item in cases)),
        "evidence_bucket_counts": counter_dict(Counter(item["evidence_bucket"] for item in cases)),
        "action_bucket_counts": counter_dict(Counter(item["action_bucket"] for item in cases)),
        "recurrence_counts": counter_dict(Counter(item["patient_seed_recurrence"] for item in cases)),
        "historically_fixable_case_count": int(
            sum(1 for item in cases if int(item["historical_fixer_count"]) > 0)
        ),
        "safe_historical_fixer_case_count": int(
            sum(
                1
                for item in cases
                if item["best_historical_fixer_broken_combo"] == 0
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dfr25_arrays = {
        seed: telemetry_arrays_model_equivalent(load_json(repo_root / relative_path))
        for seed, relative_path in DFR25_TELEMETRY.items()
    }
    combo_cache = combo_pred_cache(dfr25_arrays)
    combo = evaluate_combo(dfr25_arrays)
    references = reference_maps(dfr25_arrays, combo_cache)
    remaining = summarize_remaining_errors_after_combo(dfr25_arrays, combo_cache)
    fix_index = build_historical_fix_index(repo_root, references)
    cases = enrich_cases(repo_root, remaining, fix_index)
    patients = patient_groups(cases)
    summary = report_summary(cases)
    report = {
        "analysis": "dfr120_dfr116_false_case_report",
        "description": (
            "Branch-aware false-case report for DFR-116 strict combo.  Profiles "
            "remaining validation errors, patient recurrence, evidence buckets, "
            "and historical fixer collateral; no threshold expansion or test metrics."
        ),
        "dfr116_reference": combo["aggregate"],
        "summary": summary,
        "patient_groups": patients,
        "cases": cases,
        "recommended_next_scope": {
            "do_not_expand_posthoc_thresholds": True,
            "highest_priority_data_or_classifier_audit": [
                item["case_id"]
                for item in cases
                if item["action_bucket"] == "data_or_view_classifier_audit_first"
            ],
            "slice_sampling_or_sensitivity_audit": [
                item["case_id"]
                for item in cases
                if item["action_bucket"] == "slice_sampling_or_per_view_sensitivity_audit"
            ],
            "safe_training_mechanism_candidates": [
                item["case_id"]
                for item in cases
                if item["action_bucket"] == "candidate_for_safe_mechanism"
            ],
        },
    }
    save_json(repo_root / args.output, report)

    print(f"Saved DFR-120 false-case report to: {repo_root / args.output}")
    print("dfr116", combo["aggregate"]["mean_metrics"], combo["aggregate"]["aggregate_top_weight_count"])
    print("summary", summary)
    print("patient_groups", len(patients))
    print(
        "safe_training_candidates",
        report["recommended_next_scope"]["safe_training_mechanism_candidates"],
    )


if __name__ == "__main__":
    main()
