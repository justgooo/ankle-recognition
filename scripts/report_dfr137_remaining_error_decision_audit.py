#!/usr/bin/env python3
"""DFR-137 remaining-error decision audit.

This report consolidates the DFR-116 clean remaining false-case analysis after
the posthoc, axial-risk, and sampling-frontier families have been tested.  It
does not train, does not read test metrics, and does not modify data files.
The goal is to decide whether any immediate model-side experiment remains
justified, or whether the next step should be a data/annotation review packet.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = "autoresearch_logs/dfr137_remaining_error_decision_audit.json"
SOURCE_PATHS = {
    "dfr120_false_case_report": "autoresearch_logs/dfr120_dfr116_false_case_report.json",
    "dfr123_duplicate_scan": "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json",
    "dfr125_clean_frontier": "autoresearch_logs/dfr125_clean_frontier_false_case_audit.json",
    "dfr126_clean_remaining": "autoresearch_logs/dfr126_clean_remaining_evidence_sampling.json",
    "dfr127_axial_fp_frontier": "autoresearch_logs/dfr127_axial_fp_calibration_frontier.json",
    "dfr136_sampling_replay_frontier": "autoresearch_logs/dfr136_sampling_replay_frontier.json",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def metric_triplet(row: dict[str, Any] | None) -> dict[str, float | None]:
    if not row:
        return {"accuracy": None, "auc": None, "f1": None}
    return {
        "accuracy": row.get("accuracy"),
        "auc": row.get("auc"),
        "f1": row.get("f1"),
    }


def patient_case_count(patient: dict[str, Any]) -> int:
    return len(patient.get("cases", []))


def duplicate_member_ids(patient: dict[str, Any]) -> list[str]:
    ids: set[str] = set()
    for group in patient.get("duplicate_groups", []):
        for member in group.get("members", []):
            ids.add(str(member["patient_id"]))
    return sorted(ids)


def max_error_view_abnormal(patient: dict[str, Any]) -> float | None:
    ranges = patient["interpretation"]["error_seed_view_abnormal_ranges"]
    values = [
        item.get("max")
        for item in ranges.values()
        if item.get("max") is not None
    ]
    return max(float(value) for value in values) if values else None


def min_axial_error_abnormal(patient: dict[str, Any]) -> float | None:
    value = patient["interpretation"]["error_seed_view_abnormal_ranges"]["axial"].get("min")
    return float(value) if value is not None else None


def decide_patient_bucket(patient: dict[str, Any]) -> dict[str, Any]:
    label = int(patient["label"])
    suggested = str(patient["interpretation"]["suggested_bucket"])
    duplicate_ids = duplicate_member_ids(patient)
    sampling_flags = list(patient["interpretation"].get("sampling_flags", []))
    max_view = max_error_view_abnormal(patient)
    min_axial = min_axial_error_abnormal(patient)

    blockers: list[str] = []
    next_bucket = "data_or_annotation_review"
    model_side_next = False

    if label == 1:
        if duplicate_ids:
            blockers.append("validation duplicate content group; optimize only after duplicate-aware review")
            next_bucket = "duplicate_positive_annotation_review"
        elif max_view is not None and max_view < 0.45:
            blockers.append("no error-seed view reaches weak abnormal evidence")
            next_bucket = "positive_no_view_evidence_annotation_review"
        else:
            blockers.append("positive evidence remains subthreshold or single-view weak")
            next_bucket = "positive_weak_evidence_review"
        if sampling_flags:
            blockers.append("baseline sampling flags exist, but DFR132-136 closed global/replay sampling")
    else:
        if suggested == "negative_strong_axial_classifier_fp":
            blockers.append("strong axial FP overlaps many true positives; DFR127 found no zero-positive gate")
        if min_axial is not None and min_axial >= 0.90:
            blockers.append("axial abnormal >=0.90 in every error seed")
        if sampling_flags:
            blockers.append("sampling flags are not actionable after DFR136 sampling closure")
        next_bucket = "negative_strong_axial_fp_label_or_view_classifier_review"

    return {
        "patient_id": patient["patient_id"],
        "label": label,
        "case_ids": list(patient.get("case_ids", [])),
        "case_count": patient_case_count(patient),
        "dfr126_bucket": suggested,
        "decision_bucket": next_bucket,
        "model_side_next_experiment_supported": model_side_next,
        "blockers": blockers,
        "duplicate_member_ids": duplicate_ids,
        "sampling_flags": sampling_flags,
        "error_seed_view_abnormal_ranges": patient["interpretation"]["error_seed_view_abnormal_ranges"],
        "all_view_columns_same_path": bool(patient.get("all_view_columns_same_path")),
        "all_view_hashes_same": bool(patient.get("all_view_hashes_same")),
    }


def closure_evidence(reports: dict[str, Any]) -> dict[str, Any]:
    dfr125 = reports["dfr125_clean_frontier"]
    dfr127 = reports["dfr127_axial_fp_frontier"]
    dfr136 = reports["dfr136_sampling_replay_frontier"]

    best_full_fp = dfr127["candidate_scan"]["best_full_fp_coverage"]
    dfr136_best = dfr136["candidate_frontier"]["best_improved_no_harm_vs_dfr25"]
    return {
        "posthoc_threshold_expansion": {
            "closed": int(dfr125["improved_no_harm_candidate_count"]) == 0,
            "candidate_count": int(dfr125["candidate_count"]),
            "no_harm_candidate_count": int(dfr125["no_harm_candidate_count"]),
            "improved_no_harm_candidate_count": int(dfr125["improved_no_harm_candidate_count"]),
            "best_no_harm_metrics": metric_triplet(
                dfr125.get("best_no_harm", {}).get("aggregate", {}).get("mean_metrics")
            ),
        },
        "label_free_axial_fp_gate": {
            "closed": int(dfr127["candidate_scan"]["zero_positive_candidate_count"]) == 0,
            "candidate_count": int(dfr127["candidate_scan"]["candidate_count"]),
            "zero_positive_candidate_count": int(dfr127["candidate_scan"]["zero_positive_candidate_count"]),
            "full_fp_coverage_candidate_count": int(
                dfr127["candidate_scan"]["full_fp_coverage_candidate_count"]
            ),
            "best_full_fp_coverage_fp_count": int(best_full_fp["fp_covered_count"]),
            "best_full_fp_coverage_positive_collateral_count": int(
                best_full_fp["positive_collateral_count"]
            ),
            "best_full_fp_coverage_fp_precision_vs_positive": best_full_fp[
                "fp_precision_vs_positive"
            ],
        },
        "sampling_count_or_replay": {
            "closed": bool(dfr136["assessment"]["close_sampling_family_if_no_dfr116_safe_gain"]),
            "candidate_count": int(dfr136["candidate_frontier"]["candidate_count"]),
            "improved_no_harm_vs_dfr25_count": int(
                dfr136["candidate_frontier"]["improved_no_harm_vs_dfr25_count"]
            ),
            "dfr116_safe_count": int(dfr136["candidate_frontier"]["dfr116_safe_count"]),
            "best_improved_no_harm_name": dfr136_best["name"] if dfr136_best else None,
            "best_improved_no_harm_fixed_ids": (
                dfr136_best["vs_dfr25"]["fixed_ids"] if dfr136_best else []
            ),
            "best_improved_no_harm_broken_vs_dfr116": (
                dfr136_best["vs_dfr116"]["broken_ids"] if dfr136_best else []
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports = {
        name: load_json(REPO_ROOT / path)
        for name, path in SOURCE_PATHS.items()
    }
    patients = [
        decide_patient_bucket(patient)
        for patient in reports["dfr126_clean_remaining"]["patients"]
    ]
    decision_counts = Counter(patient["decision_bucket"] for patient in patients)
    dfr126_bucket_counts = Counter(patient["dfr126_bucket"] for patient in patients)

    source_relpaths = {
        name: str((REPO_ROOT / path).relative_to(REPO_ROOT))
        for name, path in SOURCE_PATHS.items()
    }
    clean_metrics = reports["dfr125_clean_frontier"]["baseline_dfr116_combo_clean"]["aggregate"][
        "mean_metrics"
    ]
    full_metrics = reports["dfr120_false_case_report"]["dfr116_reference"]["mean_metrics"]

    payload = {
        "analysis": "dfr137_remaining_error_decision_audit",
        "description": (
            "Read-only decision audit after DFR116 branch, clean-frontier closure, axial-risk "
            "closure, and sampling-family closure. No training, no test metrics, no data edits."
        ),
        "source_paths": source_relpaths,
        "dfr116_reference_metrics": {
            "official_full_mean": metric_triplet(full_metrics),
            "combined_clean_mean": metric_triplet(clean_metrics),
        },
        "closure_evidence": closure_evidence(reports),
        "clean_remaining_summary": {
            "target_patient_count": int(reports["dfr126_clean_remaining"]["target_patient_count"]),
            "target_case_count": int(reports["dfr126_clean_remaining"]["target_case_count"]),
            "dfr126_bucket_counts": counter_dict(dfr126_bucket_counts),
            "decision_bucket_counts": counter_dict(decision_counts),
            "model_side_supported_patient_count": int(
                sum(1 for patient in patients if patient["model_side_next_experiment_supported"])
            ),
        },
        "patients": sorted(
            patients,
            key=lambda item: (
                str(item["decision_bucket"]),
                -int(item["case_count"]),
                str(item["patient_id"]),
            ),
        ),
        "assessment": {
            "immediate_model_side_experiment_supported": False,
            "safe_training_or_eval_candidate_count": 0,
            "closed_families": [
                "posthoc threshold expansion",
                "label-free axial FP gate",
                "global slice-count training",
                "frozen sampling replay",
                "sampling replay blend/selector",
            ],
            "main_failure_mode": (
                "Remaining clean errors are no longer a gate-threshold or global sampling problem: "
                "positive cases have duplicate/subthreshold/no-view evidence, while negative cases "
                "are strong axial classifier FPs that overlap many true positives."
            ),
            "recommended_next_experiment": (
                "DFR-138 should generate a compact read-only review packet for the seven clean "
                "remaining patients, including paths/hashes, duplicate membership, view evidence, "
                "sampling flags, and model predictions. Do not start another training run until "
                "that packet identifies a model-expressible target."
            ),
        },
    }

    save_json(REPO_ROOT / args.output, payload)
    print(f"Wrote {REPO_ROOT / args.output}")
    print(json.dumps(payload["assessment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
