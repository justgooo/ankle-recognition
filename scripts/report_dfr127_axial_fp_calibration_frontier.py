#!/usr/bin/env python3
"""Axial FP calibration frontier for DFR-127.

This read-only report uses clean validation telemetry only.  It analyzes
DFR-116 negative false positives vs true negatives and true positives, then
scans simple label-free axial-risk feature rules to see whether strong axial FP
cases are separable without positive collateral.  It does not change data files
or use test metrics.
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
    counter_dict,
    load_json,
    round_float,
    save_json,
)
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402
from scripts.report_dfr122_duplicate_metric_sensitivity import auc_from_scores  # noqa: E402
from scripts.report_dfr124_duplicate_leakage_metric_sensitivity import (  # noqa: E402
    exclude_patients_mask,
    keep_first_val_val_mask,
    validation_patient_groups,
)


DEFAULT_DUPLICATE_REPORT = "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json"
DEFAULT_CLEAN_REMAINING_REPORT = "autoresearch_logs/dfr126_clean_remaining_evidence_sampling.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr127_axial_fp_calibration_frontier.json"
SEEDS = ("42", "123", "456")


def clean_mask(patient_ids: np.ndarray, groups: dict[str, Any]) -> np.ndarray:
    return keep_first_val_val_mask(patient_ids, groups["val_val_groups"]) & exclude_patients_mask(
        patient_ids,
        groups["train_val_validation_patients"],
    )


def sample_records(repo_root: Path, groups: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for seed in SEEDS:
        telemetry = load_json(repo_root / DFR116_TELEMETRY[seed])
        samples = telemetry["samples"]
        patient_ids = np.asarray([str(sample["patient_id"]) for sample in samples], dtype=object)
        mask = clean_mask(patient_ids, groups)
        for index, sample in enumerate(samples):
            if not bool(mask[index]):
                continue
            views = {
                view: float(sample["views"][view]["abnormal_prob"])
                for view in VIEWS
            }
            weights = {
                view: float(sample["views"][view]["fusion_weight"])
                for view in VIEWS
            }
            label = int(sample["label"])
            pred = int(sample["fusion_prediction"]["pred"])
            axial = views["axial"]
            coronal = views["coronal"]
            sagittal = views["sagittal"]
            max_nonaxial = max(coronal, sagittal)
            min_nonaxial_normal = min(1.0 - coronal, 1.0 - sagittal)
            max_nonaxial_normal = max(1.0 - coronal, 1.0 - sagittal)
            records.append(
                {
                    "seed": seed,
                    "patient_id": str(sample["patient_id"]),
                    "case_id": f"{seed}:{sample['patient_id']}",
                    "label": label,
                    "pred": pred,
                    "correct": bool(label == pred),
                    "error_type": "TP" if label == pred == 1 else "TN" if label == pred == 0 else "FN" if label == 1 else "FP",
                    "fusion_abnormal": float(sample["fusion_prediction"]["abnormal_prob"]),
                    "top_weight_view": str((sample.get("top_weight_views") or ["none"])[0]),
                    "views": views,
                    "weights": weights,
                    "features": {
                        "axial_abnormal": axial,
                        "coronal_abnormal": coronal,
                        "sagittal_abnormal": sagittal,
                        "sagittal_normal": 1.0 - sagittal,
                        "coronal_normal": 1.0 - coronal,
                        "max_nonaxial_abnormal": max_nonaxial,
                        "min_nonaxial_abnormal": min(coronal, sagittal),
                        "min_nonaxial_normal": min_nonaxial_normal,
                        "max_nonaxial_normal": max_nonaxial_normal,
                        "axial_minus_max_nonaxial": axial - max_nonaxial,
                        "axial_minus_sagittal": axial - sagittal,
                        "axial_minus_coronal": axial - coronal,
                    },
                }
            )
    return records


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round_float(float(np.min(arr))),
        "q25": round_float(float(np.quantile(arr, 0.25))),
        "median": round_float(float(np.quantile(arr, 0.50))),
        "q75": round_float(float(np.quantile(arr, 0.75))),
        "max": round_float(float(np.max(arr))),
        "mean": round_float(float(np.mean(arr))),
    }


def feature_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = sorted(records[0]["features"]) if records else []
    grouped: dict[str, list[dict[str, Any]]] = {
        key: [record for record in records if record["error_type"] == key]
        for key in ("TN", "FP", "TP", "FN")
    }
    distribution = {}
    for feature in feature_names:
        distribution[feature] = {
            group: quantiles([float(record["features"][feature]) for record in items])
            for group, items in grouped.items()
        }
    return distribution


def separability(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_positive = [record for record in records if int(record["pred"]) == 1]
    negative_clean = [record for record in records if int(record["label"]) == 0]
    feature_names = sorted(records[0]["features"]) if records else []
    result = {}
    for feature in feature_names:
        feature_result = {}
        if predicted_positive:
            labels = np.asarray(
                [1 if record["error_type"] == "FP" else 0 for record in predicted_positive],
                dtype=np.int64,
            )
            scores = np.asarray([float(record["features"][feature]) for record in predicted_positive])
            feature_result["fp_vs_tp_auc_predicted_positive"] = round_float(
                auc_from_scores(labels, scores)
            ) if len(set(labels.tolist())) == 2 else None
        if negative_clean:
            labels = np.asarray(
                [1 if record["error_type"] == "FP" else 0 for record in negative_clean],
                dtype=np.int64,
            )
            scores = np.asarray([float(record["features"][feature]) for record in negative_clean])
            feature_result["fp_vs_tn_auc_negative_only"] = round_float(
                auc_from_scores(labels, scores)
            ) if len(set(labels.tolist())) == 2 else None
        result[feature] = feature_result
    return result


def trigger_record(record: dict[str, Any], candidate: dict[str, Any]) -> bool:
    features = record["features"]
    if candidate["family"] == "sagittal_normal_axial_risk":
        return (
            record["pred"] == 1
            and features["axial_abnormal"] >= candidate["axial_min"]
            and record["fusion_abnormal"] >= candidate["fused_min"]
            and features["sagittal_normal"] >= candidate["sagittal_normal_min"]
            and features["coronal_abnormal"] <= candidate["coronal_max"]
        )
    if candidate["family"] == "nonaxial_low_support_axial_risk":
        return (
            record["pred"] == 1
            and features["axial_abnormal"] >= candidate["axial_min"]
            and record["fusion_abnormal"] >= candidate["fused_min"]
            and features["max_nonaxial_abnormal"] <= candidate["max_nonaxial_max"]
        )
    if candidate["family"] == "axial_gap_low_support_risk":
        return (
            record["pred"] == 1
            and features["axial_abnormal"] >= candidate["axial_min"]
            and record["fusion_abnormal"] >= candidate["fused_min"]
            and features["max_nonaxial_abnormal"] <= candidate["max_nonaxial_max"]
            and features["axial_minus_max_nonaxial"] >= candidate["gap_min"]
        )
    raise ValueError(f"Unknown candidate family: {candidate['family']}")


def candidate_grid() -> list[dict[str, Any]]:
    candidates = []
    for axial_min in (0.88, 0.90, 0.925, 0.95, 0.97, 0.985):
        for fused_min in (0.75, 0.80, 0.85, 0.90, 0.95):
            for sagittal_normal_min in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
                for coronal_max in (0.45, 0.50, 0.525, 0.55, 0.60):
                    candidates.append(
                        {
                            "family": "sagittal_normal_axial_risk",
                            "axial_min": axial_min,
                            "fused_min": fused_min,
                            "sagittal_normal_min": sagittal_normal_min,
                            "coronal_max": coronal_max,
                        }
                    )
            for max_nonaxial_max in (0.35, 0.40, 0.45, 0.50, 0.525, 0.55, 0.60):
                candidates.append(
                    {
                        "family": "nonaxial_low_support_axial_risk",
                        "axial_min": axial_min,
                        "fused_min": fused_min,
                        "max_nonaxial_max": max_nonaxial_max,
                    }
                )
                for gap_min in (0.30, 0.40, 0.50, 0.60, 0.70):
                    candidates.append(
                        {
                            "family": "axial_gap_low_support_risk",
                            "axial_min": axial_min,
                            "fused_min": fused_min,
                            "max_nonaxial_max": max_nonaxial_max,
                            "gap_min": gap_min,
                        }
                    )
    return candidates


def evaluate_candidate(records: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    triggered = [record for record in records if trigger_record(record, candidate)]
    type_counts = Counter(record["error_type"] for record in triggered)
    label_counts = Counter(str(record["label"]) for record in triggered)
    positive_records = [record for record in triggered if int(record["label"]) == 1]
    fp_records = [record for record in triggered if record["error_type"] == "FP"]
    return {
        "candidate": candidate,
        "triggered_count": int(len(triggered)),
        "triggered_type_counts": counter_dict(type_counts),
        "triggered_label_counts": counter_dict(label_counts),
        "fp_covered_count": int(len(fp_records)),
        "positive_collateral_count": int(len(positive_records)),
        "tp_collateral_count": int(type_counts.get("TP", 0)),
        "fn_collateral_count": int(type_counts.get("FN", 0)),
        "tn_triggered_count": int(type_counts.get("TN", 0)),
        "fp_precision_vs_positive": round_float(
            len(fp_records) / (len(fp_records) + len(positive_records))
        )
        if (len(fp_records) + len(positive_records)) > 0
        else None,
        "triggered_case_ids": sorted(record["case_id"] for record in triggered),
        "fp_case_ids": sorted(record["case_id"] for record in fp_records),
        "positive_case_ids": sorted(record["case_id"] for record in positive_records),
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(item["positive_collateral_count"]),
        int(item["fp_covered_count"]),
        float(item["fp_precision_vs_positive"] or 0.0),
        -int(item["triggered_count"]),
    )


def scan_candidates(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        evaluate_candidate(records, candidate)
        for candidate in candidate_grid()
    ]
    evaluated = [item for item in evaluated if int(item["triggered_count"]) > 0]
    zero_positive = [
        item for item in evaluated
        if int(item["positive_collateral_count"]) == 0 and int(item["fp_covered_count"]) > 0
    ]
    full_fp_count = sum(1 for record in records if record["error_type"] == "FP")
    full_fp_coverage = [
        item for item in evaluated
        if int(item["fp_covered_count"]) == full_fp_count
    ]
    best_zero_positive = sorted(zero_positive, key=candidate_sort_key, reverse=True)
    best_overall = sorted(evaluated, key=candidate_sort_key, reverse=True)
    best_full_fp = sorted(
        full_fp_coverage,
        key=lambda item: (
            -int(item["positive_collateral_count"]),
            float(item["fp_precision_vs_positive"] or 0.0),
            -int(item["triggered_count"]),
        ),
        reverse=True,
    )
    return {
        "candidate_count": int(len(evaluated)),
        "zero_positive_candidate_count": int(len(zero_positive)),
        "full_fp_count": int(full_fp_count),
        "full_fp_coverage_candidate_count": int(len(full_fp_coverage)),
        "best_zero_positive": best_zero_positive[0] if best_zero_positive else None,
        "top_20_zero_positive": best_zero_positive[:20],
        "best_overall": best_overall[0] if best_overall else None,
        "top_20_overall": best_overall[:20],
        "best_full_fp_coverage": best_full_fp[0] if best_full_fp else None,
        "top_20_full_fp_coverage": best_full_fp[:20],
    }


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "patient_id": record["patient_id"],
        "seed": record["seed"],
        "label": int(record["label"]),
        "pred": int(record["pred"]),
        "error_type": record["error_type"],
        "fusion_abnormal": round_float(float(record["fusion_abnormal"])),
        "top_weight_view": record["top_weight_view"],
        "features": {
            key: round_float(float(value))
            for key, value in record["features"].items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--duplicate-report", type=Path, default=Path(DEFAULT_DUPLICATE_REPORT))
    parser.add_argument("--clean-remaining-report", type=Path, default=Path(DEFAULT_CLEAN_REMAINING_REPORT))
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
    records = sample_records(repo_root, groups)
    confusion = Counter(record["error_type"] for record in records)
    scan = scan_candidates(records)
    clean_remaining_path = args.clean_remaining_report
    if not clean_remaining_path.is_absolute():
        clean_remaining_path = repo_root / clean_remaining_path
    clean_remaining = load_json(clean_remaining_path)
    report = {
        "analysis": "dfr127_axial_fp_calibration_frontier",
        "description": (
            "Read-only clean validation axial FP calibration frontier.  Scans "
            "simple label-free axial-risk feature rules for separability without "
            "data edits or test metrics."
        ),
        "duplicate_report": str(duplicate_report_path),
        "clean_remaining_report": str(clean_remaining_path),
        "clean_sample_count": int(len(records)),
        "confusion_counts": counter_dict(confusion),
        "feature_distribution_by_error_type": feature_distribution(records),
        "feature_separability_auc": separability(records),
        "candidate_scan": scan,
        "clean_fp_records": [
            compact_record(record)
            for record in records
            if record["error_type"] == "FP"
        ],
        "clean_tp_high_axial_records": [
            compact_record(record)
            for record in records
            if record["error_type"] == "TP" and float(record["features"]["axial_abnormal"]) >= 0.90
        ],
        "dfr126_bucket_counts": clean_remaining.get("bucket_counts"),
        "interpretation": {
            "zero_positive_axial_risk_candidate_exists": bool(scan["best_zero_positive"]),
            "posthoc_threshold_expansion_closed_by_dfr125": True,
            "assessment": (
                "If zero-positive candidates cover only a minority of clean FPs, "
                "label-free axial-risk gating is not enough; any model-side follow-up "
                "should be supervised/view-specific calibration with positive protection."
            ),
        },
    }
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    save_json(output, report)

    print(f"Saved DFR-127 axial FP calibration frontier to: {output}")
    print("clean_sample_count", report["clean_sample_count"], "confusion", report["confusion_counts"])
    print(
        "candidate_count",
        scan["candidate_count"],
        "zero_positive",
        scan["zero_positive_candidate_count"],
        "full_fp_coverage",
        scan["full_fp_coverage_candidate_count"],
    )
    if scan["best_zero_positive"]:
        best = scan["best_zero_positive"]
        print(
            "best_zero_positive",
            best["candidate"],
            "fp_covered",
            best["fp_covered_count"],
            "positive_collateral",
            best["positive_collateral_count"],
            "triggered",
            best["triggered_count"],
        )
    if scan["best_full_fp_coverage"]:
        best = scan["best_full_fp_coverage"]
        print(
            "best_full_fp_coverage",
            best["candidate"],
            "positive_collateral",
            best["positive_collateral_count"],
            "triggered",
            best["triggered_count"],
        )


if __name__ == "__main__":
    main()
