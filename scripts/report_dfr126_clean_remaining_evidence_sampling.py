#!/usr/bin/env python3
"""Targeted evidence/sampling audit for DFR-126 clean remaining cases.

This read-only report inspects the seven patients that remain wrong after the
DFR-116 combo under the DFR-124 combined-clean validation policy.  It records
DFR-116 per-seed evidence, NIfTI metadata, and baseline sampling coverage
proxies without reading test metrics or modifying data files.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

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
from scripts.report_dfr121_sampling_sensitivity import (  # noqa: E402
    VIEW_COLUMNS,
    resolve_path,
    sha256_file,
    summarize_view_sampling,
)
from src.dataset import canonicalize_view_columns, is_nifti_path, load_nifti_volume  # noqa: E402


DEFAULT_CLEAN_FRONTIER_REPORT = "autoresearch_logs/dfr125_clean_frontier_false_case_audit.json"
DEFAULT_DUPLICATE_REPORT = "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json"
DEFAULT_METADATA = "data/realdata/metadata.csv"
DEFAULT_OUTPUT = "autoresearch_logs/dfr126_clean_remaining_evidence_sampling.json"
SEEDS = ("42", "123", "456")


def dfr116_sample_maps(repo_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        seed: {
            str(sample["patient_id"]): sample
            for sample in load_json(repo_root / DFR116_TELEMETRY[seed])["samples"]
        }
        for seed in SEEDS
    }


def top_weight_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def view_evidence(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        view: {
            "abnormal": round_float(float(sample["views"][view]["abnormal_prob"])),
            "fusion_weight": round_float(float(sample["views"][view]["fusion_weight"])),
            "confidence_logit": round_float(float(sample["views"][view]["confidence_logit"])),
            "scaled_confidence_logit": round_float(
                float(
                    sample["views"][view].get(
                        "scaled_confidence_logit",
                        sample["views"][view]["confidence_logit"],
                    )
                )
            ),
        }
        for view in VIEWS
    }


def sample_prediction_record(
    sample: dict[str, Any],
    case_ids: list[str],
    error_cases: dict[str, dict[str, Any]],
    seed: str,
) -> dict[str, Any]:
    patient_id = str(sample["patient_id"])
    case_id = f"{seed}:{patient_id}"
    fusion = sample["fusion_prediction"]
    return {
        "seed": seed,
        "case_id": case_id,
        "is_remaining_error": case_id in error_cases,
        "remaining_case": error_cases.get(case_id),
        "label": int(sample["label"]),
        "pred": int(fusion["pred"]),
        "correct": bool(int(fusion["pred"]) == int(sample["label"])),
        "fusion_abnormal": round_float(float(fusion["abnormal_prob"])),
        "top_weight_view": top_weight_view(sample),
        "views": view_evidence(sample),
    }


def duplicate_annotations(
    duplicate_report: dict[str, Any],
    patient_id: str,
) -> list[dict[str, Any]]:
    annotations = []
    for group in duplicate_report.get("duplicate_content_groups", []):
        members = []
        contains = False
        for member in group.get("members", []):
            for patient in member.get("patients", []):
                pid = str(patient["patient_id"])
                contains = contains or pid == patient_id
                members.append(
                    {
                        "patient_id": pid,
                        "split": str(patient["split"]),
                        "label": int(patient["label"]),
                        "path": str(member["path"]),
                    }
                )
        if contains:
            annotations.append(
                {
                    "sha256": str(group["sha256"]),
                    "cross_split": bool(group["cross_split"]),
                    "cross_label": bool(group["cross_label"]),
                    "split_counts": group["split_counts"],
                    "label_counts": group["label_counts"],
                    "members": sorted(members, key=lambda item: item["patient_id"]),
                }
            )
    return annotations


def summarize_patient_sampling(
    repo_root: Path,
    row: pd.Series,
) -> dict[str, Any]:
    paths = {
        view_column: resolve_path(repo_root, str(row[view_column]))
        for view_column in VIEW_COLUMNS
    }
    path_hashes = {}
    volume_shapes = {}
    view_sampling = {}
    for view_column, path in paths.items():
        path_hashes[view_column] = {
            "path": str(path),
            "exists": bool(path.exists()),
        }
        if not path.exists():
            continue
        path_hashes[view_column]["file_size_bytes"] = int(path.stat().st_size)
        path_hashes[view_column]["sha256"] = sha256_file(path)
        if is_nifti_path(path):
            volume = load_nifti_volume(str(path.resolve()))
            volume_shapes[str(path)] = [int(value) for value in volume.shape]
            view_sampling[view_column] = summarize_view_sampling(volume, view_column)
    return {
        "view_paths": {key: str(path) for key, path in paths.items()},
        "path_hashes": path_hashes,
        "volume_shapes": volume_shapes,
        "all_view_columns_same_path": len(set(str(path) for path in paths.values())) == 1,
        "all_view_hashes_same": len(
            {
                item.get("sha256")
                for item in path_hashes.values()
                if item.get("exists")
            }
        )
        == 1,
        "view_sampling": view_sampling,
    }


def patient_level_interpretation(
    label: int,
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    sampling_flags: list[str],
    duplicate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    error_types = Counter(str(case["error_type"]) for case in cases)
    evidence_buckets = Counter(str(case["evidence_bucket"]) for case in cases)
    abnormal_by_view = {
        view: [
            float(pred["views"][view]["abnormal"])
            for pred in predictions
            if pred["is_remaining_error"]
        ]
        for view in VIEWS
    }
    repeated_error = len(cases) >= 2
    flags = set(sampling_flags)

    suggested_bucket = "manual_review"
    rationale = []
    if duplicate_groups:
        suggested_bucket = "duplicate_family_or_split_review"
        rationale.append("patient belongs to duplicate content group")
    if label == 1:
        max_error_view = max(
            (max(values) if values else 0.0)
            for values in abnormal_by_view.values()
        )
        if max_error_view < 0.45:
            suggested_bucket = "positive_no_view_reaches_weak_abnormal"
            rationale.append("all error-seed view abnormal probabilities stay below 0.45")
        elif repeated_error:
            suggested_bucket = "positive_recurrent_subthreshold_evidence"
            rationale.append("positive case recurs across seeds with only subthreshold/weak support")
        if any("low_hu300_peak_coverage" in flag for flag in flags):
            rationale.append("baseline 8-slice sampling has low HU300 peak coverage in at least one view")
        if any("baseline_misses_std_top10" in flag for flag in flags):
            rationale.append("baseline 8-slice sampling misses std top10 in at least one view")
    else:
        axial_values = abnormal_by_view["axial"]
        if axial_values and min(axial_values) >= 0.90:
            suggested_bucket = "negative_strong_axial_classifier_fp"
            rationale.append("axial abnormal probability is >=0.90 for every remaining-error seed")
        elif axial_values and max(axial_values) >= 0.88:
            suggested_bucket = "negative_axial_classifier_fp"
            rationale.append("axial abnormal probability exceeds DFR113 safety cap")
        if any(
            float(pred["views"]["sagittal"]["abnormal"]) <= 0.10
            for pred in predictions
            if pred["is_remaining_error"]
        ):
            rationale.append("at least one error seed has very normal sagittal evidence")

    return {
        "suggested_bucket": suggested_bucket,
        "rationale": rationale,
        "error_type_counts": counter_dict(error_types),
        "evidence_bucket_counts": counter_dict(evidence_buckets),
        "sampling_flags": sorted(flags),
        "error_seed_view_abnormal_ranges": {
            view: {
                "min": round_float(min(values)) if values else None,
                "max": round_float(max(values)) if values else None,
                "mean": round_float(sum(values) / len(values)) if values else None,
            }
            for view, values in abnormal_by_view.items()
        },
    }


def summarize_flags(records: list[dict[str, Any]]) -> dict[str, Any]:
    flags = Counter()
    for record in records:
        for flag in record["interpretation"]["sampling_flags"]:
            flags.update([flag])
    return counter_dict(flags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--metadata", type=Path, default=Path(DEFAULT_METADATA))
    parser.add_argument("--clean-frontier-report", type=Path, default=Path(DEFAULT_CLEAN_FRONTIER_REPORT))
    parser.add_argument("--duplicate-report", type=Path, default=Path(DEFAULT_DUPLICATE_REPORT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    metadata_path = args.metadata if args.metadata.is_absolute() else repo_root / args.metadata
    clean_report_path = (
        args.clean_frontier_report
        if args.clean_frontier_report.is_absolute()
        else repo_root / args.clean_frontier_report
    )
    duplicate_report_path = (
        args.duplicate_report
        if args.duplicate_report.is_absolute()
        else repo_root / args.duplicate_report
    )

    df = canonicalize_view_columns(pd.read_csv(metadata_path))
    clean_report = load_json(clean_report_path)
    duplicate_report = load_json(duplicate_report_path)
    dfr116_maps = dfr116_sample_maps(repo_root)

    cases_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    error_cases = {}
    for case in clean_report["remaining_errors_after_combo_clean"]:
        patient_id = str(case["patient_id"])
        cases_by_patient[patient_id].append(case)
        error_cases[str(case["case_id"])] = case

    patient_records = []
    for patient_id in sorted(cases_by_patient):
        matches = df[df["patient_id"].astype(str) == patient_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one metadata row for {patient_id}, found {len(matches)}")
        row = matches.iloc[0]
        cases = sorted(cases_by_patient[patient_id], key=lambda item: str(item["case_id"]))
        predictions = [
            sample_prediction_record(
                dfr116_maps[seed][patient_id],
                [str(case["case_id"]) for case in cases],
                error_cases,
                seed,
            )
            for seed in SEEDS
        ]
        sampling = summarize_patient_sampling(repo_root, row)
        duplicate_groups = duplicate_annotations(duplicate_report, patient_id)
        sampling_flags = []
        for view_column, view_summary in sampling["view_sampling"].items():
            for flag in view_summary.get("flags", []):
                sampling_flags.append(f"{view_summary.get('view')}:{flag}")
        interpretation = patient_level_interpretation(
            int(row["label"]),
            cases,
            predictions,
            sampling_flags,
            duplicate_groups,
        )
        patient_records.append(
            {
                "patient_id": patient_id,
                "label": int(row["label"]),
                "split": str(row["split"]),
                "case_ids": [str(case["case_id"]) for case in cases],
                "cases": cases,
                "dfr116_predictions": predictions,
                "duplicate_groups": duplicate_groups,
                **sampling,
                "interpretation": interpretation,
            }
        )

    bucket_counts = Counter(record["interpretation"]["suggested_bucket"] for record in patient_records)
    report = {
        "analysis": "dfr126_clean_remaining_evidence_sampling",
        "description": (
            "Read-only targeted evidence/sampling audit for DFR-116 combined-clean remaining "
            "cases.  No test metrics and no data edits."
        ),
        "source_clean_frontier_report": str(clean_report_path),
        "source_duplicate_report": str(duplicate_report_path),
        "target_patient_count": int(len(patient_records)),
        "target_case_count": int(sum(len(record["cases"]) for record in patient_records)),
        "bucket_counts": counter_dict(bucket_counts),
        "sampling_flag_counts": summarize_flags(patient_records),
        "patients": patient_records,
        "interpretation": {
            "posthoc_threshold_expansion_closed_by_dfr125": True,
            "suggested_next_scope": (
                "Do not expand eval-side thresholds.  If continuing model-side work, focus on "
                "view-specific classifier calibration or sampling/annotation review for the "
                "identified clean remaining buckets."
            ),
        },
    }
    output = args.output if args.output.is_absolute() else repo_root / args.output
    save_json(output, report)

    print(f"Saved DFR-126 clean remaining evidence/sampling audit to: {output}")
    print("patients", report["target_patient_count"], "cases", report["target_case_count"])
    print("bucket_counts", report["bucket_counts"])
    print("sampling_flags", report["sampling_flag_counts"])


if __name__ == "__main__":
    main()
