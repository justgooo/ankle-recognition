#!/usr/bin/env python3
"""Clean-validation sampling variable audit for DFR-131.

This read-only report extends the DFR-126 targeted sampling audit from the
seven remaining DFR-116 clean-error patients to the whole combined-clean
validation set.  It compares baseline 8-slice/trim=2 coverage proxies against
candidate runtime sampling grids and summarizes whether any configurable
sampling variable looks worth a follow-up formal run.  No test metrics are read
and no data files are modified.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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
    SAMPLING_GRIDS,
    VIEW_COLUMNS,
    VIEW_COLUMN_TO_NAME,
    resolve_path,
    summarize_grid_indices,
    summarize_metric_coverage,
)
from scripts.report_dfr124_duplicate_leakage_metric_sensitivity import (  # noqa: E402
    exclude_patients_mask,
    keep_first_val_val_mask,
    validation_patient_groups,
)
from src.dataset import (  # noqa: E402
    VIEW_AXIS_BY_COLUMN,
    canonicalize_view_columns,
    is_nifti_path,
    load_nifti_volume,
)


DEFAULT_METADATA = "data/realdata/metadata.csv"
DEFAULT_DUPLICATE_REPORT = "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json"
DEFAULT_DFR126_REPORT = "autoresearch_logs/dfr126_clean_remaining_evidence_sampling.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr131_sampling_variable_audit.json"
SEEDS = ("42", "123", "456")
BASELINE_GRID = "baseline_8_trim2"
METRICS = ("hu300_fraction", "hu500_fraction", "std", "mean")


def dfr116_sample_maps(repo_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        seed: {
            str(sample["patient_id"]): sample
            for sample in load_json(repo_root / DFR116_TELEMETRY[seed])["samples"]
        }
        for seed in SEEDS
    }


def dfr116_sample_lists(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        seed: load_json(repo_root / DFR116_TELEMETRY[seed])["samples"]
        for seed in SEEDS
    }


def clean_patient_ids_from_telemetry(
    samples: list[dict[str, Any]],
    duplicate_groups: dict[str, Any],
) -> list[str]:
    patient_ids = np.asarray([str(sample["patient_id"]) for sample in samples], dtype=object)
    mask = keep_first_val_val_mask(patient_ids, duplicate_groups["val_val_groups"])
    mask &= exclude_patients_mask(patient_ids, duplicate_groups["train_val_validation_patients"])
    return [str(patient_id) for patient_id in patient_ids[mask]]


def dfr126_remaining_cases(report: dict[str, Any]) -> dict[str, Any]:
    case_by_id = {}
    cases_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_by_patient = {}
    for patient in report.get("patients", []):
        patient_id = str(patient["patient_id"])
        bucket_by_patient[patient_id] = str(patient["interpretation"]["suggested_bucket"])
        for case in patient.get("cases", []):
            case_id = str(case["case_id"])
            case_by_id[case_id] = case
            cases_by_patient[patient_id].append(case)
    return {
        "case_by_id": case_by_id,
        "cases_by_patient": {
            patient_id: sorted(items, key=lambda item: str(item["case_id"]))
            for patient_id, items in cases_by_patient.items()
        },
        "bucket_by_patient": bucket_by_patient,
    }


def grid_flags(grid: dict[str, Any], coverage: dict[str, dict[str, Any]]) -> list[str]:
    flags = []
    if int(grid["unique_count"]) < int(grid["num_slices"]):
        flags.append("duplicate_sample_indices")
    if int(coverage["hu300_fraction"]["top10_hit_count"]) == 0:
        flags.append("misses_hu300_top10")
    if int(coverage["std"]["top10_hit_count"]) == 0:
        flags.append("misses_std_top10")
    if float(coverage["hu300_fraction"]["sampled_to_full_max_ratio"]) < 0.75:
        flags.append("low_hu300_peak_coverage")
    return flags


def metric_profiles(volume: np.ndarray, axis: int) -> dict[str, np.ndarray]:
    reduce_axes = tuple(index for index in range(volume.ndim) if index != axis)
    return {
        "hu300_fraction": np.asarray((volume >= 300.0).mean(axis=reduce_axes), dtype=np.float64),
        "hu500_fraction": np.asarray((volume >= 500.0).mean(axis=reduce_axes), dtype=np.float64),
        "std": np.asarray(volume.std(axis=reduce_axes), dtype=np.float64),
        "mean": np.asarray(volume.mean(axis=reduce_axes), dtype=np.float64),
    }


def summarize_view_sampling_extended(volume: np.ndarray, view_column: str) -> dict[str, Any]:
    axis = VIEW_AXIS_BY_COLUMN[view_column]
    length = int(volume.shape[axis])
    grids = {
        name: summarize_grid_indices(length, num_slices, trim)
        for name, num_slices, trim in SAMPLING_GRIDS
    }
    profiles = metric_profiles(volume, axis)
    coverage_by_grid = {}
    for grid_name, grid in grids.items():
        indices = [int(index) for index in grid["indices"]]
        coverage_by_grid[grid_name] = {
            metric: summarize_metric_coverage(profile, indices)
            for metric, profile in profiles.items()
        }

    baseline_flags = [
        f"baseline_{flag}"
        for flag in grid_flags(grids[BASELINE_GRID], coverage_by_grid[BASELINE_GRID])
    ]
    candidate_deltas = {}
    baseline_coverage = coverage_by_grid[BASELINE_GRID]
    for grid_name, coverage in coverage_by_grid.items():
        if grid_name == BASELINE_GRID:
            continue
        delta = {}
        for metric in ("hu300_fraction", "hu500_fraction", "std", "mean"):
            base_metric = baseline_coverage[metric]
            candidate_metric = coverage[metric]
            delta[metric] = {
                "sampled_to_full_max_ratio_delta": round_float(
                    float(candidate_metric["sampled_to_full_max_ratio"])
                    - float(base_metric["sampled_to_full_max_ratio"])
                ),
                "top10_hit_count_delta": int(candidate_metric["top10_hit_count"])
                - int(base_metric["top10_hit_count"]),
                "nearest_sample_to_full_max_distance_delta": int(
                    candidate_metric["nearest_sample_to_full_max_distance"]
                )
                - int(base_metric["nearest_sample_to_full_max_distance"]),
            }
        candidate_flags = grid_flags(grids[grid_name], coverage)
        candidate_deltas[grid_name] = {
            "metric_deltas": delta,
            "candidate_flags": candidate_flags,
            "resolved_flags": sorted(
                flag for flag in baseline_flags if flag.replace("baseline_", "") not in candidate_flags
            ),
            "created_flags": sorted(
                f"baseline_{flag}"
                for flag in candidate_flags
                if f"baseline_{flag}" not in baseline_flags
            ),
        }

    return {
        "view": VIEW_COLUMN_TO_NAME[view_column],
        "axis": int(axis),
        "length": length,
        "baseline_sampling_grid": grids[BASELINE_GRID],
        "baseline_metric_coverage": baseline_coverage,
        "baseline_flags": baseline_flags,
        "candidate_deltas": candidate_deltas,
    }


def summarize_patient_sampling(
    repo_root: Path,
    row: pd.Series,
) -> dict[str, Any]:
    paths = {
        view_column: resolve_path(repo_root, str(row[view_column]))
        for view_column in VIEW_COLUMNS
    }
    view_sampling = {}
    path_status = {}
    local_volume_cache: dict[str, np.ndarray] = {}
    for view_column, path in paths.items():
        path_status[view_column] = {"path": str(path), "exists": bool(path.exists())}
        if not path.exists():
            continue
        if not is_nifti_path(path):
            path_status[view_column]["is_nifti"] = False
            continue
        path_status[view_column]["is_nifti"] = True
        cache_key = str(path.resolve())
        if cache_key not in local_volume_cache:
            local_volume_cache[cache_key] = load_nifti_volume(cache_key)
        view_sampling[view_column] = summarize_view_sampling_extended(
            local_volume_cache[cache_key],
            view_column,
        )
    return {
        "view_paths": {key: str(path) for key, path in paths.items()},
        "path_status": path_status,
        "all_view_columns_same_path": len(set(str(path) for path in paths.values())) == 1,
        "view_sampling": view_sampling,
    }


def top_weight_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def prediction_record(sample: dict[str, Any], seed: str, remaining_case: dict[str, Any] | None) -> dict[str, Any]:
    fusion = sample["fusion_prediction"]
    return {
        "seed": seed,
        "case_id": f"{seed}:{sample['patient_id']}",
        "label": int(sample["label"]),
        "pred": int(fusion["pred"]),
        "correct": bool(int(fusion["pred"]) == int(sample["label"])),
        "fusion_abnormal": round_float(float(fusion["abnormal_prob"])),
        "top_weight_view": top_weight_view(sample),
        "is_remaining_error": remaining_case is not None,
        "remaining_bucket": remaining_case.get("evidence_bucket") if remaining_case else None,
        "views": {
            view: {
                "abnormal": round_float(float(sample["views"][view]["abnormal_prob"])),
                "fusion_weight": round_float(float(sample["views"][view]["fusion_weight"])),
            }
            for view in VIEWS
        },
    }


def patient_group_name(record: dict[str, Any]) -> str:
    if record["remaining_error_case_count"] > 0:
        return "positive_remaining" if int(record["label"]) == 1 else "negative_remaining"
    return "positive_clean_correct" if int(record["label"]) == 1 else "negative_clean_correct"


def grouped_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "all_clean": list(records),
        "remaining_error": [record for record in records if record["remaining_error_case_count"] > 0],
        "clean_correct": [record for record in records if record["remaining_error_case_count"] == 0],
        "positive_remaining": [],
        "negative_remaining": [],
        "positive_clean_correct": [],
        "negative_clean_correct": [],
    }
    for record in records:
        groups[patient_group_name(record)].append(record)
    return groups


def baseline_flag_counts(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for group_name, records in groups.items():
        counts = Counter()
        for record in records:
            for view_summary in record["sampling"]["view_sampling"].values():
                view = view_summary["view"]
                for flag in view_summary["baseline_flags"]:
                    counts.update([f"{view}:{flag}"])
        denominator = max(len(records), 1)
        summary[group_name] = {
            "patient_count": int(len(records)),
            "counts": counter_dict(counts),
            "rates": {
                key: round_float(float(value) / denominator)
                for key, value in sorted(counts.items())
            },
        }
    return summary


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def candidate_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for grid_name, _num_slices, _trim in SAMPLING_GRIDS:
        if grid_name == BASELINE_GRID:
            continue
        summary[grid_name] = {}
        for group_name, records in groups.items():
            group_summary = {}
            for view_column in VIEW_COLUMNS:
                view = VIEW_COLUMN_TO_NAME[view_column]
                resolved = Counter()
                created = Counter()
                deltas: dict[str, list[float]] = defaultdict(list)
                present = 0
                for record in records:
                    view_summary = record["sampling"]["view_sampling"].get(view_column)
                    if not view_summary:
                        continue
                    present += 1
                    item = view_summary["candidate_deltas"][grid_name]
                    for flag in item["resolved_flags"]:
                        resolved.update([flag])
                    for flag in item["created_flags"]:
                        created.update([flag])
                    for metric, metric_delta in item["metric_deltas"].items():
                        deltas[f"{metric}:ratio_delta"].append(
                            float(metric_delta["sampled_to_full_max_ratio_delta"])
                        )
                        deltas[f"{metric}:top10_hit_delta"].append(
                            float(metric_delta["top10_hit_count_delta"])
                        )
                        deltas[f"{metric}:nearest_distance_delta"].append(
                            float(metric_delta["nearest_sample_to_full_max_distance_delta"])
                        )
                group_summary[view] = {
                    "patient_count": int(present),
                    "resolved_flag_counts": counter_dict(resolved),
                    "created_flag_counts": counter_dict(created),
                    "mean_metric_deltas": {
                        key: round_float(mean(values))
                        for key, values in sorted(deltas.items())
                    },
                }
            summary[grid_name][group_name] = group_summary
    return summary


def candidate_rankings(candidate_summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for grid_name, grid_summary in candidate_summaries.items():
        pos_coronal = grid_summary["positive_remaining"]["coronal"]
        rem_coronal = grid_summary["remaining_error"]["coronal"]
        clean_coronal = grid_summary["clean_correct"]["coronal"]
        pos_resolved = pos_coronal["resolved_flag_counts"]
        rem_resolved = rem_coronal["resolved_flag_counts"]
        clean_created = clean_coronal["created_flag_counts"]
        score = (
            3.0 * pos_resolved.get("baseline_low_hu300_peak_coverage", 0)
            + 2.0 * pos_resolved.get("baseline_misses_std_top10", 0)
            + 2.0 * pos_resolved.get("baseline_misses_hu300_top10", 0)
            + 0.5 * sum(rem_resolved.values())
            - 0.25 * sum(clean_created.values())
        )
        rows.append(
            {
                "grid": grid_name,
                "score": round_float(score),
                "positive_remaining_coronal_resolved": pos_resolved,
                "remaining_error_coronal_resolved": rem_resolved,
                "clean_correct_coronal_created": clean_created,
                "positive_remaining_coronal_mean_deltas": pos_coronal["mean_metric_deltas"],
            }
        )
    return sorted(rows, key=lambda item: (-float(item["score"]), str(item["grid"])))


def group_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = grouped_records(records)
    bucket_counts = Counter()
    error_case_count = 0
    for record in records:
        bucket = record["dfr126_bucket"]
        if bucket:
            bucket_counts.update([bucket])
        error_case_count += int(record["remaining_error_case_count"])
    return {
        "clean_patient_count": int(len(records)),
        "clean_case_count_across_seeds": int(len(records) * len(SEEDS)),
        "remaining_error_patient_count": int(len(groups["remaining_error"])),
        "remaining_error_case_count": int(error_case_count),
        "clean_correct_patient_count": int(len(groups["clean_correct"])),
        "positive_remaining_patient_count": int(len(groups["positive_remaining"])),
        "negative_remaining_patient_count": int(len(groups["negative_remaining"])),
        "positive_clean_correct_patient_count": int(len(groups["positive_clean_correct"])),
        "negative_clean_correct_patient_count": int(len(groups["negative_clean_correct"])),
        "dfr126_bucket_patient_counts": counter_dict(bucket_counts),
    }


def assessment(rankings: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    best = rankings[0] if rankings else None
    positive_remaining = [
        record for record in records
        if record["remaining_error_case_count"] > 0 and int(record["label"]) == 1
    ]
    negative_remaining = [
        record for record in records
        if record["remaining_error_case_count"] > 0 and int(record["label"]) == 0
    ]
    return {
        "best_positive_coronal_sampling_grid": best,
        "positive_remaining_patient_ids": [record["patient_id"] for record in positive_remaining],
        "negative_remaining_patient_ids": [record["patient_id"] for record in negative_remaining],
        "train_py_expressible_without_dataset_code_change": {
            "global_num_slices_per_view": True,
            "global_trim_edge_slices": True,
            "view_specific_num_slices_or_trim": False,
        },
        "suggested_next_scope": (
            "Use this report to decide whether a single global sampling-variable formal probe "
            "is justified.  View-specific coronal sampling is not expressible without changing "
            "src/dataset.py, which is outside the allowed edit scope."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--metadata", type=Path, default=Path(DEFAULT_METADATA))
    parser.add_argument("--duplicate-report", type=Path, default=Path(DEFAULT_DUPLICATE_REPORT))
    parser.add_argument("--dfr126-report", type=Path, default=Path(DEFAULT_DFR126_REPORT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    metadata_path = args.metadata if args.metadata.is_absolute() else repo_root / args.metadata
    duplicate_report_path = (
        args.duplicate_report if args.duplicate_report.is_absolute() else repo_root / args.duplicate_report
    )
    dfr126_report_path = (
        args.dfr126_report if args.dfr126_report.is_absolute() else repo_root / args.dfr126_report
    )

    duplicate_report = load_json(duplicate_report_path)
    dfr126 = dfr126_remaining_cases(load_json(dfr126_report_path))
    duplicate_groups = validation_patient_groups(duplicate_report)
    sample_lists = dfr116_sample_lists(repo_root)
    sample_maps = {
        seed: {str(sample["patient_id"]): sample for sample in samples}
        for seed, samples in sample_lists.items()
    }
    clean_ids = clean_patient_ids_from_telemetry(sample_lists["42"], duplicate_groups)
    clean_id_set = set(clean_ids)

    metadata = canonicalize_view_columns(pd.read_csv(metadata_path))
    metadata = metadata[metadata["patient_id"].astype(str).isin(clean_id_set)].copy()
    metadata = metadata.sort_values("patient_id")
    if args.max_patients is None and len(metadata) != len(clean_ids):
        raise ValueError(f"Expected {len(clean_ids)} clean metadata rows, found {len(metadata)}")
    if args.max_patients is not None:
        metadata = metadata.head(int(args.max_patients))

    records = []
    total_rows = int(len(metadata))
    for position, (_, row) in enumerate(metadata.iterrows(), start=1):
        patient_id = str(row["patient_id"])
        if args.progress_every > 0 and (position == 1 or position % args.progress_every == 0):
            print(f"processing patient {position}/{total_rows}: {patient_id}", flush=True)
        remaining_cases = dfr126["cases_by_patient"].get(patient_id, [])
        predictions = [
            prediction_record(
                sample_maps[seed][patient_id],
                seed,
                dfr126["case_by_id"].get(f"{seed}:{patient_id}"),
            )
            for seed in SEEDS
        ]
        records.append(
            {
                "patient_id": patient_id,
                "label": int(row["label"]),
                "split": str(row["split"]),
                "remaining_error_case_count": int(len(remaining_cases)),
                "remaining_case_ids": [str(case["case_id"]) for case in remaining_cases],
                "dfr126_bucket": dfr126["bucket_by_patient"].get(patient_id),
                "dfr116_predictions": predictions,
                "sampling": summarize_patient_sampling(repo_root, row),
            }
        )

    groups = grouped_records(records)
    candidates = candidate_summary(groups)
    rankings = candidate_rankings(candidates)
    report = {
        "analysis": "dfr131_sampling_variable_audit",
        "description": (
            "Read-only whole combined-clean validation sampling audit for DFR-116/DFR-126. "
            "Compares baseline 8-slice trim2 sampling against runtime sampling grids; no "
            "test metrics and no data edits."
        ),
        "source_duplicate_report": str(duplicate_report_path),
        "source_dfr126_report": str(dfr126_report_path),
        "sampling_grids": [
            {"name": name, "num_slices": num_slices, "trim_edge_slices": trim}
            for name, num_slices, trim in SAMPLING_GRIDS
        ],
        "group_counts": group_counts(records),
        "baseline_flag_summary": baseline_flag_counts(groups),
        "candidate_sampling_summary": candidates,
        "candidate_rankings": rankings,
        "assessment": assessment(rankings, records),
        "patients": records,
    }
    output = args.output if args.output.is_absolute() else repo_root / args.output
    save_json(output, report)

    print(f"Saved DFR-131 sampling variable audit to: {output}")
    print("group_counts", report["group_counts"])
    print("top_candidate", rankings[0] if rankings else None)
    print("baseline_remaining_flags", report["baseline_flag_summary"]["remaining_error"]["counts"])


if __name__ == "__main__":
    main()
