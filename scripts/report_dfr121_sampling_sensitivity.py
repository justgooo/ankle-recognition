#!/usr/bin/env python3
"""Sampling/per-view sensitivity audit for DFR-116 remaining false cases.

This script is read-only.  It inspects the validation false cases that remain
after DFR-116, records the actual 8-slice/trim=2 sampling indices for each
view, and checks for repeated prediction signatures or duplicate NIfTI files.
No test split is read and no data files are modified.
"""

from __future__ import annotations

import argparse
import hashlib
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
    summarize_remaining_errors_after_combo,
)
from scripts.analyze_dfr118_variant_evidence_drift import reference_maps  # noqa: E402
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402
from scripts.report_dfr120_dfr116_false_cases import (  # noqa: E402
    build_historical_fix_index,
    enrich_cases,
)
from src.dataset import (  # noqa: E402
    VIEW_AXIS_BY_COLUMN,
    canonicalize_view_columns,
    is_nifti_path,
    load_nifti_volume,
    sample_slice_indices,
)


DEFAULT_OUTPUT = "autoresearch_logs/dfr121_sampling_sensitivity_audit.json"
SEEDS = ("42", "123", "456")
VIEW_COLUMNS = ("axial_dir", "coronal_dir", "sagittal_dir")
VIEW_COLUMN_TO_NAME = {
    "axial_dir": "axial",
    "coronal_dir": "coronal",
    "sagittal_dir": "sagittal",
}
SAMPLING_GRIDS = (
    ("baseline_8_trim2", 8, 2),
    ("n8_trim0", 8, 0),
    ("n8_trim1", 8, 1),
    ("n16_trim2", 16, 2),
    ("n24_trim2", 24, 2),
    ("n32_trim2", 32, 2),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def profile_metric(volume: np.ndarray, axis: int, metric: str) -> np.ndarray:
    moved = np.moveaxis(volume, axis, 0)
    if metric == "hu300_fraction":
        return np.asarray([(slice_ >= 300.0).mean() for slice_ in moved], dtype=np.float64)
    if metric == "hu500_fraction":
        return np.asarray([(slice_ >= 500.0).mean() for slice_ in moved], dtype=np.float64)
    if metric == "std":
        return np.asarray([slice_.std() for slice_ in moved], dtype=np.float64)
    if metric == "mean":
        return np.asarray([slice_.mean() for slice_ in moved], dtype=np.float64)
    raise ValueError(f"Unknown metric: {metric}")


def top_fraction_indices(profile: np.ndarray, fraction: float) -> set[int]:
    length = int(profile.shape[0])
    count = max(1, int(round(length * fraction)))
    order = np.argsort(profile, kind="mergesort")
    return set(int(index) for index in order[-count:])


def summarize_grid_indices(length: int, num_slices: int, trim: int) -> dict[str, Any]:
    indices = sample_slice_indices(length, num_slices, trim_edge_slices=trim)
    counts = Counter(int(index) for index in indices.tolist())
    return {
        "num_slices": int(num_slices),
        "trim_edge_slices": int(trim),
        "indices": [int(index) for index in indices.tolist()],
        "unique_count": int(len(counts)),
        "duplicate_indices": {
            str(index): int(count)
            for index, count in sorted(counts.items())
            if count > 1
        },
        "normalized_positions": [
            round_float(float(index / max(length - 1, 1))) for index in indices.tolist()
        ],
    }


def summarize_metric_coverage(profile: np.ndarray, sampled_indices: list[int]) -> dict[str, Any]:
    indices = np.asarray(sampled_indices, dtype=np.int64)
    full_max_index = int(np.argmax(profile))
    sampled_values = profile[indices]
    sampled_max_pos = int(np.argmax(sampled_values))
    sampled_max_index = int(indices[sampled_max_pos])
    full_max = float(profile[full_max_index])
    sampled_max = float(sampled_values[sampled_max_pos])
    nearest_distance = int(np.min(np.abs(indices - full_max_index)))
    top10 = top_fraction_indices(profile, 0.10)
    top25 = top_fraction_indices(profile, 0.25)
    return {
        "full_max_index": full_max_index,
        "full_max_value": round_float(full_max),
        "sampled_max_index": sampled_max_index,
        "sampled_max_value": round_float(sampled_max),
        "sampled_to_full_max_ratio": round_float(
            sampled_max / full_max if full_max > 1e-12 else 1.0
        ),
        "nearest_sample_to_full_max_distance": nearest_distance,
        "nearest_sample_to_full_max_distance_norm": round_float(
            nearest_distance / max(int(profile.shape[0]) - 1, 1)
        ),
        "top10_hit_count": int(sum(int(index) in top10 for index in indices)),
        "top25_hit_count": int(sum(int(index) in top25 for index in indices)),
    }


def summarize_view_sampling(volume: np.ndarray, view_column: str) -> dict[str, Any]:
    axis = VIEW_AXIS_BY_COLUMN[view_column]
    length = int(volume.shape[axis])
    grid_summaries = {
        name: summarize_grid_indices(length, num_slices, trim)
        for name, num_slices, trim in SAMPLING_GRIDS
    }
    baseline_indices = grid_summaries["baseline_8_trim2"]["indices"]
    metrics = {}
    for metric in ("hu300_fraction", "hu500_fraction", "std", "mean"):
        profile = profile_metric(volume, axis, metric)
        metrics[metric] = summarize_metric_coverage(profile, baseline_indices)
    flags = []
    if grid_summaries["baseline_8_trim2"]["unique_count"] < 8:
        flags.append("baseline_duplicate_sample_indices")
    if metrics["hu300_fraction"]["top10_hit_count"] == 0:
        flags.append("baseline_misses_hu300_top10")
    if metrics["std"]["top10_hit_count"] == 0:
        flags.append("baseline_misses_std_top10")
    if metrics["hu300_fraction"]["sampled_to_full_max_ratio"] < 0.75:
        flags.append("low_hu300_peak_coverage")
    return {
        "view": VIEW_COLUMN_TO_NAME[view_column],
        "axis": int(axis),
        "length": length,
        "sampling_grids": grid_summaries,
        "baseline_metric_coverage": metrics,
        "flags": flags,
    }


def dfr116_prediction_signature(seed: str, patient_id: str, sample: dict[str, Any]) -> str:
    parts = [
        seed,
        str(int(sample["label"])),
        str(int(sample["fusion_prediction"]["pred"])),
        f"{float(sample['fusion_prediction']['abnormal_prob']):.6f}",
    ]
    for view in VIEWS:
        parts.append(f"{float(sample['views'][view]['abnormal_prob']):.6f}")
        parts.append(f"{float(sample['views'][view]['fusion_weight']):.6f}")
    return "|".join(parts)


def build_case_context(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    dfr25_arrays = {
        seed: telemetry_arrays_model_equivalent(load_json(repo_root / relative_path))
        for seed, relative_path in DFR25_TELEMETRY.items()
    }
    combo_cache = combo_pred_cache(dfr25_arrays)
    references = reference_maps(dfr25_arrays, combo_cache)
    remaining = summarize_remaining_errors_after_combo(dfr25_arrays, combo_cache)
    fix_index = build_historical_fix_index(repo_root, references)
    cases = enrich_cases(repo_root, remaining, fix_index)
    dfr116_samples = {
        seed: {
            str(sample["patient_id"]): sample
            for sample in load_json(repo_root / DFR116_TELEMETRY[seed])["samples"]
        }
        for seed in SEEDS
    }
    return cases, dfr116_samples


def summarize_patient(
    repo_root: Path,
    metadata_row: dict[str, Any],
    case_ids: list[str],
    dfr116_samples: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    patient_id = str(metadata_row["patient_id"])
    base_dir = repo_root
    paths = {
        view_column: resolve_path(base_dir, str(metadata_row[view_column]))
        for view_column in VIEW_COLUMNS
    }
    path_hashes = {}
    view_summaries = {}
    shape_by_path: dict[str, list[int]] = {}
    for view_column, path in paths.items():
        if not path.exists():
            path_hashes[view_column] = {"exists": False, "path": str(path)}
            continue
        path_hashes[view_column] = {
            "exists": True,
            "path": str(path),
            "file_size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if is_nifti_path(path):
            volume = load_nifti_volume(str(path.resolve()))
            shape_by_path[str(path)] = [int(value) for value in volume.shape]
            view_summaries[view_column] = summarize_view_sampling(volume, view_column)
        else:
            view_summaries[view_column] = {
                "view": VIEW_COLUMN_TO_NAME[view_column],
                "unsupported_non_nifti_path": str(path),
            }

    prediction_signatures = {}
    for case_id in case_ids:
        seed, _pid = case_id.split(":", 1)
        sample = dfr116_samples[seed][patient_id]
        prediction_signatures[case_id] = dfr116_prediction_signature(seed, patient_id, sample)

    return {
        "patient_id": patient_id,
        "label": int(metadata_row["label"]),
        "split": str(metadata_row["split"]),
        "case_ids": sorted(case_ids),
        "view_paths": {key: str(path) for key, path in paths.items()},
        "path_hashes": path_hashes,
        "volume_shapes": shape_by_path,
        "all_view_columns_same_path": len(set(str(path) for path in paths.values())) == 1,
        "all_view_hashes_same": len(
            {
                item.get("sha256")
                for item in path_hashes.values()
                if item.get("exists")
            }
        )
        == 1,
        "dfr116_prediction_signatures": prediction_signatures,
        "view_sampling": view_summaries,
    }


def duplicate_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    hash_groups: dict[str, list[str]] = defaultdict(list)
    signature_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        first_hash = next(
            (
                item.get("sha256")
                for item in record["path_hashes"].values()
                if item.get("exists")
            ),
            None,
        )
        if first_hash:
            hash_groups[str(first_hash)].append(str(record["patient_id"]))
        for case_id, signature in record["dfr116_prediction_signatures"].items():
            signature_groups[signature].append(case_id)
    return {
        "nifti_sha256_groups": [
            {"sha256": key, "patient_ids": sorted(values)}
            for key, values in sorted(hash_groups.items())
            if len(values) > 1
        ],
        "dfr116_prediction_signature_groups": [
            {"signature": key, "case_ids": sorted(values)}
            for key, values in sorted(signature_groups.items())
            if len(values) > 1
        ],
    }


def summarize_flags(patient_records: list[dict[str, Any]]) -> dict[str, Any]:
    flags = Counter()
    for record in patient_records:
        for view_summary in record["view_sampling"].values():
            for flag in view_summary.get("flags", []):
                flags.update([f"{view_summary.get('view')}:{flag}"])
    return counter_dict(flags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    cases, dfr116_samples = build_case_context(repo_root)
    df = canonicalize_view_columns(pd.read_csv(repo_root / "data/realdata/metadata.csv"))
    case_ids_by_patient: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        case_ids_by_patient[str(case["patient_id"])].append(str(case["case_id"]))
    target_ids = sorted(case_ids_by_patient)
    target_df = df[df["patient_id"].isin(target_ids)].copy()
    records = [
        summarize_patient(
            repo_root,
            row.to_dict(),
            case_ids_by_patient[str(row["patient_id"])],
            dfr116_samples,
        )
        for _, row in target_df.sort_values("patient_id").iterrows()
    ]
    duplicates = duplicate_groups(records)
    report = {
        "analysis": "dfr121_sampling_sensitivity_audit",
        "description": (
            "Read-only sampling/per-view sensitivity audit for DFR-116 remaining "
            "validation false cases.  Records NIfTI hashes, sampled indices, "
            "coverage proxies, and repeated prediction signatures; no test split."
        ),
        "sampling_config": {
            "baseline_num_slices_per_view": 8,
            "baseline_trim_edge_slices": 2,
            "grids_reported": [
                {"name": name, "num_slices": num_slices, "trim_edge_slices": trim}
                for name, num_slices, trim in SAMPLING_GRIDS
            ],
        },
        "target_case_count": len(cases),
        "target_patient_count": len(records),
        "target_patients": records,
        "duplicate_audit": duplicates,
        "sampling_flag_counts": summarize_flags(records),
        "interpretation": {
            "no_safe_gate_or_training_mechanism_from_dfr120": True,
            "repeated_fn_patient_ids": sorted(
                record["patient_id"]
                for record in records
                if int(record["label"]) == 1 and len(record["case_ids"]) >= 2
            ),
            "repeated_fp_patient_ids": sorted(
                record["patient_id"]
                for record in records
                if int(record["label"]) == 0 and len(record["case_ids"]) >= 2
            ),
        },
    }
    save_json(repo_root / args.output, report)

    print(f"Saved DFR-121 sampling sensitivity audit to: {repo_root / args.output}")
    print("cases", len(cases), "patients", len(records))
    print("duplicate_hash_groups", len(duplicates["nifti_sha256_groups"]))
    print("duplicate_prediction_groups", len(duplicates["dfr116_prediction_signature_groups"]))
    print("sampling_flags", report["sampling_flag_counts"])


if __name__ == "__main__":
    main()
