#!/usr/bin/env python3
"""Duplicate/leakage-aware validation metric sensitivity for DFR-124.

This read-only report uses DFR-123 duplicate groups to recompute validation
metrics for DFR-25 and DFR-116 under diagnostic duplicate/leakage policies.  It
does not change the validation split and does not introduce a model-selection
metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_dfr111_confirmed_target_posthoc import (  # noqa: E402
    load_json,
    round_float,
    save_json,
)
from scripts.analyze_dfr112_multiseed_posthoc import DFR25_TELEMETRY  # noqa: E402
from scripts.report_dfr122_duplicate_metric_sensitivity import (  # noqa: E402
    auc_from_scores,
    delta_metrics,
    metrics_from_scores,
)
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402


DEFAULT_DUPLICATE_REPORT = "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr124_duplicate_leakage_metric_sensitivity.json"
SEEDS = ("42", "123", "456")
VARIANTS = {
    "dfr25": DFR25_TELEMETRY,
    "dfr116": DFR116_TELEMETRY,
}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def arrays_from_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    samples = telemetry["samples"]
    return {
        "patient_ids": np.asarray([str(sample["patient_id"]) for sample in samples], dtype=object),
        "labels": np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64),
        "abnormal": np.asarray(
            [float(sample["fusion_prediction"]["abnormal_prob"]) for sample in samples],
            dtype=np.float64,
        ),
        "preds": np.asarray(
            [int(sample["fusion_prediction"]["pred"]) for sample in samples],
            dtype=np.int64,
        ),
    }


def validation_patient_groups(duplicate_report: dict[str, Any]) -> dict[str, Any]:
    val_val_groups = []
    train_val_groups = []
    all_validation_duplicate_patients = set()
    for group in duplicate_report["duplicate_content_groups"]:
        split_counts = group["split_counts"]
        members_by_split: dict[str, list[str]] = {}
        for member in group["members"]:
            for patient in member["patients"]:
                split = str(patient["split"])
                patient_id = str(patient["patient_id"])
                members_by_split.setdefault(split, []).append(patient_id)
                if split == "val":
                    all_validation_duplicate_patients.add(patient_id)

        validation_patients = sorted(set(members_by_split.get("val", [])))
        if split_counts == {"val": len(validation_patients)} and len(validation_patients) > 1:
            val_val_groups.append(validation_patients)
        elif validation_patients and "train" in members_by_split:
            train_val_groups.append(
                {
                    "validation_patients": validation_patients,
                    "train_patients": sorted(set(members_by_split.get("train", []))),
                    "sha256": group["sha256"],
                }
            )

    return {
        "val_val_groups": val_val_groups,
        "train_val_groups": train_val_groups,
        "train_val_validation_patients": sorted(
            {
                patient_id
                for group in train_val_groups
                for patient_id in group["validation_patients"]
            }
        ),
        "all_validation_duplicate_patients": sorted(all_validation_duplicate_patients),
    }


def weights_for_val_val(patient_ids: np.ndarray, groups: list[list[str]]) -> np.ndarray:
    weights = np.ones(patient_ids.shape[0], dtype=np.float64)
    id_set = set(patient_ids.tolist())
    for group in groups:
        present = [patient_id for patient_id in group if patient_id in id_set]
        if not present:
            continue
        weight = 1.0 / len(present)
        for patient_id in present:
            weights[patient_ids == patient_id] = weight
    return weights


def keep_first_val_val_mask(patient_ids: np.ndarray, groups: list[list[str]]) -> np.ndarray:
    mask = np.ones(patient_ids.shape[0], dtype=bool)
    id_set = set(patient_ids.tolist())
    for group in groups:
        present = [patient_id for patient_id in group if patient_id in id_set]
        for patient_id in present[1:]:
            mask[patient_ids == patient_id] = False
    return mask


def exclude_patients_mask(patient_ids: np.ndarray, excluded: list[str]) -> np.ndarray:
    excluded_set = set(excluded)
    return np.asarray([patient_id not in excluded_set for patient_id in patient_ids], dtype=bool)


def metrics_for_mask(arrays: dict[str, Any], mask: np.ndarray) -> dict[str, float]:
    return metrics_from_scores(arrays["labels"][mask], arrays["abnormal"][mask])


def metrics_for_weights(arrays: dict[str, Any], weights: np.ndarray) -> dict[str, float]:
    return metrics_from_scores(arrays["labels"], arrays["abnormal"], weights)


def patient_records(arrays: dict[str, Any], patient_ids: list[str]) -> list[dict[str, Any]]:
    records = []
    for patient_id in patient_ids:
        matches = np.flatnonzero(arrays["patient_ids"] == patient_id)
        if matches.size == 0:
            records.append({"patient_id": patient_id, "present": False})
            continue
        index = int(matches[0])
        records.append(
            {
                "patient_id": patient_id,
                "present": True,
                "label": int(arrays["labels"][index]),
                "pred": int(arrays["preds"][index]),
                "abnormal": round_float(float(arrays["abnormal"][index])),
                "correct": bool(int(arrays["labels"][index]) == int(arrays["preds"][index])),
            }
        )
    return records


def evaluate_seed(path: Path, groups: dict[str, Any]) -> dict[str, Any]:
    arrays = arrays_from_telemetry(load_json(path))
    val_val_weights = weights_for_val_val(arrays["patient_ids"], groups["val_val_groups"])
    val_val_keep_first = keep_first_val_val_mask(arrays["patient_ids"], groups["val_val_groups"])
    train_val_mask = exclude_patients_mask(
        arrays["patient_ids"],
        groups["train_val_validation_patients"],
    )
    combined_weights = val_val_weights.copy()
    for patient_id in groups["train_val_validation_patients"]:
        combined_weights[arrays["patient_ids"] == patient_id] = 0.0
    combined_mask = train_val_mask & val_val_keep_first

    return {
        "full": metrics_from_scores(arrays["labels"], arrays["abnormal"]),
        "val_val_weighted": metrics_for_weights(arrays, val_val_weights),
        "val_val_keep_first": metrics_for_mask(arrays, val_val_keep_first),
        "train_val_validation_excluded": metrics_for_mask(arrays, train_val_mask),
        "combined_weighted_exclude_train_val": metrics_for_weights(arrays, combined_weights),
        "combined_keep_first_exclude_train_val": metrics_for_mask(arrays, combined_mask),
        "diagnostic_patient_records": {
            "val_val_duplicate_patients": patient_records(
                arrays,
                sorted({patient_id for group in groups["val_val_groups"] for patient_id in group}),
            ),
            "train_val_duplicated_validation_patients": patient_records(
                arrays,
                groups["train_val_validation_patients"],
            ),
        },
    }


def aggregate(per_seed: dict[str, dict[str, Any]], key: str) -> dict[str, float]:
    metric_keys = ["accuracy", "auc", "f1", "specificity", "sensitivity", "effective_n"]
    items = [seed_result[key] for seed_result in per_seed.values()]
    return {metric: round_float(mean([float(item[metric]) for item in items])) for metric in metric_keys}


def evaluate_variant(repo_root: Path, variant: str, paths: dict[str, str], groups: dict[str, Any]) -> dict[str, Any]:
    per_seed = {
        seed: evaluate_seed(repo_root / paths[seed], groups)
        for seed in SEEDS
    }
    modes = (
        "full",
        "val_val_weighted",
        "val_val_keep_first",
        "train_val_validation_excluded",
        "combined_weighted_exclude_train_val",
        "combined_keep_first_exclude_train_val",
    )
    return {
        "variant": variant,
        "paths": paths,
        "per_seed": per_seed,
        "aggregate": {mode: aggregate(per_seed, mode) for mode in modes},
        "mode_minus_full": {
            mode: delta_metrics(aggregate(per_seed, mode), aggregate(per_seed, "full"))
            for mode in modes
            if mode != "full"
        },
    }


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
    variants = {
        name: evaluate_variant(repo_root, name, paths, groups)
        for name, paths in VARIANTS.items()
    }
    modes = tuple(variants["dfr25"]["aggregate"].keys())
    report = {
        "analysis": "dfr124_duplicate_leakage_metric_sensitivity",
        "description": (
            "Read-only diagnostic validation metric sensitivity using DFR-123 duplicate groups. "
            "These metrics are not official model-selection metrics."
        ),
        "duplicate_report": str(duplicate_report_path),
        "groups": groups,
        "variants": variants,
        "dfr116_minus_dfr25": {
            mode: delta_metrics(
                variants["dfr116"]["aggregate"][mode],
                variants["dfr25"]["aggregate"][mode],
            )
            for mode in modes
        },
    }
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    save_json(output, report)
    print(f"Saved DFR-124 duplicate/leakage metric sensitivity to: {output}")
    for mode in modes:
        dfr25 = variants["dfr25"]["aggregate"][mode]
        dfr116 = variants["dfr116"]["aggregate"][mode]
        delta = report["dfr116_minus_dfr25"][mode]
        print(
            f"{mode}: "
            f"DFR25={dfr25['accuracy']:.6f}/{dfr25['auc']:.6f}/{dfr25['f1']:.6f} "
            f"DFR116={dfr116['accuracy']:.6f}/{dfr116['auc']:.6f}/{dfr116['f1']:.6f} "
            f"delta={delta['accuracy']:.6f}/{delta['auc']:.6f}/{delta['f1']:.6f}"
        )


if __name__ == "__main__":
    main()
