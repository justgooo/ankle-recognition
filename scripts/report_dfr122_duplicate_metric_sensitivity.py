#!/usr/bin/env python3
"""Duplicate-aware validation metric sensitivity for DFR-25 and DFR-116.

This is a read-only report.  It does not change the validation split and does
not introduce a new model-selection metric.  It quantifies how known duplicate
validation volumes affect DFR-25 and DFR-116 validation metrics.
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
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402


DEFAULT_OUTPUT = "autoresearch_logs/dfr122_duplicate_metric_sensitivity.json"
SEEDS = ("42", "123", "456")
VARIANTS = {
    "dfr25": DFR25_TELEMETRY,
    "dfr116": DFR116_TELEMETRY,
}
DUPLICATE_GROUPS = [
    [
        "CTyang__CT24yang1__CT2412yang3",
        "CTyang__CT24yang2__CT2412yang1",
    ],
]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def auc_from_scores(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(scores, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_weight = float(weights[pos_mask].sum())
    neg_weight = float(weights[neg_mask].sum())
    if pos_weight <= 0 or neg_weight <= 0:
        return float("nan")

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    pos_weights = weights[pos_mask]
    neg_weights = weights[neg_mask]
    total = 0.0
    for score, weight in zip(pos_scores, pos_weights):
        less = neg_weights[neg_scores < score].sum()
        tied = neg_weights[neg_scores == score].sum()
        total += float(weight) * float(less + 0.5 * tied)
    return total / (pos_weight * neg_weight)


def metrics_from_scores(
    labels: np.ndarray,
    abnormal: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    abnormal = np.asarray(abnormal, dtype=np.float64)
    preds = (abnormal >= 0.5).astype(np.int64)
    if weights is None:
        weights = np.ones_like(abnormal, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    total = float(weights.sum())
    correct = float(weights[preds == labels].sum())
    tp = float(weights[(preds == 1) & (labels == 1)].sum())
    fp = float(weights[(preds == 1) & (labels == 0)].sum())
    fn = float(weights[(preds == 0) & (labels == 1)].sum())
    tn = float(weights[(preds == 0) & (labels == 0)].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    return {
        "accuracy": round_float(correct / total if total else float("nan")),
        "auc": round_float(auc_from_scores(labels, abnormal, weights)),
        "f1": round_float(f1),
        "specificity": round_float(specificity),
        "sensitivity": round_float(recall),
        "effective_n": round_float(total),
    }


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


def duplicate_weights(patient_ids: np.ndarray) -> np.ndarray:
    weights = np.ones(patient_ids.shape[0], dtype=np.float64)
    for group in DUPLICATE_GROUPS:
        present = [patient_id for patient_id in group if patient_id in set(patient_ids.tolist())]
        if not present:
            continue
        group_weight = 1.0 / len(present)
        for patient_id in present:
            weights[patient_ids == patient_id] = group_weight
    return weights


def keep_first_mask(patient_ids: np.ndarray) -> np.ndarray:
    mask = np.ones(patient_ids.shape[0], dtype=bool)
    for group in DUPLICATE_GROUPS:
        present = [patient_id for patient_id in group if patient_id in set(patient_ids.tolist())]
        for patient_id in present[1:]:
            mask[patient_ids == patient_id] = False
    return mask


def duplicate_group_records(arrays: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    patient_ids = arrays["patient_ids"]
    for group in DUPLICATE_GROUPS:
        members = []
        for patient_id in group:
            matches = np.flatnonzero(patient_ids == patient_id)
            if matches.size == 0:
                continue
            index = int(matches[0])
            members.append(
                {
                    "patient_id": patient_id,
                    "label": int(arrays["labels"][index]),
                    "pred": int(arrays["preds"][index]),
                    "abnormal": round_float(float(arrays["abnormal"][index])),
                    "correct": bool(int(arrays["preds"][index]) == int(arrays["labels"][index])),
                }
            )
        records.append({"group": group, "members": members})
    return records


def evaluate_variant_seed(path: Path) -> dict[str, Any]:
    telemetry = load_json(path)
    arrays = arrays_from_telemetry(telemetry)
    weights = duplicate_weights(arrays["patient_ids"])
    mask = keep_first_mask(arrays["patient_ids"])
    return {
        "full": metrics_from_scores(arrays["labels"], arrays["abnormal"]),
        "duplicate_group_weighted": metrics_from_scores(
            arrays["labels"],
            arrays["abnormal"],
            weights,
        ),
        "keep_first_duplicate_only": metrics_from_scores(
            arrays["labels"][mask],
            arrays["abnormal"][mask],
        ),
        "duplicate_groups": duplicate_group_records(arrays),
    }


def aggregate_variant(per_seed: dict[str, dict[str, Any]], key: str) -> dict[str, float]:
    metrics = [item[key] for item in per_seed.values()]
    metric_keys = ["accuracy", "auc", "f1", "specificity", "sensitivity", "effective_n"]
    return {metric: round_float(mean([float(item[metric]) for item in metrics])) for metric in metric_keys}


def evaluate_variant(repo_root: Path, variant: str, paths: dict[str, str]) -> dict[str, Any]:
    per_seed = {
        seed: evaluate_variant_seed(repo_root / paths[seed])
        for seed in SEEDS
    }
    return {
        "variant": variant,
        "paths": paths,
        "per_seed": per_seed,
        "aggregate": {
            key: aggregate_variant(per_seed, key)
            for key in ("full", "duplicate_group_weighted", "keep_first_duplicate_only")
        },
    }


def delta_metrics(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        key: round_float(float(after[key]) - float(before[key]))
        for key in ("accuracy", "auc", "f1", "specificity", "sensitivity", "effective_n")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    variants = {
        name: evaluate_variant(repo_root, name, paths)
        for name, paths in VARIANTS.items()
    }
    sensitivity = {}
    for variant_name, variant in variants.items():
        full = variant["aggregate"]["full"]
        weighted = variant["aggregate"]["duplicate_group_weighted"]
        keep_first = variant["aggregate"]["keep_first_duplicate_only"]
        sensitivity[variant_name] = {
            "weighted_minus_full": delta_metrics(weighted, full),
            "keep_first_minus_full": delta_metrics(keep_first, full),
        }
    report = {
        "analysis": "dfr122_duplicate_metric_sensitivity",
        "description": (
            "Read-only duplicate-aware validation metric sensitivity for DFR-25 "
            "and DFR-116.  These are diagnostic metrics only, not model-selection metrics."
        ),
        "duplicate_groups": DUPLICATE_GROUPS,
        "variants": variants,
        "sensitivity": sensitivity,
        "dfr116_minus_dfr25": {
            mode: delta_metrics(
                variants["dfr116"]["aggregate"][mode],
                variants["dfr25"]["aggregate"][mode],
            )
            for mode in ("full", "duplicate_group_weighted", "keep_first_duplicate_only")
        },
    }
    save_json(repo_root / args.output, report)

    print(f"Saved DFR-122 duplicate metric sensitivity to: {repo_root / args.output}")
    for name, variant in variants.items():
        print(name, variant["aggregate"])
        print("sensitivity", sensitivity[name])
    print("dfr116_minus_dfr25", report["dfr116_minus_dfr25"])


if __name__ == "__main__":
    main()
