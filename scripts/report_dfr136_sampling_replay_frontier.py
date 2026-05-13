#!/usr/bin/env python3
"""DFR-136 sampling replay ensemble/frontier audit.

This read-only report tests whether DFR-135 frozen n24/n32 validation
telemetry contains any label-free sampling ensemble signal that can improve on
DFR-25 without breaking DFR-25-correct samples.  It uses validation telemetry
only, reads no test metrics, and modifies no data files.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("axial", "coronal", "sagittal")

DEFAULT_OUTPUT = "autoresearch_logs/dfr136_sampling_replay_frontier.json"
TELEMETRY_PATHS = {
    "dfr25_8slice": (
        "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
        "trials/trial_0001/run/fusion_weight_analysis.json"
    ),
    "dfr116_combo_8slice": (
        "runs/resnext_decision_256x8_mainline/"
        "dfr116_posthoc_combo_gate_formal_s42/fusion_weight_analysis.json"
    ),
    "dfr135_frozen_n24": (
        "runs/resnext_decision_256x8_mainline/"
        "dfr135_frozen_dfr25_replay_256x24_s42/fusion_weight_analysis.json"
    ),
    "dfr135_frozen_n32": (
        "runs/resnext_decision_256x8_mainline/"
        "dfr135_frozen_dfr25_replay_256x32_s42/fusion_weight_analysis.json"
    ),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = int(labels.shape[0] - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def metrics_from_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    preds = (scores >= 0.5).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = int(labels.shape[0])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round_float((tp + tn) / total if total else float("nan")),
        "auc": round_float(auc_from_scores(labels, scores)),
        "f1": round_float(f1),
        "specificity": round_float(specificity),
        "sensitivity": round_float(recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def sample_label(sample: dict[str, Any]) -> int:
    return int(sample["label"])


def sample_score(sample: dict[str, Any]) -> float:
    return float(sample["fusion_prediction"]["abnormal_prob"])


def sample_pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def sample_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def view_value(sample: dict[str, Any], view: str, key: str) -> float:
    return float(sample["views"][view][key])


def telemetry_metrics(telemetry: dict[str, Any]) -> dict[str, float | None]:
    metrics = telemetry["summary"]["full_fusion_metrics"]
    return {
        key: round_float(float(metrics[key]))
        for key in ("accuracy", "auc", "f1", "specificity", "sensitivity")
        if key in metrics
    }


def telemetry_summary(path: Path, telemetry: dict[str, Any]) -> dict[str, Any]:
    per_view = {str(row["view"]): row for row in telemetry["per_view"]}
    samples = telemetry["samples"]
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "metrics": telemetry_metrics(telemetry),
        "sample_count": int(telemetry["total_samples"]),
        "top_weight_view_distribution": {
            view: int(telemetry["summary"]["top_weight_view_distribution"].get(view, 0))
            for view in VIEWS
        },
        "mean_fusion_weight": {
            view: round_float(float(per_view[view]["mean_fusion_weight"]))
            for view in VIEWS
        },
        "per_view_metrics": {
            view: {
                key: round_float(float(per_view[view]["metrics"][key]))
                for key in ("accuracy", "auc", "f1", "specificity", "sensitivity")
                if key in per_view[view]["metrics"]
            }
            for view in VIEWS
        },
        "confusion_counts": counter_dict(
            Counter(
                ("T" if sample_correct(sample) else "F") + str(sample_label(sample))
                for sample in samples
            )
        ),
    }


def arrays_for_ids(
    samples: dict[str, dict[str, Any]],
    patient_ids: list[str],
) -> dict[str, Any]:
    labels = np.asarray([sample_label(samples[patient_id]) for patient_id in patient_ids], dtype=np.int64)
    scores = np.asarray([sample_score(samples[patient_id]) for patient_id in patient_ids], dtype=np.float64)
    preds = np.asarray([sample_pred(samples[patient_id]) for patient_id in patient_ids], dtype=np.int64)
    view_probs = np.asarray(
        [
            [view_value(samples[patient_id], view, "abnormal_prob") for view in VIEWS]
            for patient_id in patient_ids
        ],
        dtype=np.float64,
    )
    view_weights = np.asarray(
        [
            [view_value(samples[patient_id], view, "fusion_weight") for view in VIEWS]
            for patient_id in patient_ids
        ],
        dtype=np.float64,
    )
    confidence_logits = np.asarray(
        [
            [view_value(samples[patient_id], view, "scaled_confidence_logit") for view in VIEWS]
            for patient_id in patient_ids
        ],
        dtype=np.float64,
    )
    return {
        "patient_ids": patient_ids,
        "labels": labels,
        "scores": scores,
        "preds": preds,
        "correct": preds == labels,
        "view_probs": view_probs,
        "view_weights": view_weights,
        "confidence_logits": confidence_logits,
    }


def compare_scores(
    *,
    patient_ids: list[str],
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    reference_preds: np.ndarray,
) -> dict[str, Any]:
    candidate_preds = (np.asarray(candidate_scores, dtype=np.float64) >= 0.5).astype(np.int64)
    reference_correct = reference_preds == labels
    candidate_correct = candidate_preds == labels

    fixed_indices = np.flatnonzero((~reference_correct) & candidate_correct)
    broken_indices = np.flatnonzero(reference_correct & (~candidate_correct))
    changed_indices = np.flatnonzero(candidate_preds != reference_preds)

    return {
        "fixed_count": int(fixed_indices.size),
        "broken_count": int(broken_indices.size),
        "net_delta": int(fixed_indices.size - broken_indices.size),
        "changed_prediction_count": int(changed_indices.size),
        "fixed_ids": [patient_ids[int(index)] for index in fixed_indices],
        "broken_ids": [patient_ids[int(index)] for index in broken_indices],
        "changed_prediction_ids": [patient_ids[int(index)] for index in changed_indices],
        "fixed_by_label": counter_dict(Counter(str(int(labels[index])) for index in fixed_indices)),
        "broken_by_label": counter_dict(Counter(str(int(labels[index])) for index in broken_indices)),
    }


def candidate_record(
    *,
    name: str,
    family: str,
    params: dict[str, Any],
    scores: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    dfr25_preds: np.ndarray,
    dfr116_preds: np.ndarray,
    trigger_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    record = {
        "name": name,
        "family": family,
        "params": params,
        "metrics": metrics_from_scores(labels, scores),
        "vs_dfr25": compare_scores(
            patient_ids=patient_ids,
            labels=labels,
            candidate_scores=scores,
            reference_preds=dfr25_preds,
        ),
        "vs_dfr116": compare_scores(
            patient_ids=patient_ids,
            labels=labels,
            candidate_scores=scores,
            reference_preds=dfr116_preds,
        ),
    }
    if trigger_mask is not None:
        rows = np.flatnonzero(trigger_mask)
        record["triggered_count"] = int(rows.size)
        record["triggered_ids"] = [patient_ids[int(index)] for index in rows]
        record["triggered_by_label"] = counter_dict(Counter(str(int(labels[index])) for index in rows))
    return record


def score_key(record: dict[str, Any]) -> tuple[float, float, int, int, str]:
    metrics = record["metrics"]
    return (
        float(metrics["accuracy"] or 0.0),
        float(metrics["auc"] or 0.0),
        int(record["vs_dfr25"]["net_delta"]),
        -int(record["vs_dfr25"]["broken_count"]),
        str(record["name"]),
    )


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "name": record["name"],
        "family": record["family"],
        "params": record["params"],
        "metrics": record["metrics"],
        "vs_dfr25": record["vs_dfr25"],
        "vs_dfr116": record["vs_dfr116"],
    }
    if "triggered_count" in record:
        keep["triggered_count"] = record["triggered_count"]
        keep["triggered_by_label"] = record["triggered_by_label"]
        keep["triggered_ids"] = record["triggered_ids"]
    return keep


def candidate_sort(records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        compact_record(record)
        for record in sorted(records, key=score_key, reverse=True)[:limit]
    ]


def grid_values(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + step * index, 6) for index in range(count + 1)]


def fixed_blend_candidates(
    *,
    base: np.ndarray,
    n24: np.ndarray,
    n32: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    dfr25_preds: np.ndarray,
    dfr116_preds: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for alt_name, alt_scores in (("n24", n24), ("n32", n32)):
        for alpha in grid_values(0.05, 0.95, 0.05):
            scores = (1.0 - alpha) * base + alpha * alt_scores
            records.append(
                candidate_record(
                    name=f"blend_base_{alt_name}_a{alpha:.2f}",
                    family="fixed_two_way_blend",
                    params={"alt": alt_name, "alpha": round_float(alpha)},
                    scores=scores,
                    labels=labels,
                    patient_ids=patient_ids,
                    dfr25_preds=dfr25_preds,
                    dfr116_preds=dfr116_preds,
                )
            )

    for n24_weight in grid_values(0.0, 1.0, 0.05):
        for n32_weight in grid_values(0.0, 1.0, 0.05):
            if n24_weight == 0.0 and n32_weight == 0.0:
                continue
            if n24_weight + n32_weight > 1.000001:
                continue
            base_weight = 1.0 - n24_weight - n32_weight
            scores = base_weight * base + n24_weight * n24 + n32_weight * n32
            records.append(
                candidate_record(
                    name=f"blend3_b{base_weight:.2f}_n24{n24_weight:.2f}_n32{n32_weight:.2f}",
                    family="fixed_three_way_blend",
                    params={
                        "base_weight": round_float(base_weight),
                        "n24_weight": round_float(n24_weight),
                        "n32_weight": round_float(n32_weight),
                    },
                    scores=scores,
                    labels=labels,
                    patient_ids=patient_ids,
                    dfr25_preds=dfr25_preds,
                    dfr116_preds=dfr116_preds,
                )
            )

    stacks = {
        "mean_n24_n32": np.stack([n24, n32], axis=0).mean(axis=0),
        "min_n24_n32": np.minimum(n24, n32),
        "max_n24_n32": np.maximum(n24, n32),
        "mean_base_n24_n32": np.stack([base, n24, n32], axis=0).mean(axis=0),
        "median_base_n24_n32": np.median(np.stack([base, n24, n32], axis=0), axis=0),
        "min_base_n24_n32": np.minimum(np.minimum(base, n24), n32),
        "max_base_n24_n32": np.maximum(np.maximum(base, n24), n32),
    }
    for name, scores in stacks.items():
        records.append(
            candidate_record(
                name=name,
                family="fixed_order_statistic",
                params={},
                scores=scores,
                labels=labels,
                patient_ids=patient_ids,
                dfr25_preds=dfr25_preds,
                dfr116_preds=dfr116_preds,
            )
        )
    return records


def selector_candidate(
    *,
    name: str,
    family: str,
    params: dict[str, Any],
    base: np.ndarray,
    replacement: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    dfr25_preds: np.ndarray,
    dfr116_preds: np.ndarray,
) -> dict[str, Any]:
    scores = base.copy()
    scores[mask] = replacement[mask]
    return candidate_record(
        name=name,
        family=family,
        params=params,
        scores=scores,
        labels=labels,
        patient_ids=patient_ids,
        dfr25_preds=dfr25_preds,
        dfr116_preds=dfr116_preds,
        trigger_mask=mask,
    )


def selector_candidates(
    *,
    base: np.ndarray,
    n24: np.ndarray,
    n32: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    dfr25_preds: np.ndarray,
    dfr116_preds: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    alts = {
        "n24": n24,
        "n32": n32,
        "mean_n24_n32": np.stack([n24, n32], axis=0).mean(axis=0),
        "min_n24_n32": np.minimum(n24, n32),
        "max_n24_n32": np.maximum(n24, n32),
    }

    for alt_name, alt in alts.items():
        delta_down = base - alt
        delta_up = alt - base
        base_gap = np.abs(base - 0.5)
        alt_gap = np.abs(alt - 0.5)

        for base_min in grid_values(0.50, 0.90, 0.05):
            for alt_max in grid_values(0.35, 0.55, 0.025):
                for delta_min in grid_values(0.05, 0.40, 0.05):
                    mask = (base >= base_min) & (alt <= alt_max) & (delta_down >= delta_min)
                    if not mask.any():
                        continue
                    records.append(
                        selector_candidate(
                            name=(
                                f"lower_{alt_name}_base{base_min:.2f}_alt{alt_max:.3f}"
                                f"_d{delta_min:.2f}"
                            ),
                            family="selector_lower_abnormal",
                            params={
                                "alt": alt_name,
                                "base_min": round_float(base_min),
                                "alt_max": round_float(alt_max),
                                "delta_min": round_float(delta_min),
                            },
                            base=base,
                            replacement=alt,
                            mask=mask,
                            labels=labels,
                            patient_ids=patient_ids,
                            dfr25_preds=dfr25_preds,
                            dfr116_preds=dfr116_preds,
                        )
                    )

        for base_max in grid_values(0.10, 0.50, 0.05):
            for alt_min in grid_values(0.50, 0.90, 0.05):
                for delta_min in grid_values(0.05, 0.40, 0.05):
                    mask = (base <= base_max) & (alt >= alt_min) & (delta_up >= delta_min)
                    if not mask.any():
                        continue
                    records.append(
                        selector_candidate(
                            name=(
                                f"raise_{alt_name}_base{base_max:.2f}_alt{alt_min:.2f}"
                                f"_d{delta_min:.2f}"
                            ),
                            family="selector_raise_abnormal",
                            params={
                                "alt": alt_name,
                                "base_max": round_float(base_max),
                                "alt_min": round_float(alt_min),
                                "delta_min": round_float(delta_min),
                            },
                            base=base,
                            replacement=alt,
                            mask=mask,
                            labels=labels,
                            patient_ids=patient_ids,
                            dfr25_preds=dfr25_preds,
                            dfr116_preds=dfr116_preds,
                        )
                    )

        for base_margin_max in grid_values(0.025, 0.35, 0.025):
            for alt_margin_min in grid_values(0.0, 0.35, 0.05):
                mask = (base_gap <= base_margin_max) & (alt_gap >= alt_margin_min)
                if not mask.any():
                    continue
                records.append(
                    selector_candidate(
                        name=(
                            f"near_boundary_{alt_name}_basegap{base_margin_max:.3f}"
                            f"_altgap{alt_margin_min:.2f}"
                        ),
                        family="selector_near_boundary_replace",
                        params={
                            "alt": alt_name,
                            "base_margin_max": round_float(base_margin_max),
                            "alt_margin_min": round_float(alt_margin_min),
                        },
                        base=base,
                        replacement=alt,
                        mask=mask,
                        labels=labels,
                        patient_ids=patient_ids,
                        dfr25_preds=dfr25_preds,
                        dfr116_preds=dfr116_preds,
                    )
                )

    both_lower = (n24 < base) & (n32 < base)
    both_higher = (n24 > base) & (n32 > base)
    for delta_min in grid_values(0.05, 0.40, 0.05):
        for alt_max in grid_values(0.40, 0.60, 0.025):
            replacement = np.minimum(n24, n32)
            mask = both_lower & ((base - replacement) >= delta_min) & (replacement <= alt_max)
            if mask.any():
                records.append(
                    selector_candidate(
                        name=f"consensus_lower_min_alt{alt_max:.3f}_d{delta_min:.2f}",
                        family="selector_consensus_lower",
                        params={"replacement": "min_n24_n32", "alt_max": round_float(alt_max), "delta_min": round_float(delta_min)},
                        base=base,
                        replacement=replacement,
                        mask=mask,
                        labels=labels,
                        patient_ids=patient_ids,
                        dfr25_preds=dfr25_preds,
                        dfr116_preds=dfr116_preds,
                    )
                )

        for alt_min in grid_values(0.40, 0.80, 0.05):
            replacement = np.maximum(n24, n32)
            mask = both_higher & ((replacement - base) >= delta_min) & (replacement >= alt_min)
            if mask.any():
                records.append(
                    selector_candidate(
                        name=f"consensus_raise_max_alt{alt_min:.2f}_d{delta_min:.2f}",
                        family="selector_consensus_raise",
                        params={"replacement": "max_n24_n32", "alt_min": round_float(alt_min), "delta_min": round_float(delta_min)},
                        base=base,
                        replacement=replacement,
                        mask=mask,
                        labels=labels,
                        patient_ids=patient_ids,
                        dfr25_preds=dfr25_preds,
                        dfr116_preds=dfr116_preds,
                    )
                )

    return records


def variant_compare(
    arrays: dict[str, Any],
    reference: dict[str, Any],
    patient_ids: list[str],
) -> dict[str, Any]:
    return compare_scores(
        patient_ids=patient_ids,
        labels=reference["labels"],
        candidate_scores=arrays["scores"],
        reference_preds=reference["preds"],
    )


def score_drift_summary(
    *,
    base: np.ndarray,
    alt: np.ndarray,
    labels: np.ndarray,
    base_preds: np.ndarray,
) -> dict[str, Any]:
    delta = alt - base
    groups = {
        "all": np.ones_like(labels, dtype=bool),
        "dfr25_correct": base_preds == labels,
        "dfr25_error": base_preds != labels,
        "dfr25_fp": (base_preds == 1) & (labels == 0),
        "dfr25_fn": (base_preds == 0) & (labels == 1),
    }
    out: dict[str, Any] = {}
    for name, mask in groups.items():
        values = delta[mask]
        if values.size == 0:
            out[name] = {"count": 0}
            continue
        helpful = (
            ((labels[mask] == 0) & (values < 0.0))
            | ((labels[mask] == 1) & (values > 0.0))
        )
        out[name] = {
            "count": int(values.size),
            "mean_delta": round_float(float(values.mean())),
            "min_delta": round_float(float(values.min())),
            "max_delta": round_float(float(values.max())),
            "helpful_direction_count": int(helpful.sum()),
            "harmful_direction_count": int(values.size - helpful.sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    telemetry_paths = {name: REPO_ROOT / path for name, path in TELEMETRY_PATHS.items()}
    telemetry = {name: load_json(path) for name, path in telemetry_paths.items()}
    maps = {name: sample_map(payload) for name, payload in telemetry.items()}

    base_patient_ids = [str(sample["patient_id"]) for sample in telemetry["dfr25_8slice"]["samples"]]
    common_ids = [
        patient_id
        for patient_id in base_patient_ids
        if all(patient_id in samples for samples in maps.values())
    ]
    if len(common_ids) != len(base_patient_ids):
        raise RuntimeError(
            f"Expected all baseline patients in every telemetry file, got {len(common_ids)} of {len(base_patient_ids)}"
        )

    arrays = {
        name: arrays_for_ids(samples, common_ids)
        for name, samples in maps.items()
    }
    labels = arrays["dfr25_8slice"]["labels"]
    for name, current in arrays.items():
        if not np.array_equal(labels, current["labels"]):
            raise RuntimeError(f"Label mismatch in {name}")

    base_scores = arrays["dfr25_8slice"]["scores"]
    n24_scores = arrays["dfr135_frozen_n24"]["scores"]
    n32_scores = arrays["dfr135_frozen_n32"]["scores"]
    dfr25_preds = arrays["dfr25_8slice"]["preds"]
    dfr116_preds = arrays["dfr116_combo_8slice"]["preds"]

    direct_candidates = [
        candidate_record(
            name=name,
            family="direct_variant",
            params={},
            scores=arrays[name]["scores"],
            labels=labels,
            patient_ids=common_ids,
            dfr25_preds=dfr25_preds,
            dfr116_preds=dfr116_preds,
        )
        for name in ("dfr135_frozen_n24", "dfr135_frozen_n32")
    ]

    candidates = []
    candidates.extend(direct_candidates)
    candidates.extend(
        fixed_blend_candidates(
            base=base_scores,
            n24=n24_scores,
            n32=n32_scores,
            labels=labels,
            patient_ids=common_ids,
            dfr25_preds=dfr25_preds,
            dfr116_preds=dfr116_preds,
        )
    )
    candidates.extend(
        selector_candidates(
            base=base_scores,
            n24=n24_scores,
            n32=n32_scores,
            labels=labels,
            patient_ids=common_ids,
            dfr25_preds=dfr25_preds,
            dfr116_preds=dfr116_preds,
        )
    )

    family_counts = Counter(record["family"] for record in candidates)
    no_harm = [
        record
        for record in candidates
        if record["vs_dfr25"]["broken_count"] == 0
    ]
    improved_no_harm = [
        record
        for record in no_harm
        if record["vs_dfr25"]["fixed_count"] > 0
    ]
    dfr116_safe = [
        record
        for record in candidates
        if record["vs_dfr116"]["broken_count"] == 0
    ]
    dfr116_improved_safe = [
        record
        for record in dfr116_safe
        if record["vs_dfr116"]["fixed_count"] > 0
    ]

    dfr25_metrics = telemetry_metrics(telemetry["dfr25_8slice"])
    dfr116_metrics = telemetry_metrics(telemetry["dfr116_combo_8slice"])
    best_overall = sorted(candidates, key=score_key, reverse=True)[0]
    best_no_harm = sorted(no_harm, key=score_key, reverse=True)[0] if no_harm else None
    best_improved_no_harm = (
        sorted(improved_no_harm, key=score_key, reverse=True)[0]
        if improved_no_harm
        else None
    )
    best_dfr116_safe = (
        sorted(dfr116_safe, key=score_key, reverse=True)[0]
        if dfr116_safe
        else None
    )

    top_by_family: dict[str, list[dict[str, Any]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[record["family"]].append(record)
    for family, records in sorted(grouped.items()):
        top_by_family[family] = candidate_sort(records, limit=5)

    payload = {
        "analysis": "dfr136_sampling_replay_frontier_audit",
        "description": (
            "Read-only validation telemetry frontier for DFR25 baseline plus DFR135 "
            "frozen n24/n32 sampling replays. Candidate rules are label-free probability "
            "blends or selectors; labels are used only to score the frontier."
        ),
        "source_paths": {
            name: str(path.relative_to(REPO_ROOT))
            for name, path in telemetry_paths.items()
        },
        "common_patient_count": len(common_ids),
        "telemetry_summary": {
            name: telemetry_summary(telemetry_paths[name], telemetry[name])
            for name in TELEMETRY_PATHS
        },
        "direct_variant_vs_dfr25": {
            "dfr135_frozen_n24": variant_compare(
                arrays["dfr135_frozen_n24"], arrays["dfr25_8slice"], common_ids
            ),
            "dfr135_frozen_n32": variant_compare(
                arrays["dfr135_frozen_n32"], arrays["dfr25_8slice"], common_ids
            ),
        },
        "direct_variant_vs_dfr116": {
            "dfr135_frozen_n24": compare_scores(
                patient_ids=common_ids,
                labels=labels,
                candidate_scores=n24_scores,
                reference_preds=dfr116_preds,
            ),
            "dfr135_frozen_n32": compare_scores(
                patient_ids=common_ids,
                labels=labels,
                candidate_scores=n32_scores,
                reference_preds=dfr116_preds,
            ),
        },
        "score_drift_summary": {
            "n24_minus_dfr25": score_drift_summary(
                base=base_scores,
                alt=n24_scores,
                labels=labels,
                base_preds=dfr25_preds,
            ),
            "n32_minus_dfr25": score_drift_summary(
                base=base_scores,
                alt=n32_scores,
                labels=labels,
                base_preds=dfr25_preds,
            ),
        },
        "candidate_frontier": {
            "candidate_count": len(candidates),
            "family_counts": counter_dict(family_counts),
            "no_harm_vs_dfr25_count": len(no_harm),
            "improved_no_harm_vs_dfr25_count": len(improved_no_harm),
            "dfr116_safe_count": len(dfr116_safe),
            "dfr116_improved_safe_count": len(dfr116_improved_safe),
            "best_overall_by_accuracy": compact_record(best_overall),
            "best_no_harm_vs_dfr25": compact_record(best_no_harm) if best_no_harm else None,
            "best_improved_no_harm_vs_dfr25": (
                compact_record(best_improved_no_harm) if best_improved_no_harm else None
            ),
            "best_dfr116_safe": compact_record(best_dfr116_safe) if best_dfr116_safe else None,
            "top_overall": candidate_sort(candidates, limit=20),
            "top_no_harm_vs_dfr25": candidate_sort(no_harm, limit=20),
            "top_improved_no_harm_vs_dfr25": candidate_sort(improved_no_harm, limit=20),
            "top_dfr116_safe": candidate_sort(dfr116_safe, limit=20),
            "top_dfr116_improved_safe": candidate_sort(dfr116_improved_safe, limit=20),
            "top_by_family": top_by_family,
        },
        "assessment": {
            "dfr25_seed42_metrics": dfr25_metrics,
            "dfr116_seed42_metrics": dfr116_metrics,
            "best_sampling_frontier_metrics": compact_record(best_overall)["metrics"],
            "has_no_harm_improvement_vs_dfr25": bool(improved_no_harm),
            "has_no_harm_improvement_vs_dfr116": bool(dfr116_improved_safe),
            "best_improved_no_harm_vs_dfr25_name": best_improved_no_harm["name"] if best_improved_no_harm else None,
            "best_improved_no_harm_vs_dfr25_fixed_ids": (
                best_improved_no_harm["vs_dfr25"]["fixed_ids"] if best_improved_no_harm else []
            ),
            "best_improved_no_harm_vs_dfr25_broken_vs_dfr116": (
                best_improved_no_harm["vs_dfr116"]["broken_ids"] if best_improved_no_harm else []
            ),
            "sampling_frontier_exceeds_dfr116_seed42": (
                float(best_overall["metrics"]["accuracy"] or 0.0)
                > float(dfr116_metrics["accuracy"] or 0.0)
            ),
            "close_sampling_family_if_no_dfr116_safe_gain": not bool(dfr116_improved_safe),
            "recommended_next_experiment": (
                "If no candidate improves DFR116 without breaking DFR116-correct samples, "
                "stop the sampling-count/replay family and move to view-evidence calibration "
                "or annotation/data-quality audit. If a candidate only fixes a DFR25 error already "
                "covered by DFR116, do not promote sampling; treat it as redundant."
            ),
        },
    }

    save_json(REPO_ROOT / args.output, payload)
    print(f"Wrote {REPO_ROOT / args.output}")
    print(json.dumps(payload["assessment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
