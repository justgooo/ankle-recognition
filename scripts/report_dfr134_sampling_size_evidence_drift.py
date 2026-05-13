#!/usr/bin/env python3
"""DFR-134 sampling-size evidence drift audit.

This read-only report compares DFR-25 seed42, DFR-132 n32/trim2, and
DFR-133 n24/trim2 validation decision-fusion telemetry.  It cross-references
the DFR-131 static sampling coverage flags to determine why larger global
slice counts did not translate into better coronal/sagittal classifier
evidence.  No test metrics are read and no data files are modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("axial", "coronal", "sagittal")
VIEW_COLUMNS = ("axial_dir", "coronal_dir", "sagittal_dir")

DEFAULT_OUTPUT = "autoresearch_logs/dfr134_sampling_size_evidence_drift.json"
DEFAULT_DFR131_REPORT = "autoresearch_logs/dfr131_sampling_variable_audit.json"
TELEMETRY_PATHS = {
    "dfr25_8slice": (
        "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
        "trials/trial_0001/run/fusion_weight_analysis.json"
    ),
    "dfr116_combo_8slice": (
        "runs/resnext_decision_256x8_mainline/"
        "dfr116_posthoc_combo_gate_formal_s42/fusion_weight_analysis.json"
    ),
    "dfr132_n32": (
        "runs/resnext_decision_256x8_mainline/"
        "dfr132_sampling_256x32_formal_s42/fusion_weight_analysis.json"
    ),
    "dfr133_n24": (
        "runs/resnext_decision_256x8_mainline/"
        "dfr133_sampling_256x24_formal_s42/fusion_weight_analysis.json"
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
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "mean": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round_float(float(np.min(arr))),
        "q25": round_float(float(np.quantile(arr, 0.25))),
        "median": round_float(float(np.quantile(arr, 0.50))),
        "q75": round_float(float(np.quantile(arr, 0.75))),
        "max": round_float(float(np.max(arr))),
        "mean": round_float(float(np.mean(arr))),
    }


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def label(sample: dict[str, Any]) -> int:
    return int(sample["label"])


def correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def fused_abnormal(sample: dict[str, Any]) -> float:
    return float(sample["fusion_prediction"]["abnormal_prob"])


def top_weight_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def top_true_margin_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_true_margin_views") or []
    return str(views[0]) if views else "none"


def view_value(sample: dict[str, Any], view: str, key: str) -> float:
    return float(sample["views"][view][key])


def confusion_type(sample: dict[str, Any]) -> str:
    if label(sample) == 0 and pred(sample) == 0:
        return "TN"
    if label(sample) == 1 and pred(sample) == 1:
        return "TP"
    if label(sample) == 0 and pred(sample) == 1:
        return "FP"
    return "FN"


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def metric_triplet(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        key: round_float(float(metrics[key]))
        for key in ("accuracy", "auc", "f1", "specificity", "sensitivity")
        if key in metrics
    }


def telemetry_summary(path: Path, telemetry: dict[str, Any]) -> dict[str, Any]:
    per_view = {str(item["view"]): item for item in telemetry["per_view"]}
    samples = telemetry["samples"]
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "metrics": metric_triplet(telemetry["summary"]["full_fusion_metrics"]),
        "confusion_counts": counter_dict(Counter(confusion_type(sample) for sample in samples)),
        "mean_fused_abnormal_by_label": {
            str(group_label): round_float(
                np.mean([fused_abnormal(sample) for sample in samples if label(sample) == group_label])
            )
            for group_label in (0, 1)
        },
        "per_view": {
            view: {
                "metrics": metric_triplet(per_view[view]["metrics"]),
                "mean_fusion_weight": round_float(float(per_view[view]["mean_fusion_weight"])),
                "top_weight_count": int(per_view[view]["top_weight_count"]),
                "top_true_margin_count": int(per_view[view]["top_true_margin_count"]),
                "mean_abnormal_by_label": {
                    str(group_label): round_float(
                        np.mean(
                            [
                                view_value(sample, view, "abnormal_prob")
                                for sample in samples
                                if label(sample) == group_label
                            ]
                        )
                    )
                    for group_label in (0, 1)
                },
                "mean_true_margin_by_label": {
                    str(group_label): round_float(
                        np.mean(
                            [
                                view_value(sample, view, "true_margin")
                                for sample in samples
                                if label(sample) == group_label
                            ]
                        )
                    )
                    for group_label in (0, 1)
                },
            }
            for view in VIEWS
        },
    }


def subset_stats(
    samples: dict[str, dict[str, Any]],
    patient_ids: list[str],
) -> dict[str, Any]:
    present = [samples[patient_id] for patient_id in patient_ids if patient_id in samples]
    return {
        "patient_count": len(present),
        "confusion_counts": counter_dict(Counter(confusion_type(sample) for sample in present)),
        "accuracy": round_float(
            sum(1 for sample in present if correct(sample)) / len(present)
            if present
            else float("nan")
        ),
        "mean_fused_abnormal": round_float(np.mean([fused_abnormal(sample) for sample in present]))
        if present
        else None,
        "per_view": {
            view: {
                "abnormal": quantiles([view_value(sample, view, "abnormal_prob") for sample in present]),
                "true_margin": quantiles([view_value(sample, view, "true_margin") for sample in present]),
                "fusion_weight": quantiles([view_value(sample, view, "fusion_weight") for sample in present]),
            }
            for view in VIEWS
        },
    }


def compare_to_reference(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common_ids = sorted(reference.keys() & candidate.keys())
    fixed: list[str] = []
    broken: list[str] = []
    changed_pred: list[str] = []
    changed_top_weight: list[str] = []
    changed_top_true_margin: list[str] = []

    for patient_id in common_ids:
        ref = reference[patient_id]
        cand = candidate[patient_id]
        if pred(ref) != pred(cand):
            changed_pred.append(patient_id)
        if top_weight_view(ref) != top_weight_view(cand):
            changed_top_weight.append(patient_id)
        if top_true_margin_view(ref) != top_true_margin_view(cand):
            changed_top_true_margin.append(patient_id)
        if not correct(ref) and correct(cand):
            fixed.append(patient_id)
        if correct(ref) and not correct(cand):
            broken.append(patient_id)

    deltas = {
        "fused_abnormal": [
            fused_abnormal(candidate[patient_id]) - fused_abnormal(reference[patient_id])
            for patient_id in common_ids
        ],
        "views": {
            view: {
                "abnormal": [
                    view_value(candidate[patient_id], view, "abnormal_prob")
                    - view_value(reference[patient_id], view, "abnormal_prob")
                    for patient_id in common_ids
                ],
                "true_margin": [
                    view_value(candidate[patient_id], view, "true_margin")
                    - view_value(reference[patient_id], view, "true_margin")
                    for patient_id in common_ids
                ],
                "fusion_weight": [
                    view_value(candidate[patient_id], view, "fusion_weight")
                    - view_value(reference[patient_id], view, "fusion_weight")
                    for patient_id in common_ids
                ],
            }
            for view in VIEWS
        },
    }
    return {
        "common_patient_count": len(common_ids),
        "fixed_count": len(fixed),
        "fixed_ids": fixed,
        "broken_count": len(broken),
        "broken_ids": broken,
        "changed_prediction_count": len(changed_pred),
        "changed_prediction_ids": changed_pred,
        "changed_top_weight_count": len(changed_top_weight),
        "changed_top_weight_ids": changed_top_weight,
        "changed_top_true_margin_count": len(changed_top_true_margin),
        "changed_top_true_margin_ids": changed_top_true_margin,
        "delta_summary": {
            "fused_abnormal": quantiles(deltas["fused_abnormal"]),
            "views": {
                view: {
                    key: quantiles(values)
                    for key, values in deltas["views"][view].items()
                }
                for view in VIEWS
            },
        },
        "broken_by_label": counter_dict(Counter(str(label(reference[patient_id])) for patient_id in broken)),
        "fixed_by_label": counter_dict(Counter(str(label(reference[patient_id])) for patient_id in fixed)),
    }


def state_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label(sample),
        "pred": pred(sample),
        "correct": correct(sample),
        "confusion_type": confusion_type(sample),
        "fusion_abnormal": round_float(fused_abnormal(sample)),
        "top_weight_view": top_weight_view(sample),
        "top_true_margin_view": top_true_margin_view(sample),
        "views": {
            view: {
                "abnormal": round_float(view_value(sample, view, "abnormal_prob")),
                "true_margin": round_float(view_value(sample, view, "true_margin")),
                "fusion_weight": round_float(view_value(sample, view, "fusion_weight")),
                "pred": int(sample["views"][view]["pred"]),
            }
            for view in VIEWS
        },
    }


def sampling_flags_for_patient(patient: dict[str, Any]) -> dict[str, Any]:
    view_sampling = patient.get("sampling", {}).get("view_sampling", {})
    result: dict[str, Any] = {}
    for view_column in VIEW_COLUMNS:
        view_name = view_column.removesuffix("_dir")
        view_info = view_sampling.get(view_column, {})
        candidates = view_info.get("candidate_deltas", {})
        result[view_name] = {
            "baseline_flags": list(view_info.get("baseline_flags", [])),
            "n24_resolved_flags": list(candidates.get("n24_trim2", {}).get("resolved_flags", [])),
            "n24_created_flags": list(candidates.get("n24_trim2", {}).get("created_flags", [])),
            "n24_metric_deltas": candidates.get("n24_trim2", {}).get("metric_deltas", {}),
            "n32_resolved_flags": list(candidates.get("n32_trim2", {}).get("resolved_flags", [])),
            "n32_created_flags": list(candidates.get("n32_trim2", {}).get("created_flags", [])),
            "n32_metric_deltas": candidates.get("n32_trim2", {}).get("metric_deltas", {}),
        }
    return result


def patient_details(
    patient_ids: list[str],
    maps: dict[str, dict[str, dict[str, Any]]],
    dfr131_patients: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    details = []
    for patient_id in patient_ids:
        row = dfr131_patients.get(patient_id, {})
        states = {
            name: state_for_sample(samples[patient_id])
            for name, samples in maps.items()
            if patient_id in samples
        }
        deltas_vs_dfr25 = {}
        if "dfr25_8slice" in maps and patient_id in maps["dfr25_8slice"]:
            ref = maps["dfr25_8slice"][patient_id]
            for name in ("dfr132_n32", "dfr133_n24"):
                if patient_id not in maps[name]:
                    continue
                cand = maps[name][patient_id]
                deltas_vs_dfr25[name] = {
                    "fusion_abnormal": round_float(fused_abnormal(cand) - fused_abnormal(ref)),
                    "views": {
                        view: {
                            "abnormal": round_float(
                                view_value(cand, view, "abnormal_prob")
                                - view_value(ref, view, "abnormal_prob")
                            ),
                            "true_margin": round_float(
                                view_value(cand, view, "true_margin")
                                - view_value(ref, view, "true_margin")
                            ),
                            "fusion_weight": round_float(
                                view_value(cand, view, "fusion_weight")
                                - view_value(ref, view, "fusion_weight")
                            ),
                        }
                        for view in VIEWS
                    },
                }
        details.append(
            {
                "patient_id": patient_id,
                "label": row.get("label"),
                "dfr126_bucket": row.get("dfr126_bucket"),
                "remaining_case_ids": row.get("remaining_case_ids", []),
                "sampling_flags": sampling_flags_for_patient(row) if row else {},
                "states": states,
                "deltas_vs_dfr25": deltas_vs_dfr25,
            }
        )
    return details


def mean_delta_for_group(
    ref: dict[str, dict[str, Any]],
    cand: dict[str, dict[str, Any]],
    patient_ids: list[str],
    view: str,
    key: str,
) -> float | None:
    values = []
    for patient_id in patient_ids:
        if patient_id not in ref or patient_id not in cand:
            continue
        values.append(view_value(cand[patient_id], view, key) - view_value(ref[patient_id], view, key))
    if not values:
        return None
    return round_float(float(np.mean(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dfr131-report", default=DEFAULT_DFR131_REPORT)
    args = parser.parse_args()

    telemetry_paths = {name: REPO_ROOT / path for name, path in TELEMETRY_PATHS.items()}
    telemetry = {name: load_json(path) for name, path in telemetry_paths.items()}
    maps = {name: sample_map(payload) for name, payload in telemetry.items()}

    dfr131 = load_json(REPO_ROOT / args.dfr131_report)
    dfr131_patients = {
        str(patient["patient_id"]): patient
        for patient in dfr131["patients"]
    }
    positive_remaining_ids = list(dfr131["assessment"]["positive_remaining_patient_ids"])
    negative_remaining_ids = list(dfr131["assessment"]["negative_remaining_patient_ids"])
    remaining_ids = sorted(positive_remaining_ids + negative_remaining_ids)

    dfr25 = maps["dfr25_8slice"]
    dfr132 = maps["dfr132_n32"]
    dfr133 = maps["dfr133_n24"]
    dfr116 = maps["dfr116_combo_8slice"]

    candidate_vs_dfr25 = {
        "dfr132_n32": compare_to_reference(dfr25, dfr132),
        "dfr133_n24": compare_to_reference(dfr25, dfr133),
    }
    candidate_vs_dfr116 = {
        "dfr132_n32": compare_to_reference(dfr116, dfr132),
        "dfr133_n24": compare_to_reference(dfr116, dfr133),
    }

    non_axial_f1_zero = {
        name: {
            view: round_float(float(
                {item["view"]: item for item in payload["per_view"]}[view]["metrics"]["f1"]
            ))
            for view in ("coronal", "sagittal")
        }
        for name, payload in (("dfr132_n32", telemetry["dfr132_n32"]), ("dfr133_n24", telemetry["dfr133_n24"]))
    }

    positive_coronal_drift = {
        name: {
            "abnormal_delta": mean_delta_for_group(dfr25, maps[name], positive_remaining_ids, "coronal", "abnormal_prob"),
            "true_margin_delta": mean_delta_for_group(dfr25, maps[name], positive_remaining_ids, "coronal", "true_margin"),
            "fusion_weight_delta": mean_delta_for_group(dfr25, maps[name], positive_remaining_ids, "coronal", "fusion_weight"),
        }
        for name in ("dfr132_n32", "dfr133_n24")
    }

    payload = {
        "analysis": "dfr134_sampling_size_evidence_drift_audit",
        "description": (
            "Read-only validation telemetry audit comparing DFR25 baseline, DFR132 n32, "
            "and DFR133 n24 sampling-size formal runs, cross-referenced with DFR131 static "
            "sampling coverage flags. No test metrics or data edits."
        ),
        "source_paths": {
            **{name: str(path.relative_to(REPO_ROOT)) for name, path in telemetry_paths.items()},
            "dfr131_report": str((REPO_ROOT / args.dfr131_report).relative_to(REPO_ROOT)),
        },
        "telemetry_summary": {
            name: telemetry_summary(telemetry_paths[name], telemetry[name])
            for name in ("dfr25_8slice", "dfr116_combo_8slice", "dfr132_n32", "dfr133_n24")
        },
        "candidate_vs_dfr25": candidate_vs_dfr25,
        "candidate_vs_dfr116": candidate_vs_dfr116,
        "dfr131_candidate_rankings": dfr131["candidate_rankings"],
        "remaining_group_stats": {
            name: {
                "positive_remaining": subset_stats(samples, positive_remaining_ids),
                "negative_remaining": subset_stats(samples, negative_remaining_ids),
                "all_remaining": subset_stats(samples, remaining_ids),
            }
            for name, samples in maps.items()
        },
        "remaining_patient_details": patient_details(remaining_ids, maps, dfr131_patients),
        "changed_patient_details": patient_details(
            sorted(
                set(candidate_vs_dfr25["dfr132_n32"]["fixed_ids"])
                | set(candidate_vs_dfr25["dfr132_n32"]["broken_ids"])
                | set(candidate_vs_dfr25["dfr133_n24"]["fixed_ids"])
                | set(candidate_vs_dfr25["dfr133_n24"]["broken_ids"])
            ),
            maps,
            dfr131_patients,
        ),
        "assessment": {
            "n24_better_than_n32_but_still_discard": True,
            "close_global_slice_count_formal_family": True,
            "do_not_run_n16_n24_n32_multiseed": True,
            "non_axial_f1_by_candidate": non_axial_f1_zero,
            "positive_remaining_coronal_drift_vs_dfr25": positive_coronal_drift,
            "static_coverage_did_not_translate_to_classifier_evidence": all(
                values["abnormal_delta"] is not None and values["abnormal_delta"] < 0.0
                for values in positive_coronal_drift.values()
            ),
            "main_failure_mode": (
                "Increasing global num_slices_per_view improves static HU/STD coverage in DFR131 "
                "but trained n24/n32 checkpoints reduce positive-remaining coronal abnormal evidence "
                "and keep top-weight routing axial-only; this is classifier/evidence drift, not a "
                "simple gate-only failure."
            ),
            "recommended_next_experiment": (
                "DFR-135 frozen-checkpoint eval-side sampling replay: run DFR25 seed42 validation "
                "telemetry with alternative num_slices_per_view configs against the same checkpoint. "
                "If frozen replay improves coronal evidence, training with larger S is the problem; "
                "if it also fails, DFR131 static coverage is not predictive enough and the next move "
                "should be view-evidence calibration or label/annotation audit rather than more slice counts."
            ),
        },
    }
    save_json(REPO_ROOT / args.output, payload)
    print(f"Wrote {REPO_ROOT / args.output}")
    print(json.dumps(payload["assessment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
