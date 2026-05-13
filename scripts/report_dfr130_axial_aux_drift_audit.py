#!/usr/bin/env python3
"""Drift audit for the DFR-129 axial FP-risk normal auxiliary.

This read-only report compares DFR-25 seed42, DFR-129 best trial, and the
retained DFR-116 post-hoc combo telemetry.  It focuses on whether the
train-time axial-normal auxiliary changed only its intended high axial-risk
samples, or whether it caused off-target classifier drift.  No training, test
metrics, or data files are used.
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
from scripts.analyze_dfr112_multiseed_posthoc import DFR25_TELEMETRY  # noqa: E402
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402


DEFAULT_OUTPUT = "autoresearch_logs/dfr130_axial_aux_drift_audit.json"
DFR129_BEST_TELEMETRY = (
    "runs/optuna_main_resnext_decision_256x8_dfr129_axial_fp_risk_normal_aux/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR129_MONITOR_REPORT = (
    "runs/optuna_main_resnext_decision_256x8_dfr129_axial_fp_risk_normal_aux/"
    "monitor/report.json"
)
TRIGGER_THRESHOLDS = {
    "fused_abnormal_min": 0.75,
    "axial_abnormal_min": 0.925,
    "sagittal_normal_min": 0.60,
    "coronal_abnormal_max": 0.525,
}


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def top_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def error_type(sample: dict[str, Any]) -> str:
    label = int(sample["label"])
    pred = int(sample["fusion_prediction"]["pred"])
    if label == pred == 0:
        return "TN"
    if label == pred == 1:
        return "TP"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def is_correct(sample: dict[str, Any]) -> bool:
    return int(sample["label"]) == int(sample["fusion_prediction"]["pred"])


def abnormal_prob(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["abnormal_prob"])


def fusion_weight(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["fusion_weight"])


def fused_abnormal(sample: dict[str, Any]) -> float:
    return float(sample["fusion_prediction"]["abnormal_prob"])


def axial_aux_trigger(
    sample: dict[str, Any],
    thresholds: dict[str, float] = TRIGGER_THRESHOLDS,
) -> bool:
    return (
        fused_abnormal(sample) >= thresholds["fused_abnormal_min"]
        and abnormal_prob(sample, "axial") >= thresholds["axial_abnormal_min"]
        and (1.0 - abnormal_prob(sample, "sagittal")) >= thresholds["sagittal_normal_min"]
        and abnormal_prob(sample, "coronal") <= thresholds["coronal_abnormal_max"]
    )


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q10": None, "q25": None, "median": None, "q75": None, "q90": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round_float(float(np.min(arr))),
        "q10": round_float(float(np.quantile(arr, 0.10))),
        "q25": round_float(float(np.quantile(arr, 0.25))),
        "median": round_float(float(np.quantile(arr, 0.50))),
        "q75": round_float(float(np.quantile(arr, 0.75))),
        "q90": round_float(float(np.quantile(arr, 0.90))),
        "max": round_float(float(np.max(arr))),
        "mean": round_float(float(np.mean(arr))),
    }


def telemetry_summary(path: Path, telemetry: dict[str, Any]) -> dict[str, Any]:
    per_view = {str(item["view"]): item for item in telemetry["per_view"]}
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "metrics": telemetry["summary"]["full_fusion_metrics"],
        "mean_fusion_weight": {
            view: round_float(float(per_view[view]["mean_fusion_weight"]))
            for view in VIEWS
        },
        "top_weight_count": {
            view: int(per_view[view]["top_weight_count"])
            for view in VIEWS
        },
        "top_true_margin_count": {
            view: int(per_view[view]["top_true_margin_count"])
            for view in VIEWS
        },
        "per_view_accuracy": {
            view: round_float(float(per_view[view]["metrics"]["accuracy"]))
            for view in VIEWS
        },
        "per_view_auc": {
            view: round_float(float(per_view[view]["metrics"]["auc"]))
            for view in VIEWS
        },
    }


def trigger_summary(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    triggered = [sample for sample in samples.values() if axial_aux_trigger(sample)]
    return {
        "triggered_count": len(triggered),
        "triggered_ids": sorted(str(sample["patient_id"]) for sample in triggered),
        "by_error_type": counter_dict(Counter(error_type(sample) for sample in triggered)),
        "by_label": counter_dict(Counter(int(sample["label"]) for sample in triggered)),
        "by_prediction": counter_dict(
            Counter(int(sample["fusion_prediction"]["pred"]) for sample in triggered)
        ),
    }


def state_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "pred": int(sample["fusion_prediction"]["pred"]),
        "correct": bool(is_correct(sample)),
        "fusion_abnormal": round_float(fused_abnormal(sample)),
        "top_weight_view": top_view(sample),
        "triggered": bool(axial_aux_trigger(sample)),
        "views": {
            view: {
                "abnormal": round_float(abnormal_prob(sample, view)),
                "fusion_weight": round_float(fusion_weight(sample, view)),
                "pred": int(sample["views"][view]["pred"]),
                "true_margin": round_float(float(sample["views"][view]["true_margin"])),
            }
            for view in VIEWS
        },
    }


def changed_case_detail(
    patient_id: str,
    base_sample: dict[str, Any],
    candidate_sample: dict[str, Any],
    combo_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detail = {
        "patient_id": patient_id,
        "label": int(base_sample["label"]),
        "base_error_type": error_type(base_sample),
        "base": state_for_sample(base_sample),
        "candidate": state_for_sample(candidate_sample),
        "deltas": {
            "fusion_abnormal": round_float(fused_abnormal(candidate_sample) - fused_abnormal(base_sample)),
            "views": {
                view: {
                    "abnormal": round_float(abnormal_prob(candidate_sample, view) - abnormal_prob(base_sample, view)),
                    "fusion_weight": round_float(fusion_weight(candidate_sample, view) - fusion_weight(base_sample, view)),
                }
                for view in VIEWS
            },
        },
    }
    if combo_sample is not None:
        detail["dfr116_combo"] = state_for_sample(combo_sample)
    return detail


def compare_predictions(
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    combo: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fixed: list[str] = []
    broken: list[str] = []
    pred_changed: list[str] = []
    top_changed: list[str] = []
    for patient_id in sorted(base.keys() & candidate.keys()):
        base_sample = base[patient_id]
        candidate_sample = candidate[patient_id]
        if not is_correct(base_sample) and is_correct(candidate_sample):
            fixed.append(patient_id)
        if is_correct(base_sample) and not is_correct(candidate_sample):
            broken.append(patient_id)
        if int(base_sample["fusion_prediction"]["pred"]) != int(candidate_sample["fusion_prediction"]["pred"]):
            pred_changed.append(patient_id)
        if top_view(base_sample) != top_view(candidate_sample):
            top_changed.append(patient_id)

    def detail(patient_id: str) -> dict[str, Any]:
        combo_sample = combo.get(patient_id) if combo is not None else None
        return changed_case_detail(patient_id, base[patient_id], candidate[patient_id], combo_sample)

    return {
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "net_delta": len(fixed) - len(broken),
        "pred_changed_count": len(pred_changed),
        "top_changed_count": len(top_changed),
        "fixed_ids": fixed,
        "broken_ids": broken,
        "pred_changed_ids": pred_changed,
        "top_changed_ids": top_changed,
        "fixed_details": [detail(patient_id) for patient_id in fixed],
        "broken_details": [detail(patient_id) for patient_id in broken],
        "pred_changed_details": [detail(patient_id) for patient_id in pred_changed],
    }


def drift_records(
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for patient_id in sorted(base.keys() & candidate.keys()):
        base_sample = base[patient_id]
        candidate_sample = candidate[patient_id]
        records.append(
            {
                "patient_id": patient_id,
                "label": int(base_sample["label"]),
                "base_error_type": error_type(base_sample),
                "base_triggered": bool(axial_aux_trigger(base_sample)),
                "candidate_triggered": bool(axial_aux_trigger(candidate_sample)),
                "base_correct": bool(is_correct(base_sample)),
                "candidate_correct": bool(is_correct(candidate_sample)),
                "fusion_abnormal_delta": fused_abnormal(candidate_sample) - fused_abnormal(base_sample),
                "axial_abnormal_delta": abnormal_prob(candidate_sample, "axial") - abnormal_prob(base_sample, "axial"),
                "coronal_abnormal_delta": abnormal_prob(candidate_sample, "coronal") - abnormal_prob(base_sample, "coronal"),
                "sagittal_abnormal_delta": abnormal_prob(candidate_sample, "sagittal") - abnormal_prob(base_sample, "sagittal"),
                "axial_weight_delta": fusion_weight(candidate_sample, "axial") - fusion_weight(base_sample, "axial"),
                "coronal_weight_delta": fusion_weight(candidate_sample, "coronal") - fusion_weight(base_sample, "coronal"),
                "sagittal_weight_delta": fusion_weight(candidate_sample, "sagittal") - fusion_weight(base_sample, "sagittal"),
            }
        )
    return records


def drift_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "fusion_abnormal_delta",
        "axial_abnormal_delta",
        "coronal_abnormal_delta",
        "sagittal_abnormal_delta",
        "axial_weight_delta",
        "coronal_weight_delta",
        "sagittal_weight_delta",
    ]
    result: dict[str, Any] = {
        "overall": {
            metric: quantiles([float(record[metric]) for record in records])
            for metric in metric_names
        }
    }
    group_specs = {
        "by_label": lambda record: f"label{record['label']}",
        "by_base_error_type": lambda record: str(record["base_error_type"]),
        "by_base_triggered": lambda record: "triggered" if record["base_triggered"] else "not_triggered",
    }
    for group_name, grouper in group_specs.items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(grouper(record), []).append(record)
        result[group_name] = {
            key: {
                metric: quantiles([float(record[metric]) for record in items])
                for metric in metric_names
            }
            for key, items in sorted(grouped.items())
        }

    result["largest_axial_abnormal_increases"] = [
        {
            "patient_id": record["patient_id"],
            "label": int(record["label"]),
            "base_error_type": str(record["base_error_type"]),
            "base_triggered": bool(record["base_triggered"]),
            "candidate_triggered": bool(record["candidate_triggered"]),
            "base_correct": bool(record["base_correct"]),
            "candidate_correct": bool(record["candidate_correct"]),
            "axial_abnormal_delta": round_float(float(record["axial_abnormal_delta"])),
            "fusion_abnormal_delta": round_float(float(record["fusion_abnormal_delta"])),
        }
        for record in sorted(records, key=lambda item: float(item["axial_abnormal_delta"]), reverse=True)[:10]
    ]
    result["largest_axial_abnormal_decreases"] = [
        {
            "patient_id": record["patient_id"],
            "label": int(record["label"]),
            "base_error_type": str(record["base_error_type"]),
            "base_triggered": bool(record["base_triggered"]),
            "candidate_triggered": bool(record["candidate_triggered"]),
            "base_correct": bool(record["base_correct"]),
            "candidate_correct": bool(record["candidate_correct"]),
            "axial_abnormal_delta": round_float(float(record["axial_abnormal_delta"])),
            "fusion_abnormal_delta": round_float(float(record["fusion_abnormal_delta"])),
        }
        for record in sorted(records, key=lambda item: float(item["axial_abnormal_delta"]))[:10]
    ]
    return result


def dfr116_fixed_target_status(
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    combo: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixed_by_combo = [
        patient_id
        for patient_id in sorted(base.keys() & combo.keys())
        if not is_correct(base[patient_id]) and is_correct(combo[patient_id])
    ]
    details = []
    for patient_id in fixed_by_combo:
        candidate_sample = candidate.get(patient_id)
        details.append(
            {
                "patient_id": patient_id,
                "label": int(base[patient_id]["label"]),
                "base": state_for_sample(base[patient_id]),
                "dfr129": state_for_sample(candidate_sample) if candidate_sample is not None else None,
                "dfr116_combo": state_for_sample(combo[patient_id]),
                "dfr129_fixed": bool(candidate_sample is not None and is_correct(candidate_sample)),
            }
        )
    return {
        "dfr116_fixed_count": len(fixed_by_combo),
        "dfr116_fixed_ids": fixed_by_combo,
        "dfr129_fixed_these_count": sum(1 for item in details if item["dfr129_fixed"]),
        "details": details,
    }


def load_inputs(repo_root: Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    dfr25_path = repo_root / DFR25_TELEMETRY["42"]
    dfr129_path = repo_root / DFR129_BEST_TELEMETRY
    dfr116_path = repo_root / DFR116_TELEMETRY["42"]
    monitor_path = repo_root / DFR129_MONITOR_REPORT
    monitor = load_json(monitor_path) if monitor_path.exists() else {}
    return (
        dfr25_path,
        load_json(dfr25_path),
        dfr129_path,
        load_json(dfr129_path),
        dfr116_path,
        load_json(dfr116_path),
        monitor,
    )


def build_report(repo_root: Path) -> dict[str, Any]:
    dfr25_path, dfr25, dfr129_path, dfr129, dfr116_path, dfr116, monitor = load_inputs(repo_root)
    dfr25_samples = sample_map(dfr25)
    dfr129_samples = sample_map(dfr129)
    dfr116_samples = sample_map(dfr116)

    dfr129_vs_dfr25 = compare_predictions(dfr25_samples, dfr129_samples, dfr116_samples)
    dfr116_vs_dfr25 = compare_predictions(dfr25_samples, dfr116_samples, dfr116_samples)
    drifts = drift_records(dfr25_samples, dfr129_samples)

    off_trigger_broken = [
        patient_id
        for patient_id in dfr129_vs_dfr25["broken_ids"]
        if not axial_aux_trigger(dfr25_samples[patient_id])
    ]
    intended_trigger_fixed = [
        patient_id
        for patient_id in dfr129_vs_dfr25["fixed_ids"]
        if axial_aux_trigger(dfr25_samples[patient_id])
    ]

    assessment = {
        "status": "discard_stop_train_time_axial_aux",
        "main_reason": (
            "DFR-129 tied DFR-25 seed42 accuracy but did not move any sample's top fusion route away "
            "from axial; the only fixed sample was offset by an off-trigger broken sample."
        ),
        "should_continue_weight_or_threshold_sweep": False,
        "evidence": {
            "dfr129_net_delta_vs_dfr25": int(dfr129_vs_dfr25["net_delta"]),
            "dfr129_top_changed_count": int(dfr129_vs_dfr25["top_changed_count"]),
            "off_trigger_broken_ids": off_trigger_broken,
            "intended_trigger_fixed_ids": intended_trigger_fixed,
            "dfr116_fixed_targets_missed_by_dfr129": [
                item["patient_id"]
                for item in dfr116_fixed_target_status(dfr25_samples, dfr129_samples, dfr116_samples)["details"]
                if not item["dfr129_fixed"]
            ],
        },
        "next_recommendation": (
            "Do not extend DFR-129 as a training auxiliary. Return to DFR-116 style eval-side "
            "calibration or run a separate sampling/evidence audit for remaining clean FN/FP cases."
        ),
    }

    return {
        "analysis": "dfr130_axial_aux_drift_audit",
        "inputs": {
            "dfr25_seed42": str(dfr25_path.relative_to(repo_root)),
            "dfr129_best_trial1": str(dfr129_path.relative_to(repo_root)),
            "dfr116_combo_seed42": str(dfr116_path.relative_to(repo_root)),
            "dfr129_monitor_report": DFR129_MONITOR_REPORT if monitor else None,
        },
        "trigger_thresholds": TRIGGER_THRESHOLDS,
        "telemetry_summary": {
            "dfr25_seed42": telemetry_summary(dfr25_path, dfr25),
            "dfr129_best_trial1": telemetry_summary(dfr129_path, dfr129),
            "dfr116_combo_seed42": telemetry_summary(dfr116_path, dfr116),
        },
        "dfr129_monitor_best": monitor.get("best_by_accuracy") if monitor else None,
        "trigger_summary": {
            "dfr25_seed42": trigger_summary(dfr25_samples),
            "dfr129_best_trial1": trigger_summary(dfr129_samples),
        },
        "prediction_comparison": {
            "dfr129_vs_dfr25": dfr129_vs_dfr25,
            "dfr116_combo_vs_dfr25": {
                key: value
                for key, value in dfr116_vs_dfr25.items()
                if key not in {"fixed_details", "broken_details", "pred_changed_details"}
            },
        },
        "dfr116_fixed_target_status_under_dfr129": dfr116_fixed_target_status(
            dfr25_samples,
            dfr129_samples,
            dfr116_samples,
        ),
        "drift_distribution": drift_distribution(drifts),
        "assessment": assessment,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DFR-129 axial auxiliary drift.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report = build_report(repo_root)
    output = repo_root / args.output
    save_json(output, report)

    dfr129_summary = report["telemetry_summary"]["dfr129_best_trial1"]
    comparison = report["prediction_comparison"]["dfr129_vs_dfr25"]
    trigger = report["trigger_summary"]["dfr25_seed42"]
    print(f"Saved DFR-130 axial aux drift audit to: {output}")
    print("dfr129_best", dfr129_summary["metrics"], dfr129_summary["top_weight_count"])
    print(
        "dfr129_vs_dfr25",
        "fixed=",
        comparison["fixed_count"],
        "broken=",
        comparison["broken_count"],
        "net=",
        comparison["net_delta"],
        "top_changed=",
        comparison["top_changed_count"],
    )
    print("dfr25_trigger", trigger["triggered_count"], trigger["by_error_type"], trigger["triggered_ids"])
    print("assessment", report["assessment"]["status"])


if __name__ == "__main__":
    main()
