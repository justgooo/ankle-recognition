#!/usr/bin/env python3
"""Audit DFR-105 targeted evidence-rank auxiliary trigger coverage.

This analysis-only follow-up reads existing validation telemetry and replays the
DFR-105 trigger rule:

    best_nonaxial_log_prob[class] - axial_log_prob[class] >= gap

It reports whether the resulting supervision would actually land on DFR-25
errors or non-axial true-margin samples, and whether the DFR-105 best trial
turned those targets into top or near-top routing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


VIEWS = ("axial", "coronal", "sagittal")
NONAXIAL = ("coronal", "sagittal")
CLASSES = (0, 1)

DFR25_SEED42 = (
    "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR105_BEST = (
    "runs/optuna_main_autoloop/iter_0010_20260513_010547/"
    "trials/trial_0000/run/fusion_weight_analysis.json"
)
DFR105_STUDY = "runs/optuna_main_autoloop/iter_0010_20260513_010547"
DEFAULT_GAPS = (0.02, 0.05, 0.10)
NEAR_TOP_TOLERANCE = 0.05


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def samples_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in load_json(path)["samples"]}


def top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str:
    values = sample.get(key) or []
    return str(values[0]) if values else "none"


def label(sample: dict[str, Any]) -> int:
    return int(sample["label"])


def fusion_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def fusion_pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def view_correct(sample: dict[str, Any], view: str) -> bool:
    return bool(sample["views"][view]["correct"])


def view_pred(sample: dict[str, Any], view: str) -> int:
    return int(sample["views"][view]["pred"])


def abnormal_prob(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["abnormal_prob"])


def class_prob(sample: dict[str, Any], view: str, class_index: int) -> float:
    prob = abnormal_prob(sample, view)
    return prob if int(class_index) == 1 else 1.0 - prob


def class_log_prob(sample: dict[str, Any], view: str, class_index: int) -> float:
    return math.log(max(class_prob(sample, view, class_index), 1e-8))


def true_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["true_margin"])


def pred_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["pred_margin"])


def fusion_weight(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["fusion_weight"])


def confidence_logit(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["confidence_logit"])


def error_type(sample: dict[str, Any]) -> str:
    if fusion_correct(sample):
        return "TP" if label(sample) == 1 else "TN"
    if label(sample) == 1 and fusion_pred(sample) == 0:
        return "FN"
    if label(sample) == 0 and fusion_pred(sample) == 1:
        return "FP"
    return "ERR"


def view_correct_pattern(sample: dict[str, Any]) -> str:
    return "".join(view[0].upper() if view_correct(sample, view) else "-" for view in VIEWS)


def summarize_telemetry(path: Path) -> dict[str, Any]:
    telemetry = load_json(path)
    per_view = {
        item["view"]: {
            "mean_fusion_weight": round_float(item["mean_fusion_weight"]),
            "top_weight_count": int(item["top_weight_count"]),
            "top_weight_rate": round_float(item["top_weight_rate"]),
            "top_true_margin_count": int(item["top_true_margin_count"]),
            "top_true_margin_rate": round_float(item["top_true_margin_rate"]),
            "accuracy": round_float(item["metrics"]["accuracy"]),
            "f1": round_float(item["metrics"]["f1"]),
        }
        for item in telemetry["per_view"]
    }
    metrics = telemetry["summary"]["full_fusion_metrics"]
    return {
        "path": str(path),
        "accuracy": round_float(metrics["accuracy"]),
        "auc": round_float(metrics["auc"]),
        "f1": round_float(metrics["f1"]),
        "top_weight_view_distribution": telemetry["summary"]["top_weight_view_distribution"],
        "top_true_margin_view_distribution": telemetry["summary"]["top_true_margin_view_distribution"],
        "top_weight_hit_rate_true_margin": round_float(
            telemetry["summary"]["top_weight_hit_rate"]["true_margin"]
        ),
        "per_view": per_view,
    }


def summarize_cases(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "labels": {},
            "error_types": {},
            "top_weight": {},
            "top_true_margin": {},
            "view_correct_patterns": {},
        }
    return {
        "count": len(samples),
        "labels": counter_dict(Counter(label(sample) for sample in samples)),
        "error_types": counter_dict(Counter(error_type(sample) for sample in samples)),
        "top_weight": counter_dict(Counter(top_view(sample) for sample in samples)),
        "top_true_margin": counter_dict(Counter(top_view(sample, "top_true_margin_views") for sample in samples)),
        "view_correct_patterns": counter_dict(Counter(view_correct_pattern(sample) for sample in samples)),
        "mean_axial_weight": round_float(mean(fusion_weight(sample, "axial") for sample in samples)),
        "mean_coronal_weight": round_float(mean(fusion_weight(sample, "coronal") for sample in samples)),
        "mean_sagittal_weight": round_float(mean(fusion_weight(sample, "sagittal") for sample in samples)),
        "mean_best_nonaxial_true_margin": round_float(
            mean(max(true_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
        "mean_best_nonaxial_pred_margin": round_float(
            mean(max(pred_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
    }


def dfr105_class_targets(sample: dict[str, Any], gap: float) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for class_index in CLASSES:
        axial_log = class_log_prob(sample, "axial", class_index)
        scored = [
            (
                class_log_prob(sample, view, class_index) - axial_log,
                view,
            )
            for view in NONAXIAL
        ]
        advantage, target_view = max(scored, key=lambda item: item[0])
        targets[class_index] = {
            "active": bool(advantage >= gap),
            "target_view": target_view,
            "advantage": float(advantage),
            "axial_log_prob": axial_log,
            "target_log_prob": class_log_prob(sample, target_view, class_index),
        }
    return targets


def classify_trigger(sample: dict[str, Any], gap: float) -> dict[str, Any]:
    targets = dfr105_class_targets(sample, gap)
    true_class = label(sample)
    other_class = 1 - true_class
    true_active = bool(targets[true_class]["active"])
    other_active = bool(targets[other_class]["active"])
    if true_active and other_active:
        mode = "both_true_and_non_target"
    elif true_active:
        mode = "true_class_only"
    elif other_active:
        mode = "non_target_only"
    else:
        mode = "inactive"
    target_view = str(targets[true_class]["target_view"]) if true_active else "axial"
    return {
        "class_targets": targets,
        "true_class_active": true_active,
        "non_target_class_active": other_active,
        "mode": mode,
        "true_class_target_view": target_view,
        "true_class_advantage": float(targets[true_class]["advantage"]),
        "non_target_advantage": float(targets[other_class]["advantage"]),
    }


def is_near_top(sample: dict[str, Any], view: str, tolerance: float) -> bool:
    target_weight = fusion_weight(sample, view)
    top_weight = max(fusion_weight(sample, item) for item in VIEWS)
    return target_weight >= top_weight - tolerance


def best_nonaxial_true_view(sample: dict[str, Any]) -> str:
    return max(NONAXIAL, key=lambda view: true_margin(sample, view))


def evaluate_gap(
    source_samples: dict[str, dict[str, Any]],
    dfr25_samples: dict[str, dict[str, Any]],
    dfr105_samples: dict[str, dict[str, Any]],
    gap: float,
) -> dict[str, Any]:
    shared_ids = sorted(set(source_samples) & set(dfr25_samples) & set(dfr105_samples))
    dfr25_wrong_ids = [pid for pid in shared_ids if not fusion_correct(dfr25_samples[pid])]
    dfr25_nonaxial_true_margin_ids = [
        pid for pid in shared_ids if top_view(dfr25_samples[pid], "top_true_margin_views") in NONAXIAL
    ]

    records: list[dict[str, Any]] = []
    fired_ids: list[str] = []
    true_active_ids: list[str] = []
    non_target_only_ids: list[str] = []
    both_active_ids: list[str] = []
    helpful_ids: list[str] = []
    harmful_ids: list[str] = []
    confirmed_scope_ids: list[str] = []
    target_views: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    target_labels: Counter[int] = Counter()
    target_error_types: Counter[str] = Counter()
    target_top_after_dfr105: Counter[str] = Counter()
    target_near_top_after_dfr105 = 0
    target_became_top_after_dfr105 = 0
    fixed_triggered = 0
    broken_triggered = 0

    for pid in shared_ids:
        source = source_samples[pid]
        dfr25 = dfr25_samples[pid]
        dfr105 = dfr105_samples[pid]
        trigger = classify_trigger(source, gap)
        mode = str(trigger["mode"])
        modes[mode] += 1
        true_target_view = str(trigger["true_class_target_view"])
        true_active = bool(trigger["true_class_active"])
        if true_active:
            fired_ids.append(pid)
            true_active_ids.append(pid)
            target_views[true_target_view] += 1
            target_labels[label(dfr25)] += 1
            target_error_types[error_type(dfr25)] += 1
            if not fusion_correct(dfr25) and view_correct(dfr25, true_target_view):
                helpful_ids.append(pid)
            if fusion_correct(dfr25) and not view_correct(dfr25, true_target_view):
                harmful_ids.append(pid)
            if (not fusion_correct(dfr25)) or top_view(dfr25, "top_true_margin_views") in NONAXIAL:
                confirmed_scope_ids.append(pid)
            if top_view(dfr105) == true_target_view:
                target_became_top_after_dfr105 += 1
            if is_near_top(dfr105, true_target_view, NEAR_TOP_TOLERANCE):
                target_near_top_after_dfr105 += 1
            target_top_after_dfr105[top_view(dfr105)] += 1
            if not fusion_correct(dfr25) and fusion_correct(dfr105):
                fixed_triggered += 1
            if fusion_correct(dfr25) and not fusion_correct(dfr105):
                broken_triggered += 1
        if mode == "non_target_only":
            non_target_only_ids.append(pid)
        elif mode == "both_true_and_non_target":
            both_active_ids.append(pid)

        records.append(
            {
                "patient_id": pid,
                "label": label(dfr25),
                "dfr25_correct": fusion_correct(dfr25),
                "dfr25_error_type": error_type(dfr25),
                "dfr25_top_true_margin": top_view(dfr25, "top_true_margin_views"),
                "dfr25_view_correct_pattern": view_correct_pattern(dfr25),
                "true_class_active": true_active,
                "non_target_class_active": bool(trigger["non_target_class_active"]),
                "trigger_mode": mode,
                "true_class_target_view": true_target_view,
                "true_class_advantage": round_float(trigger["true_class_advantage"]),
                "non_target_advantage": round_float(trigger["non_target_advantage"]),
                "best_nonaxial_true_view": best_nonaxial_true_view(dfr25),
                "dfr105_correct": fusion_correct(dfr105),
                "dfr105_top_weight": top_view(dfr105),
                "dfr105_target_view_is_top": bool(true_active and top_view(dfr105) == true_target_view),
                "dfr105_target_view_near_top": bool(
                    true_active and is_near_top(dfr105, true_target_view, NEAR_TOP_TOLERANCE)
                ),
                "dfr105_weight_axial": round_float(fusion_weight(dfr105, "axial")),
                "dfr105_weight_coronal": round_float(fusion_weight(dfr105, "coronal")),
                "dfr105_weight_sagittal": round_float(fusion_weight(dfr105, "sagittal")),
            }
        )

    fired_set = set(fired_ids)
    non_target_only_set = set(non_target_only_ids)
    both_active_set = set(both_active_ids)
    dfr25_wrong_set = set(dfr25_wrong_ids)
    nonaxial_true_margin_set = set(dfr25_nonaxial_true_margin_ids)
    confirmed_scope_set = set(confirmed_scope_ids)
    true_active_set = set(true_active_ids)

    return {
        "gap": round_float(gap),
        "source_samples": len(shared_ids),
        "dfr25_wrong_total": len(dfr25_wrong_ids),
        "dfr25_nonaxial_true_margin_total": len(dfr25_nonaxial_true_margin_ids),
        "true_class_triggered_samples": len(true_active_ids),
        "true_class_triggered_rate": round_float(safe_rate(len(true_active_ids), len(shared_ids))),
        "true_class_target_views": counter_dict(target_views),
        "true_class_triggered_labels": counter_dict(target_labels),
        "true_class_triggered_dfr25_error_types": counter_dict(target_error_types),
        "trigger_modes": counter_dict(modes),
        "non_target_only_samples": len(non_target_only_ids),
        "both_true_and_non_target_samples": len(both_active_ids),
        "dfr25_wrong_covered": len(fired_set & dfr25_wrong_set),
        "dfr25_wrong_coverage_rate": round_float(safe_rate(len(fired_set & dfr25_wrong_set), len(dfr25_wrong_ids))),
        "dfr25_nonaxial_true_margin_covered": len(fired_set & nonaxial_true_margin_set),
        "dfr25_nonaxial_true_margin_coverage_rate": round_float(
            safe_rate(len(fired_set & nonaxial_true_margin_set), len(dfr25_nonaxial_true_margin_ids))
        ),
        "confirmed_scope_triggered": len(confirmed_scope_set),
        "confirmed_scope_rate_within_triggered": round_float(
            safe_rate(len(confirmed_scope_set), len(true_active_ids))
        ),
        "helpful_if_promoted": len(helpful_ids),
        "harmful_if_promoted": len(harmful_ids),
        "net_oracle_delta_upper_bound": len(helpful_ids) - len(harmful_ids),
        "helpful_rate_within_triggered": round_float(safe_rate(len(helpful_ids), len(true_active_ids))),
        "harmful_rate_within_triggered": round_float(safe_rate(len(harmful_ids), len(true_active_ids))),
        "dfr105_target_became_top": target_became_top_after_dfr105,
        "dfr105_target_became_top_rate": round_float(
            safe_rate(target_became_top_after_dfr105, len(true_active_ids))
        ),
        "dfr105_target_near_top": target_near_top_after_dfr105,
        "dfr105_target_near_top_rate": round_float(
            safe_rate(target_near_top_after_dfr105, len(true_active_ids))
        ),
        "dfr105_top_weight_on_triggered": counter_dict(target_top_after_dfr105),
        "dfr105_fixed_triggered_dfr25_errors": fixed_triggered,
        "dfr105_broken_triggered_dfr25_correct": broken_triggered,
        "case_summaries": {
            "true_class_triggered": summarize_cases([dfr25_samples[pid] for pid in sorted(true_active_set)]),
            "confirmed_scope_triggered": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(confirmed_scope_set)]
            ),
            "non_target_only": summarize_cases([dfr25_samples[pid] for pid in sorted(non_target_only_set)]),
            "both_true_and_non_target": summarize_cases([dfr25_samples[pid] for pid in sorted(both_active_set)]),
            "helpful_if_promoted": summarize_cases([dfr25_samples[pid] for pid in sorted(helpful_ids)]),
            "harmful_if_promoted": summarize_cases([dfr25_samples[pid] for pid in sorted(harmful_ids)]),
            "dfr25_wrong_uncovered": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(dfr25_wrong_set - fired_set)]
            ),
            "nonaxial_true_margin_uncovered": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(nonaxial_true_margin_set - fired_set)]
            ),
        },
        "notable_case_ids": {
            "true_class_triggered": sorted(true_active_set),
            "confirmed_scope_triggered": sorted(confirmed_scope_set),
            "dfr25_wrong_covered": sorted(fired_set & dfr25_wrong_set),
            "dfr25_wrong_uncovered": sorted(dfr25_wrong_set - fired_set),
            "nonaxial_true_margin_covered": sorted(fired_set & nonaxial_true_margin_set),
            "nonaxial_true_margin_uncovered": sorted(nonaxial_true_margin_set - fired_set),
            "helpful_if_promoted": sorted(helpful_ids),
            "harmful_if_promoted": sorted(harmful_ids),
        },
        "sample_records": records,
    }


def dfr105_trial_metrics(study_dir: Path) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for trial_path in sorted(study_dir.glob("trials/trial_*/trial.json")):
        trial = load_json(trial_path)
        params = trial.get("params", {})
        trials.append(
            {
                "trial_number": int(trial.get("trial_number", -1)),
                "status": trial.get("status"),
                "val_accuracy": round_float(float(trial.get("val_accuracy", 0.0))),
                "val_auc": round_float(float(trial.get("val_auc", 0.0))),
                "val_f1": round_float(float(trial.get("val_f1", 0.0))),
                "peak_vram_mb": round_float(float(trial.get("peak_vram_mb", 0.0)), 3),
                "gap": params.get("runtime_env.ANKLE_DECISION_GATE_TARGETED_EVIDENCE_RANK_AUX_GAP"),
                "run_dir": trial.get("paths", {}).get("run_dir"),
            }
        )
    return trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autoresearch_logs/dfr105_targeted_aux_audit.json"),
    )
    parser.add_argument(
        "--gaps",
        type=float,
        nargs="*",
        default=list(DEFAULT_GAPS),
        help="DFR-105 targeted evidence rank gaps to replay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dfr25_path = repo_root / DFR25_SEED42
    dfr105_path = repo_root / DFR105_BEST
    study_dir = repo_root / DFR105_STUDY
    dfr25_samples = samples_by_id(dfr25_path)
    dfr105_samples = samples_by_id(dfr105_path)

    report = {
        "analysis": "dfr105_targeted_aux_trigger_audit",
        "description": (
            "Analysis-only audit of DFR-105 targeted non-axial evidence-rank auxiliary. "
            "It replays the aux trigger on DFR-25 seed42 and DFR-105 best-trial "
            "validation telemetry, then joins the targets with DFR-25 errors, "
            "non-axial true-margin samples, and DFR-105 final routing."
        ),
        "dfr25_seed42_input": DFR25_SEED42,
        "dfr105_best_input": DFR105_BEST,
        "dfr105_study": DFR105_STUDY,
        "near_top_tolerance": NEAR_TOP_TOLERANCE,
        "dfr25_seed42_telemetry": summarize_telemetry(dfr25_path),
        "dfr105_best_telemetry": summarize_telemetry(dfr105_path),
        "dfr105_trials": dfr105_trial_metrics(study_dir),
        "gap_audits": {
            "dfr25_seed42_source": {
                f"{gap:.2f}": evaluate_gap(dfr25_samples, dfr25_samples, dfr105_samples, gap)
                for gap in args.gaps
            },
            "dfr105_best_source": {
                f"{gap:.2f}": evaluate_gap(dfr105_samples, dfr25_samples, dfr105_samples, gap)
                for gap in args.gaps
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    dfr25 = report["dfr25_seed42_telemetry"]
    dfr105 = report["dfr105_best_telemetry"]
    best_gap = report["gap_audits"]["dfr25_seed42_source"]["0.02"]
    print(f"Saved DFR-105 targeted-aux audit to: {args.output}")
    print(
        "DFR25 seed42: "
        f"acc={dfr25['accuracy']:.6f} auc={dfr25['auc']:.6f} f1={dfr25['f1']:.6f} "
        f"mean_weights={dfr25['per_view']['axial']['mean_fusion_weight']:.6f}/"
        f"{dfr25['per_view']['coronal']['mean_fusion_weight']:.6f}/"
        f"{dfr25['per_view']['sagittal']['mean_fusion_weight']:.6f} "
        f"top={dfr25['top_weight_view_distribution']}"
    )
    print(
        "DFR105 best: "
        f"acc={dfr105['accuracy']:.6f} auc={dfr105['auc']:.6f} f1={dfr105['f1']:.6f} "
        f"mean_weights={dfr105['per_view']['axial']['mean_fusion_weight']:.6f}/"
        f"{dfr105['per_view']['coronal']['mean_fusion_weight']:.6f}/"
        f"{dfr105['per_view']['sagittal']['mean_fusion_weight']:.6f} "
        f"top={dfr105['top_weight_view_distribution']}"
    )
    print(
        "DFR25-source gap=0.02: "
        f"triggered={best_gap['true_class_triggered_samples']} "
        f"target_views={best_gap['true_class_target_views']} "
        f"wrong_covered={best_gap['dfr25_wrong_covered']}/{best_gap['dfr25_wrong_total']} "
        f"nonaxial_true_margin_covered="
        f"{best_gap['dfr25_nonaxial_true_margin_covered']}/"
        f"{best_gap['dfr25_nonaxial_true_margin_total']} "
        f"confirmed_scope_rate={best_gap['confirmed_scope_rate_within_triggered']:.6f} "
        f"target_top_after_dfr105={best_gap['dfr105_target_became_top']} "
        f"target_near_top_after_dfr105={best_gap['dfr105_target_near_top']}"
    )


if __name__ == "__main__":
    main()
