#!/usr/bin/env python3
"""Audit DFR-107 confirmed-target rank-margin auxiliary encoding.

This analysis-only follow-up replays the DFR-107 confirmed-target rule on
existing validation telemetry, then computes the effective CrossEntropy
gradient direction produced by the current train.py auxiliary hook.

The key question is whether the binary patient-label CE encoding promotes the
intended non-axial target, cancels it, or applies axial-reinforcing pressure
when only the non-target class fires.
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
DFR107_BEST = (
    "runs/optuna_main_autoloop/iter_0002_20260513_023419/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR107_STUDY = "runs/optuna_main_autoloop/iter_0002_20260513_023419"
DEFAULT_GAP = 0.02
DEFAULT_MARGIN = 0.25
NEAR_TOP_TOLERANCE = 0.05
EPS = 1e-8
EFFECT_EPS = 1e-9


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def round_float(value: float, digits: int = 6) -> float:
    if math.isinf(value) or math.isnan(value):
        return float(value)
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


def abnormal_prob(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["abnormal_prob"])


def class_prob(sample: dict[str, Any], view: str, class_index: int) -> float:
    prob = abnormal_prob(sample, view)
    return prob if int(class_index) == 1 else 1.0 - prob


def class_log_prob(sample: dict[str, Any], view: str, class_index: int) -> float:
    return math.log(max(class_prob(sample, view, class_index), EPS))


def class_margin(sample: dict[str, Any], view: str, class_index: int) -> float:
    this_prob = class_prob(sample, view, class_index)
    other_prob = class_prob(sample, view, 1 - class_index)
    return math.log(max(this_prob, EPS)) - math.log(max(other_prob, EPS))


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


def best_nonaxial_true_view(sample: dict[str, Any]) -> str:
    return max(NONAXIAL, key=lambda view: true_margin(sample, view))


def is_near_top(sample: dict[str, Any], view: str, tolerance: float) -> bool:
    target_weight = fusion_weight(sample, view)
    top_weight = max(fusion_weight(sample, item) for item in VIEWS)
    return target_weight >= top_weight - tolerance


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
        "mean_axial_confidence_logit": round_float(mean(confidence_logit(sample, "axial") for sample in samples)),
        "mean_best_nonaxial_confidence_gap": round_float(
            mean(
                max(confidence_logit(sample, view) for view in NONAXIAL)
                - confidence_logit(sample, "axial")
                for sample in samples
            )
        ),
        "mean_best_nonaxial_true_margin": round_float(
            mean(max(true_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
        "mean_best_nonaxial_pred_margin": round_float(
            mean(max(pred_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
    }


def confirmed_class_targets(sample: dict[str, Any], gap: float, margin: float) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    axial_log_weight = math.log(max(fusion_weight(sample, "axial"), EPS))
    for class_index in CLASSES:
        axial_log_prob = class_log_prob(sample, "axial", class_index)
        scored = []
        for view in NONAXIAL:
            if class_margin(sample, view, class_index) < 0.0:
                advantage = -math.inf
            else:
                advantage = class_log_prob(sample, view, class_index) - axial_log_prob
            scored.append((advantage, view))
        best_advantage, target_view = max(scored, key=lambda item: item[0])
        active = bool(best_advantage >= gap)
        if active:
            target_log_weight = math.log(max(fusion_weight(sample, target_view), EPS))
            rank_margin_logit = target_log_weight - axial_log_weight - margin
        else:
            rank_margin_logit = 0.0
        targets[class_index] = {
            "active": active,
            "target_view": target_view,
            "advantage": float(best_advantage),
            "rank_margin_logit": float(rank_margin_logit),
            "target_class_margin": round_float(class_margin(sample, target_view, class_index)),
            "axial_log_prob": float(axial_log_prob),
            "target_log_prob": float(class_log_prob(sample, target_view, class_index)),
        }
    return targets


def softmax2(z0: float, z1: float) -> tuple[float, float]:
    base = max(z0, z1)
    e0 = math.exp(z0 - base)
    e1 = math.exp(z1 - base)
    denom = e0 + e1
    return e0 / denom, e1 / denom


def aux_ce_effect(sample: dict[str, Any], gap: float, margin: float) -> dict[str, Any]:
    targets = confirmed_class_targets(sample, gap, margin)
    aux_logits = {class_index: targets[class_index]["rank_margin_logit"] for class_index in CLASSES}
    prob0, prob1 = softmax2(aux_logits[0], aux_logits[1])
    probs = {0: prob0, 1: prob1}
    true_label = label(sample)
    grad_z = {
        class_index: probs[class_index] - (1.0 if class_index == true_label else 0.0)
        for class_index in CLASSES
    }
    grad_log_weight = {view: 0.0 for view in VIEWS}
    for class_index in CLASSES:
        if not targets[class_index]["active"]:
            continue
        target_view = str(targets[class_index]["target_view"])
        grad_log_weight[target_view] += grad_z[class_index]
        grad_log_weight["axial"] -= grad_z[class_index]

    descent_delta = {view: -value for view, value in grad_log_weight.items()}
    true_target = targets[true_label]
    true_active = bool(true_target["active"])
    non_target = targets[1 - true_label]
    non_target_active = bool(non_target["active"])
    true_target_view = str(true_target["target_view"])
    if true_active:
        target_vs_axial_descent = (
            descent_delta[true_target_view] - descent_delta["axial"]
        )
        if target_vs_axial_descent > EFFECT_EPS:
            effect_kind = "promotes_true_target_vs_axial"
        elif target_vs_axial_descent < -EFFECT_EPS:
            effect_kind = "reverses_true_target_vs_axial"
        else:
            effect_kind = "cancels_true_target_vs_axial"
    elif non_target_active:
        target_vs_axial_descent = None
        effect_kind = "non_target_only_reinforces_axial"
    else:
        target_vs_axial_descent = None
        effect_kind = "inactive"

    if true_active and non_target_active:
        if true_target_view == str(non_target["target_view"]):
            overlap_kind = "both_active_same_target_view"
        else:
            overlap_kind = "both_active_different_target_view"
    elif true_active:
        overlap_kind = "true_class_only"
    elif non_target_active:
        overlap_kind = "non_target_only"
    else:
        overlap_kind = "inactive"

    return {
        "class_targets": targets,
        "aux_logits": {str(key): round_float(value) for key, value in aux_logits.items()},
        "aux_probabilities": {"0": round_float(prob0), "1": round_float(prob1)},
        "grad_z": {str(key): round_float(value) for key, value in grad_z.items()},
        "descent_delta_log_weight": {
            view: round_float(value) for view, value in descent_delta.items()
        },
        "true_class_active": true_active,
        "non_target_class_active": non_target_active,
        "true_class_target_view": true_target_view if true_active else "axial",
        "non_target_class_target_view": str(non_target["target_view"]) if non_target_active else "none",
        "target_vs_axial_descent": (
            round_float(target_vs_axial_descent)
            if target_vs_axial_descent is not None
            else None
        ),
        "effect_kind": effect_kind,
        "overlap_kind": overlap_kind,
    }


def audit_source(
    source_samples: dict[str, dict[str, Any]],
    dfr25_samples: dict[str, dict[str, Any]],
    dfr107_samples: dict[str, dict[str, Any]],
    gap: float,
    margin: float,
) -> dict[str, Any]:
    shared_ids = sorted(set(source_samples) & set(dfr25_samples) & set(dfr107_samples))
    dfr25_wrong = {pid for pid in shared_ids if not fusion_correct(dfr25_samples[pid])}
    dfr25_nonaxial_true = {
        pid for pid in shared_ids if top_view(dfr25_samples[pid], "top_true_margin_views") in NONAXIAL
    }

    records: list[dict[str, Any]] = []
    true_active_ids: list[str] = []
    non_target_only_ids: list[str] = []
    both_same_ids: list[str] = []
    both_different_ids: list[str] = []
    promote_ids: list[str] = []
    cancel_ids: list[str] = []
    reverse_ids: list[str] = []
    helpful_ids: list[str] = []
    harmful_ids: list[str] = []
    normal_rescue_ids: list[str] = []
    normal_rescue_promote_ids: list[str] = []
    normal_rescue_cancel_ids: list[str] = []

    labels = Counter()
    true_target_views = Counter()
    effect_kinds = Counter()
    overlap_kinds = Counter()
    non_target_only_labels = Counter()
    target_top_after_dfr107 = Counter()
    target_near_top_after_dfr107 = 0

    for pid in shared_ids:
        source = source_samples[pid]
        dfr25 = dfr25_samples[pid]
        dfr107 = dfr107_samples[pid]
        effect = aux_ce_effect(source, gap, margin)
        true_active = bool(effect["true_class_active"])
        non_target_active = bool(effect["non_target_class_active"])
        effect_kind = str(effect["effect_kind"])
        overlap_kind = str(effect["overlap_kind"])
        true_target_view = str(effect["true_class_target_view"])

        effect_kinds[effect_kind] += 1
        overlap_kinds[overlap_kind] += 1
        if true_active:
            true_active_ids.append(pid)
            labels[label(dfr25)] += 1
            true_target_views[true_target_view] += 1
            target_top_after_dfr107[top_view(dfr107)] += 1
            if is_near_top(dfr107, true_target_view, NEAR_TOP_TOLERANCE):
                target_near_top_after_dfr107 += 1
            if not fusion_correct(dfr25) and view_correct(dfr25, true_target_view):
                helpful_ids.append(pid)
            if fusion_correct(dfr25) and not view_correct(dfr25, true_target_view):
                harmful_ids.append(pid)
            if label(dfr25) == 0 and not fusion_correct(dfr25) and view_correct(dfr25, true_target_view):
                normal_rescue_ids.append(pid)
            if effect_kind == "promotes_true_target_vs_axial":
                promote_ids.append(pid)
                if pid in normal_rescue_ids:
                    normal_rescue_promote_ids.append(pid)
            elif effect_kind == "cancels_true_target_vs_axial":
                cancel_ids.append(pid)
                if pid in normal_rescue_ids:
                    normal_rescue_cancel_ids.append(pid)
            elif effect_kind == "reverses_true_target_vs_axial":
                reverse_ids.append(pid)
        if non_target_active and not true_active:
            non_target_only_ids.append(pid)
            non_target_only_labels[label(dfr25)] += 1
        if overlap_kind == "both_active_same_target_view":
            both_same_ids.append(pid)
        elif overlap_kind == "both_active_different_target_view":
            both_different_ids.append(pid)

        records.append(
            {
                "patient_id": pid,
                "label": label(dfr25),
                "dfr25_correct": fusion_correct(dfr25),
                "dfr25_error_type": error_type(dfr25),
                "dfr25_top_true_margin": top_view(dfr25, "top_true_margin_views"),
                "dfr25_view_correct_pattern": view_correct_pattern(dfr25),
                "dfr25_best_nonaxial_true_view": best_nonaxial_true_view(dfr25),
                "source_top_weight": top_view(source),
                "source_weight_axial": round_float(fusion_weight(source, "axial")),
                "source_weight_coronal": round_float(fusion_weight(source, "coronal")),
                "source_weight_sagittal": round_float(fusion_weight(source, "sagittal")),
                "true_class_active": true_active,
                "non_target_class_active": non_target_active,
                "true_class_target_view": true_target_view,
                "non_target_class_target_view": effect["non_target_class_target_view"],
                "overlap_kind": overlap_kind,
                "effect_kind": effect_kind,
                "target_vs_axial_descent": effect["target_vs_axial_descent"],
                "aux_logits": effect["aux_logits"],
                "aux_probabilities": effect["aux_probabilities"],
                "descent_delta_log_weight": effect["descent_delta_log_weight"],
                "class_targets": effect["class_targets"],
                "helpful_if_promoted": bool(
                    true_active and not fusion_correct(dfr25) and view_correct(dfr25, true_target_view)
                ),
                "harmful_if_promoted": bool(
                    true_active and fusion_correct(dfr25) and not view_correct(dfr25, true_target_view)
                ),
                "dfr107_correct": fusion_correct(dfr107),
                "dfr107_top_weight": top_view(dfr107),
                "dfr107_target_view_is_top": bool(true_active and top_view(dfr107) == true_target_view),
                "dfr107_target_view_near_top": bool(
                    true_active and is_near_top(dfr107, true_target_view, NEAR_TOP_TOLERANCE)
                ),
                "dfr107_weight_axial": round_float(fusion_weight(dfr107, "axial")),
                "dfr107_weight_coronal": round_float(fusion_weight(dfr107, "coronal")),
                "dfr107_weight_sagittal": round_float(fusion_weight(dfr107, "sagittal")),
            }
        )

    true_active_set = set(true_active_ids)
    dfr25_wrong_covered = true_active_set & dfr25_wrong
    nonaxial_true_covered = true_active_set & dfr25_nonaxial_true
    confirmed_scope = true_active_set & (dfr25_wrong | dfr25_nonaxial_true)

    return {
        "gap": round_float(gap),
        "margin": round_float(margin),
        "source_samples": len(shared_ids),
        "dfr25_wrong_total": len(dfr25_wrong),
        "dfr25_nonaxial_true_margin_total": len(dfr25_nonaxial_true),
        "true_class_triggered_samples": len(true_active_ids),
        "true_class_triggered_rate": round_float(safe_rate(len(true_active_ids), len(shared_ids))),
        "true_class_triggered_labels": counter_dict(labels),
        "true_class_target_views": counter_dict(true_target_views),
        "effect_kinds": counter_dict(effect_kinds),
        "overlap_kinds": counter_dict(overlap_kinds),
        "non_target_only_samples": len(non_target_only_ids),
        "non_target_only_rate": round_float(safe_rate(len(non_target_only_ids), len(shared_ids))),
        "non_target_only_labels": counter_dict(non_target_only_labels),
        "both_active_same_target_view": len(both_same_ids),
        "both_active_different_target_view": len(both_different_ids),
        "true_target_promoted_by_ce_direction": len(promote_ids),
        "true_target_canceled_by_ce_direction": len(cancel_ids),
        "true_target_reversed_by_ce_direction": len(reverse_ids),
        "dfr25_wrong_covered": len(dfr25_wrong_covered),
        "dfr25_wrong_coverage_rate": round_float(safe_rate(len(dfr25_wrong_covered), len(dfr25_wrong))),
        "dfr25_nonaxial_true_margin_covered": len(nonaxial_true_covered),
        "dfr25_nonaxial_true_margin_coverage_rate": round_float(
            safe_rate(len(nonaxial_true_covered), len(dfr25_nonaxial_true))
        ),
        "confirmed_scope_triggered": len(confirmed_scope),
        "confirmed_scope_rate_within_triggered": round_float(
            safe_rate(len(confirmed_scope), len(true_active_ids))
        ),
        "helpful_if_promoted": len(helpful_ids),
        "harmful_if_promoted": len(harmful_ids),
        "normal_rescue_targets": len(normal_rescue_ids),
        "normal_rescue_promoted_by_ce_direction": len(normal_rescue_promote_ids),
        "normal_rescue_canceled_by_ce_direction": len(normal_rescue_cancel_ids),
        "dfr107_target_became_top": sum(
            1
            for pid in true_active_ids
            if top_view(dfr107_samples[pid]) == str(aux_ce_effect(source_samples[pid], gap, margin)["true_class_target_view"])
        ),
        "dfr107_target_became_top_rate": round_float(
            safe_rate(
                sum(
                    1
                    for pid in true_active_ids
                    if top_view(dfr107_samples[pid])
                    == str(aux_ce_effect(source_samples[pid], gap, margin)["true_class_target_view"])
                ),
                len(true_active_ids),
            )
        ),
        "dfr107_target_near_top": target_near_top_after_dfr107,
        "dfr107_target_near_top_rate": round_float(
            safe_rate(target_near_top_after_dfr107, len(true_active_ids))
        ),
        "dfr107_top_weight_on_triggered": counter_dict(target_top_after_dfr107),
        "case_summaries": {
            "true_class_triggered": summarize_cases([dfr25_samples[pid] for pid in sorted(true_active_set)]),
            "confirmed_scope_triggered": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(confirmed_scope)]
            ),
            "non_target_only": summarize_cases([dfr25_samples[pid] for pid in sorted(non_target_only_ids)]),
            "both_active_same_target_view": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(both_same_ids)]
            ),
            "both_active_different_target_view": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(both_different_ids)]
            ),
            "helpful_if_promoted": summarize_cases([dfr25_samples[pid] for pid in sorted(helpful_ids)]),
            "harmful_if_promoted": summarize_cases([dfr25_samples[pid] for pid in sorted(harmful_ids)]),
            "normal_rescue_targets": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(normal_rescue_ids)]
            ),
            "dfr25_wrong_uncovered": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(dfr25_wrong - true_active_set)]
            ),
            "nonaxial_true_margin_uncovered": summarize_cases(
                [dfr25_samples[pid] for pid in sorted(dfr25_nonaxial_true - true_active_set)]
            ),
        },
        "notable_case_ids": {
            "true_class_triggered": sorted(true_active_set),
            "confirmed_scope_triggered": sorted(confirmed_scope),
            "non_target_only": sorted(non_target_only_ids),
            "both_active_same_target_view": sorted(both_same_ids),
            "both_active_different_target_view": sorted(both_different_ids),
            "true_target_promoted_by_ce_direction": sorted(promote_ids),
            "true_target_canceled_by_ce_direction": sorted(cancel_ids),
            "true_target_reversed_by_ce_direction": sorted(reverse_ids),
            "dfr25_wrong_covered": sorted(dfr25_wrong_covered),
            "dfr25_wrong_uncovered": sorted(dfr25_wrong - true_active_set),
            "nonaxial_true_margin_covered": sorted(nonaxial_true_covered),
            "nonaxial_true_margin_uncovered": sorted(dfr25_nonaxial_true - true_active_set),
            "helpful_if_promoted": sorted(helpful_ids),
            "harmful_if_promoted": sorted(harmful_ids),
            "normal_rescue_targets": sorted(normal_rescue_ids),
            "normal_rescue_promoted_by_ce_direction": sorted(normal_rescue_promote_ids),
            "normal_rescue_canceled_by_ce_direction": sorted(normal_rescue_cancel_ids),
        },
        "sample_records": records,
    }


def dfr107_trial_metrics(study_dir: Path) -> list[dict[str, Any]]:
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
                "rank_margin": params.get(
                    "runtime_env.ANKLE_DECISION_GATE_CONFIRMED_TARGET_RANK_MARGIN_AUX_MARGIN"
                ),
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
        default=Path("autoresearch_logs/dfr107_confirmed_target_aux_audit.json"),
    )
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dfr25_path = repo_root / DFR25_SEED42
    dfr107_path = repo_root / DFR107_BEST
    study_dir = repo_root / DFR107_STUDY

    dfr25_samples = samples_by_id(dfr25_path)
    dfr107_samples = samples_by_id(dfr107_path)

    dfr25_source_audit = audit_source(
        dfr25_samples,
        dfr25_samples,
        dfr107_samples,
        gap=args.gap,
        margin=args.margin,
    )
    dfr107_source_audit = audit_source(
        dfr107_samples,
        dfr25_samples,
        dfr107_samples,
        gap=args.gap,
        margin=args.margin,
    )

    report = {
        "analysis": "dfr107_confirmed_target_aux_ce_audit",
        "description": (
            "Analysis-only audit of DFR-107 confirmed-target rank-margin auxiliary. "
            "It replays the target selection and computes the effective train.py "
            "CrossEntropy gradient direction for the binary aux logits, with focus "
            "on label-0 normal rescue targets and non-target-only axial reinforcement."
        ),
        "dfr25_seed42_input": DFR25_SEED42,
        "dfr107_best_input": DFR107_BEST,
        "dfr107_study": DFR107_STUDY,
        "gap": round_float(args.gap),
        "margin": round_float(args.margin),
        "near_top_tolerance": NEAR_TOP_TOLERANCE,
        "dfr25_seed42_telemetry": summarize_telemetry(dfr25_path),
        "dfr107_best_telemetry": summarize_telemetry(dfr107_path),
        "dfr107_trials": dfr107_trial_metrics(study_dir),
        "audits": {
            "dfr25_seed42_source": dfr25_source_audit,
            "dfr107_best_source": dfr107_source_audit,
        },
        "conclusion_hint": (
            "If many true-class targets are canceled by same-view non-target activation, "
            "or many non-target-only samples reinforce axial, the next mechanism should "
            "avoid patient-label CE over two class-coded rank logits and use a "
            "label-invariant target-vs-axial gate objective restricted to confirmed "
            "DFR25 error / non-axial true-margin samples."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    dfr25 = report["dfr25_seed42_telemetry"]
    dfr107 = report["dfr107_best_telemetry"]
    source = report["audits"]["dfr25_seed42_source"]
    print(f"Saved DFR-107 confirmed-target aux audit to: {args.output}")
    print(
        "DFR25 seed42 telemetry: "
        f"acc={dfr25['accuracy']:.6f} auc={dfr25['auc']:.6f} f1={dfr25['f1']:.6f} "
        f"mean_weights={dfr25['per_view']['axial']['mean_fusion_weight']:.6f}/"
        f"{dfr25['per_view']['coronal']['mean_fusion_weight']:.6f}/"
        f"{dfr25['per_view']['sagittal']['mean_fusion_weight']:.6f} "
        f"top={dfr25['top_weight_view_distribution']}"
    )
    print(
        "DFR107 best telemetry: "
        f"acc={dfr107['accuracy']:.6f} auc={dfr107['auc']:.6f} f1={dfr107['f1']:.6f} "
        f"mean_weights={dfr107['per_view']['axial']['mean_fusion_weight']:.6f}/"
        f"{dfr107['per_view']['coronal']['mean_fusion_weight']:.6f}/"
        f"{dfr107['per_view']['sagittal']['mean_fusion_weight']:.6f} "
        f"top={dfr107['top_weight_view_distribution']}"
    )
    print(
        "DFR25-source CE audit: "
        f"true_targets={source['true_class_triggered_samples']} "
        f"target_views={source['true_class_target_views']} "
        f"effects={source['effect_kinds']} "
        f"overlap={source['overlap_kinds']} "
        f"non_target_only={source['non_target_only_samples']} "
        f"wrong_covered={source['dfr25_wrong_covered']}/{source['dfr25_wrong_total']} "
        f"nonaxial_true_covered={source['dfr25_nonaxial_true_margin_covered']}/"
        f"{source['dfr25_nonaxial_true_margin_total']} "
        f"normal_rescue={source['normal_rescue_targets']} "
        f"target_top_after_dfr107={source['dfr107_target_became_top']} "
        f"target_near_top_after_dfr107={source['dfr107_target_near_top']}"
    )


if __name__ == "__main__":
    main()
