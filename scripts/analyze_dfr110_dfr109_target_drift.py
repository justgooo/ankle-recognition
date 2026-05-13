#!/usr/bin/env python3
"""Audit why DFR-109 failed to release sagittal routing.

This analysis-only follow-up compares DFR-25, DFR-107, and DFR-109 telemetry
on the DFR-108 confirmed target set and on the non-target-only positive samples
that previously produced axial-reinforcing pressure.
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
DFR25 = (
    "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR107 = (
    "runs/optuna_main_autoloop/iter_0002_20260513_023419/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR109 = (
    "runs/optuna_main_resnext_decision_256x8_dfr109_fp_risk_axsag_rank_aux/"
    "trials/trial_0002/run/fusion_weight_analysis.json"
)
DEFAULT_OUTPUT = "autoresearch_logs/dfr110_dfr109_target_drift_audit.json"
EPS = 1e-8


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def round_float(value: float, digits: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        return float(value)
    return round(float(value), digits)


def safe_mean(values: list[float]) -> float:
    return round_float(mean(values)) if values else 0.0


def safe_rate(count: int, total: int) -> float:
    return round_float(count / total) if total else 0.0


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def samples_by_id(path: Path) -> dict[str, dict[str, Any]]:
    telemetry = load_json(path)
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str:
    values = sample.get(key) or []
    return str(values[0]) if values else "none"


def label(sample: dict[str, Any]) -> int:
    return int(sample["label"])


def fusion_pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def fusion_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def fused_abnormal(sample: dict[str, Any]) -> float:
    return float(sample["fusion_prediction"]["abnormal_prob"])


def view_correct(sample: dict[str, Any], view: str) -> bool:
    return bool(sample["views"][view]["correct"])


def abnormal_prob(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["abnormal_prob"])


def class_prob(sample: dict[str, Any], view: str, class_index: int) -> float:
    value = abnormal_prob(sample, view)
    return value if class_index == 1 else 1.0 - value


def class_log_prob(sample: dict[str, Any], view: str, class_index: int) -> float:
    return math.log(max(class_prob(sample, view, class_index), EPS))


def class_margin(sample: dict[str, Any], view: str, class_index: int) -> float:
    return class_log_prob(sample, view, class_index) - class_log_prob(
        sample,
        view,
        1 - class_index,
    )


def fusion_weight(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["fusion_weight"])


def confidence_logit(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["confidence_logit"])


def true_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["true_margin"])


def error_type(sample: dict[str, Any]) -> str:
    if fusion_correct(sample):
        return "TP" if label(sample) == 1 else "TN"
    if label(sample) == 1 and fusion_pred(sample) == 0:
        return "FN"
    if label(sample) == 0 and fusion_pred(sample) == 1:
        return "FP"
    return "ERR"


def correctness_pattern(sample: dict[str, Any]) -> str:
    return "".join(view[0].upper() if view_correct(sample, view) else "-" for view in VIEWS)


def telemetry_summary(path: Path) -> dict[str, Any]:
    telemetry = load_json(path)
    summary = telemetry["summary"]
    metrics = summary["full_fusion_metrics"]
    return {
        "path": str(path),
        "accuracy": round_float(metrics["accuracy"]),
        "auc": round_float(metrics["auc"]),
        "f1": round_float(metrics["f1"]),
        "top_weight_view_distribution": summary["top_weight_view_distribution"],
        "top_true_margin_view_distribution": summary["top_true_margin_view_distribution"],
        "top_weight_hit_rate_true_margin": round_float(
            summary["top_weight_hit_rate"]["true_margin"]
        ),
        "per_view": {
            item["view"]: {
                "accuracy": round_float(item["metrics"]["accuracy"]),
                "auc": round_float(item["metrics"]["auc"]),
                "f1": round_float(item["metrics"]["f1"]),
                "sensitivity": round_float(item["metrics"]["sensitivity"]),
                "specificity": round_float(item["metrics"]["specificity"]),
                "mean_fusion_weight": round_float(item["mean_fusion_weight"]),
                "top_weight_count": int(item["top_weight_count"]),
                "top_true_margin_count": int(item["top_true_margin_count"]),
            }
            for item in telemetry["per_view"]
        },
    }


def confirmed_true_target(
    sample: dict[str, Any],
    gap: float,
) -> dict[str, Any]:
    class_index = label(sample)
    axial_log_prob = class_log_prob(sample, "axial", class_index)
    candidates: list[tuple[float, str]] = []
    for view in NONAXIAL:
        if class_margin(sample, view, class_index) < 0.0:
            candidates.append((-math.inf, view))
        else:
            candidates.append((class_log_prob(sample, view, class_index) - axial_log_prob, view))
    advantage, target_view = max(candidates, key=lambda item: item[0])
    active = bool(advantage >= gap)
    return {
        "active": active,
        "target_view": target_view,
        "advantage": round_float(advantage),
        "target_class_margin": round_float(class_margin(sample, target_view, class_index)),
    }


def confirmed_non_target_only(
    sample: dict[str, Any],
    gap: float,
) -> dict[str, Any]:
    true_info = confirmed_true_target(sample, gap)
    other_class = 1 - label(sample)
    axial_log_prob = class_log_prob(sample, "axial", other_class)
    candidates: list[tuple[float, str]] = []
    for view in NONAXIAL:
        if class_margin(sample, view, other_class) < 0.0:
            candidates.append((-math.inf, view))
        else:
            candidates.append((class_log_prob(sample, view, other_class) - axial_log_prob, view))
    advantage, target_view = max(candidates, key=lambda item: item[0])
    other_active = bool(advantage >= gap)
    return {
        "active": bool(other_active and not true_info["active"]),
        "other_class_active": other_active,
        "true_class_active": bool(true_info["active"]),
        "target_view": target_view,
        "advantage": round_float(advantage),
    }


def dfr109_fp_risk_trigger(
    sample: dict[str, Any],
    *,
    axial_min: float,
    axial_max: float,
    sagittal_normal_min: float,
    fused_abnormal_min: float,
    coronal_abnormal_max: float,
) -> bool:
    axial_abnormal = abnormal_prob(sample, "axial")
    sagittal_normal = 1.0 - abnormal_prob(sample, "sagittal")
    coronal_abnormal = abnormal_prob(sample, "coronal")
    return bool(
        fused_abnormal(sample) >= fused_abnormal_min
        and axial_abnormal >= axial_min
        and axial_abnormal <= axial_max
        and sagittal_normal >= sagittal_normal_min
        and coronal_abnormal <= coronal_abnormal_max
    )


def sample_snapshot(sample: dict[str, Any], target_view: str | None = None) -> dict[str, Any]:
    return {
        "label": label(sample),
        "fusion_pred": fusion_pred(sample),
        "fusion_correct": fusion_correct(sample),
        "error_type": error_type(sample),
        "fused_abnormal": round_float(fused_abnormal(sample)),
        "top_weight": top_view(sample),
        "top_true_margin": top_view(sample, "top_true_margin_views"),
        "view_correct_pattern": correctness_pattern(sample),
        "target_is_top": bool(target_view and top_view(sample) == target_view),
        "target_weight": round_float(fusion_weight(sample, target_view))
        if target_view
        else None,
        "weights": {view: round_float(fusion_weight(sample, view)) for view in VIEWS},
        "confidence_logits": {view: round_float(confidence_logit(sample, view)) for view in VIEWS},
        "abnormal_probs": {view: round_float(abnormal_prob(sample, view)) for view in VIEWS},
        "true_margins": {view: round_float(true_margin(sample, view)) for view in VIEWS},
    }


def summarize_ids(ids: set[str], samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    subset = [samples[pid] for pid in sorted(ids)]
    return {
        "count": len(subset),
        "labels": counter_dict(Counter(label(sample) for sample in subset)),
        "error_types": counter_dict(Counter(error_type(sample) for sample in subset)),
        "top_weight": counter_dict(Counter(top_view(sample) for sample in subset)),
        "top_true_margin": counter_dict(
            Counter(top_view(sample, "top_true_margin_views") for sample in subset)
        ),
        "view_correct_patterns": counter_dict(Counter(correctness_pattern(sample) for sample in subset)),
        "mean_fused_abnormal": safe_mean([fused_abnormal(sample) for sample in subset]),
        "mean_axial_weight": safe_mean([fusion_weight(sample, "axial") for sample in subset]),
        "mean_coronal_weight": safe_mean([fusion_weight(sample, "coronal") for sample in subset]),
        "mean_sagittal_weight": safe_mean([fusion_weight(sample, "sagittal") for sample in subset]),
        "mean_axial_abnormal": safe_mean([abnormal_prob(sample, "axial") for sample in subset]),
        "mean_coronal_abnormal": safe_mean([abnormal_prob(sample, "coronal") for sample in subset]),
        "mean_sagittal_abnormal": safe_mean([abnormal_prob(sample, "sagittal") for sample in subset]),
    }


def transition_summary(
    ids: set[str],
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixed = sorted(pid for pid in ids if not fusion_correct(before[pid]) and fusion_correct(after[pid]))
    broken = sorted(pid for pid in ids if fusion_correct(before[pid]) and not fusion_correct(after[pid]))
    stayed_wrong = sorted(
        pid for pid in ids if not fusion_correct(before[pid]) and not fusion_correct(after[pid])
    )
    stayed_correct = sorted(
        pid for pid in ids if fusion_correct(before[pid]) and fusion_correct(after[pid])
    )
    return {
        "fixed": len(fixed),
        "broken": len(broken),
        "stayed_wrong": len(stayed_wrong),
        "stayed_correct": len(stayed_correct),
        "net_correct": len(fixed) - len(broken),
        "fixed_ids": fixed,
        "broken_ids": broken,
        "stayed_wrong_ids": stayed_wrong,
    }


def target_case_records(
    ids: set[str],
    dfr25: dict[str, dict[str, Any]],
    dfr107: dict[str, dict[str, Any]],
    dfr109: dict[str, dict[str, Any]],
    gap: float,
    trigger_kwargs: dict[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pid in sorted(ids):
        target = confirmed_true_target(dfr25[pid], gap)
        target_view = str(target["target_view"])
        records.append(
            {
                "patient_id": pid,
                "target": target,
                "dfr109_trigger_on_dfr25": dfr109_fp_risk_trigger(dfr25[pid], **trigger_kwargs),
                "dfr109_trigger_on_dfr109": dfr109_fp_risk_trigger(dfr109[pid], **trigger_kwargs),
                "helpful_if_promoted": bool(
                    not fusion_correct(dfr25[pid]) and view_correct(dfr25[pid], target_view)
                ),
                "dfr25": sample_snapshot(dfr25[pid], target_view),
                "dfr107": sample_snapshot(dfr107[pid], target_view),
                "dfr109": sample_snapshot(dfr109[pid], target_view),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--gap", type=float, default=0.02)
    parser.add_argument("--axial-min", type=float, default=0.84)
    parser.add_argument("--axial-max", type=float, default=0.95)
    parser.add_argument("--sagittal-normal-min", type=float, default=0.64)
    parser.add_argument("--fused-abnormal-min", type=float, default=0.75)
    parser.add_argument("--coronal-abnormal-max", type=float, default=0.525)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    paths = {
        "dfr25": repo_root / DFR25,
        "dfr107": repo_root / DFR107,
        "dfr109": repo_root / DFR109,
    }
    dfr25 = samples_by_id(paths["dfr25"])
    dfr107 = samples_by_id(paths["dfr107"])
    dfr109 = samples_by_id(paths["dfr109"])
    shared_ids = set(dfr25) & set(dfr107) & set(dfr109)
    trigger_kwargs = {
        "axial_min": args.axial_min,
        "axial_max": args.axial_max,
        "sagittal_normal_min": args.sagittal_normal_min,
        "fused_abnormal_min": args.fused_abnormal_min,
        "coronal_abnormal_max": args.coronal_abnormal_max,
    }

    true_target_ids = {
        pid for pid in shared_ids if confirmed_true_target(dfr25[pid], args.gap)["active"]
    }
    non_target_only_ids = {
        pid for pid in shared_ids if confirmed_non_target_only(dfr25[pid], args.gap)["active"]
    }
    positive_non_target_only_ids = {
        pid for pid in non_target_only_ids if label(dfr25[pid]) == 1
    }
    dfr25_wrong_ids = {pid for pid in shared_ids if not fusion_correct(dfr25[pid])}
    dfr109_trigger_dfr25_ids = {
        pid for pid in shared_ids if dfr109_fp_risk_trigger(dfr25[pid], **trigger_kwargs)
    }
    dfr109_trigger_dfr109_ids = {
        pid for pid in shared_ids if dfr109_fp_risk_trigger(dfr109[pid], **trigger_kwargs)
    }

    report = {
        "analysis": "dfr110_dfr109_target_drift_audit",
        "description": (
            "Compare DFR-25, DFR-107, and DFR-109 telemetry on DFR-108 confirmed "
            "true targets and non-target-only positive pressure samples."
        ),
        "inputs": {name: str(path) for name, path in paths.items()},
        "parameters": {"confirmed_gap": args.gap, "dfr109_trigger": trigger_kwargs},
        "telemetry_summary": {
            name: telemetry_summary(path) for name, path in paths.items()
        },
        "global_transitions_vs_dfr25": {
            "dfr107": transition_summary(shared_ids, dfr25, dfr107),
            "dfr109": transition_summary(shared_ids, dfr25, dfr109),
        },
        "sets": {
            "shared_samples": len(shared_ids),
            "dfr25_wrong_total": len(dfr25_wrong_ids),
            "true_target_ids": sorted(true_target_ids),
            "true_target_summary_dfr25": summarize_ids(true_target_ids, dfr25),
            "true_target_summary_dfr107": summarize_ids(true_target_ids, dfr107),
            "true_target_summary_dfr109": summarize_ids(true_target_ids, dfr109),
            "true_target_transitions": {
                "dfr107_vs_dfr25": transition_summary(true_target_ids, dfr25, dfr107),
                "dfr109_vs_dfr25": transition_summary(true_target_ids, dfr25, dfr109),
            },
            "non_target_only_ids": sorted(non_target_only_ids),
            "non_target_only_summary_dfr25": summarize_ids(non_target_only_ids, dfr25),
            "non_target_only_summary_dfr109": summarize_ids(non_target_only_ids, dfr109),
            "positive_non_target_only_ids": sorted(positive_non_target_only_ids),
            "positive_non_target_only_transitions": {
                "dfr109_vs_dfr25": transition_summary(
                    positive_non_target_only_ids,
                    dfr25,
                    dfr109,
                )
            },
            "dfr109_trigger_on_dfr25_ids": sorted(dfr109_trigger_dfr25_ids),
            "dfr109_trigger_on_dfr25_summary": summarize_ids(
                dfr109_trigger_dfr25_ids,
                dfr25,
            ),
            "dfr109_trigger_on_dfr109_ids": sorted(dfr109_trigger_dfr109_ids),
            "dfr109_trigger_on_dfr109_summary": summarize_ids(
                dfr109_trigger_dfr109_ids,
                dfr109,
            ),
            "dfr109_trigger_true_target_overlap_dfr25": {
                "count": len(dfr109_trigger_dfr25_ids & true_target_ids),
                "rate_within_true_targets": safe_rate(
                    len(dfr109_trigger_dfr25_ids & true_target_ids),
                    len(true_target_ids),
                ),
                "ids": sorted(dfr109_trigger_dfr25_ids & true_target_ids),
            },
            "dfr109_trigger_positive_non_target_overlap_dfr25": {
                "count": len(dfr109_trigger_dfr25_ids & positive_non_target_only_ids),
                "ids": sorted(dfr109_trigger_dfr25_ids & positive_non_target_only_ids),
            },
        },
        "target_case_records": target_case_records(
            true_target_ids,
            dfr25,
            dfr107,
            dfr109,
            args.gap,
            trigger_kwargs,
        ),
    }

    write_json(repo_root / args.output, report)
    print(f"Saved DFR-110 audit to: {repo_root / args.output}")
    print(
        "true_targets=",
        len(true_target_ids),
        "trigger_overlap=",
        len(dfr109_trigger_dfr25_ids & true_target_ids),
        "positive_non_target_overlap=",
        len(dfr109_trigger_dfr25_ids & positive_non_target_only_ids),
    )
    print(
        "dfr109 true-target transition:",
        report["sets"]["true_target_transitions"]["dfr109_vs_dfr25"],
    )


if __name__ == "__main__":
    main()
