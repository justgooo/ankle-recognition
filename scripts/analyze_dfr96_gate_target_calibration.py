#!/usr/bin/env python3
"""Audit DFR-90..95 telemetry for a smaller non-axial gate target.

This is an analysis-only DFR follow-up.  It reads existing validation
fusion-weight telemetry and asks whether previous non-axial routing repairs
identify a compact set of samples where coronal/sagittal should be allowed to
take over, while protecting DFR-25-correct axial samples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


VIEWS = ("axial", "coronal", "sagittal")
NONAXIAL = ("coronal", "sagittal")

DFR25_SEED42 = (
    "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/"
    "trials/trial_0001/run/fusion_weight_analysis.json"
)
DFR25_MULTI_SEED = {
    "42": DFR25_SEED42,
    "123": (
        "runs/optuna_main_autoloop/iter_0002_20260423_031222/"
        "trials/trial_0000/run/fusion_weight_analysis.json"
    ),
    "456": (
        "runs/optuna_main_autoloop/iter_0003_20260423_033048/"
        "trials/trial_0000/run/fusion_weight_analysis.json"
    ),
}

COMPARATOR_RUNS = {
    "dfr90_nonaxial_abnormal_aux": (
        "runs/optuna_main_resnext_decision_256x8_dfr90_nonaxial_abnormal_aux/"
        "trials/trial_0002/run/fusion_weight_analysis.json"
    ),
    "dfr91_coronal_abnormal_aux": (
        "runs/optuna_main_resnext_decision_256x8_dfr91_coronal_abnormal_aux/"
        "trials/trial_0000/run/fusion_weight_analysis.json"
    ),
    "dfr92_low_coronal_abnormal_aux": (
        "runs/optuna_main_resnext_decision_256x8_dfr92_coronal_abnormal_aux_low/"
        "trials/trial_0001/run/fusion_weight_analysis.json"
    ),
    "dfr93_coronal_pair_abnormal_aux": (
        "runs/optuna_main_resnext_decision_256x8_dfr93_coronal_pair_abnormal_aux/"
        "trials/trial_0000/run/fusion_weight_analysis.json"
    ),
    "dfr94_forced_axcor_pair_aux": (
        "runs/optuna_main_resnext_decision_256x8_dfr94_forced_axcor_pair_aux/"
        "trials/trial_0001/run/fusion_weight_analysis.json"
    ),
    "dfr95_active_mask_control": (
        "runs/optuna_main_resnext_decision_256x8_dfr95_active_mask_control/"
        "trials/trial_0001/run/fusion_weight_analysis.json"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def safe_rate(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def samples_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in load_json(path)["samples"]}


def top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str:
    views = sample.get(key) or []
    return str(views[0]) if views else "none"


def fusion_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def fusion_pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def view_correct(sample: dict[str, Any], view: str) -> bool:
    return bool(sample["views"][view]["correct"])


def view_pred(sample: dict[str, Any], view: str) -> int:
    return int(sample["views"][view]["pred"])


def pred_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["pred_margin"])


def true_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["true_margin"])


def abnormal_prob(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["abnormal_prob"])


def confidence_logit(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["confidence_logit"])


def fusion_weight(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["fusion_weight"])


def signed_margin(sample: dict[str, Any], view: str) -> float:
    margin = pred_margin(sample, view)
    return margin if view_pred(sample, view) == 1 else -margin


def view_correct_pattern(sample: dict[str, Any]) -> str:
    return "".join(view[0].upper() if view_correct(sample, view) else "-" for view in VIEWS)


def error_type(sample: dict[str, Any]) -> str:
    label = int(sample["label"])
    if fusion_correct(sample):
        return "TP" if label == 1 else "TN"
    if label == 1 and fusion_pred(sample) == 0:
        return "FN"
    if label == 0 and fusion_pred(sample) == 1:
        return "FP"
    return "ERR"


def best_nonaxial_true_view(sample: dict[str, Any]) -> str:
    return max(NONAXIAL, key=lambda view: true_margin(sample, view))


def best_nonaxial_pred_view(sample: dict[str, Any]) -> str:
    return max(NONAXIAL, key=lambda view: pred_margin(sample, view))


def best_correct_nonaxial(sample: dict[str, Any]) -> str | None:
    correct = [view for view in NONAXIAL if view_correct(sample, view)]
    if not correct:
        return None
    return max(correct, key=lambda view: true_margin(sample, view))


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
        "labels": counter_dict(Counter(int(sample["label"]) for sample in samples)),
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


def compare_run(
    dfr25_samples: dict[str, dict[str, Any]],
    run_name: str,
    samples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared_ids = sorted(set(dfr25_samples) & set(samples))
    fixed_ids = [
        pid
        for pid in shared_ids
        if not fusion_correct(dfr25_samples[pid]) and fusion_correct(samples[pid])
    ]
    broken_ids = [
        pid
        for pid in shared_ids
        if fusion_correct(dfr25_samples[pid]) and not fusion_correct(samples[pid])
    ]
    nonaxial_top_ids = [pid for pid in shared_ids if top_view(samples[pid]) in NONAXIAL]
    nonaxial_top_fixed_ids = [pid for pid in nonaxial_top_ids if pid in fixed_ids]
    nonaxial_top_broken_ids = [pid for pid in nonaxial_top_ids if pid in broken_ids]
    top_migration_ids = [
        pid
        for pid in shared_ids
        if top_view(dfr25_samples[pid]) == "axial" and top_view(samples[pid]) in NONAXIAL
    ]

    return {
        "run": run_name,
        "shared_samples": len(shared_ids),
        "fixed_dfr25_errors": len(fixed_ids),
        "broken_dfr25_correct": len(broken_ids),
        "net_delta_vs_dfr25_seed42": len(fixed_ids) - len(broken_ids),
        "nonaxial_top_samples": len(nonaxial_top_ids),
        "nonaxial_top_rate": round_float(safe_rate(len(nonaxial_top_ids), len(shared_ids))),
        "nonaxial_top_fixed": len(nonaxial_top_fixed_ids),
        "nonaxial_top_broken": len(nonaxial_top_broken_ids),
        "top_migration_from_axial": len(top_migration_ids),
        "top_weight_distribution": counter_dict(Counter(top_view(samples[pid]) for pid in shared_ids)),
        "fixed_summary": summarize_cases([samples[pid] for pid in fixed_ids]),
        "broken_summary": summarize_cases([samples[pid] for pid in broken_ids]),
        "nonaxial_top_summary": summarize_cases([samples[pid] for pid in nonaxial_top_ids]),
        "fixed_case_ids": fixed_ids,
        "broken_case_ids": broken_ids,
        "nonaxial_top_case_ids": nonaxial_top_ids,
    }


def score_values(sample: dict[str, Any], view: str) -> dict[str, float]:
    return {
        "oracle_true_margin_gap": true_margin(sample, view) - true_margin(sample, "axial"),
        "pred_margin_gap": pred_margin(sample, view) - pred_margin(sample, "axial"),
        "signed_margin_gap": signed_margin(sample, view) - signed_margin(sample, "axial"),
        "abnormal_prob_gap": abnormal_prob(sample, view) - abnormal_prob(sample, "axial"),
        "confidence_logit_gap": confidence_logit(sample, view) - confidence_logit(sample, "axial"),
        "fusion_weight_gap": fusion_weight(sample, view) - fusion_weight(sample, "axial"),
    }


def evaluate_threshold(
    dfr25_samples: dict[str, dict[str, Any]],
    score_name: str,
    threshold: float,
) -> dict[str, Any]:
    fired: list[tuple[str, str]] = []
    fired_sample_ids: set[str] = set()
    helpful_sample_ids: set[str] = set()
    harmful_sample_ids: set[str] = set()
    candidate_views: Counter[str] = Counter()
    for pid, sample in dfr25_samples.items():
        for view in NONAXIAL:
            if score_values(sample, view)[score_name] < threshold:
                continue
            fired.append((pid, view))
            fired_sample_ids.add(pid)
            candidate_views[view] += 1
            if not fusion_correct(sample) and view_correct(sample, view):
                helpful_sample_ids.add(pid)
            if fusion_correct(sample) and not view_correct(sample, view):
                harmful_sample_ids.add(pid)
    return {
        "score": score_name,
        "threshold": round_float(threshold),
        "fired_view_candidates": len(fired),
        "fired_samples": len(fired_sample_ids),
        "candidate_views": counter_dict(candidate_views),
        "helpful_if_promoted": len(helpful_sample_ids),
        "harmful_if_promoted": len(harmful_sample_ids),
        "net_oracle_delta_upper_bound": len(helpful_sample_ids) - len(harmful_sample_ids),
        "precision_for_dfr25_error_fix": round_float(
            safe_rate(len(helpful_sample_ids), len(fired_sample_ids))
        ),
        "harm_rate": round_float(safe_rate(len(harmful_sample_ids), len(fired_sample_ids))),
        "helpful_case_ids": sorted(helpful_sample_ids),
        "harmful_case_ids": sorted(harmful_sample_ids),
    }


def threshold_grid_report(dfr25_samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grids = {
        "oracle_true_margin_gap": [-0.25, 0.0, 0.25, 0.5, 1.0],
        "pred_margin_gap": [-0.25, 0.0, 0.25, 0.5, 1.0],
        "signed_margin_gap": [-0.5, 0.0, 0.5, 1.0, 2.0],
        "abnormal_prob_gap": [-0.1, 0.0, 0.05, 0.1, 0.2],
        "confidence_logit_gap": [-2.0, -1.0, -0.5, 0.0, 0.5],
        "fusion_weight_gap": [-0.9, -0.75, -0.5, -0.25, 0.0],
    }
    report: dict[str, Any] = {}
    for score_name, thresholds in grids.items():
        items = [evaluate_threshold(dfr25_samples, score_name, threshold) for threshold in thresholds]
        best = max(
            items,
            key=lambda item: (
                item["net_oracle_delta_upper_bound"],
                item["precision_for_dfr25_error_fix"],
                -item["harmful_if_promoted"],
                -item["fired_samples"],
            ),
        )
        report[score_name] = {
            "best_by_net_then_precision": best,
            "thresholds": items,
        }
    return report


def sample_level_calibration(
    dfr25_samples: dict[str, dict[str, Any]],
    comparator_samples: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    shared_ids = sorted(
        set(dfr25_samples).intersection(*(set(samples) for samples in comparator_samples.values()))
    )
    records: list[dict[str, Any]] = []
    category_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_distribution: Counter[str] = Counter()

    for pid in shared_ids:
        dfr25_sample = dfr25_samples[pid]
        fixed_by: list[str] = []
        broken_by: list[str] = []
        nonaxial_top_by: list[str] = []
        comparator_top_views: Counter[str] = Counter()
        for run_name, samples in comparator_samples.items():
            sample = samples[pid]
            comparator_top_views[top_view(sample)] += 1
            if not fusion_correct(dfr25_sample) and fusion_correct(sample):
                fixed_by.append(run_name)
            if fusion_correct(dfr25_sample) and not fusion_correct(sample):
                broken_by.append(run_name)
            if top_view(sample) in NONAXIAL:
                nonaxial_top_by.append(run_name)

        correct_nonaxial_view = best_correct_nonaxial(dfr25_sample)
        needs_nonaxial = (
            not fusion_correct(dfr25_sample)
            and not view_correct(dfr25_sample, "axial")
            and correct_nonaxial_view is not None
        )
        validated_promote = needs_nonaxial and bool(fixed_by)
        missed_promote = needs_nonaxial and not fixed_by
        protect_axial = (
            fusion_correct(dfr25_sample)
            and view_correct(dfr25_sample, "axial")
            and any(not view_correct(dfr25_sample, view) for view in NONAXIAL)
        )
        fragile_protect = protect_axial and bool(broken_by)
        if validated_promote:
            target_view = str(correct_nonaxial_view)
            category_cases["validated_promote_nonaxial"].append(dfr25_sample)
        elif missed_promote:
            target_view = str(correct_nonaxial_view)
            category_cases["missed_or_unvalidated_nonaxial_need"].append(dfr25_sample)
        else:
            target_view = "axial"
        if protect_axial:
            category_cases["protect_axial"].append(dfr25_sample)
        if fragile_protect:
            category_cases["fragile_protect_axial"].append(dfr25_sample)
        if not fusion_correct(dfr25_sample):
            category_cases["dfr25_wrong"].append(dfr25_sample)
        target_distribution[target_view] += 1

        scores_by_view = {
            view: {key: round_float(value) for key, value in score_values(dfr25_sample, view).items()}
            for view in NONAXIAL
        }
        records.append(
            {
                "patient_id": pid,
                "label": int(dfr25_sample["label"]),
                "dfr25_correct": fusion_correct(dfr25_sample),
                "dfr25_error_type": error_type(dfr25_sample),
                "view_correct_pattern": view_correct_pattern(dfr25_sample),
                "dfr25_top_weight": top_view(dfr25_sample),
                "dfr25_top_true_margin": top_view(dfr25_sample, "top_true_margin_views"),
                "best_correct_nonaxial": correct_nonaxial_view,
                "needs_nonaxial": needs_nonaxial,
                "validated_promote": validated_promote,
                "protect_axial": protect_axial,
                "fragile_protect": fragile_protect,
                "fixed_by": fixed_by,
                "broken_by": broken_by,
                "nonaxial_top_by": nonaxial_top_by,
                "comparator_top_views": counter_dict(comparator_top_views),
                "recommended_target_view": target_view,
                "scores": scores_by_view,
            }
        )

    target_total = sum(target_distribution.values())
    return {
        "shared_samples": len(shared_ids),
        "recommended_target_top_distribution": counter_dict(target_distribution),
        "recommended_target_top_rates": {
            view: round_float(safe_rate(target_distribution[view], target_total))
            for view in VIEWS
        },
        "categories": {
            category: summarize_cases(samples)
            for category, samples in sorted(category_cases.items())
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing existing telemetry files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autoresearch_logs/dfr96_gate_target_calibration_audit/report.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dfr25_seed42_path = repo_root / DFR25_SEED42
    dfr25_samples = samples_by_id(dfr25_seed42_path)
    comparator_samples = {
        name: samples_by_id(repo_root / relative_path)
        for name, relative_path in COMPARATOR_RUNS.items()
    }
    calibration = sample_level_calibration(dfr25_samples, comparator_samples)
    report = {
        "analysis": "dfr96_gate_target_calibration_audit",
        "description": (
            "Reads DFR-25 seed42 and DFR-90..95 validation fusion telemetry to "
            "estimate whether prior non-axial interventions expose a compact "
            "gate calibration target.  No training or checkpoint selection is "
            "performed by this script."
        ),
        "dfr25_seed42_input": DFR25_SEED42,
        "dfr25_multiseed_inputs": DFR25_MULTI_SEED,
        "comparator_inputs": COMPARATOR_RUNS,
        "dfr25_seed42_telemetry": summarize_telemetry(dfr25_seed42_path),
        "dfr25_multiseed_telemetry": {
            seed: summarize_telemetry(repo_root / relative_path)
            for seed, relative_path in DFR25_MULTI_SEED.items()
        },
        "comparators": {
            name: {
                "telemetry": summarize_telemetry(repo_root / COMPARATOR_RUNS[name]),
                "vs_dfr25_seed42": compare_run(dfr25_samples, name, samples),
            }
            for name, samples in comparator_samples.items()
        },
        "sample_level_calibration": calibration,
        "threshold_grid_against_dfr25_seed42": threshold_grid_report(dfr25_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    dfr25 = report["dfr25_seed42_telemetry"]
    target = calibration["recommended_target_top_distribution"]
    target_rates = calibration["recommended_target_top_rates"]
    categories = calibration["categories"]
    print(f"Saved DFR-96 gate-target calibration audit to: {args.output}")
    print(
        "DFR25 seed42: "
        f"acc={dfr25['accuracy']:.6f} auc={dfr25['auc']:.6f} f1={dfr25['f1']:.6f} "
        f"top={dfr25['top_weight_view_distribution']} "
        f"mean_weights="
        f"{dfr25['per_view']['axial']['mean_fusion_weight']:.6f}/"
        f"{dfr25['per_view']['coronal']['mean_fusion_weight']:.6f}/"
        f"{dfr25['per_view']['sagittal']['mean_fusion_weight']:.6f}"
    )
    print(f"recommended_target_top={target} rates={target_rates}")
    for category in (
        "dfr25_wrong",
        "validated_promote_nonaxial",
        "missed_or_unvalidated_nonaxial_need",
        "fragile_protect_axial",
    ):
        item = categories.get(category, {"count": 0})
        print(f"{category}: count={item['count']}")
    for name, item in report["comparators"].items():
        compare = item["vs_dfr25_seed42"]
        print(
            f"{name}: fixed={compare['fixed_dfr25_errors']} "
            f"broken={compare['broken_dfr25_correct']} "
            f"net={compare['net_delta_vs_dfr25_seed42']} "
            f"nonaxial_top={compare['nonaxial_top_samples']}"
        )


if __name__ == "__main__":
    main()
