#!/usr/bin/env python3
"""Audit label-free non-axial routing triggers against DFR-25 telemetry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


VIEWS = ("axial", "coronal", "sagittal")
NONAXIAL = ("coronal", "sagittal")

DEFAULT_DFR25_RUNS = {
    "42": "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/trials/trial_0001/run/fusion_weight_analysis.json",
    "123": "runs/optuna_main_autoloop/iter_0002_20260423_031222/trials/trial_0000/run/fusion_weight_analysis.json",
    "456": "runs/optuna_main_autoloop/iter_0003_20260423_033048/trials/trial_0000/run/fusion_weight_analysis.json",
}

DEFAULT_COMPARATORS = {
    "dfr76_seed42": "runs/optuna_main_autoloop/iter_0003_20260512_040326/trials/trial_0002/run/fusion_weight_analysis.json",
    "dfr81_seed42_trial1": "runs/optuna_main_autoloop/iter_0008_20260512_055427/trials/trial_0001/run/fusion_weight_analysis.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str | None:
    values = sample.get(key) or []
    return str(values[0]) if values else None


def fusion_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def fusion_pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def error_type(sample: dict[str, Any]) -> str:
    label = int(sample["label"])
    if fusion_correct(sample):
        return "TP" if label == 1 else "TN"
    if label == 1 and fusion_pred(sample) == 0:
        return "FN"
    if label == 0 and fusion_pred(sample) == 1:
        return "FP"
    return "ERR"


def signed_margin(sample: dict[str, Any], view: str) -> float:
    views = sample["views"]
    abnormal_prob = float(views[view]["abnormal_prob"])
    pred = int(views[view]["pred"])
    pred_margin = float(views[view]["pred_margin"])
    return pred_margin if pred == 1 else -pred_margin


def pred_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["pred_margin"])


def true_margin(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["true_margin"])


def abnormal_prob(sample: dict[str, Any], view: str) -> float:
    return float(sample["views"][view]["abnormal_prob"])


def view_correct(sample: dict[str, Any], view: str) -> bool:
    return bool(sample["views"][view]["correct"])


def view_pred(sample: dict[str, Any], view: str) -> int:
    return int(sample["views"][view]["pred"])


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def summarize_cases(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "labels": {},
            "error_types": {},
            "top_true_margin": {},
            "top_pred_margin": {},
            "view_correct_patterns": {},
        }

    def pattern(sample: dict[str, Any]) -> str:
        return "".join(view[0].upper() if view_correct(sample, view) else "-" for view in VIEWS)

    return {
        "count": len(samples),
        "labels": counter_dict(Counter(int(sample["label"]) for sample in samples)),
        "error_types": counter_dict(Counter(error_type(sample) for sample in samples)),
        "top_true_margin": counter_dict(Counter(top_view(sample, "top_true_margin_views") for sample in samples)),
        "top_pred_margin": counter_dict(Counter(top_view(sample, "top_pred_margin_views") for sample in samples)),
        "view_correct_patterns": counter_dict(Counter(pattern(sample) for sample in samples)),
        "mean_axial_weight": round_float(mean(float(sample["views"]["axial"]["fusion_weight"]) for sample in samples)),
        "mean_best_nonaxial_true_margin": round_float(
            mean(max(true_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
        "mean_best_nonaxial_pred_margin": round_float(
            mean(max(pred_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
    }


def candidate_views(sample: dict[str, Any], trigger: str) -> list[str]:
    axial_pred_margin = pred_margin(sample, "axial")
    candidates: list[str] = []
    for view in NONAXIAL:
        view_signed = signed_margin(sample, view)
        other_nonaxial = "sagittal" if view == "coronal" else "coronal"
        if trigger == "true_margin_oracle":
            if true_margin(sample, view) > true_margin(sample, "axial"):
                candidates.append(view)
        elif trigger == "view_correct_oracle":
            if view_correct(sample, view) and not view_correct(sample, "axial"):
                candidates.append(view)
        elif trigger == "pred_margin_gap_0.5":
            if pred_margin(sample, view) - axial_pred_margin >= 0.5:
                candidates.append(view)
        elif trigger == "class_consensus_0.35":
            consensus_support = max(
                view_signed * signed_margin(sample, "axial"),
                view_signed * signed_margin(sample, other_nonaxial),
            )
            if (
                pred_margin(sample, view) - axial_pred_margin >= 0.5
                and consensus_support >= 0.35
            ):
                candidates.append(view)
        elif trigger == "pair_consensus_0.2":
            consensus_support = max(
                view_signed * signed_margin(sample, "axial"),
                view_signed * signed_margin(sample, other_nonaxial),
            )
            nonaxial_pair_support = view_signed * signed_margin(sample, other_nonaxial)
            if (
                pred_margin(sample, view) - axial_pred_margin >= 0.5
                and consensus_support >= 0.35
                and nonaxial_pair_support >= 0.2
            ):
                candidates.append(view)
        elif trigger == "nonaxial_abnormal_over_axial":
            if (
                abnormal_prob(sample, view) > abnormal_prob(sample, "axial")
                and abnormal_prob(sample, view) >= 0.5
            ):
                candidates.append(view)
        else:
            raise ValueError(f"unknown trigger: {trigger}")
    return candidates


def evaluate_trigger(samples: list[dict[str, Any]], trigger: str) -> dict[str, Any]:
    fired_samples: list[dict[str, Any]] = []
    helpful_samples: list[dict[str, Any]] = []
    harmful_samples: list[dict[str, Any]] = []
    dfr25_wrong_samples: list[dict[str, Any]] = []
    correct_nonaxial_wrong_axial: list[dict[str, Any]] = []
    candidate_view_counter: Counter[str] = Counter()
    fired_error_counter: Counter[str] = Counter()
    fired_label_counter: Counter[int] = Counter()

    for sample in samples:
        if not fusion_correct(sample):
            dfr25_wrong_samples.append(sample)
        if any(view_correct(sample, view) for view in NONAXIAL) and not view_correct(sample, "axial"):
            correct_nonaxial_wrong_axial.append(sample)

        views = candidate_views(sample, trigger)
        if not views:
            continue
        fired_samples.append(sample)
        candidate_view_counter.update(views)
        fired_error_counter[error_type(sample)] += 1
        fired_label_counter[int(sample["label"])] += 1
        candidate_is_correct = any(view_correct(sample, view) for view in views)
        if not fusion_correct(sample) and candidate_is_correct:
            helpful_samples.append(sample)
        if fusion_correct(sample) and not candidate_is_correct:
            harmful_samples.append(sample)

    dfr25_wrong = len(dfr25_wrong_samples)
    possible_nonaxial = len(correct_nonaxial_wrong_axial)
    return {
        "trigger": trigger,
        "fired": len(fired_samples),
        "fired_rate": round_float(safe_rate(len(fired_samples), len(samples))),
        "candidate_views": counter_dict(candidate_view_counter),
        "fired_labels": counter_dict(fired_label_counter),
        "fired_dfr25_error_types": counter_dict(fired_error_counter),
        "helpful_if_promoted": len(helpful_samples),
        "harmful_if_promoted": len(harmful_samples),
        "net_oracle_delta_upper_bound": len(helpful_samples) - len(harmful_samples),
        "helpful_coverage_of_dfr25_errors": round_float(safe_rate(len(helpful_samples), dfr25_wrong)),
        "helpful_coverage_of_correct_nonaxial_wrong_axial": round_float(
            safe_rate(len(helpful_samples), possible_nonaxial)
        ),
        "precision_among_fired_for_fixing_dfr25_error": round_float(
            safe_rate(len(helpful_samples), len(fired_samples))
        ),
        "harm_rate_among_fired": round_float(safe_rate(len(harmful_samples), len(fired_samples))),
        "fired_summary": summarize_cases(fired_samples),
        "helpful_summary": summarize_cases(helpful_samples),
        "harmful_summary": summarize_cases(harmful_samples),
        "helpful_case_ids": [str(sample["patient_id"]) for sample in helpful_samples],
        "harmful_case_ids": [str(sample["patient_id"]) for sample in harmful_samples],
    }


def load_samples(path: Path) -> list[dict[str, Any]]:
    telemetry = load_json(path)
    return list(telemetry["samples"])


def telemetry_summary(path: Path) -> dict[str, Any]:
    telemetry = load_json(path)
    per_view = {
        item["view"]: {
            "mean_fusion_weight": round_float(item["mean_fusion_weight"]),
            "top_weight_count": int(item["top_weight_count"]),
            "top_weight_rate": round_float(item["top_weight_rate"]),
            "top_true_margin_count": int(item["top_true_margin_count"]),
            "top_true_margin_rate": round_float(item["top_true_margin_rate"]),
            "accuracy": round_float(item["metrics"]["accuracy"]),
        }
        for item in telemetry["per_view"]
    }
    return {
        "path": str(path),
        "accuracy": round_float(telemetry["summary"]["full_fusion_metrics"]["accuracy"]),
        "auc": round_float(telemetry["summary"]["full_fusion_metrics"]["auc"]),
        "f1": round_float(telemetry["summary"]["full_fusion_metrics"]["f1"]),
        "top_weight_view_distribution": telemetry["summary"]["top_weight_view_distribution"],
        "top_true_margin_view_distribution": telemetry["summary"]["top_true_margin_view_distribution"],
        "top_weight_hit_rate_true_margin": round_float(
            telemetry["summary"]["top_weight_hit_rate"]["true_margin"]
        ),
        "per_view": per_view,
    }


def analyze_dfr25(repo_root: Path, triggers: list[str]) -> dict[str, Any]:
    seed_reports: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []
    for seed, relative_path in DEFAULT_DFR25_RUNS.items():
        path = repo_root / relative_path
        samples = load_samples(path)
        all_samples.extend(samples)
        seed_reports[seed] = {
            "telemetry": telemetry_summary(path),
            "dfr25_wrong_summary": summarize_cases(
                [sample for sample in samples if not fusion_correct(sample)]
            ),
            "correct_nonaxial_wrong_axial_summary": summarize_cases(
                [
                    sample
                    for sample in samples
                    if any(view_correct(sample, view) for view in NONAXIAL)
                    and not view_correct(sample, "axial")
                ]
            ),
            "triggers": {
                trigger: evaluate_trigger(samples, trigger)
                for trigger in triggers
            },
        }

    return {
        "seeds": seed_reports,
        "aggregate": {
            "num_samples": len(all_samples),
            "dfr25_wrong_summary": summarize_cases(
                [sample for sample in all_samples if not fusion_correct(sample)]
            ),
            "correct_nonaxial_wrong_axial_summary": summarize_cases(
                [
                    sample
                    for sample in all_samples
                    if any(view_correct(sample, view) for view in NONAXIAL)
                    and not view_correct(sample, "axial")
                ]
            ),
            "triggers": {
                trigger: evaluate_trigger(all_samples, trigger)
                for trigger in triggers
            },
        },
    }


def compare_with_dfr25_seed42(repo_root: Path) -> dict[str, Any]:
    dfr25_path = repo_root / DEFAULT_DFR25_RUNS["42"]
    dfr25_samples = {
        str(sample["patient_id"]): sample
        for sample in load_samples(dfr25_path)
    }
    reports: dict[str, Any] = {}
    for name, relative_path in DEFAULT_COMPARATORS.items():
        path = repo_root / relative_path
        samples = {
            str(sample["patient_id"]): sample
            for sample in load_samples(path)
        }
        shared_ids = sorted(set(dfr25_samples) & set(samples))
        fixed = [
            pid
            for pid in shared_ids
            if not fusion_correct(dfr25_samples[pid]) and fusion_correct(samples[pid])
        ]
        broken = [
            pid
            for pid in shared_ids
            if fusion_correct(dfr25_samples[pid]) and not fusion_correct(samples[pid])
        ]
        reports[name] = {
            "telemetry": telemetry_summary(path),
            "fixed_dfr25_errors": len(fixed),
            "broken_dfr25_correct": len(broken),
            "net_delta_vs_dfr25_seed42": len(fixed) - len(broken),
            "fixed_summary": summarize_cases([samples[pid] for pid in fixed]),
            "broken_summary": summarize_cases([samples[pid] for pid in broken]),
            "fixed_case_ids": fixed,
            "broken_case_ids": broken,
        }
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing existing fusion_weight_analysis.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autoresearch_logs/dfr82_trigger_audit/report.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--triggers",
        nargs="+",
        default=[
            "true_margin_oracle",
            "view_correct_oracle",
            "pred_margin_gap_0.5",
            "class_consensus_0.35",
            "pair_consensus_0.2",
            "nonaxial_abnormal_over_axial",
        ],
        help="Trigger names to audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report = {
        "analysis": "dfr25_nonaxial_trigger_audit",
        "description": (
            "Reads existing fusion_weight_analysis.json telemetry and estimates "
            "whether oracle upper bounds and label-free non-axial routing triggers "
            "can repair DFR-25 errors without breaking currently correct samples."
        ),
        "dfr25_inputs": DEFAULT_DFR25_RUNS,
        "comparator_inputs": DEFAULT_COMPARATORS,
        "dfr25": analyze_dfr25(repo_root, args.triggers),
        "comparators_vs_dfr25_seed42": compare_with_dfr25_seed42(repo_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    aggregate = report["dfr25"]["aggregate"]
    print(f"Saved DFR-25 trigger audit to: {args.output}")
    print(f"DFR-25 aggregate wrong samples: {aggregate['dfr25_wrong_summary']['count']}")
    for trigger, item in aggregate["triggers"].items():
        print(
            f"{trigger}: fired={item['fired']} helpful={item['helpful_if_promoted']} "
            f"harmful={item['harmful_if_promoted']} net={item['net_oracle_delta_upper_bound']}"
        )


if __name__ == "__main__":
    main()
