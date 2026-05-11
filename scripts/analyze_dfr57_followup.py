#!/usr/bin/env python3
"""Compare DFR-57 against DFR-25 at sample level.

The script reads existing fusion_weight_analysis.json files only. It does not
touch checkpoints, datasets, or training code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VIEWS = ("axial", "coronal", "sagittal")
DEFAULT_RUNS = {
    "42": {
        "dfr25": "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/trials/trial_0001/run/fusion_weight_analysis.json",
        "dfr57": "runs/resnext_decision_256x8_mainline/dfr57_view_role_confidence_formal_s42/fusion_weight_analysis.json",
    },
    "123": {
        "dfr25": "runs/optuna_main_autoloop/iter_0002_20260423_031222/trials/trial_0000/run/fusion_weight_analysis.json",
        "dfr57": "runs/resnext_decision_256x8_mainline/dfr57_view_role_confidence_formal_s123/fusion_weight_analysis.json",
    },
    "456": {
        "dfr25": "runs/optuna_main_autoloop/iter_0003_20260423_033048/trials/trial_0000/run/fusion_weight_analysis.json",
        "dfr57": "runs/resnext_decision_256x8_mainline/dfr57_view_role_confidence_formal_s456/fusion_weight_analysis.json",
    },
}


def load_telemetry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {sample["patient_id"]: sample for sample in telemetry["samples"]}


def top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str | None:
    values = sample.get(key) or []
    return values[0] if values else None


def fusion_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["fusion_prediction"]["correct"])


def fusion_pred(sample: dict[str, Any]) -> int:
    return int(sample["fusion_prediction"]["pred"])


def fusion_prob(sample: dict[str, Any]) -> float:
    return float(sample["fusion_prediction"]["abnormal_prob"])


def error_type(sample: dict[str, Any]) -> str:
    if fusion_correct(sample):
        return "TP" if int(sample["label"]) == 1 else "TN"
    if int(sample["label"]) == 1 and fusion_pred(sample) == 0:
        return "FN"
    if int(sample["label"]) == 0 and fusion_pred(sample) == 1:
        return "FP"
    return "ERR"


def view_correct_pattern(sample: dict[str, Any]) -> str:
    return "".join(
        view[0].upper() if sample["views"][view]["correct"] else "-"
        for view in VIEWS
    )


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def view_weights(sample: dict[str, Any]) -> dict[str, float]:
    return {
        view: round_float(sample["views"][view]["fusion_weight"], 4)
        for view in VIEWS
    }


def view_probs(sample: dict[str, Any]) -> dict[str, float]:
    return {
        view: round_float(sample["views"][view]["abnormal_prob"], 4)
        for view in VIEWS
    }


def view_true_margins(sample: dict[str, Any]) -> dict[str, float]:
    return {
        view: round_float(sample["views"][view]["true_margin"], 4)
        for view in VIEWS
    }


def changed_case(pid: str, dfr25: dict[str, Any], dfr57: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": pid,
        "label": int(dfr57["label"]),
        "dfr25": {
            "pred": fusion_pred(dfr25),
            "abnormal_prob": round_float(fusion_prob(dfr25), 4),
            "error_type": error_type(dfr25),
            "top_weight": top_view(dfr25),
            "top_true_margin": top_view(dfr25, "top_true_margin_views"),
            "view_correct_pattern": view_correct_pattern(dfr25),
            "weights": view_weights(dfr25),
            "view_abnormal_probs": view_probs(dfr25),
            "view_true_margins": view_true_margins(dfr25),
        },
        "dfr57": {
            "pred": fusion_pred(dfr57),
            "abnormal_prob": round_float(fusion_prob(dfr57), 4),
            "error_type": error_type(dfr57),
            "top_weight": top_view(dfr57),
            "top_true_margin": top_view(dfr57, "top_true_margin_views"),
            "view_correct_pattern": view_correct_pattern(dfr57),
            "weights": view_weights(dfr57),
            "view_abnormal_probs": view_probs(dfr57),
            "view_true_margins": view_true_margins(dfr57),
        },
    }


def per_view_accuracy_delta(dfr25: dict[str, Any], dfr57: dict[str, Any]) -> dict[str, dict[str, float]]:
    by25 = {item["view"]: item["metrics"]["accuracy"] for item in dfr25["per_view"]}
    by57 = {item["view"]: item["metrics"]["accuracy"] for item in dfr57["per_view"]}
    return {
        view: {
            "dfr25": round_float(by25[view]),
            "dfr57": round_float(by57[view]),
            "delta": round_float(by57[view] - by25[view]),
        }
        for view in VIEWS
    }


def average_weights(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {view: 0.0 for view in VIEWS}
    return {
        view: round_float(
            sum(sample["views"][view]["fusion_weight"] for sample in samples) / len(samples),
            4,
        )
        for view in VIEWS
    }


def analyze_seed(seed: str, dfr25_path: Path, dfr57_path: Path) -> dict[str, Any]:
    dfr25 = load_telemetry(dfr25_path)
    dfr57 = load_telemetry(dfr57_path)
    samples25 = sample_map(dfr25)
    samples57 = sample_map(dfr57)
    patient_ids = sorted(set(samples25) & set(samples57))
    if len(patient_ids) != len(samples25) or len(patient_ids) != len(samples57):
        raise ValueError(f"seed {seed}: sample ids differ between DFR-25 and DFR-57")

    fixed = [
        pid
        for pid in patient_ids
        if not fusion_correct(samples25[pid]) and fusion_correct(samples57[pid])
    ]
    broken = [
        pid
        for pid in patient_ids
        if fusion_correct(samples25[pid]) and not fusion_correct(samples57[pid])
    ]
    both_correct = [
        pid
        for pid in patient_ids
        if fusion_correct(samples25[pid]) and fusion_correct(samples57[pid])
    ]
    both_wrong = [
        pid
        for pid in patient_ids
        if not fusion_correct(samples25[pid]) and not fusion_correct(samples57[pid])
    ]
    all57_samples = [samples57[pid] for pid in patient_ids]
    correct57_samples = [samples57[pid] for pid in patient_ids if fusion_correct(samples57[pid])]
    wrong57_samples = [samples57[pid] for pid in patient_ids if not fusion_correct(samples57[pid])]

    return {
        "seed": seed,
        "num_samples": len(patient_ids),
        "dfr25_acc": round_float(dfr25["summary"]["full_fusion_metrics"]["accuracy"]),
        "dfr57_acc": round_float(dfr57["summary"]["full_fusion_metrics"]["accuracy"]),
        "correctness_transition": {
            "both_correct": len(both_correct),
            "both_wrong": len(both_wrong),
            "fixed_by_dfr57": len(fixed),
            "broken_by_dfr57": len(broken),
            "net_correct_delta": len(fixed) - len(broken),
        },
        "dfr25_error_types": counter_dict(Counter(error_type(samples25[pid]) for pid in patient_ids)),
        "dfr57_error_types": counter_dict(Counter(error_type(samples57[pid]) for pid in patient_ids)),
        "fixed_labels": counter_dict(Counter(samples57[pid]["label"] for pid in fixed)),
        "broken_labels": counter_dict(Counter(samples57[pid]["label"] for pid in broken)),
        "fixed_dfr57_top_weight": counter_dict(Counter(top_view(samples57[pid]) for pid in fixed)),
        "broken_dfr57_top_weight": counter_dict(Counter(top_view(samples57[pid]) for pid in broken)),
        "top_weight_migration": counter_dict(
            Counter((top_view(samples25[pid]), top_view(samples57[pid])) for pid in patient_ids)
        ),
        "dfr57_top_weight_by_final_group": {
            "all": counter_dict(Counter(top_view(sample) for sample in all57_samples)),
            "correct": counter_dict(Counter(top_view(sample) for sample in correct57_samples)),
            "wrong": counter_dict(Counter(top_view(sample) for sample in wrong57_samples)),
        },
        "dfr57_average_weights_by_final_group": {
            "all": average_weights(all57_samples),
            "correct": average_weights(correct57_samples),
            "wrong": average_weights(wrong57_samples),
        },
        "per_view_accuracy_delta": per_view_accuracy_delta(dfr25, dfr57),
        "fixed_cases": [changed_case(pid, samples25[pid], samples57[pid]) for pid in fixed],
        "broken_cases": [changed_case(pid, samples25[pid], samples57[pid]) for pid in broken],
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    seed_reports = []
    for seed, paths in DEFAULT_RUNS.items():
        seed_reports.append(
            analyze_seed(seed, repo_root / paths["dfr25"], repo_root / paths["dfr57"])
        )

    fixed_patient_ids = [
        case["patient_id"]
        for seed_report in seed_reports
        for case in seed_report["fixed_cases"]
    ]
    broken_patient_ids = [
        case["patient_id"]
        for seed_report in seed_reports
        for case in seed_report["broken_cases"]
    ]
    aggregate = {
        "fixed_by_dfr57": sum(
            item["correctness_transition"]["fixed_by_dfr57"] for item in seed_reports
        ),
        "broken_by_dfr57": sum(
            item["correctness_transition"]["broken_by_dfr57"] for item in seed_reports
        ),
        "net_correct_delta": sum(
            item["correctness_transition"]["net_correct_delta"] for item in seed_reports
        ),
        "fixed_by_seed": {
            item["seed"]: item["correctness_transition"]["fixed_by_dfr57"]
            for item in seed_reports
        },
        "broken_by_seed": {
            item["seed"]: item["correctness_transition"]["broken_by_dfr57"]
            for item in seed_reports
        },
        "repeated_fixed_patients": counter_dict(
            Counter(pid for pid in fixed_patient_ids if fixed_patient_ids.count(pid) > 1)
        ),
        "repeated_broken_patients": counter_dict(
            Counter(pid for pid in broken_patient_ids if broken_patient_ids.count(pid) > 1)
        ),
    }
    return {
        "analysis": "dfr57_vs_dfr25_followup",
        "inputs": DEFAULT_RUNS,
        "aggregate": aggregate,
        "seeds": seed_reports,
    }


def print_summary(report: dict[str, Any]) -> None:
    agg = report["aggregate"]
    print("DFR57 vs DFR25 follow-up")
    print(
        f"aggregate fixed={agg['fixed_by_dfr57']} broken={agg['broken_by_dfr57']} "
        f"net={agg['net_correct_delta']}"
    )
    for item in report["seeds"]:
        transition = item["correctness_transition"]
        print(
            f"seed {item['seed']}: acc {item['dfr25_acc']:.6f} -> {item['dfr57_acc']:.6f}; "
            f"fixed={transition['fixed_by_dfr57']} broken={transition['broken_by_dfr57']} "
            f"net={transition['net_correct_delta']}"
        )
        print(f"  DFR57 top-weight all: {item['dfr57_top_weight_by_final_group']['all']}")
        print(f"  per-view acc delta: {item['per_view_accuracy_delta']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing runs/ telemetry files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full comparison report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report = build_report(repo_root)
    print_summary(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
