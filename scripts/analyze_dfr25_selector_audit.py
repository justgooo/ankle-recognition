#!/usr/bin/env python3
"""Audit whether DFR-25 telemetry supports a learned view selector target."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VIEWS = ("axial", "coronal", "sagittal")
VIEW_TO_INDEX = {view: index for index, view in enumerate(VIEWS)}
NONAXIAL = ("coronal", "sagittal")

DEFAULT_DFR25_RUNS = {
    "42": "runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/trials/trial_0001/run/fusion_weight_analysis.json",
    "123": "runs/optuna_main_autoloop/iter_0002_20260423_031222/trials/trial_0000/run/fusion_weight_analysis.json",
    "456": "runs/optuna_main_autoloop/iter_0003_20260423_033048/trials/trial_0000/run/fusion_weight_analysis.json",
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


def top_view(sample: dict[str, Any], key: str = "top_weight_views") -> str:
    values = sample.get(key) or []
    return str(values[0]) if values else "none"


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


def signed_margin(sample: dict[str, Any], view: str) -> float:
    margin = pred_margin(sample, view)
    return margin if view_pred(sample, view) == 1 else -margin


def load_seed_samples(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    seed_samples: dict[str, list[dict[str, Any]]] = {}
    for seed, relative_path in DEFAULT_DFR25_RUNS.items():
        path = repo_root / relative_path
        seed_samples[seed] = list(load_json(path)["samples"])
    return seed_samples


def telemetry_summary(repo_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for seed, relative_path in DEFAULT_DFR25_RUNS.items():
        path = repo_root / relative_path
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
        summary[seed] = {
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
    return summary


def view_feature(sample: dict[str, Any], view: str) -> list[float]:
    data = sample["views"][view]
    index = VIEW_TO_INDEX[view]
    prob = abnormal_prob(sample, view)
    one_hot = [1.0 if index == item else 0.0 for item in range(len(VIEWS))]
    return [
        float(data["fusion_weight"]),
        float(data["confidence_logit"]),
        float(data["scaled_confidence_logit"]),
        prob,
        abs(prob - 0.5),
        pred_margin(sample, view),
        signed_margin(sample, view),
        1.0 if view_pred(sample, view) == 1 else 0.0,
        *one_hot,
    ]


def sample_feature(sample: dict[str, Any]) -> list[float]:
    features: list[float] = []
    for view in VIEWS:
        features.extend(view_feature(sample, view))
    for left, right in (("coronal", "axial"), ("sagittal", "axial"), ("sagittal", "coronal")):
        features.extend(
            [
                pred_margin(sample, left) - pred_margin(sample, right),
                signed_margin(sample, left) - signed_margin(sample, right),
                abnormal_prob(sample, left) - abnormal_prob(sample, right),
                float(sample["views"][left]["confidence_logit"])
                - float(sample["views"][right]["confidence_logit"]),
                float(sample["views"][left]["fusion_weight"])
                - float(sample["views"][right]["fusion_weight"]),
            ]
        )
    return features


def pair_feature(sample: dict[str, Any], left: str, right: str) -> list[float]:
    left_features = view_feature(sample, left)
    right_features = view_feature(sample, right)
    diff = [left_value - right_value for left_value, right_value in zip(left_features, right_features)]
    return [
        *diff,
        abs(pred_margin(sample, left) - pred_margin(sample, right)),
        abs(signed_margin(sample, left) - signed_margin(sample, right)),
        1.0 if view_pred(sample, left) == view_pred(sample, right) else 0.0,
    ]


def selector_target(sample: dict[str, Any]) -> int:
    correct_views = [view for view in VIEWS if view_correct(sample, view)]
    if correct_views:
        target = max(correct_views, key=lambda view: true_margin(sample, view))
    else:
        target = max(VIEWS, key=lambda view: true_margin(sample, view))
    return VIEW_TO_INDEX[target]


def view_correct_pattern(sample: dict[str, Any]) -> str:
    return "".join(view[0].upper() if view_correct(sample, view) else "-" for view in VIEWS)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "labels": {},
            "error_types": {},
            "top_true_margin": {},
            "top_pred_margin": {},
            "view_correct_patterns": {},
        }
    return {
        "count": len(samples),
        "labels": counter_dict(Counter(int(sample["label"]) for sample in samples)),
        "error_types": counter_dict(Counter(error_type(sample) for sample in samples)),
        "top_true_margin": counter_dict(Counter(top_view(sample, "top_true_margin_views") for sample in samples)),
        "top_pred_margin": counter_dict(Counter(top_view(sample, "top_pred_margin_views") for sample in samples)),
        "view_correct_patterns": counter_dict(Counter(view_correct_pattern(sample) for sample in samples)),
        "mean_axial_weight": round_float(
            mean(float(sample["views"]["axial"]["fusion_weight"]) for sample in samples)
        ),
        "mean_best_nonaxial_true_margin": round_float(
            mean(max(true_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
        "mean_best_nonaxial_pred_margin": round_float(
            mean(max(pred_margin(sample, view) for view in NONAXIAL) for sample in samples)
        ),
    }


def summarize_selection(
    samples: list[dict[str, Any]],
    selected_indices: list[int],
) -> dict[str, Any]:
    selected_views = [VIEWS[index] for index in selected_indices]
    selected_correct = [
        sample
        for sample, selected_view in zip(samples, selected_views)
        if view_correct(sample, selected_view)
    ]
    fixed = [
        sample
        for sample, selected_view in zip(samples, selected_views)
        if not fusion_correct(sample) and view_correct(sample, selected_view)
    ]
    broken = [
        sample
        for sample, selected_view in zip(samples, selected_views)
        if fusion_correct(sample) and not view_correct(sample, selected_view)
    ]
    nonaxial_selected = [
        sample
        for sample, selected_view in zip(samples, selected_views)
        if selected_view in NONAXIAL
    ]
    return {
        "selected_view_distribution": counter_dict(Counter(selected_views)),
        "selected_view_accuracy": round_float(safe_rate(len(selected_correct), len(samples))),
        "fixed_dfr25_errors": len(fixed),
        "broken_dfr25_correct": len(broken),
        "net_delta_vs_dfr25": len(fixed) - len(broken),
        "nonaxial_selected": len(nonaxial_selected),
        "nonaxial_selected_rate": round_float(safe_rate(len(nonaxial_selected), len(samples))),
        "fixed_summary": summarize_samples(fixed),
        "broken_summary": summarize_samples(broken),
        "nonaxial_selected_summary": summarize_samples(nonaxial_selected),
        "fixed_case_ids": [str(sample["patient_id"]) for sample in fixed],
        "broken_case_ids": [str(sample["patient_id"]) for sample in broken],
    }


def oracle_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    target_indices = [selector_target(sample) for sample in samples]
    any_correct = [sample for sample in samples if any(view_correct(sample, view) for view in VIEWS)]
    dfr25_wrong = [sample for sample in samples if not fusion_correct(sample)]
    correctable_wrong = [
        sample
        for sample in dfr25_wrong
        if any(view_correct(sample, view) for view in VIEWS)
    ]
    nonaxial_rescue_wrong = [
        sample
        for sample in dfr25_wrong
        if not view_correct(sample, "axial")
        and any(view_correct(sample, view) for view in NONAXIAL)
    ]
    axial_protection_cases = [
        sample
        for sample in samples
        if fusion_correct(sample)
        and view_correct(sample, "axial")
        and any(not view_correct(sample, view) for view in NONAXIAL)
    ]
    return {
        "any_view_correct_accuracy_upper_bound": round_float(safe_rate(len(any_correct), len(samples))),
        "dfr25_wrong": len(dfr25_wrong),
        "dfr25_wrong_correctable_by_any_view": len(correctable_wrong),
        "dfr25_wrong_correctable_by_nonaxial_when_axial_wrong": len(nonaxial_rescue_wrong),
        "axial_correct_nonaxial_wrong_protection_cases": len(axial_protection_cases),
        "oracle_selected_view_distribution": counter_dict(Counter(VIEWS[index] for index in target_indices)),
        "oracle_selection": summarize_selection(samples, target_indices),
        "dfr25_wrong_summary": summarize_samples(dfr25_wrong),
        "nonaxial_rescue_wrong_summary": summarize_samples(nonaxial_rescue_wrong),
        "axial_protection_summary": summarize_samples(axial_protection_cases),
    }


def fit_multiclass_selector(
    train_samples: list[dict[str, Any]],
) -> Any:
    x_train = np.asarray([sample_feature(sample) for sample in train_samples], dtype=np.float64)
    y_train = np.asarray([selector_target(sample) for sample in train_samples], dtype=np.int64)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    ).fit(x_train, y_train)


def pairwise_training_data(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        for left_index, left in enumerate(VIEWS):
            for right in VIEWS[left_index + 1 :]:
                if view_correct(sample, left) == view_correct(sample, right):
                    continue
                features.append(pair_feature(sample, left, right))
                labels.append(1 if view_correct(sample, left) else 0)
                features.append(pair_feature(sample, right, left))
                labels.append(1 if view_correct(sample, right) else 0)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def fit_pairwise_selector(train_samples: list[dict[str, Any]]) -> Any:
    x_train, y_train = pairwise_training_data(train_samples)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("pairwise training data must contain both classes")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    ).fit(x_train, y_train)


def pairwise_tournament_select(model: Any, samples: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []
    for sample in samples:
        scores = {view: 0.0 for view in VIEWS}
        for left_index, left in enumerate(VIEWS):
            for right in VIEWS[left_index + 1 :]:
                probability_left = float(
                    model.predict_proba(
                        np.asarray([pair_feature(sample, left, right)], dtype=np.float64)
                    )[0, 1]
                )
                scores[left] += probability_left
                scores[right] += 1.0 - probability_left
        selected_view = max(VIEWS, key=lambda view: (scores[view], -VIEW_TO_INDEX[view]))
        selected.append(VIEW_TO_INDEX[selected_view])
    return selected


def pairwise_holdout_metrics(model: Any, samples: list[dict[str, Any]]) -> dict[str, Any]:
    x_test, y_test = pairwise_training_data(samples)
    if len(y_test) == 0:
        return {"pair_examples": 0, "accuracy": 0.0, "auc": 0.0}
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    auc = roc_auc_score(y_test, probabilities) if len(set(y_test.tolist())) > 1 else 0.0
    return {
        "pair_examples": int(len(y_test)),
        "accuracy": round_float(float((predictions == y_test).mean())),
        "auc": round_float(float(auc)),
    }


def cross_seed_audit(seed_samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    multiclass_reports: dict[str, Any] = {}
    pairwise_reports: dict[str, Any] = {}
    all_multiclass_samples: list[dict[str, Any]] = []
    all_multiclass_selected: list[int] = []
    all_pairwise_samples: list[dict[str, Any]] = []
    all_pairwise_selected: list[int] = []

    for heldout_seed, test_samples in seed_samples.items():
        train_samples = [
            sample
            for seed, samples in seed_samples.items()
            if seed != heldout_seed
            for sample in samples
        ]

        multiclass_model = fit_multiclass_selector(train_samples)
        x_test = np.asarray([sample_feature(sample) for sample in test_samples], dtype=np.float64)
        multiclass_selected = multiclass_model.predict(x_test).astype(int).tolist()
        multiclass_reports[heldout_seed] = summarize_selection(test_samples, multiclass_selected)
        all_multiclass_samples.extend(test_samples)
        all_multiclass_selected.extend(multiclass_selected)

        pairwise_model = fit_pairwise_selector(train_samples)
        pairwise_selected = pairwise_tournament_select(pairwise_model, test_samples)
        pairwise_reports[heldout_seed] = {
            **summarize_selection(test_samples, pairwise_selected),
            "pairwise_holdout": pairwise_holdout_metrics(pairwise_model, test_samples),
        }
        all_pairwise_samples.extend(test_samples)
        all_pairwise_selected.extend(pairwise_selected)

    return {
        "multiclass_selector_leave_one_seed_out": {
            "by_seed": multiclass_reports,
            "aggregate": summarize_selection(all_multiclass_samples, all_multiclass_selected),
        },
        "pairwise_selector_leave_one_seed_out": {
            "by_seed": pairwise_reports,
            "aggregate": summarize_selection(all_pairwise_samples, all_pairwise_selected),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing existing DFR-25 fusion telemetry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autoresearch_logs/dfr85_selector_audit/report.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    seed_samples = load_seed_samples(repo_root)
    all_samples = [sample for samples in seed_samples.values() for sample in samples]
    report = {
        "analysis": "dfr25_selector_audit",
        "description": (
            "Reads existing DFR-25 fusion_weight_analysis telemetry and audits "
            "whether a supervised sample-level or pairwise view selector target "
            "has enough cross-seed signal to justify a future routing mechanism."
        ),
        "dfr25_inputs": DEFAULT_DFR25_RUNS,
        "dfr25_telemetry": telemetry_summary(repo_root),
        "oracle_upper_bound": oracle_report(all_samples),
        "cross_seed_learnability": cross_seed_audit(seed_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    oracle = report["oracle_upper_bound"]
    multiclass = report["cross_seed_learnability"]["multiclass_selector_leave_one_seed_out"]["aggregate"]
    pairwise = report["cross_seed_learnability"]["pairwise_selector_leave_one_seed_out"]["aggregate"]
    print(f"Saved DFR-25 selector audit to: {args.output}")
    print(
        "oracle: any_view_correct_acc="
        f"{oracle['any_view_correct_accuracy_upper_bound']:.6f} "
        f"correctable_wrong={oracle['dfr25_wrong_correctable_by_any_view']} "
        f"nonaxial_rescue_wrong={oracle['dfr25_wrong_correctable_by_nonaxial_when_axial_wrong']} "
        f"axial_protection={oracle['axial_correct_nonaxial_wrong_protection_cases']}"
    )
    print(
        "multiclass_loso: selected_acc="
        f"{multiclass['selected_view_accuracy']:.6f} fixed={multiclass['fixed_dfr25_errors']} "
        f"broken={multiclass['broken_dfr25_correct']} net={multiclass['net_delta_vs_dfr25']} "
        f"top={multiclass['selected_view_distribution']}"
    )
    print(
        "pairwise_loso: selected_acc="
        f"{pairwise['selected_view_accuracy']:.6f} fixed={pairwise['fixed_dfr25_errors']} "
        f"broken={pairwise['broken_dfr25_correct']} net={pairwise['net_delta_vs_dfr25']} "
        f"top={pairwise['selected_view_distribution']}"
    )


if __name__ == "__main__":
    main()
