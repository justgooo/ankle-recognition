#!/usr/bin/env python3
"""Trainability audit for supervised axial FP-risk calibration (DFR-128).

This report is intentionally read-only.  It checks whether the current
train.py auxiliary CE hook and src/model.py output path can express a
positive-protected axial classifier calibration objective after DFR-127 closed
label-free axial-risk gating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DFR127_REPORT = "autoresearch_logs/dfr127_axial_fp_calibration_frontier.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr128_axial_calibration_trainability_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def source_matches(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        for pattern in patterns:
            if pattern in line:
                matches.append(
                    {
                        "line": lineno,
                        "pattern": pattern,
                        "text": line.strip(),
                    }
                )
                break
    return matches


def results_rows(path: Path, keys: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t") if lines else []
    for line in lines[1:]:
        if not any(key in line for key in keys):
            continue
        parts = line.split("\t")
        row = {header[i]: parts[i] for i in range(min(len(header), len(parts)))}
        rows.append(row)
    return rows


def summarize_dfr127(report: dict[str, Any]) -> dict[str, Any]:
    best_full = report["candidate_scan"]["best_full_fp_coverage"]
    return {
        "clean_sample_count": report["clean_sample_count"],
        "confusion_counts": report["confusion_counts"],
        "zero_positive_candidate_count": report["candidate_scan"][
            "zero_positive_candidate_count"
        ],
        "full_fp_coverage_candidate_count": report["candidate_scan"][
            "full_fp_coverage_candidate_count"
        ],
        "best_full_fp_coverage": {
            "candidate": best_full["candidate"],
            "fp_covered_count": best_full["fp_covered_count"],
            "positive_collateral_count": best_full["positive_collateral_count"],
            "triggered_count": best_full["triggered_count"],
            "triggered_label_counts": best_full["triggered_label_counts"],
            "fp_precision_vs_positive": best_full["fp_precision_vs_positive"],
        },
        "key_auc": {
            "axial_abnormal": report["feature_separability_auc"]["axial_abnormal"],
            "axial_minus_max_nonaxial": report["feature_separability_auc"][
                "axial_minus_max_nonaxial"
            ],
            "sagittal_normal": report["feature_separability_auc"]["sagittal_normal"],
        },
        "clean_tp_high_axial_count": len(report["clean_tp_high_axial_records"]),
        "interpretation": report["interpretation"],
    }


def train_hook_capability(repo_root: Path) -> dict[str, Any]:
    train_py = repo_root / "train.py"
    model_py = repo_root / "src" / "model.py"
    return {
        "train_py_aux_hook": {
            "source_matches": source_matches(
                train_py,
                [
                    "getattr(model, '_view_logits'",
                    "getattr(model, '_log_vars'",
                    "_F.cross_entropy(_vl[:, _v], labels",
                    "loss = loss + _aux / _num_views",
                ],
            ),
            "accepts_model_supplied_aux_logits": True,
            "label_source": "train.py passes batch labels into CE; model forward does not receive labels.",
            "class_weighted_aux_ce": False,
            "sample_weighted_aux_ce": False,
        },
        "model_aux_output_path": {
            "source_matches": source_matches(
                model_py,
                [
                    "def _set_aux_view_loss_state",
                    "self._view_logits = aux_logits",
                    "self._log_vars = torch.full",
                    "aux_logits = torch.where",
                    "aux_logits = view_logits",
                ],
            ),
            "inactive_noop_possible": (
                "Existing aux modes already use detached logits or torch.where(..., detached) "
                "to make inactive samples no-op under the unchanged train.py CE hook."
            ),
        },
    }


def existing_aux_inventory(repo_root: Path) -> list[dict[str, Any]]:
    model_py = repo_root / "src" / "model.py"
    results = results_rows(
        repo_root / "results.tsv",
        [
            "DFR-66-",
            "DFR-90-",
            "DFR-91-",
            "DFR-92-",
            "DFR-93-",
            "DFR-109-",
        ],
    )
    return [
        {
            "mode": "global_per_view_ce",
            "env": "ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS",
            "source_present": bool(
                source_matches(model_py, ["ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS"])
            ),
            "scope": "all view logits, all samples",
            "historical_signal": [
                row for row in results if "DFR-66-" in row.get("description", "")
            ],
            "fit_for_dfr127_issue": "No. Prior DFR-66 showed global per-view CE destabilized seed-specific routing and is not targeted to high-axial-risk negatives.",
        },
        {
            "mode": "nonaxial_abnormal_ce",
            "env": "ANKLE_DECISION_ENABLE_NONAXIAL_ABNORMAL_AUX_LOSS",
            "source_present": bool(
                source_matches(
                    model_py, ["ANKLE_DECISION_ENABLE_NONAXIAL_ABNORMAL_AUX_LOSS"]
                )
            ),
            "scope": "coronal/sagittal abnormal logits, not axial calibration",
            "historical_signal": [
                row
                for row in results
                if any(tag in row.get("description", "") for tag in ["DFR-90-", "DFR-91-", "DFR-92-"])
            ],
            "fit_for_dfr127_issue": "No. It improves or probes non-axial abnormal evidence, while DFR-127 isolated high axial abnormal false positives.",
        },
        {
            "mode": "fp_risk_axial_sagittal_gate_rank",
            "env": "ANKLE_DECISION_ENABLE_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_LOSS",
            "source_present": bool(
                source_matches(
                    model_py,
                    ["ANKLE_DECISION_ENABLE_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_LOSS"],
                )
            ),
            "scope": "fusion-weight rank between axial and sagittal, not axial classifier logits",
            "historical_signal": [
                row for row in results if "DFR-109-" in row.get("description", "")
            ],
            "fit_for_dfr127_issue": "Partial but insufficient. It targets routing under FP-risk and already failed to release sagittal top routing; it does not recalibrate the axial classifier that DFR-127 identified as the error source.",
        },
        {
            "mode": "coronal_pair_abnormal_ce",
            "env": "ANKLE_DECISION_ENABLE_CORONAL_PAIR_ABNORMAL_AUX_LOSS",
            "source_present": bool(
                source_matches(
                    model_py, ["ANKLE_DECISION_ENABLE_CORONAL_PAIR_ABNORMAL_AUX_LOSS"]
                )
            ),
            "scope": "coronal abnormal plus axial-coronal pair selector",
            "historical_signal": [
                row for row in results if "DFR-93-" in row.get("description", "")
            ],
            "fit_for_dfr127_issue": "No. It addresses coronal positive evidence and released wrong competition rather than correcting high axial FP.",
        },
    ]


def proposed_objective(dfr127_summary: dict[str, Any]) -> dict[str, Any]:
    candidate = dfr127_summary["best_full_fp_coverage"]["candidate"]
    return {
        "name": "axial_fp_risk_normal_aux_loss",
        "can_be_expressed_without_train_py_change": True,
        "requires_src_model_change": True,
        "requires_data_or_split_change": False,
        "requires_test_metrics": False,
        "trigger": {
            "source": "DFR-127 best full-FP coverage observable frontier; thresholds should be treated as a first candidate, not a final broad search.",
            "conditions": {
                "predicted_positive": True,
                "fused_abnormal_min": candidate["fused_min"],
                "axial_abnormal_min": candidate["axial_min"],
                "sagittal_normal_min": candidate["sagittal_normal_min"],
                "coronal_abnormal_max": candidate["coronal_max"],
            },
            "validation_activation_from_dfr127": dfr127_summary[
                "best_full_fp_coverage"
            ],
        },
        "aux_logits_design": {
            "active_samples": (
                "Expose one aux view with logits [axial_normal_logit, "
                "detach(axial_abnormal_logit)]."
            ),
            "inactive_samples": (
                "Expose detached copies of the same logits so train.py CE is a no-op."
            ),
            "why_normal_only": (
                "Label 0 increases axial normal evidence; label 1 decreases axial normal "
                "evidence while leaving abnormal logit untouched, so positives are protected "
                "without needing labels inside model.forward."
            ),
        },
        "ce_gradient_sign": {
            "label_0_negative": {
                "dL_d_normal_logit": "p_normal - 1 < 0",
                "optimizer_step_effect": "increase axial normal logit; lower axial abnormal probability",
            },
            "label_1_positive": {
                "dL_d_normal_logit": "p_normal > 0",
                "optimizer_step_effect": "decrease axial normal logit; preserve or sharpen axial abnormal probability",
            },
            "abnormal_logit": "detached in the proposed first implementation",
        },
        "limitations": [
            "The unchanged train.py aux CE has no class weights and no per-sample weights.",
            "model.forward cannot select only negative examples because labels are not passed into the model.",
            "DFR-127 validation activation is positive-heavy (51 TP vs 7 FP), so the first run should use very low aux weights.",
            "Thresholds are derived from validation telemetry and should be treated as a bounded hypothesis, not a final clinical rule.",
        ],
        "recommended_next_experiment": {
            "id": "DFR-129",
            "edit_scope": "src/model.py plus a narrow optuna/main-search config",
            "env_flag": "ANKLE_DECISION_ENABLE_AXIAL_FP_RISK_NORMAL_AUX_LOSS",
            "weight_candidates": [0.0005, 0.001, 0.0025],
            "expected_signal": (
                "DFR-25/DFR-116 style axial-majority routing should remain; the target "
                "is lower axial FP recurrence without broad sagittal/coronal rerouting."
            ),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    dfr127 = load_json(repo_root / args.dfr127_report)
    dfr127_summary = summarize_dfr127(dfr127)
    capability = train_hook_capability(repo_root)
    inventory = existing_aux_inventory(repo_root)
    proposal = proposed_objective(dfr127_summary)

    return {
        "analysis": "dfr128_axial_calibration_trainability_audit",
        "description": (
            "Read-only audit of whether supervised positive-protected axial classifier "
            "calibration is expressible through the existing train.py aux hook after "
            "DFR-127 ruled out label-free axial-risk gating."
        ),
        "dfr127_summary": dfr127_summary,
        "trainability": capability,
        "existing_aux_inventory": inventory,
        "proposal": proposal,
        "assessment": {
            "label_free_gate_closed": True,
            "existing_gate_rank_aux_already_tried": True,
            "positive_protected_axial_classifier_aux_feasible": True,
            "can_run_without_train_py_change": True,
            "needs_model_code_before_training": True,
            "promote_now": False,
            "status": "analysis_positive_keep",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--dfr127-report", default=DEFAULT_DFR127_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.repo_root).resolve() / args.output
    save_json(report, output_path)
    print(f"Wrote {output_path}")
    print(
        "DFR-128 assessment: "
        f"positive_protected_axial_classifier_aux_feasible="
        f"{report['assessment']['positive_protected_axial_classifier_aux_feasible']}, "
        f"can_run_without_train_py_change={report['assessment']['can_run_without_train_py_change']}"
    )


if __name__ == "__main__":
    main()
