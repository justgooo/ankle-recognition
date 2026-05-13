#!/usr/bin/env python3
"""Reproducibility report for the DFR-116 posthoc combo branch.

This is a packaging/checking step, not a new model search.  It verifies that
the retained DFR-116 eval-side combo branch is reproducible from committed
configs plus existing DFR-25 checkpoints, and records the exact patient-level
diff against DFR-25 validation telemetry.  No training or test split is used.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_dfr111_confirmed_target_posthoc import (  # noqa: E402
    VIEWS,
    base_error_type,
    counter_dict,
    load_json,
    round_float,
    save_json,
)
from scripts.analyze_dfr112_multiseed_posthoc import DFR25_TELEMETRY  # noqa: E402


DEFAULT_OUTPUT = "autoresearch_logs/dfr119_dfr116_branch_repro_report.json"
SEEDS = ("42", "123", "456")
DFR116_CONFIGS = {
    "42": "configs/cmp_resnext_decision_256x8_dfr116_posthoc_combo_gate_formal_s42.yaml",
    "123": "configs/cmp_resnext_decision_256x8_dfr116_posthoc_combo_gate_formal_s123.yaml",
    "456": "configs/cmp_resnext_decision_256x8_dfr116_posthoc_combo_gate_formal_s456.yaml",
}
DFR116_TELEMETRY = {
    "42": "runs/resnext_decision_256x8_mainline/dfr116_posthoc_combo_gate_formal_s42/fusion_weight_analysis.json",
    "123": "runs/resnext_decision_256x8_mainline/dfr116_posthoc_combo_gate_formal_s123/fusion_weight_analysis.json",
    "456": "runs/resnext_decision_256x8_mainline/dfr116_posthoc_combo_gate_formal_s456/fusion_weight_analysis.json",
}
EXPECTED_RUNTIME_ENV = {
    "ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER": "1",
    "ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB": "0.25",
    "ANKLE_DECISION_ENABLE_POSTHOC_FP_RISK_SAGITTAL_GATE": "1",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_RESIDUAL": "5.0",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_AXIAL_ABNORMAL_MIN": "0.4",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_AXIAL_ABNORMAL_MAX": "0.88",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_FUSED_ABNORMAL_MIN": "0.8",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_SAGITTAL_NORMAL_MIN": "0.55",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_CORONAL_ABNORMAL_MAX": "0.525",
    "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_CONFIDENCE_GAP_MAX": "5.0",
    "ANKLE_DECISION_ENABLE_POSTHOC_FN_ABNORMAL_RESCUE_GATE": "1",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_RESIDUAL": "6.0",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_FUSED_ABNORMAL_MAX": "0.2",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_AXIAL_ABNORMAL_MIN": "0.45",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_MAX_ABNORMAL_MIN": "0.45",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_SECOND_ABNORMAL_MIN": "0.0",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_CONFIDENCE_GAP_MAX": "1.0",
    "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_AXIAL_ABNORMAL_MAX": "0.65",
}
EXPECTED_MEAN = {
    "accuracy": 0.9503546099290779,
    "auc": 0.9715151515151516,
    "f1": 0.9469648049655799,
}
EXPECTED_TOP_COUNTS = {"axial": 236, "coronal": 0, "sagittal": 46}
EXPECTED_FIXED = {
    "42:CTyin__CT24yin21",
    "42:CTyin__CT24yin95",
    "123:CTyang__CT24yang1__CT2412yang147",
}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expected_checkpoint_for_seed(seed: str) -> str:
    return str(Path(DFR25_TELEMETRY[seed]).with_name("best.pt"))


def normalize_env(env: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in dict(env).items()}


def top_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def metric_summary(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in per_seed.values()]
    top_counts = Counter()
    for item in per_seed.values():
        top_counts.update(item["top_weight_count"])
    return {
        "mean_metrics": {
            "accuracy": round_float(mean([float(item["accuracy"]) for item in metrics])),
            "auc": round_float(mean([float(item["auc"]) for item in metrics])),
            "f1": round_float(mean([float(item["f1"]) for item in metrics])),
        },
        "aggregate_top_weight_count": {view: int(top_counts.get(view, 0)) for view in VIEWS},
    }


def per_seed_summary(telemetry: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": {
            key: round_float(float(value))
            for key, value in telemetry["summary"]["full_fusion_metrics"].items()
        },
        "top_weight_count": {
            view: int(telemetry["summary"]["top_weight_view_distribution"].get(view, 0))
            for view in VIEWS
        },
        "mean_fusion_weight": {
            view: round_float(
                mean([float(sample["views"][view]["fusion_weight"]) for sample in telemetry["samples"]])
            )
            for view in VIEWS
        },
    }


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def patient_diff(seed: str, dfr25: dict[str, Any], dfr116: dict[str, Any]) -> dict[str, Any]:
    base_samples = sample_map(dfr25)
    combo_samples = sample_map(dfr116)
    fixed: list[dict[str, Any]] = []
    broken: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for patient_id, base_sample in base_samples.items():
        combo_sample = combo_samples[patient_id]
        label = int(base_sample["label"])
        base_pred = int(base_sample["fusion_prediction"]["pred"])
        combo_pred = int(combo_sample["fusion_prediction"]["pred"])
        if base_pred == combo_pred:
            continue
        record = {
            "seed": seed,
            "patient_id": patient_id,
            "label": label,
            "dfr25_pred": base_pred,
            "dfr116_pred": combo_pred,
            "dfr25_error_type": base_error_type(label, base_pred),
            "dfr116_error_type": base_error_type(label, combo_pred),
            "dfr25_abnormal": round_float(float(base_sample["fusion_prediction"]["abnormal_prob"])),
            "dfr116_abnormal": round_float(float(combo_sample["fusion_prediction"]["abnormal_prob"])),
            "dfr25_top_weight": top_view(base_sample),
            "dfr116_top_weight": top_view(combo_sample),
            "dfr25_view_abnormal": {
                view: round_float(float(base_sample["views"][view]["abnormal_prob"]))
                for view in VIEWS
            },
            "dfr116_view_abnormal": {
                view: round_float(float(combo_sample["views"][view]["abnormal_prob"]))
                for view in VIEWS
            },
        }
        changed.append(record)
        if base_pred != label and combo_pred == label:
            fixed.append(record)
        elif base_pred == label and combo_pred != label:
            broken.append(record)
    return {
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
    }


def close_enough(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def reproduction_command(seed: str) -> str:
    return (
        ".venv/bin/python scripts/analyze_fusion_weights.py "
        f"--config {DFR116_CONFIGS[seed]} "
        f"--checkpoint {expected_checkpoint_for_seed(seed)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    dfr25_telemetry = {
        seed: load_json(repo_root / DFR25_TELEMETRY[seed]) for seed in SEEDS
    }
    dfr116_telemetry = {
        seed: load_json(repo_root / DFR116_TELEMETRY[seed]) for seed in SEEDS
    }
    configs = {seed: load_yaml(repo_root / DFR116_CONFIGS[seed]) for seed in SEEDS}

    dfr25_summary = {
        seed: per_seed_summary(dfr25_telemetry[seed]) for seed in SEEDS
    }
    dfr116_summary = {
        seed: per_seed_summary(dfr116_telemetry[seed]) for seed in SEEDS
    }
    diffs = {
        seed: patient_diff(seed, dfr25_telemetry[seed], dfr116_telemetry[seed])
        for seed in SEEDS
    }
    fixed_ids = {
        f"{seed}:{item['patient_id']}"
        for seed in SEEDS
        for item in diffs[seed]["fixed"]
    }
    broken_ids = {
        f"{seed}:{item['patient_id']}"
        for seed in SEEDS
        for item in diffs[seed]["broken"]
    }
    changed_ids = {
        f"{seed}:{item['patient_id']}"
        for seed in SEEDS
        for item in diffs[seed]["changed"]
    }

    config_env = {
        seed: normalize_env(configs[seed].get("runtime_env", {})) for seed in SEEDS
    }
    telemetry_env = {
        seed: normalize_env(dfr116_telemetry[seed].get("runtime_env", {}))
        for seed in SEEDS
    }

    aggregate = metric_summary(dfr116_summary)
    checks = []
    checks.append(
        check(
            "all_telemetry_uses_val_split",
            all(dfr116_telemetry[seed].get("split") == "val" for seed in SEEDS),
            {seed: dfr116_telemetry[seed].get("split") for seed in SEEDS},
        )
    )
    checks.append(
        check(
            "config_runtime_env_matches_expected",
            all(config_env[seed] == EXPECTED_RUNTIME_ENV for seed in SEEDS),
            config_env,
        )
    )
    checks.append(
        check(
            "telemetry_runtime_env_matches_config",
            all(telemetry_env[seed] == config_env[seed] for seed in SEEDS),
            telemetry_env,
        )
    )
    checks.append(
        check(
            "telemetry_config_paths_match",
            all(dfr116_telemetry[seed].get("config_path") == DFR116_CONFIGS[seed] for seed in SEEDS),
            {seed: dfr116_telemetry[seed].get("config_path") for seed in SEEDS},
        )
    )
    checks.append(
        check(
            "telemetry_checkpoint_paths_match_dfr25_checkpoints",
            all(
                dfr116_telemetry[seed].get("checkpoint_path")
                == expected_checkpoint_for_seed(seed)
                for seed in SEEDS
            ),
            {seed: dfr116_telemetry[seed].get("checkpoint_path") for seed in SEEDS},
        )
    )
    checks.append(
        check(
            "patient_order_and_labels_match_dfr25",
            all(
                [
                    [
                        (base["patient_id"], base["label"])
                        for base in dfr25_telemetry[seed]["samples"]
                    ]
                    == [
                        (combo["patient_id"], combo["label"])
                        for combo in dfr116_telemetry[seed]["samples"]
                    ]
                    for seed in SEEDS
                ]
            ),
        )
    )
    checks.append(
        check(
            "aggregate_metrics_match_dfr116_expected",
            all(
                close_enough(aggregate["mean_metrics"][key], EXPECTED_MEAN[key])
                for key in EXPECTED_MEAN
            ),
            aggregate["mean_metrics"],
        )
    )
    checks.append(
        check(
            "aggregate_top_weight_matches_expected",
            aggregate["aggregate_top_weight_count"] == EXPECTED_TOP_COUNTS,
            aggregate["aggregate_top_weight_count"],
        )
    )
    checks.append(
        check(
            "patient_diff_matches_expected",
            fixed_ids == EXPECTED_FIXED and not broken_ids and changed_ids == EXPECTED_FIXED,
            {
                "fixed_ids": sorted(fixed_ids),
                "broken_ids": sorted(broken_ids),
                "changed_ids": sorted(changed_ids),
            },
        )
    )

    report = {
        "analysis": "dfr119_dfr116_branch_repro_report",
        "description": (
            "Reproducibility/package report for the retained DFR-116 posthoc "
            "combo calibration branch.  Uses validation telemetry only."
        ),
        "status": "pass" if all(item["ok"] for item in checks) else "fail",
        "validation_checks": checks,
        "configs": DFR116_CONFIGS,
        "telemetry": DFR116_TELEMETRY,
        "checkpoints": {
            seed: expected_checkpoint_for_seed(seed) for seed in SEEDS
        },
        "reproduction_commands": {
            seed: reproduction_command(seed) for seed in SEEDS
        },
        "runtime_env": EXPECTED_RUNTIME_ENV,
        "dfr25_reference": {
            "per_seed": dfr25_summary,
            "aggregate": metric_summary(dfr25_summary),
        },
        "dfr116_combo": {
            "per_seed": dfr116_summary,
            "aggregate": aggregate,
            "fixed_ids": sorted(fixed_ids),
            "broken_ids": sorted(broken_ids),
            "patient_diff": diffs,
        },
    }
    save_json(repo_root / args.output, report)

    print(f"Saved DFR-119 DFR-116 branch report to: {repo_root / args.output}")
    print("status", report["status"])
    print("dfr116", aggregate["mean_metrics"], aggregate["aggregate_top_weight_count"])
    print("fixed", sorted(fixed_ids), "broken", sorted(broken_ids))
    failed = [item for item in checks if not item["ok"]]
    if failed:
        print("failed_checks", [item["name"] for item in failed])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
