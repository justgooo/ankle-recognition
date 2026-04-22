from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_config, save_json

VIEW_NAMES = ("axial", "coronal", "sagittal")
CONTROL_NAMES = (
    "single_view_axial",
    "single_view_coronal",
    "single_view_sagittal",
    "leave_out_axial",
    "leave_out_coronal",
    "leave_out_sagittal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched fusion-weight, leave-one-view-out, and perturbation telemetry "
            "across multiple decision-fusion checkpoints, then write one aggregate report."
        )
    )
    parser.add_argument(
        "--trial-config",
        action="append",
        dest="trial_configs",
        required=True,
        help="Repeat once per trial config to include in the multiseed routing report.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--perturb-view",
        choices=VIEW_NAMES,
        default="axial",
        help="Which view to perturb for the migration analysis.",
    )
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=17,
        help="Odd Gaussian blur kernel size passed to the perturbation analysis.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=4.0,
        help="Gaussian blur sigma passed to the perturbation analysis.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to the aggregate multiseed routing summary JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-run telemetry JSONs even if they already exist.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def sigma_tag(value: float) -> str:
    return format(value, "g").replace(".", "p")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_analysis(command: list[str], output_json: Path, force: bool) -> None:
    if output_json.exists() and not force:
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, check=True)


def extract_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "auc": float(metrics["auc"]),
        "f1": float(metrics["f1"]),
    }


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: safe_mean([float(item[metric]) for item in items])
        for metric in ("accuracy", "auc", "f1")
    }


def mean_mapping(items: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    return {
        key: safe_mean([float(item[key]) for item in items])
        for key in keys
    }


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    script_dir = Path(__file__).resolve().parent
    perturb_output_name = (
        f"perturbation_summary_{args.perturb_view}_blur"
        f"{args.blur_kernel_size}_sigma{sigma_tag(args.blur_sigma)}.json"
    )

    run_reports: list[dict[str, Any]] = []
    for trial_config_arg in args.trial_configs:
        config_path = Path(trial_config_arg)
        config = load_config(config_path)
        seed = int(config["seed"])
        output_dir = Path(config["output_dir"])
        checkpoint_path = output_dir / "best.pt"
        summary_path = output_dir / "summary.json"
        fusion_json = output_dir / "fusion_weight_analysis.json"
        controls_json = output_dir / "view_ablation_summary.json"
        perturb_json = output_dir / perturb_output_name

        base_cmd = [sys.executable]
        run_analysis(
            base_cmd
            + [
                str(script_dir / "analyze_fusion_weights.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                args.split,
                "--output-json",
                str(fusion_json),
            ],
            output_json=fusion_json,
            force=args.force,
        )
        run_analysis(
            base_cmd
            + [
                str(script_dir / "analyze_view_controls.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                args.split,
                "--output-json",
                str(controls_json),
            ],
            output_json=controls_json,
            force=args.force,
        )
        run_analysis(
            base_cmd
            + [
                str(script_dir / "analyze_fusion_perturbations.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                args.split,
                "--perturb-view",
                args.perturb_view,
                "--blur-kernel-size",
                str(args.blur_kernel_size),
                "--blur-sigma",
                str(args.blur_sigma),
                "--output-json",
                str(perturb_json),
            ],
            output_json=perturb_json,
            force=args.force,
        )

        summary = load_json(summary_path)
        fusion = load_json(fusion_json)
        controls = load_json(controls_json)
        perturbation = load_json(perturb_json)

        per_view = {
            item["view"]: item
            for item in fusion["per_view"]
        }
        control_map = {
            item["name"]: item
            for item in controls["controls"]
        }
        top_weight_distribution = fusion["summary"]["top_weight_view_distribution"]
        top_true_margin_distribution = fusion["summary"]["top_true_margin_view_distribution"]
        control_delta_accuracy = {
            name: float(control_map[name]["delta_vs_full"]["accuracy"])
            for name in CONTROL_NAMES
        }
        control_delta_auc = {
            name: float(control_map[name]["delta_vs_full"]["auc"])
            for name in CONTROL_NAMES
        }
        control_delta_f1 = {
            name: float(control_map[name]["delta_vs_full"]["f1"])
            for name in CONTROL_NAMES
        }

        run_reports.append(
            {
                "seed": seed,
                "config_path": repo_relative(config_path),
                "output_dir": repo_relative(output_dir),
                "checkpoint_path": repo_relative(checkpoint_path),
                "summary_path": repo_relative(summary_path),
                "artifacts": {
                    "fusion_weight_analysis": repo_relative(fusion_json),
                    "view_ablation_summary": repo_relative(controls_json),
                    "perturbation_summary": repo_relative(perturb_json),
                },
                "best_val": {
                    "accuracy": float(summary["best_val"]["accuracy"]),
                    "auc": float(summary["best_val"]["auc"]),
                    "f1": float(summary["best_val"]["f1"]),
                },
                "runtime": {
                    "peak_vram_mb": float(summary["runtime"]["peak_vram_mb"]),
                    "total_seconds": float(summary["runtime"]["total_seconds"]),
                },
                "fusion_weight": {
                    "top_weight_hit_rate_true_margin": float(
                        fusion["summary"]["top_weight_hit_rate"]["true_margin"]
                    ),
                    "mixed_correctness_top_weight_correct_rate": float(
                        fusion["summary"]["mixed_correctness_top_weight_correct_rate"]
                    ),
                    "mean_weight_by_view": {
                        view: float(per_view[view]["mean_fusion_weight"])
                        for view in VIEW_NAMES
                    },
                    "top_weight_rate_by_view": {
                        view: float(per_view[view]["top_weight_rate"])
                        for view in VIEW_NAMES
                    },
                    "top_true_margin_rate_by_view": {
                        view: float(per_view[view]["top_true_margin_rate"])
                        for view in VIEW_NAMES
                    },
                    "top_weight_view_distribution": {
                        view: int(top_weight_distribution[view])
                        for view in VIEW_NAMES
                    },
                    "top_true_margin_view_distribution": {
                        view: int(top_true_margin_distribution[view])
                        for view in VIEW_NAMES
                    },
                },
                "view_controls": {
                    "full_fusion_metrics": extract_metrics(
                        controls["summary"]["full_fusion"]["metrics"]
                    ),
                    "best_single_view": {
                        "name": controls["summary"]["best_single_view"]["name"],
                        "metrics": extract_metrics(controls["summary"]["best_single_view"]["metrics"]),
                    },
                    "best_leave_one_out": {
                        "name": controls["summary"]["best_leave_one_out"]["name"],
                        "metrics": extract_metrics(controls["summary"]["best_leave_one_out"]["metrics"]),
                    },
                    "full_minus_best_single_view": {
                        metric: float(controls["summary"]["full_minus_best_single_view"][metric])
                        for metric in ("accuracy", "auc", "f1")
                    },
                    "full_minus_best_leave_one_out": {
                        metric: float(controls["summary"]["full_minus_best_leave_one_out"][metric])
                        for metric in ("accuracy", "auc", "f1")
                    },
                    "control_delta_accuracy": control_delta_accuracy,
                    "control_delta_auc": control_delta_auc,
                    "control_delta_f1": control_delta_f1,
                },
                "perturbation": {
                    "baseline_metrics": extract_metrics(perturbation["summary"]["baseline_metrics"]),
                    "perturbed_metrics": extract_metrics(perturbation["summary"]["perturbed_metrics"]),
                    "delta_vs_baseline": {
                        metric: float(perturbation["summary"]["delta_vs_baseline"][metric])
                        for metric in ("accuracy", "auc", "f1")
                    },
                    "top_weight_switch_rate": float(
                        perturbation["summary"]["top_weight_switch_rate"]
                    ),
                    "target_view": {
                        "name": str(perturbation["summary"]["target_view"]["name"]),
                        "mean_weight_delta": float(
                            perturbation["summary"]["target_view"]["mean_weight_delta"]
                        ),
                        "mean_true_margin_delta": float(
                            perturbation["summary"]["target_view"]["mean_true_margin_delta"]
                        ),
                        "weight_drop_rate": float(
                            perturbation["summary"]["target_view"]["weight_drop_rate"]
                        ),
                        "top_weight_rate_before": float(
                            perturbation["summary"]["target_view"]["top_weight_rate_before"]
                        ),
                        "top_weight_rate_after": float(
                            perturbation["summary"]["target_view"]["top_weight_rate_after"]
                        ),
                        "migrated_away_rate": float(
                            perturbation["summary"]["target_view"]["migrated_away_rate"]
                        ),
                        "margin_drop_and_weight_drop_rate": float(
                            perturbation["summary"]["target_view"]["margin_drop_and_weight_drop_rate"]
                        ),
                    },
                    "top_weight_view_distribution": {
                        phase: {
                            view: int(perturbation["summary"]["top_weight_view_distribution"][phase][view])
                            for view in VIEW_NAMES
                        }
                        for phase in ("before", "after", "migration_destinations_after_leaving_target")
                    },
                },
            }
        )

    run_reports.sort(key=lambda item: item["seed"])

    aggregate = {
        "analysis": "multiseed_decision_routing",
        "split": args.split,
        "perturbation": {
            "view": args.perturb_view,
            "blur_kernel_size": int(args.blur_kernel_size),
            "blur_sigma": float(args.blur_sigma),
        },
        "num_runs": len(run_reports),
        "best_val_mean": mean_metrics([item["best_val"] for item in run_reports]),
        "runtime_mean": {
            "peak_vram_mb": safe_mean(
                [float(item["runtime"]["peak_vram_mb"]) for item in run_reports]
            ),
            "total_seconds": safe_mean(
                [float(item["runtime"]["total_seconds"]) for item in run_reports]
            ),
        },
        "fusion_weight_mean": {
            "top_weight_hit_rate_true_margin": safe_mean(
                [
                    float(item["fusion_weight"]["top_weight_hit_rate_true_margin"])
                    for item in run_reports
                ]
            ),
            "mixed_correctness_top_weight_correct_rate": safe_mean(
                [
                    float(item["fusion_weight"]["mixed_correctness_top_weight_correct_rate"])
                    for item in run_reports
                ]
            ),
            "mean_weight_by_view": {
                view: safe_mean(
                    [
                        float(item["fusion_weight"]["mean_weight_by_view"][view])
                        for item in run_reports
                    ]
                )
                for view in VIEW_NAMES
            },
            "top_weight_rate_by_view": {
                view: safe_mean(
                    [
                        float(item["fusion_weight"]["top_weight_rate_by_view"][view])
                        for item in run_reports
                    ]
                )
                for view in VIEW_NAMES
            },
            "top_true_margin_rate_by_view": {
                view: safe_mean(
                    [
                        float(item["fusion_weight"]["top_true_margin_rate_by_view"][view])
                        for item in run_reports
                    ]
                )
                for view in VIEW_NAMES
            },
        },
        "view_control_mean": {
            "full_minus_best_single_view": mean_metrics(
                [item["view_controls"]["full_minus_best_single_view"] for item in run_reports]
            ),
            "full_minus_best_leave_one_out": mean_metrics(
                [item["view_controls"]["full_minus_best_leave_one_out"] for item in run_reports]
            ),
            "control_delta_accuracy": {
                name: safe_mean(
                    [
                        float(item["view_controls"]["control_delta_accuracy"][name])
                        for item in run_reports
                    ]
                )
                for name in CONTROL_NAMES
            },
            "control_delta_auc": {
                name: safe_mean(
                    [
                        float(item["view_controls"]["control_delta_auc"][name])
                        for item in run_reports
                    ]
                )
                for name in CONTROL_NAMES
            },
            "control_delta_f1": {
                name: safe_mean(
                    [
                        float(item["view_controls"]["control_delta_f1"][name])
                        for item in run_reports
                    ]
                )
                for name in CONTROL_NAMES
            },
        },
        "perturbation_mean": {
            "delta_vs_baseline": mean_metrics(
                [item["perturbation"]["delta_vs_baseline"] for item in run_reports]
            ),
            "top_weight_switch_rate": safe_mean(
                [float(item["perturbation"]["top_weight_switch_rate"]) for item in run_reports]
            ),
            "target_view": mean_mapping(
                [item["perturbation"]["target_view"] for item in run_reports],
                (
                    "mean_weight_delta",
                    "mean_true_margin_delta",
                    "weight_drop_rate",
                    "top_weight_rate_before",
                    "top_weight_rate_after",
                    "migrated_away_rate",
                    "margin_drop_and_weight_drop_rate",
                ),
            ),
            "top_weight_view_distribution": {
                phase: {
                    view: safe_mean(
                        [
                            float(item["perturbation"]["top_weight_view_distribution"][phase][view])
                            for item in run_reports
                        ]
                    )
                    for view in VIEW_NAMES
                }
                for phase in ("before", "after", "migration_destinations_after_leaving_target")
            },
        },
        "routing_health": {
            "full_minus_single_view_axial_accuracy": safe_mean(
                [
                    -float(item["view_controls"]["control_delta_accuracy"]["single_view_axial"])
                    for item in run_reports
                ]
            ),
            "full_minus_leave_out_coronal_accuracy": safe_mean(
                [
                    -float(item["view_controls"]["control_delta_accuracy"]["leave_out_coronal"])
                    for item in run_reports
                ]
            ),
            "full_minus_leave_out_sagittal_accuracy": safe_mean(
                [
                    -float(item["view_controls"]["control_delta_accuracy"]["leave_out_sagittal"])
                    for item in run_reports
                ]
            ),
            "non_axial_top_weight_rate": safe_mean(
                [
                    1.0 - float(item["fusion_weight"]["top_weight_rate_by_view"]["axial"])
                    for item in run_reports
                ]
            ),
            "axial_migrated_away_rate_under_perturbation": safe_mean(
                [
                    float(item["perturbation"]["target_view"]["migrated_away_rate"])
                    for item in run_reports
                ]
            ),
        },
    }

    report = {
        "analysis": "multiseed_decision_routing",
        "aggregate": aggregate,
        "runs": run_reports,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output_json)
    print(f"Saved multiseed routing report to: {output_json}")


if __name__ == "__main__":
    main()
