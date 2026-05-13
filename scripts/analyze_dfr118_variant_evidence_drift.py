#!/usr/bin/env python3
"""DFR-118 evidence-drift audit across existing 3-seed variants.

This analysis is read-only.  It compares existing formal validation telemetry
against the DFR-116 posthoc combo branch, then asks whether any historical
training-side variant fixes remaining combo errors while protecting the three
DFR-116 fixed samples.  It does not train and does not read test metrics.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_dfr111_confirmed_target_posthoc import (  # noqa: E402
    VIEWS,
    VIEW_INDEX,
    base_error_type,
    counter_dict,
    load_json,
    round_float,
    save_json,
    summarize_weights,
)
from scripts.analyze_dfr112_multiseed_posthoc import (  # noqa: E402
    DFR25_TELEMETRY,
    model_equivalent_abnormal,
    telemetry_arrays_model_equivalent,
)
from scripts.analyze_dfr114_fp_risk_coverage import (  # noqa: E402
    metrics_from_abnormal_fast,
)
from scripts.analyze_dfr117_combo_frontier_audit import (  # noqa: E402
    apply_combo_logits,
    combo_pred_cache,
    evaluate_combo,
    summarize_remaining_errors_after_combo,
)


DEFAULT_OUTPUT = "autoresearch_logs/dfr118_variant_evidence_drift_audit.json"
SEEDS = ("42", "123", "456")
MAINLINE_ROOT = "runs/resnext_decision_256x8_mainline"
GROUP_RE = re.compile(r"^(?P<name>.+)_formal_s(?P<seed>42|123|456)$")
COMPATIBILITY_TOLERANCE = 1e-4
SUPPORT_DELTA = 0.05


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def stored_weights(arrays: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [float(sample["views"][view]["fusion_weight"]) for view in VIEWS]
            for sample in arrays["samples"]
        ],
        dtype=np.float64,
    )


def telemetry_predictions(arrays: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(patient_id): {
            "label": int(arrays["labels"][index]),
            "pred": int(arrays["base_preds"][index]),
            "abnormal": float(arrays["base_abnormal"][index]),
            "view_abnormal": {
                view: float(arrays["abnormal_probs"][index, VIEW_INDEX[view]])
                for view in VIEWS
            },
        }
        for index, patient_id in enumerate(arrays["patient_ids"])
    }


def logits_predictions(
    arrays: dict[str, Any],
    logits: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    abnormal, weights = model_equivalent_abnormal(arrays, logits)
    preds = (abnormal >= 0.5).astype(np.int64)
    by_id = {
        str(patient_id): {
            "label": int(arrays["labels"][index]),
            "pred": int(preds[index]),
            "abnormal": float(abnormal[index]),
            "view_abnormal": {
                view: float(arrays["abnormal_probs"][index, VIEW_INDEX[view]])
                for view in VIEWS
            },
        }
        for index, patient_id in enumerate(arrays["patient_ids"])
    }
    return by_id, {
        "metrics": metrics_from_abnormal_fast(arrays["labels"], abnormal),
        "weight_summary": summarize_weights(weights),
    }


def model_compatibility(arrays: dict[str, Any]) -> dict[str, Any]:
    abnormal, weights = model_equivalent_abnormal(
        arrays,
        arrays["scaled_confidence_logits"],
    )
    stored = stored_weights(arrays)
    return {
        "max_abs_abnormal_delta": round_float(
            float(np.max(np.abs(abnormal - arrays["base_abnormal"]))),
            digits=10,
        ),
        "max_abs_weight_delta": round_float(
            float(np.max(np.abs(weights - stored))),
            digits=10,
        ),
        "is_model_equivalent": bool(
            np.max(np.abs(abnormal - arrays["base_abnormal"]))
            <= COMPATIBILITY_TOLERANCE
        ),
    }


def discover_complete_groups(repo_root: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    root = repo_root / MAINLINE_ROOT
    for path in sorted(root.glob("*_formal_s*/fusion_weight_analysis.json")):
        match = GROUP_RE.match(path.parent.name)
        if not match:
            continue
        name = match.group("name")
        seed = match.group("seed")
        groups.setdefault(name, {})[seed] = path
    return {
        name: by_seed
        for name, by_seed in groups.items()
        if all(seed in by_seed for seed in SEEDS)
    }


def load_seed_arrays(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {seed: telemetry_arrays_model_equivalent(load_json(path)) for seed, path in paths.items()}


def aggregate_metrics(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in seed_summaries.values()]
    top_counts = Counter()
    mean_weights = {view: [] for view in VIEWS}
    for item in seed_summaries.values():
        summary = item["weight_summary"]
        for view in VIEWS:
            top_counts[view] += int(summary["top_weight_count"][view])
            mean_weights[view].append(float(summary["mean_fusion_weight"][view]))
    return {
        "mean_metrics": {
            "accuracy": round_float(mean([float(item["accuracy"]) for item in metrics])),
            "auc": round_float(mean([float(item["auc"]) for item in metrics])),
            "f1": round_float(mean([float(item["f1"]) for item in metrics])),
        },
        "min_seed_accuracy": round_float(
            min(float(item["accuracy"]) for item in metrics)
        ),
        "aggregate_top_weight_count": {view: int(top_counts[view]) for view in VIEWS},
        "mean_fusion_weight_across_seeds": {
            view: round_float(mean(values)) for view, values in mean_weights.items()
        },
    }


def reference_maps(
    seed_arrays: dict[str, dict[str, Any]],
    combo_cache: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, Any]]]:
    references: dict[str, dict[str, dict[str, Any]]] = {}
    for seed, arrays in seed_arrays.items():
        by_id = {}
        for index, patient_id in enumerate(arrays["patient_ids"]):
            label = int(arrays["labels"][index])
            dfr25_pred = int(arrays["base_preds"][index])
            combo_pred = int(combo_cache[seed][index])
            by_id[str(patient_id)] = {
                "label": label,
                "dfr25_pred": dfr25_pred,
                "combo_pred": combo_pred,
                "combo_error_type": base_error_type(label, combo_pred),
                "dfr25_error_type": base_error_type(label, dfr25_pred),
                "dfr25_view_abnormal": {
                    view: float(arrays["abnormal_probs"][index, VIEW_INDEX[view]])
                    for view in VIEWS
                },
                "dfr25_abnormal": float(arrays["base_abnormal"][index]),
            }
        references[seed] = by_id
    return references


def compare_to_reference(
    predictions: dict[str, dict[str, dict[str, Any]]],
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    fixed_combo: list[str] = []
    broken_combo: list[str] = []
    fixed_dfr25: list[str] = []
    broken_dfr25: list[str] = []
    protected_broken: list[str] = []
    remaining_error_records: list[dict[str, Any]] = []
    error_type_fixed = Counter()
    error_type_broken = Counter()

    for seed in SEEDS:
        for patient_id, ref in references[seed].items():
            pred = predictions[seed][patient_id]
            label = int(ref["label"])
            candidate_pred = int(pred["pred"])
            combo_pred = int(ref["combo_pred"])
            dfr25_pred = int(ref["dfr25_pred"])
            key = f"{seed}:{patient_id}"
            if combo_pred != label and candidate_pred == label:
                fixed_combo.append(key)
                error_type_fixed.update([ref["combo_error_type"]])
            elif combo_pred == label and candidate_pred != label:
                broken_combo.append(key)
                error_type_broken.update([base_error_type(label, candidate_pred)])
            if dfr25_pred != label and candidate_pred == label:
                fixed_dfr25.append(key)
            elif dfr25_pred == label and candidate_pred != label:
                broken_dfr25.append(key)
            if dfr25_pred != label and combo_pred == label and candidate_pred != label:
                protected_broken.append(key)

            if combo_pred != label:
                dfr25_views = ref["dfr25_view_abnormal"]
                candidate_views = pred["view_abnormal"]
                dfr25_max = max(dfr25_views.values())
                candidate_max = max(candidate_views.values())
                if label == 1:
                    support_delta = candidate_max - dfr25_max
                    support_direction = "higher_abnormal_is_helpful"
                else:
                    support_delta = dfr25_max - candidate_max
                    support_direction = "lower_abnormal_is_helpful"
                remaining_error_records.append(
                    {
                        "seed": seed,
                        "patient_id": patient_id,
                        "label": label,
                        "combo_pred": combo_pred,
                        "candidate_pred": candidate_pred,
                        "combo_error_type": ref["combo_error_type"],
                        "fixed_by_candidate": bool(candidate_pred == label),
                        "support_direction": support_direction,
                        "support_delta_vs_dfr25_max_view": round_float(support_delta),
                        "dfr25_max_view_abnormal": round_float(dfr25_max),
                        "candidate_max_view_abnormal": round_float(candidate_max),
                        "dfr25_view_abnormal": {
                            view: round_float(value)
                            for view, value in dfr25_views.items()
                        },
                        "candidate_view_abnormal": {
                            view: round_float(value)
                            for view, value in candidate_views.items()
                        },
                    }
                )

    supportive_records = [
        item
        for item in remaining_error_records
        if float(item["support_delta_vs_dfr25_max_view"]) >= SUPPORT_DELTA
    ]
    return {
        "fixed_dfr116_combo_errors": len(fixed_combo),
        "broken_dfr116_combo_correct": len(broken_combo),
        "net_delta_vs_dfr116_combo": len(fixed_combo) - len(broken_combo),
        "fixed_dfr25_errors": len(fixed_dfr25),
        "broken_dfr25_correct": len(broken_dfr25),
        "net_delta_vs_dfr25": len(fixed_dfr25) - len(broken_dfr25),
        "protected_dfr116_fixed_broken": len(protected_broken),
        "fixed_combo_ids": sorted(fixed_combo),
        "broken_combo_ids": sorted(broken_combo),
        "protected_broken_ids": sorted(protected_broken),
        "fixed_combo_error_types": counter_dict(error_type_fixed),
        "broken_combo_error_types": counter_dict(error_type_broken),
        "remaining_combo_error_supportive_drift_count": len(supportive_records),
        "remaining_combo_error_records": sorted(
            remaining_error_records,
            key=lambda item: (
                not bool(item["fixed_by_candidate"]),
                -float(item["support_delta_vs_dfr25_max_view"]),
                item["seed"],
                item["patient_id"],
            ),
        ),
    }


def summarize_stored_group(
    name: str,
    paths: dict[str, Path],
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    arrays_by_seed = load_seed_arrays(paths)
    seed_summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    compatibility = {}
    for seed, arrays in arrays_by_seed.items():
        weights = stored_weights(arrays)
        seed_summaries[seed] = {
            "metrics": metrics_from_abnormal_fast(arrays["labels"], arrays["base_abnormal"]),
            "weight_summary": summarize_weights(weights),
        }
        predictions[seed] = telemetry_predictions(arrays)
        compatibility[seed] = model_compatibility(arrays)
    group_type = "posthoc" if name.startswith(("dfr113_", "dfr116_")) else "trained"
    return {
        "name": name,
        "group_type": group_type,
        "paths": {seed: str(paths[seed]) for seed in SEEDS},
        "stored": aggregate_metrics(seed_summaries),
        "compatibility": compatibility,
        "all_seeds_model_equivalent": all(
            bool(item["is_model_equivalent"]) for item in compatibility.values()
        ),
        "comparison_vs_dfr116_combo": compare_to_reference(predictions, references),
    }


def summarize_combo_replay(
    name: str,
    paths: dict[str, Path],
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    if name.startswith(("dfr113_", "dfr116_")):
        return None
    arrays_by_seed = load_seed_arrays(paths)
    if not all(model_compatibility(arrays)["is_model_equivalent"] for arrays in arrays_by_seed.values()):
        return None

    seed_summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    action_records: dict[str, Any] = {}
    for seed, arrays in arrays_by_seed.items():
        logits, records = apply_combo_logits(arrays)
        by_id, summary = logits_predictions(arrays, logits)
        seed_summaries[seed] = summary
        predictions[seed] = by_id
        action_records[seed] = records
    return {
        "name": f"{name}+dfr116_combo_replay",
        "source_name": name,
        "stored_source_group_type": "trained",
        "combo_replay": aggregate_metrics(seed_summaries),
        "comparison_vs_dfr116_combo": compare_to_reference(predictions, references),
        "action_records": action_records,
    }


def ranking_key(item: dict[str, Any], metrics_key: str) -> tuple[Any, ...]:
    comparison = item["comparison_vs_dfr116_combo"]
    metrics = item[metrics_key]["mean_metrics"]
    return (
        -int(comparison["protected_dfr116_fixed_broken"]),
        -int(comparison["broken_dfr116_combo_correct"]),
        int(comparison["fixed_dfr116_combo_errors"]),
        float(metrics["accuracy"]),
        float(metrics["auc"]),
        float(item[metrics_key]["min_seed_accuracy"]),
    )


def top_items(
    items: list[dict[str, Any]],
    metrics_key: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: ranking_key(item, metrics_key), reverse=True)
    return ranked[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    dfr25_arrays = {
        seed: telemetry_arrays_model_equivalent(load_json(repo_root / relative_path))
        for seed, relative_path in DFR25_TELEMETRY.items()
    }
    combo_cache = combo_pred_cache(dfr25_arrays)
    references = reference_maps(dfr25_arrays, combo_cache)
    combo_baseline = evaluate_combo(dfr25_arrays)
    remaining_errors = summarize_remaining_errors_after_combo(dfr25_arrays, combo_cache)

    groups = discover_complete_groups(repo_root)
    stored_groups = [
        summarize_stored_group(name, paths, references)
        for name, paths in sorted(groups.items())
    ]
    trained_groups = [
        item for item in stored_groups if item["group_type"] == "trained"
    ]
    replay_groups = [
        replay
        for name, paths in sorted(groups.items())
        for replay in [summarize_combo_replay(name, paths, references)]
        if replay is not None
    ]

    safe_stored = [
        item
        for item in trained_groups
        if item["comparison_vs_dfr116_combo"]["broken_dfr116_combo_correct"] == 0
        and item["comparison_vs_dfr116_combo"]["protected_dfr116_fixed_broken"] == 0
    ]
    safe_replay = [
        item
        for item in replay_groups
        if item["comparison_vs_dfr116_combo"]["broken_dfr116_combo_correct"] == 0
        and item["comparison_vs_dfr116_combo"]["protected_dfr116_fixed_broken"] == 0
    ]
    improved_safe_stored = [
        item
        for item in safe_stored
        if item["comparison_vs_dfr116_combo"]["fixed_dfr116_combo_errors"] > 0
    ]
    improved_safe_replay = [
        item
        for item in safe_replay
        if item["comparison_vs_dfr116_combo"]["fixed_dfr116_combo_errors"] > 0
    ]

    report = {
        "analysis": "dfr118_variant_evidence_drift_audit",
        "description": (
            "Analysis-only audit over existing 3-seed formal validation telemetry. "
            "Compares stored variants and model-equivalent DFR-116 combo replays "
            "against the DFR-116 strict combo reference; no training or test metrics."
        ),
        "dfr116_reference": {
            "aggregate": combo_baseline["aggregate"],
            "remaining_errors_after_combo": remaining_errors,
        },
        "discovered_complete_group_count": len(groups),
        "trained_group_count": len(trained_groups),
        "combo_replay_group_count": len(replay_groups),
        "safe_stored_group_count": len(safe_stored),
        "safe_replay_group_count": len(safe_replay),
        "improved_safe_stored_group_count": len(improved_safe_stored),
        "improved_safe_replay_group_count": len(improved_safe_replay),
        "best_stored_trained_by_safety": top_items(trained_groups, "stored", limit=20),
        "best_combo_replay_by_safety": top_items(replay_groups, "combo_replay", limit=20),
        "safe_stored_groups": top_items(safe_stored, "stored", limit=20),
        "safe_combo_replay_groups": top_items(safe_replay, "combo_replay", limit=20),
        "improved_safe_stored_groups": top_items(improved_safe_stored, "stored", limit=20),
        "improved_safe_combo_replay_groups": top_items(improved_safe_replay, "combo_replay", limit=20),
        "all_stored_groups_compact": [
            {
                "name": item["name"],
                "group_type": item["group_type"],
                "stored": item["stored"],
                "all_seeds_model_equivalent": item["all_seeds_model_equivalent"],
                "comparison_vs_dfr116_combo": {
                    key: value
                    for key, value in item["comparison_vs_dfr116_combo"].items()
                    if key
                    not in {
                        "fixed_combo_ids",
                        "broken_combo_ids",
                        "protected_broken_ids",
                        "remaining_combo_error_records",
                    }
                },
            }
            for item in sorted(
                stored_groups,
                key=lambda item: ranking_key(item, "stored"),
                reverse=True,
            )
        ],
    }
    save_json(repo_root / args.output, report)

    print(f"Saved DFR-118 variant evidence-drift audit to: {repo_root / args.output}")
    print(
        "dfr116",
        combo_baseline["aggregate"]["mean_metrics"],
        combo_baseline["aggregate"]["aggregate_top_weight_count"],
        "remaining_errors",
        len(remaining_errors),
    )
    print(
        "groups",
        "complete",
        len(groups),
        "trained",
        len(trained_groups),
        "combo_replay",
        len(replay_groups),
    )
    print(
        "safe",
        "stored",
        len(safe_stored),
        "replay",
        len(safe_replay),
        "improved_safe_stored",
        len(improved_safe_stored),
        "improved_safe_replay",
        len(improved_safe_replay),
    )
    if trained_groups:
        best = top_items(trained_groups, "stored", limit=1)[0]
        cmp_ = best["comparison_vs_dfr116_combo"]
        print(
            "best_stored",
            best["name"],
            best["stored"]["mean_metrics"],
            "fixed_combo",
            cmp_["fixed_dfr116_combo_errors"],
            "broken_combo",
            cmp_["broken_dfr116_combo_correct"],
            "protected_broken",
            cmp_["protected_dfr116_fixed_broken"],
        )
    if replay_groups:
        best = top_items(replay_groups, "combo_replay", limit=1)[0]
        cmp_ = best["comparison_vs_dfr116_combo"]
        print(
            "best_replay",
            best["name"],
            best["combo_replay"]["mean_metrics"],
            "fixed_combo",
            cmp_["fixed_dfr116_combo_errors"],
            "broken_combo",
            cmp_["broken_dfr116_combo_correct"],
            "protected_broken",
            cmp_["protected_dfr116_fixed_broken"],
        )


if __name__ == "__main__":
    main()
