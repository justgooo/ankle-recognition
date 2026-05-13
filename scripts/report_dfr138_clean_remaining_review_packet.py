#!/usr/bin/env python3
"""DFR-138 clean remaining patient review packet.

This read-only packet condenses the seven DFR-116 combined-clean remaining
patients into a JSON report and a Markdown review file.  It is intended for
human/data-quality review before any further model-side training.  It reads no
test metrics and modifies no data files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_dfr111_confirmed_target_posthoc import (  # noqa: E402
    VIEWS,
    load_json,
    round_float,
    save_json,
)
from scripts.analyze_dfr112_multiseed_posthoc import DFR25_TELEMETRY  # noqa: E402
from scripts.report_dfr119_dfr116_branch import DFR116_TELEMETRY  # noqa: E402


DEFAULT_DFR126 = "autoresearch_logs/dfr126_clean_remaining_evidence_sampling.json"
DEFAULT_DFR137 = "autoresearch_logs/dfr137_remaining_error_decision_audit.json"
DEFAULT_JSON_OUTPUT = "autoresearch_logs/dfr138_clean_remaining_review_packet.json"
DEFAULT_MARKDOWN_OUTPUT = "autoresearch_logs/dfr138_clean_remaining_review_packet.md"
SEEDS = ("42", "123", "456")


def sample_map(telemetry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(sample["patient_id"]): sample for sample in telemetry["samples"]}


def top_weight_view(sample: dict[str, Any]) -> str:
    views = sample.get("top_weight_views") or []
    return str(views[0]) if views else "none"


def view_compact(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        view: {
            "abnormal": round_float(float(sample["views"][view]["abnormal_prob"])),
            "weight": round_float(float(sample["views"][view]["fusion_weight"])),
        }
        for view in VIEWS
    }


def pred_compact(sample: dict[str, Any]) -> dict[str, Any]:
    label = int(sample["label"])
    pred = int(sample["fusion_prediction"]["pred"])
    return {
        "label": label,
        "pred": pred,
        "correct": bool(label == pred),
        "abnormal": round_float(float(sample["fusion_prediction"]["abnormal_prob"])),
        "top_weight": top_weight_view(sample),
        "views": view_compact(sample),
    }


def case_id(seed: str, patient_id: str) -> str:
    return f"{seed}:{patient_id}"


def flag_summary(patient: dict[str, Any]) -> list[str]:
    flags = list(patient["interpretation"].get("sampling_flags", []))
    return sorted(str(flag) for flag in flags)


def duplicate_summary(patient: dict[str, Any]) -> dict[str, Any]:
    groups = patient.get("duplicate_groups", [])
    members: list[dict[str, Any]] = []
    for group in groups:
        for member in group.get("members", []):
            members.append(
                {
                    "patient_id": str(member["patient_id"]),
                    "split": str(member["split"]),
                    "label": int(member["label"]),
                    "path": str(member["path"]),
                }
            )
    return {
        "group_count": len(groups),
        "members": sorted(members, key=lambda item: item["patient_id"]),
    }


def path_summary(patient: dict[str, Any]) -> dict[str, Any]:
    path_hashes = patient.get("path_hashes", {})
    return {
        key: {
            "path": value.get("path"),
            "exists": bool(value.get("exists")),
            "file_size_bytes": value.get("file_size_bytes"),
            "sha256": value.get("sha256"),
        }
        for key, value in sorted(path_hashes.items())
    }


def baseline_sampling_indices(patient: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view_column, view_info in sorted(patient.get("view_sampling", {}).items()):
        grid = view_info.get("sampling_grids", {}).get("baseline_8_trim2", {})
        coverage = view_info.get("baseline_metric_coverage", {})
        result[view_column] = {
            "view": view_info.get("view"),
            "axis": view_info.get("axis"),
            "length": view_info.get("length"),
            "indices": grid.get("indices", []),
            "flags": list(view_info.get("flags", [])),
            "hu300_ratio": coverage.get("hu300_fraction", {}).get("sampled_to_full_max_ratio"),
            "hu300_top10_hit_count": coverage.get("hu300_fraction", {}).get("top10_hit_count"),
            "std_top10_hit_count": coverage.get("std", {}).get("top10_hit_count"),
        }
    return result


def review_prompt(decision_bucket: str, label: int) -> list[str]:
    if decision_bucket == "duplicate_positive_annotation_review":
        return [
            "Confirm whether duplicate validation volumes should be interpreted as separate clinical cases.",
            "Review positive annotation visibility in the duplicated volume before treating this as model failure.",
        ]
    if decision_bucket == "positive_no_view_evidence_annotation_review":
        return [
            "Review whether the positive label is visible in the current NIfTI volume.",
            "If visible, identify which anatomical plane/slice contains evidence; current model views stay below weak abnormal evidence.",
        ]
    if label == 0:
        return [
            "Review whether axial high-abnormal evidence reflects subtle pathology, acquisition artifact, or label noise.",
            "If label is confirmed negative, mark this as view-classifier calibration target only after positive-protection signal is found.",
        ]
    return ["Review label/evidence consistency before model-side changes."]


def build_packet(
    repo_root: Path,
    dfr126_path: Path,
    dfr137_path: Path,
) -> dict[str, Any]:
    dfr126 = load_json(dfr126_path)
    dfr137 = load_json(dfr137_path)
    decisions = {
        str(patient["patient_id"]): patient
        for patient in dfr137["patients"]
    }
    dfr25_maps = {
        seed: sample_map(load_json(repo_root / DFR25_TELEMETRY[seed]))
        for seed in SEEDS
    }
    dfr116_maps = {
        seed: sample_map(load_json(repo_root / DFR116_TELEMETRY[seed]))
        for seed in SEEDS
    }

    patient_packets: list[dict[str, Any]] = []
    for patient in dfr126["patients"]:
        patient_id = str(patient["patient_id"])
        decision = decisions[patient_id]
        error_case_ids = {str(case["case_id"]) for case in patient.get("cases", [])}
        per_seed = {}
        for seed in SEEDS:
            dfr25 = pred_compact(dfr25_maps[seed][patient_id])
            dfr116 = pred_compact(dfr116_maps[seed][patient_id])
            cid = case_id(seed, patient_id)
            per_seed[seed] = {
                "case_id": cid,
                "is_clean_remaining_error": cid in error_case_ids,
                "dfr25": dfr25,
                "dfr116": dfr116,
                "prediction_changed_by_dfr116": dfr25["pred"] != dfr116["pred"],
                "abnormal_delta_dfr116_minus_dfr25": round_float(
                    float(dfr116["abnormal"]) - float(dfr25["abnormal"])
                ),
            }
        patient_packets.append(
            {
                "patient_id": patient_id,
                "label": int(patient["label"]),
                "split": str(patient["split"]),
                "decision_bucket": decision["decision_bucket"],
                "dfr126_bucket": decision["dfr126_bucket"],
                "case_ids": list(patient.get("case_ids", [])),
                "blockers": list(decision.get("blockers", [])),
                "review_prompts": review_prompt(decision["decision_bucket"], int(patient["label"])),
                "duplicate_summary": duplicate_summary(patient),
                "path_summary": path_summary(patient),
                "volume_shapes": patient.get("volume_shapes", {}),
                "all_view_columns_same_path": bool(patient.get("all_view_columns_same_path")),
                "all_view_hashes_same": bool(patient.get("all_view_hashes_same")),
                "sampling_flags": flag_summary(patient),
                "baseline_sampling": baseline_sampling_indices(patient),
                "error_seed_view_abnormal_ranges": patient["interpretation"][
                    "error_seed_view_abnormal_ranges"
                ],
                "per_seed_predictions": per_seed,
            }
        )

    return {
        "analysis": "dfr138_clean_remaining_review_packet",
        "description": (
            "Compact read-only review packet for seven DFR-116 combined-clean remaining "
            "patients. No training, no test metrics, no data edits."
        ),
        "source_paths": {
            "dfr126_clean_remaining": str(dfr126_path.relative_to(repo_root)),
            "dfr137_decision_audit": str(dfr137_path.relative_to(repo_root)),
            "dfr25_telemetry": DFR25_TELEMETRY,
            "dfr116_telemetry": DFR116_TELEMETRY,
        },
        "summary": {
            "patient_count": len(patient_packets),
            "case_count": sum(len(packet["case_ids"]) for packet in patient_packets),
            "decision_bucket_counts": dfr137["clean_remaining_summary"]["decision_bucket_counts"],
            "model_side_supported_patient_count": dfr137["clean_remaining_summary"][
                "model_side_supported_patient_count"
            ],
            "dfr116_combined_clean_metrics": dfr137["dfr116_reference_metrics"]["combined_clean_mean"],
            "dfr116_official_full_metrics": dfr137["dfr116_reference_metrics"]["official_full_mean"],
        },
        "patients": sorted(
            patient_packets,
            key=lambda item: (
                str(item["decision_bucket"]),
                -len(item["case_ids"]),
                str(item["patient_id"]),
            ),
        ),
        "assessment": {
            "ready_for_human_review": True,
            "immediate_training_recommended": False,
            "reason": (
                "DFR-137 found zero model-side-supported patients after closure of posthoc, "
                "axial-risk, and sampling families; this packet is for evidence/label review."
            ),
        },
    }


def markdown_table(packet: dict[str, Any]) -> str:
    lines = [
        "# DFR-138 Clean Remaining Review Packet",
        "",
        "This packet is read-only and uses validation telemetry plus metadata reports only.",
        "",
        "## Summary",
        "",
        f"- Patients: {packet['summary']['patient_count']}",
        f"- Clean remaining cases: {packet['summary']['case_count']}",
        f"- DFR-116 combined-clean metrics: {packet['summary']['dfr116_combined_clean_metrics']}",
        f"- Decision buckets: {packet['summary']['decision_bucket_counts']}",
        f"- Immediate training recommended: {packet['assessment']['immediate_training_recommended']}",
        "",
        "## Patients",
        "",
    ]
    for patient in packet["patients"]:
        lines.extend(
            [
                f"### {patient['patient_id']}",
                "",
                f"- Label: {patient['label']}",
                f"- Decision bucket: `{patient['decision_bucket']}`",
                f"- Cases: {', '.join(patient['case_ids'])}",
                f"- Duplicate members: {', '.join(item['patient_id'] for item in patient['duplicate_summary']['members']) or 'none'}",
                f"- Sampling flags: {', '.join(patient['sampling_flags']) or 'none'}",
                f"- Blockers: {'; '.join(patient['blockers']) or 'none'}",
                f"- Review prompts: {'; '.join(patient['review_prompts'])}",
                "",
                "| Seed | Error? | DFR25 pred/prob/top | DFR116 pred/prob/top | View abnormal DFR116 (A/C/S) |",
                "|---|---:|---|---|---|",
            ]
        )
        for seed, row in patient["per_seed_predictions"].items():
            d25 = row["dfr25"]
            d116 = row["dfr116"]
            view_probs = "/".join(
                f"{d116['views'][view]['abnormal']:.3f}" for view in VIEWS
            )
            lines.append(
                "| {seed} | {err} | {p25}/{a25:.3f}/{t25} | {p116}/{a116:.3f}/{t116} | {views} |".format(
                    seed=seed,
                    err="yes" if row["is_clean_remaining_error"] else "no",
                    p25=d25["pred"],
                    a25=d25["abnormal"],
                    t25=d25["top_weight"],
                    p116=d116["pred"],
                    a116=d116["abnormal"],
                    t116=d116["top_weight"],
                    views=view_probs,
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--dfr126", default=DEFAULT_DFR126)
    parser.add_argument("--dfr137", default=DEFAULT_DFR137)
    args = parser.parse_args()

    packet = build_packet(
        REPO_ROOT,
        REPO_ROOT / args.dfr126,
        REPO_ROOT / args.dfr137,
    )
    json_path = REPO_ROOT / args.json_output
    markdown_path = REPO_ROOT / args.markdown_output
    save_json(json_path, packet)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_table(packet), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(json.dumps(packet["assessment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
