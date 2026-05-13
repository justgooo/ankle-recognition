#!/usr/bin/env python3
"""DFR-139 training gate from DFR-138 review packet.

This read-only report converts the DFR-138 patient review packet into a
training go/no-go decision.  It intentionally does not launch training.  It
records the minimum external review answers required before any new model-side
experiment is justified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = "autoresearch_logs/dfr138_clean_remaining_review_packet.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr139_training_gate_from_review_packet.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def gate_patient(patient: dict[str, Any]) -> dict[str, Any]:
    bucket = str(patient["decision_bucket"])
    blockers = list(patient["blockers"])
    required_answers: list[str] = []
    allowed_after_review: list[str] = []

    if bucket == "duplicate_positive_annotation_review":
        required_answers = [
            "Are duplicate validation volumes separate clinical cases or duplicate records?",
            "Is the positive finding visible in the shared NIfTI volume?",
            "If visible, which view/slice range contains the finding?",
        ]
        allowed_after_review = [
            "If duplicate/label issue confirmed: do not train; report validation caveat.",
            "If positive finding is visible and localized: design a view/slice-targeted sensitivity audit before training.",
        ]
    elif bucket == "positive_no_view_evidence_annotation_review":
        required_answers = [
            "Is the positive label visible in the NIfTI volume?",
            "If visible, which view/slice range contains evidence missed by all three view classifiers?",
        ]
        allowed_after_review = [
            "If label/evidence mismatch: do not train; mark data-quality issue.",
            "If evidence exists but all view probabilities are weak: only then consider a supervised sensitivity target.",
        ]
    elif bucket == "negative_strong_axial_fp_label_or_view_classifier_review":
        required_answers = [
            "Does axial high-abnormal evidence correspond to subtle pathology, artifact, implant, or normal anatomy?",
            "Is the negative label clinically correct?",
            "Can positives with similar high axial abnormal be separated by a reviewed feature?",
        ]
        allowed_after_review = [
            "If negative label is suspect: do not train; mark annotation issue.",
            "If label is confirmed and a positive-protection feature exists: consider narrow supervised axial calibration.",
            "If no positive-protection feature exists: do not run axial FP training; DFR127 already showed label-free overlap.",
        ]
    else:
        required_answers = ["Unrecognized bucket; inspect packet manually before training."]

    return {
        "patient_id": patient["patient_id"],
        "label": int(patient["label"]),
        "case_ids": list(patient["case_ids"]),
        "decision_bucket": bucket,
        "training_allowed_now": False,
        "blockers": blockers,
        "required_review_answers": required_answers,
        "allowed_actions_after_review": allowed_after_review,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default=DEFAULT_PACKET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    packet_path = REPO_ROOT / args.packet
    packet = load_json(packet_path)
    patient_gates = [gate_patient(patient) for patient in packet["patients"]]
    bucket_counts = Counter(patient["decision_bucket"] for patient in patient_gates)

    payload = {
        "analysis": "dfr139_training_gate_from_review_packet",
        "description": (
            "Read-only go/no-go decision derived from DFR-138. No training, no test metrics, "
            "no data edits. The report blocks training until review answers identify a "
            "model-expressible target."
        ),
        "source_paths": {
            "dfr138_packet": str(packet_path.relative_to(REPO_ROOT)),
        },
        "summary": {
            "patient_count": len(patient_gates),
            "case_count": sum(len(patient["case_ids"]) for patient in patient_gates),
            "decision_bucket_counts": counter_dict(bucket_counts),
            "training_allowed_now_count": int(
                sum(1 for patient in patient_gates if patient["training_allowed_now"])
            ),
        },
        "patients": patient_gates,
        "assessment": {
            "training_go": False,
            "training_blocked_reason": (
                "No reviewed patient currently identifies a model-expressible target; all seven "
                "remaining patients require duplicate/annotation or strong axial FP review first."
            ),
            "minimum_next_step": (
                "Complete the required review answers in this report. Only after that should a new "
                "experiment be proposed; otherwise keep DFR-116 as the current posthoc branch."
            ),
            "current_best_branch": "DFR-116 strict posthoc combo",
        },
    }
    save_json(REPO_ROOT / args.output, payload)
    print(f"Wrote {REPO_ROOT / args.output}")
    print(json.dumps(payload["assessment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
