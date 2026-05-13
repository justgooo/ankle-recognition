#!/usr/bin/env python3
"""DFR-142 human review answer template.

Exports a structured, fillable template for the DFR-139/141 review questions.
It does not infer answers, edit data, or launch training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "autoresearch_logs/dfr141_repro_manifest.json"
DEFAULT_OUTPUT_JSON = "autoresearch_logs/dfr142_review_answer_template.json"
DEFAULT_OUTPUT_MD = "autoresearch_logs/dfr142_review_answer_template.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def schema_for_bucket(bucket: str) -> dict[str, Any]:
    if bucket == "duplicate_positive_annotation_review":
        return {
            "fields": {
                "duplicate_status": {
                    "allowed_values": [
                        "unknown",
                        "separate_clinical_cases",
                        "duplicate_record",
                    ],
                    "value": "unknown",
                },
                "positive_finding_visible": {
                    "allowed_values": ["unknown", "yes", "no"],
                    "value": "unknown",
                },
                "visible_views_or_slice_range": "",
                "reviewer_notes": "",
            },
            "go_rule": "Training remains blocked unless positive_finding_visible=yes and a specific view/slice target is documented.",
        }
    if bucket == "positive_no_view_evidence_annotation_review":
        return {
            "fields": {
                "positive_label_supported_in_volume": {
                    "allowed_values": ["unknown", "yes", "no"],
                    "value": "unknown",
                },
                "evidence_views_or_slice_range": "",
                "missed_evidence_pattern": {
                    "allowed_values": [
                        "unknown",
                        "subtle_visible_target",
                        "outside_sampled_slices",
                        "label_or_volume_mismatch",
                    ],
                    "value": "unknown",
                },
                "reviewer_notes": "",
            },
            "go_rule": "Training remains blocked unless the positive label is supported and a view/slice-targeted sensitivity target is documented.",
        }
    if bucket == "negative_strong_axial_fp_label_or_view_classifier_review":
        return {
            "fields": {
                "negative_label_confirmed": {
                    "allowed_values": ["unknown", "yes", "no"],
                    "value": "unknown",
                },
                "axial_high_abnormal_explanation": {
                    "allowed_values": [
                        "unknown",
                        "normal_anatomy",
                        "artifact_or_implant",
                        "subtle_pathology",
                        "label_suspect",
                    ],
                    "value": "unknown",
                },
                "positive_protection_feature_exists": {
                    "allowed_values": ["unknown", "yes", "no"],
                    "value": "unknown",
                },
                "positive_protection_feature": "",
                "reviewer_notes": "",
            },
            "go_rule": "Training remains blocked unless the negative label is confirmed and a positive-protection feature is documented.",
        }
    return {
        "fields": {"reviewer_notes": ""},
        "go_rule": "Training remains blocked until the bucket-specific review is resolved.",
    }


def action_hint(bucket: str) -> str:
    if bucket == "duplicate_positive_annotation_review":
        return (
            "If duplicate/label issue is confirmed, keep DFR-116 and report a "
            "validation caveat. If visible localized evidence is confirmed, "
            "open a narrow view/slice sensitivity audit before any training."
        )
    if bucket == "positive_no_view_evidence_annotation_review":
        return (
            "If the label/evidence mismatches, do not train. If evidence exists "
            "but all view probabilities are weak, only then propose a supervised "
            "sensitivity target."
        )
    if bucket == "negative_strong_axial_fp_label_or_view_classifier_review":
        return (
            "If the negative label is suspect, do not train. If label is "
            "confirmed and a positive-protection feature exists, consider only "
            "a narrow supervised axial calibration."
        )
    return "Resolve the review bucket before proposing training."


def build_template(manifest: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for patient in manifest["review_checklist"]:
        bucket = patient["decision_bucket"]
        schema = schema_for_bucket(bucket)
        entries.append(
            {
                "patient_id": patient["patient_id"],
                "case_ids": patient["case_ids"],
                "label": patient["label"],
                "decision_bucket": bucket,
                "training_allowed_now": False,
                "blockers": patient["blockers"],
                "required_review_answers": patient["required_review_answers"],
                "answer_fields": schema["fields"],
                "go_rule": schema["go_rule"],
                "action_hint": action_hint(bucket),
                "allowed_actions_after_review": patient["allowed_actions_after_review"],
            }
        )

    return {
        "description": "Fillable human review answer template for DFR-139/141 blocked remaining cases.",
        "source_manifest": DEFAULT_MANIFEST,
        "training_gate_at_export": manifest["training_gate"],
        "review_summary": manifest["review_summary"],
        "completion_rule": "Do not launch training until each proposed model-side action has explicit filled answer fields and maps to a documented go_rule.",
        "entries": entries,
    }


def write_markdown(template: dict[str, Any], path: Path) -> None:
    lines = [
        "# DFR-142 Review Answer Template",
        "",
        "## Gate",
        "",
        f"- training_go at export: `{template['training_gate_at_export']['training_go']}`",
        f"- current best branch: `{template['training_gate_at_export']['current_best_branch']}`",
        f"- completion rule: {template['completion_rule']}",
        "",
        "## Summary",
        "",
        f"- patients/cases: `{template['review_summary']['patient_count']} / {template['review_summary']['case_count']}`",
        f"- training_allowed_now_count: `{template['review_summary']['training_allowed_now_count']}`",
        f"- decision buckets: `{template['review_summary']['decision_bucket_counts']}`",
        "",
        "## Fillable Entries",
        "",
    ]

    for entry in template["entries"]:
        lines.append(f"### {entry['patient_id']}")
        lines.append("")
        lines.append(f"- bucket: `{entry['decision_bucket']}`")
        lines.append(f"- cases: `{', '.join(entry['case_ids'])}`")
        lines.append(f"- go rule: {entry['go_rule']}")
        lines.append(f"- action hint: {entry['action_hint']}")
        lines.append("- required answers:")
        for question in entry["required_review_answers"]:
            lines.append(f"  - {question}")
        lines.append("- answer fields:")
        for field, spec in entry["answer_fields"].items():
            if isinstance(spec, dict):
                allowed = ", ".join(spec.get("allowed_values", []))
                value = spec.get("value", "")
                lines.append(f"  - `{field}`: `{value}` allowed=`{allowed}`")
            else:
                lines.append(f"  - `{field}`: `{spec}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    manifest = load_json(REPO_ROOT / args.manifest)
    template = build_template(manifest)

    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(template, output_md)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print(
        json.dumps(
            {
                "entry_count": len(template["entries"]),
                "training_go": template["training_gate_at_export"]["training_go"],
                "training_allowed_now_count": template["review_summary"][
                    "training_allowed_now_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
