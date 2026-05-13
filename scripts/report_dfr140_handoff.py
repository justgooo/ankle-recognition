#!/usr/bin/env python3
"""DFR-140 handoff export.

Creates a compact Markdown handoff for the current DFR-116 branch, the DFR-138
review packet, and the DFR-139 training gate.  This is read-only and launches
no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DFR119 = "autoresearch_logs/dfr119_dfr116_branch_repro_report.json"
DEFAULT_DFR138 = "autoresearch_logs/dfr138_clean_remaining_review_packet.json"
DEFAULT_DFR139 = "autoresearch_logs/dfr139_training_gate_from_review_packet.json"
DEFAULT_OUTPUT = "autoresearch_logs/dfr140_handoff.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_metrics(metrics: dict[str, Any]) -> str:
    return "{accuracy:.6f}/{auc:.6f}/{f1:.6f}".format(
        accuracy=float(metrics["accuracy"]),
        auc=float(metrics["auc"]),
        f1=float(metrics["f1"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfr119", default=DEFAULT_DFR119)
    parser.add_argument("--dfr138", default=DEFAULT_DFR138)
    parser.add_argument("--dfr139", default=DEFAULT_DFR139)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dfr119 = load_json(REPO_ROOT / args.dfr119)
    dfr138 = load_json(REPO_ROOT / args.dfr138)
    dfr139 = load_json(REPO_ROOT / args.dfr139)

    combo = dfr119["dfr116_combo"]
    branch = combo["aggregate"]
    reference = dfr119["dfr25_reference"]["aggregate"]
    packet_summary = dfr138["summary"]
    gate = dfr139["assessment"]

    lines = [
        "# DFR-140 Autoresearch Handoff",
        "",
        "## Current Branch",
        "",
        f"- Current best posthoc branch: `{gate['current_best_branch']}`",
        f"- DFR-25 reference mean val_acc/auc/f1: `{fmt_metrics(reference['mean_metrics'])}`",
        f"- DFR-116 combo mean val_acc/auc/f1: `{fmt_metrics(branch['mean_metrics'])}`",
        f"- DFR-116 fixed cases: `{', '.join(combo['fixed_ids'])}`",
        f"- DFR-116 broken cases: `{', '.join(combo['broken_ids']) or 'none'}`",
        "",
        "## Remaining Review Packet",
        "",
        f"- Packet JSON: `{args.dfr138}`",
        "- Packet Markdown: `autoresearch_logs/dfr138_clean_remaining_review_packet.md`",
        f"- Patients/cases: `{packet_summary['patient_count']} / {packet_summary['case_count']}`",
        f"- Decision buckets: `{packet_summary['decision_bucket_counts']}`",
        f"- Combined-clean DFR-116 val_acc/auc/f1: `{fmt_metrics(packet_summary['dfr116_combined_clean_metrics'])}`",
        "",
        "## Training Gate",
        "",
        f"- training_go: `{gate['training_go']}`",
        f"- blocked reason: {gate['training_blocked_reason']}",
        f"- minimum next step: {gate['minimum_next_step']}",
        "",
        "## Required Review Questions",
        "",
    ]
    for patient in dfr139["patients"]:
        lines.append(f"### {patient['patient_id']}")
        lines.append("")
        lines.append(f"- bucket: `{patient['decision_bucket']}`")
        lines.append(f"- cases: `{', '.join(patient['case_ids'])}`")
        lines.append("- required answers:")
        for answer in patient["required_review_answers"]:
            lines.append(f"  - {answer}")
        lines.append("- allowed actions after review:")
        for action in patient["allowed_actions_after_review"]:
            lines.append(f"  - {action}")
        lines.append("")

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
