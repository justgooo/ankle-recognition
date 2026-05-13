#!/usr/bin/env python3
"""DFR-141 reproducibility/archive manifest.

Collects the retained DFR-116 branch assets, DFR-138/139/140 governance
reports, and the required human review checklist into JSON and Markdown.
This is read-only with respect to training/data and launches no jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DFR119 = "autoresearch_logs/dfr119_dfr116_branch_repro_report.json"
DEFAULT_DFR138 = "autoresearch_logs/dfr138_clean_remaining_review_packet.json"
DEFAULT_DFR139 = "autoresearch_logs/dfr139_training_gate_from_review_packet.json"
DEFAULT_DFR140 = "autoresearch_logs/dfr140_handoff.md"
DEFAULT_OUTPUT_JSON = "autoresearch_logs/dfr141_repro_manifest.json"
DEFAULT_OUTPUT_MD = "autoresearch_logs/dfr141_repro_manifest.md"

TEXT_HASH_LIMIT_BYTES = 64 * 1024 * 1024


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_metrics(metrics: dict[str, Any]) -> str:
    return "{accuracy:.6f}/{auc:.6f}/{f1:.6f}".format(
        accuracy=float(metrics["accuracy"]),
        auc=float(metrics["auc"]),
        f1=float(metrics["f1"]),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_record(path_str: str, *, role: str) -> dict[str, Any]:
    path = REPO_ROOT / path_str
    record: dict[str, Any] = {
        "path": path_str,
        "role": role,
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    stat = path.stat()
    record["size_bytes"] = stat.st_size
    record["mtime_ns"] = stat.st_mtime_ns
    if stat.st_size <= TEXT_HASH_LIMIT_BYTES:
        record["sha256"] = sha256_file(path)
    else:
        record["sha256"] = None
        record["hash_note"] = "skipped_large_file"
    return record


def add_seed_assets(
    assets: list[dict[str, Any]],
    seed_map: dict[str, str],
    *,
    role_prefix: str,
) -> None:
    for seed in sorted(seed_map, key=lambda value: int(value)):
        assets.append(asset_record(seed_map[seed], role=f"{role_prefix}_seed{seed}"))


def build_manifest(
    dfr119: dict[str, Any],
    dfr138: dict[str, Any],
    dfr139: dict[str, Any],
    dfr140_path: str,
) -> dict[str, Any]:
    dfr116 = dfr119["dfr116_combo"]
    dfr25 = dfr119["dfr25_reference"]
    gate = dfr139["assessment"]
    dfr138_summary = dfr138["summary"]

    validation_checks = dfr119["validation_checks"]
    all_checks_pass = all(check.get("ok") for check in validation_checks)

    assets: list[dict[str, Any]] = [
        asset_record(DEFAULT_DFR119, role="dfr119_branch_repro_report"),
        asset_record(DEFAULT_DFR138, role="dfr138_review_packet_json"),
        asset_record(
            "autoresearch_logs/dfr138_clean_remaining_review_packet.md",
            role="dfr138_review_packet_markdown",
        ),
        asset_record(DEFAULT_DFR139, role="dfr139_training_gate_json"),
        asset_record(dfr140_path, role="dfr140_handoff_markdown"),
        asset_record("backlog.md", role="experiment_backlog"),
        asset_record("results.tsv", role="experiment_results_ledger"),
        asset_record("scripts/report_dfr119_dfr116_branch.py", role="dfr119_script"),
        asset_record(
            "scripts/report_dfr138_clean_remaining_review_packet.py",
            role="dfr138_script",
        ),
        asset_record(
            "scripts/report_dfr139_training_gate_from_review_packet.py",
            role="dfr139_script",
        ),
        asset_record("scripts/report_dfr140_handoff.py", role="dfr140_script"),
        asset_record("scripts/report_dfr141_repro_manifest.py", role="dfr141_script"),
    ]

    add_seed_assets(assets, dfr119["configs"], role_prefix="dfr116_config")
    add_seed_assets(assets, dfr119["checkpoints"], role_prefix="dfr25_checkpoint")
    add_seed_assets(assets, dfr119["telemetry"], role_prefix="dfr116_telemetry")
    add_seed_assets(
        assets,
        dfr138["source_paths"]["dfr25_telemetry"],
        role_prefix="dfr25_telemetry",
    )

    missing_assets = [asset["path"] for asset in assets if not asset["exists"]]
    review_rows = []
    for patient in dfr139["patients"]:
        review_rows.append(
            {
                "patient_id": patient["patient_id"],
                "case_ids": patient["case_ids"],
                "decision_bucket": patient["decision_bucket"],
                "label": patient["label"],
                "training_allowed_now": patient["training_allowed_now"],
                "blockers": patient["blockers"],
                "required_review_answers": patient["required_review_answers"],
                "allowed_actions_after_review": patient["allowed_actions_after_review"],
            }
        )

    return {
        "description": "DFR-141 reproducibility/archive manifest for the retained DFR-116 branch and review gate.",
        "status": "pass" if all_checks_pass and not missing_assets else "needs_attention",
        "training_gate": {
            "training_go": gate["training_go"],
            "blocked_reason": gate["training_blocked_reason"],
            "minimum_next_step": gate["minimum_next_step"],
            "current_best_branch": gate["current_best_branch"],
        },
        "metrics": {
            "dfr25_reference_mean": dfr25["aggregate"]["mean_metrics"],
            "dfr25_reference_top_weight": dfr25["aggregate"][
                "aggregate_top_weight_count"
            ],
            "dfr116_combo_mean": dfr116["aggregate"]["mean_metrics"],
            "dfr116_combo_top_weight": dfr116["aggregate"][
                "aggregate_top_weight_count"
            ],
            "dfr116_combined_clean": dfr138_summary[
                "dfr116_combined_clean_metrics"
            ],
        },
        "dfr116_patient_diff": {
            "fixed_ids": dfr116["fixed_ids"],
            "broken_ids": dfr116["broken_ids"],
        },
        "review_summary": {
            "patient_count": dfr139["summary"]["patient_count"],
            "case_count": dfr139["summary"]["case_count"],
            "decision_bucket_counts": dfr139["summary"]["decision_bucket_counts"],
            "training_allowed_now_count": dfr139["summary"][
                "training_allowed_now_count"
            ],
        },
        "review_checklist": review_rows,
        "validation_checks": validation_checks,
        "all_validation_checks_pass": all_checks_pass,
        "assets": assets,
        "missing_assets": missing_assets,
        "reproduction_commands": dfr119["reproduction_commands"],
        "source_reports": {
            "dfr119": DEFAULT_DFR119,
            "dfr138": DEFAULT_DFR138,
            "dfr139": DEFAULT_DFR139,
            "dfr140": dfr140_path,
        },
    }


def write_markdown(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# DFR-141 Reproducibility Manifest",
        "",
        "## Status",
        "",
        f"- manifest status: `{manifest['status']}`",
        f"- current best branch: `{manifest['training_gate']['current_best_branch']}`",
        f"- training_go: `{manifest['training_gate']['training_go']}`",
        f"- blocked reason: {manifest['training_gate']['blocked_reason']}",
        "",
        "## Metrics",
        "",
        f"- DFR-25 reference mean val_acc/auc/f1: `{fmt_metrics(manifest['metrics']['dfr25_reference_mean'])}`",
        f"- DFR-116 combo mean val_acc/auc/f1: `{fmt_metrics(manifest['metrics']['dfr116_combo_mean'])}`",
        f"- DFR-116 combined-clean diagnostic val_acc/auc/f1: `{fmt_metrics(manifest['metrics']['dfr116_combined_clean'])}`",
        f"- DFR-116 fixed cases: `{', '.join(manifest['dfr116_patient_diff']['fixed_ids'])}`",
        f"- DFR-116 broken cases: `{', '.join(manifest['dfr116_patient_diff']['broken_ids']) or 'none'}`",
        "",
        "## Validation Checks",
        "",
        f"- all DFR-119 validation checks pass: `{manifest['all_validation_checks_pass']}`",
    ]
    for check in manifest["validation_checks"]:
        lines.append(f"- `{check['name']}`: `{check['ok']}`")

    lines.extend(
        [
            "",
            "## Assets",
            "",
            f"- asset count: `{len(manifest['assets'])}`",
            f"- missing assets: `{', '.join(manifest['missing_assets']) or 'none'}`",
        ]
    )
    for asset in manifest["assets"]:
        status = "present" if asset["exists"] else "missing"
        size = asset.get("size_bytes", "n/a")
        lines.append(f"- `{asset['role']}`: `{status}` `{asset['path']}` size=`{size}`")

    lines.extend(
        [
            "",
            "## Required Review Checklist",
            "",
            f"- patients/cases: `{manifest['review_summary']['patient_count']} / {manifest['review_summary']['case_count']}`",
            f"- training_allowed_now_count: `{manifest['review_summary']['training_allowed_now_count']}`",
        ]
    )
    for patient in manifest["review_checklist"]:
        lines.append("")
        lines.append(f"### {patient['patient_id']}")
        lines.append("")
        lines.append(f"- bucket: `{patient['decision_bucket']}`")
        lines.append(f"- cases: `{', '.join(patient['case_ids'])}`")
        lines.append(f"- training_allowed_now: `{patient['training_allowed_now']}`")
        lines.append(f"- blockers: `{', '.join(patient['blockers'])}`")
        lines.append("- required answers:")
        for answer in patient["required_review_answers"]:
            lines.append(f"  - {answer}")
        lines.append("- allowed actions after review:")
        for action in patient["allowed_actions_after_review"]:
            lines.append(f"  - {action}")

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
        ]
    )
    for seed, command in sorted(
        manifest["reproduction_commands"].items(), key=lambda item: int(item[0])
    ):
        lines.append(f"- seed `{seed}`: `{command}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfr119", default=DEFAULT_DFR119)
    parser.add_argument("--dfr138", default=DEFAULT_DFR138)
    parser.add_argument("--dfr139", default=DEFAULT_DFR139)
    parser.add_argument("--dfr140", default=DEFAULT_DFR140)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    dfr119 = load_json(REPO_ROOT / args.dfr119)
    dfr138 = load_json(REPO_ROOT / args.dfr138)
    dfr139 = load_json(REPO_ROOT / args.dfr139)

    manifest = build_manifest(dfr119, dfr138, dfr139, args.dfr140)

    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(manifest, output_md)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "missing_asset_count": len(manifest["missing_assets"]),
                "training_go": manifest["training_gate"]["training_go"],
                "review_patient_count": manifest["review_summary"]["patient_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
