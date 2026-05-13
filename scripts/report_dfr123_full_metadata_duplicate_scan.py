#!/usr/bin/env python3
"""Full-metadata NIfTI duplicate scan for DFR-123.

This is a read-only data audit.  It scans train/val/test metadata paths,
reports path/hash/split duplicate groups, and does not inspect predictions or
test metrics.  Exact duplicate byte content must have the same file size, so
SHA256 is computed only inside same-size NIfTI candidate groups.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_dfr111_confirmed_target_posthoc import (  # noqa: E402
    counter_dict,
    save_json,
)
from src.dataset import VIEW_COLUMNS, canonicalize_view_columns, is_nifti_path  # noqa: E402


DEFAULT_METADATA = "data/realdata/metadata.csv"
DEFAULT_OUTPUT = "autoresearch_logs/dfr123_full_metadata_duplicate_scan.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def rel_or_abs(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return counter_dict(counter)


def required_columns() -> tuple[str, ...]:
    return ("patient_id", "label", "split", *VIEW_COLUMNS)


def load_metadata(path: Path) -> pd.DataFrame:
    df = canonicalize_view_columns(pd.read_csv(path))
    missing = [column for column in required_columns() if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    return df


def patient_ref(row: pd.Series, row_index: int, view_column: str) -> dict[str, Any]:
    return {
        "row_index": int(row_index),
        "patient_id": str(row["patient_id"]),
        "split": str(row["split"]),
        "label": int(row["label"]),
        "view": str(view_column),
    }


def path_level_patient_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for ref in refs:
        key = (str(ref["patient_id"]), int(ref["row_index"]))
        item = grouped.setdefault(
            key,
            {
                "row_index": int(ref["row_index"]),
                "patient_id": str(ref["patient_id"]),
                "split": str(ref["split"]),
                "label": int(ref["label"]),
                "views": [],
            },
        )
        item["views"].append(str(ref["view"]))

    return [
        {
            **item,
            "views": sorted(set(item["views"])),
        }
        for item in sorted(grouped.values(), key=lambda value: (value["split"], value["patient_id"]))
    ]


def summarize_refs(refs: list[dict[str, Any]]) -> dict[str, Any]:
    patients = path_level_patient_refs(refs)
    return {
        "patient_count": int(len({item["patient_id"] for item in patients})),
        "row_count": int(len({item["row_index"] for item in patients})),
        "split_counts": sorted_counter(Counter(item["split"] for item in patients)),
        "label_counts": sorted_counter(Counter(str(item["label"]) for item in patients)),
        "patients": patients,
    }


def collect_paths(repo_root: Path, df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    paths: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        for view_column in VIEW_COLUMNS:
            resolved = resolve_path(repo_root, str(row[view_column]))
            key = str(resolved)
            entry = paths.setdefault(
                key,
                {
                    "path": rel_or_abs(repo_root, resolved),
                    "absolute_path": str(resolved),
                    "refs": [],
                },
            )
            entry["refs"].append(patient_ref(row, int(row_index), view_column))

    for entry in paths.values():
        path = Path(str(entry["absolute_path"]))
        entry["exists"] = bool(path.exists())
        entry["is_file"] = bool(path.is_file())
        entry["is_nifti"] = bool(is_nifti_path(path))
        if not path.exists():
            errors.append({"path": entry["path"], "error": "missing_path"})
            continue
        if not path.is_file():
            errors.append({"path": entry["path"], "error": "not_file"})
            continue
        try:
            entry["file_size_bytes"] = int(path.stat().st_size)
        except OSError as exc:
            errors.append({"path": entry["path"], "error": f"stat_failed:{exc}"})
            entry["file_size_bytes"] = None
    return paths, errors


def path_duplicate_groups(paths: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for entry in paths.values():
        summary = summarize_refs(entry["refs"])
        if summary["patient_count"] <= 1:
            continue
        groups.append(
            {
                "path": entry["path"],
                "exists": bool(entry["exists"]),
                "is_nifti": bool(entry["is_nifti"]),
                "file_size_bytes": entry.get("file_size_bytes"),
                **summary,
            }
        )
    return sorted(groups, key=lambda item: (-item["patient_count"], item["path"]))


def hash_candidate_paths(paths: dict[str, dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in paths.values():
        if not entry.get("exists") or not entry.get("is_file") or not entry.get("is_nifti"):
            continue
        size = entry.get("file_size_bytes")
        if size is None:
            continue
        by_size[int(size)].append(entry)
    return {size: items for size, items in by_size.items() if len(items) > 1}


def compute_hashes(
    candidates_by_size: dict[int, list[dict[str, Any]]],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hash_errors = []
    hashed_entries = []
    for size, entries in sorted(candidates_by_size.items()):
        for entry in sorted(entries, key=lambda item: item["path"]):
            path = Path(str(entry["absolute_path"]))
            try:
                sha256 = sha256_file(path)
            except OSError as exc:
                hash_errors.append({"path": rel_or_abs(repo_root, path), "error": f"hash_failed:{exc}"})
                continue
            hashed_entry = dict(entry)
            hashed_entry["sha256"] = sha256
            hashed_entry["file_size_bytes"] = int(size)
            hashed_entries.append(hashed_entry)
    return hashed_entries, hash_errors


def content_duplicate_groups(hashed_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in hashed_entries:
        by_hash[str(entry["sha256"])].append(entry)

    groups = []
    for sha256, entries in by_hash.items():
        if len(entries) <= 1:
            continue
        all_patients = []
        members = []
        for entry in sorted(entries, key=lambda item: item["path"]):
            summary = summarize_refs(entry["refs"])
            all_patients.extend(summary["patients"])
            members.append(
                {
                    "path": entry["path"],
                    "file_size_bytes": int(entry["file_size_bytes"]),
                    **summary,
                }
            )

        split_counts = Counter(item["split"] for item in all_patients)
        label_counts = Counter(str(item["label"]) for item in all_patients)
        patient_ids = sorted({str(item["patient_id"]) for item in all_patients})
        groups.append(
            {
                "sha256": sha256,
                "file_size_bytes": int(entries[0]["file_size_bytes"]),
                "path_count": int(len(entries)),
                "patient_count": int(len(patient_ids)),
                "patient_ids": patient_ids,
                "split_counts": sorted_counter(split_counts),
                "label_counts": sorted_counter(label_counts),
                "cross_split": bool(len(split_counts) > 1),
                "cross_label": bool(len(label_counts) > 1),
                "members": members,
            }
        )
    return sorted(
        groups,
        key=lambda item: (
            not item["cross_split"],
            -item["patient_count"],
            item["sha256"],
        ),
    )


def duplicate_patient_id_groups(df: pd.DataFrame) -> list[dict[str, Any]]:
    groups = []
    for patient_id, group in df.reset_index(drop=True).groupby("patient_id", sort=True):
        if len(group) <= 1:
            continue
        rows = []
        for row_index, row in group.iterrows():
            rows.append(
                {
                    "row_index": int(row_index),
                    "patient_id": str(patient_id),
                    "split": str(row["split"]),
                    "label": int(row["label"]),
                }
            )
        groups.append(
            {
                "patient_id": str(patient_id),
                "row_count": int(len(group)),
                "split_counts": sorted_counter(Counter(row["split"] for row in rows)),
                "label_counts": sorted_counter(Counter(str(row["label"]) for row in rows)),
                "rows": rows,
            }
        )
    return groups


def build_report(repo_root: Path, metadata_path: Path) -> dict[str, Any]:
    df = load_metadata(metadata_path)
    paths, path_errors = collect_paths(repo_root, df)
    multi_patient_path_groups = path_duplicate_groups(paths)
    candidates_by_size = hash_candidate_paths(paths)
    hashed_entries, hash_errors = compute_hashes(candidates_by_size, repo_root)
    duplicate_content = content_duplicate_groups(hashed_entries)
    duplicate_patients = duplicate_patient_id_groups(df)

    existing_nifti_paths = [
        entry
        for entry in paths.values()
        if entry.get("exists") and entry.get("is_file") and entry.get("is_nifti")
    ]
    hashed_path_set = {entry["absolute_path"] for entry in hashed_entries}
    size_candidate_path_count = sum(len(items) for items in candidates_by_size.values())
    unique_size_path_count = len(existing_nifti_paths) - size_candidate_path_count

    cross_split_content = [group for group in duplicate_content if group["cross_split"]]
    cross_label_content = [group for group in duplicate_content if group["cross_label"]]
    duplicate_content_patient_ids = sorted(
        {
            patient_id
            for group in duplicate_content
            for patient_id in group["patient_ids"]
        }
    )

    return {
        "analysis": "dfr123_full_metadata_duplicate_scan",
        "description": (
            "Read-only full-metadata scan of train/val/test NIfTI path and byte-content "
            "duplicate groups.  No predictions or test metrics are read."
        ),
        "metadata": rel_or_abs(repo_root, metadata_path),
        "summary": {
            "metadata_rows": int(len(df)),
            "unique_patient_ids": int(df["patient_id"].astype(str).nunique()),
            "split_counts": sorted_counter(Counter(df["split"].astype(str))),
            "label_counts_by_split": {
                split: sorted_counter(Counter(group["label"].astype(str)))
                for split, group in df.groupby("split", sort=True)
            },
            "view_reference_count": int(len(df) * len(VIEW_COLUMNS)),
            "unique_resolved_path_count": int(len(paths)),
            "existing_nifti_path_count": int(len(existing_nifti_paths)),
            "missing_or_invalid_path_count": int(len(path_errors)),
            "duplicate_patient_id_group_count": int(len(duplicate_patients)),
            "multi_patient_same_path_group_count": int(len(multi_patient_path_groups)),
            "same_size_candidate_group_count": int(len(candidates_by_size)),
            "size_candidate_path_count": int(size_candidate_path_count),
            "sha256_hashed_path_count": int(len(hashed_path_set)),
            "unique_size_nifti_path_count": int(unique_size_path_count),
            "duplicate_content_group_count": int(len(duplicate_content)),
            "duplicate_content_patient_count": int(len(duplicate_content_patient_ids)),
            "cross_split_duplicate_content_group_count": int(len(cross_split_content)),
            "cross_label_duplicate_content_group_count": int(len(cross_label_content)),
        },
        "hash_scope": {
            "method": (
                "SHA256 was computed only for existing NIfTI unique paths whose "
                "file_size_bytes occurs in at least two unique paths. Exact byte "
                "duplicates cannot have different file sizes."
            ),
            "same_size_candidate_sizes": [
                {
                    "file_size_bytes": int(size),
                    "path_count": int(len(entries)),
                }
                for size, entries in sorted(candidates_by_size.items())
            ],
            "unique_size_nifti_path_count": int(unique_size_path_count),
        },
        "duplicate_patient_id_groups": duplicate_patients,
        "multi_patient_same_path_groups": multi_patient_path_groups,
        "duplicate_content_groups": duplicate_content,
        "path_errors": path_errors,
        "hash_errors": hash_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--metadata", type=Path, default=Path(DEFAULT_METADATA))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    metadata_path = args.metadata
    if not metadata_path.is_absolute():
        metadata_path = repo_root / metadata_path
    report = build_report(repo_root, metadata_path.resolve())
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    save_json(output, report)
    summary = report["summary"]
    print(f"Saved DFR-123 duplicate scan to: {output}")
    print(
        "Summary: "
        f"rows={summary['metadata_rows']} "
        f"unique_paths={summary['unique_resolved_path_count']} "
        f"hashed_paths={summary['sha256_hashed_path_count']} "
        f"duplicate_content_groups={summary['duplicate_content_group_count']} "
        f"cross_split_duplicate_content_groups={summary['cross_split_duplicate_content_group_count']}"
    )


if __name__ == "__main__":
    main()
