from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VIEW_COLUMNS = ("axial_dir", "coronal_dir", "sagittal_dir")


def build_candidate(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def normalize_path(raw_path: str) -> str:
    return raw_path.replace("\\", "/")


def repair_path(base_dir: Path, raw_path: str) -> tuple[str, bool]:
    normalized = normalize_path(raw_path)
    if build_candidate(base_dir, normalized).exists():
        return normalized, False

    fallback = normalized.replace("/CTdata2/", "/")
    if fallback != normalized and build_candidate(base_dir, fallback).exists():
        return fallback, True

    raise FileNotFoundError(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local metadata mirror with repaired view paths without modifying the source CSV."
    )
    parser.add_argument(
        "--src",
        default="data/realdata/metadata.csv",
        help="Source metadata CSV.",
    )
    parser.add_argument(
        "--dst",
        default="autoresearch_logs/local_metadata_fixed.csv",
        help="Destination CSV for the repaired local mirror.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory used to resolve relative paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_path = Path(args.src)
    dst_path = Path(args.dst)
    base_dir = Path(args.base_dir).resolve()

    df = pd.read_csv(src_path)
    fixed_counts: dict[str, int] = {column: 0 for column in VIEW_COLUMNS}
    unresolved: list[tuple[int, str, str]] = []

    for column in VIEW_COLUMNS:
        repaired_values: list[str] = []
        for row_index, raw_value in enumerate(df[column].astype(str).tolist()):
            try:
                repaired_value, changed = repair_path(base_dir, raw_value)
            except FileNotFoundError:
                unresolved.append((row_index, column, raw_value))
                repaired_values.append(normalize_path(raw_value))
                continue
            repaired_values.append(repaired_value)
            if changed:
                fixed_counts[column] += 1
        df[column] = repaired_values

    if unresolved:
        preview = ", ".join(
            f"row={row_index} col={column} path={raw_value}"
            for row_index, column, raw_value in unresolved[:5]
        )
        raise RuntimeError(
            f"Unable to resolve {len(unresolved)} metadata paths. Examples: {preview}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)

    fix_summary = ", ".join(f"{column}={count}" for column, count in fixed_counts.items())
    print(f"Wrote repaired metadata mirror to: {dst_path}")
    print(f"Applied path repairs: {fix_summary}")


if __name__ == "__main__":
    main()
