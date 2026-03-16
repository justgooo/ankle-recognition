from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NIFTI_PATTERNS = ("*.nii", "*.nii.gz", "*.gz")
OUTPUT_COLUMNS = [
    "patient_id",
    "label",
    "view1_dir",
    "view2_dir",
    "view3_dir",
    "split",
]


def discover_volume_files(input_dir: Path) -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in NIFTI_PATTERNS:
        for path in input_dir.glob(pattern):
            if path.is_file():
                files[path.resolve()] = None
    return sorted(files)


def patient_id_from_path(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    if lowered.endswith(".nii.gz"):
        return name[:-7]
    if lowered.endswith(".nii"):
        return name[:-4]
    if lowered.endswith(".gz"):
        return name[:-3]
    return path.stem


def build_base_rows(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        patient_id = patient_id_from_path(path)
        source = str(path).replace("\\", "/")
        rows.append(
            {
                "patient_id": patient_id,
                "label": pd.NA,
                "view1_dir": source,
                "view2_dir": source,
                "view3_dir": source,
                "split": pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def merge_optional_metadata(base_df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    extra_df = pd.read_csv(metadata_path)
    if "source_file" in extra_df.columns:
        base_df = base_df.copy()
        base_df["source_file"] = [Path(path).name for path in base_df["view1_dir"]]
        merged = base_df.merge(extra_df, on="source_file", how="left", suffixes=("", "_extra"))
    elif "patient_id" in extra_df.columns:
        merged = base_df.merge(extra_df, on="patient_id", how="left", suffixes=("", "_extra"))
    else:
        raise ValueError("Metadata CSV must contain either 'source_file' or 'patient_id'.")

    for column in ["label", "split"]:
        extra_column = f"{column}_extra"
        if extra_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[extra_column])
            merged = merged.drop(columns=[extra_column])

    return merged[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create metadata CSV for NIfTI real datasets.")
    parser.add_argument("--input_dir", default="data/realdata", help="Folder containing NIfTI volumes.")
    parser.add_argument(
        "--output_csv",
        default="data/realdata/metadata_template.csv",
        help="Where to save the generated metadata template.",
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Optional CSV with label/split columns keyed by patient_id or source_file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = discover_volume_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI volumes found in: {input_dir}")

    metadata = build_base_rows(files)
    if args.metadata_csv:
        metadata = merge_optional_metadata(metadata, Path(args.metadata_csv))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_csv, index=False)

    missing_label_count = int(metadata["label"].isna().sum())
    print(f"Found {len(metadata)} volumes in: {input_dir}")
    print(f"Saved metadata template to: {output_csv}")
    if missing_label_count > 0:
        print(f"{missing_label_count} rows are missing labels; fill them before training.")
    else:
        print("Labels are complete; this CSV can be used directly for training.")


if __name__ == "__main__":
    main()
