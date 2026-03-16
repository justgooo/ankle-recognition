from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_pattern(label: int, view_index: int, slice_index: int, image_size: int) -> np.ndarray:
    image = np.random.normal(loc=0.35, scale=0.08, size=(image_size, image_size)).astype(np.float32)
    yy, xx = np.mgrid[0:image_size, 0:image_size]

    center_x = image_size * (0.35 + 0.15 * view_index)
    center_y = image_size * (0.35 + 0.05 * ((slice_index % 5) - 2))
    radius = image_size * (0.14 + 0.02 * label)
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 < radius**2

    if label == 1:
        image[mask] += 0.45
        image[:, image_size // 2 - 3 : image_size // 2 + 3] += 0.08
    else:
        image[mask] += 0.15
        image[image_size // 3 : image_size // 3 + 8, :] += 0.05

    image += np.random.normal(loc=0.0, scale=0.02, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((array * 255.0).astype(np.uint8), mode="L")
    image.save(path)


def build_split(index: int, total: int) -> str:
    if index < int(total * 0.7):
        return "train"
    if index < int(total * 0.85):
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/demo", help="Output folder.")
    parser.add_argument("--num_patients", type=int, default=30, help="Number of demo patients.")
    parser.add_argument("--num_slices", type=int, default=16, help="Slices per view.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    patients_dir = output_dir / "patients"
    rows = []

    rng = np.random.default_rng(42)
    labels = rng.integers(low=0, high=2, size=args.num_patients)

    for index in range(args.num_patients):
        patient_id = f"P{index + 1:03d}"
        label = int(labels[index])
        age = int(rng.normal(loc=48 + 12 * label, scale=6))
        sex = "M" if rng.random() > 0.5 else "F"
        split = build_split(index, args.num_patients)

        patient_root = patients_dir / patient_id
        view_paths = []
        for view_index in range(3):
            view_dir = patient_root / f"view{view_index + 1}"
            view_paths.append(view_dir)
            for slice_index in range(args.num_slices):
                image = make_pattern(
                    label=label,
                    view_index=view_index,
                    slice_index=slice_index,
                    image_size=args.image_size,
                )
                save_image(image, view_dir / f"{slice_index + 1:04d}.png")

        rows.append(
            {
                "patient_id": patient_id,
                "label": label,
                "age": age,
                "sex": sex,
                "view1_dir": str(view_paths[0]).replace("\\", "/"),
                "view2_dir": str(view_paths[1]).replace("\\", "/"),
                "view3_dir": str(view_paths[2]).replace("\\", "/"),
                "split": split,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Dummy dataset created at: {output_dir}")
    print(f"Metadata CSV: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
