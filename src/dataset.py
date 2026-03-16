from __future__ import annotations

import gzip
import struct
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import pydicom
except ImportError:  # pragma: no cover - optional for non-DICOM datasets
    pydicom = None


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".dcm"}
VIEW_COLUMNS = ["view1_dir", "view2_dir", "view3_dir"]
NIFTI_SUFFIXES = (".nii", ".nii.gz", ".gz")


def normalize_ct_image(image: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    image = np.clip(image.astype(np.float32), hu_min, hu_max)
    image = (image - hu_min) / max(hu_max - hu_min, 1e-6)
    return image


def normalize_standard_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max - image_min < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L")
    pil_image = pil_image.resize((size, size), Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.float32) / 255.0


def sample_slice_paths(paths: List[Path], num_slices: int, trim_edge_slices: int = 0) -> List[Path]:
    if len(paths) == 0:
        return []
    total_samples = num_slices + max(trim_edge_slices, 0) * 2
    indices = np.linspace(0, len(paths) - 1, total_samples)
    indices = np.round(indices).astype(int)
    if trim_edge_slices > 0:
        indices = indices[trim_edge_slices:-trim_edge_slices]
    return [paths[i] for i in indices]


def sample_slice_indices(length: int, num_slices: int, trim_edge_slices: int = 0) -> np.ndarray:
    if length <= 0:
        raise ValueError("Slice axis must be positive.")
    total_samples = num_slices + max(trim_edge_slices, 0) * 2
    indices = np.linspace(0, length - 1, total_samples)
    indices = np.round(indices).astype(int)
    if trim_edge_slices > 0:
        indices = indices[trim_edge_slices:-trim_edge_slices]
    return indices


def is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def _dtype_from_nifti(datatype: int, endian: str):
    dtype_map = {
        2: np.dtype(f"{endian}u1"),
        4: np.dtype(f"{endian}i2"),
        8: np.dtype(f"{endian}i4"),
        16: np.dtype(f"{endian}f4"),
        64: np.dtype(f"{endian}f8"),
        256: np.dtype(f"{endian}i1"),
        512: np.dtype(f"{endian}u2"),
        768: np.dtype(f"{endian}u4"),
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype: {datatype}")
    return dtype_map[datatype]


@lru_cache(maxsize=32)
def load_nifti_volume(path_str: str) -> np.ndarray:
    path = Path(path_str)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        header = f.read(348)
        if len(header) < 348:
            raise ValueError(f"Incomplete NIfTI header: {path}")

        sizeof_hdr = struct.unpack("<I", header[:4])[0]
        endian = "<"
        if sizeof_hdr != 348:
            sizeof_hdr = struct.unpack(">I", header[:4])[0]
            if sizeof_hdr != 348:
                raise ValueError(f"Not a valid NIfTI-1 file: {path}")
            endian = ">"

        dim = struct.unpack(f"{endian}8h", header[40:56])
        ndim = dim[0]
        shape = tuple(int(x) for x in dim[1 : ndim + 1])
        datatype = struct.unpack(f"{endian}h", header[70:72])[0]
        vox_offset = int(round(struct.unpack(f"{endian}f", header[108:112])[0]))
        scl_slope = struct.unpack(f"{endian}f", header[112:116])[0]
        scl_inter = struct.unpack(f"{endian}f", header[116:120])[0]

        if ndim != 3:
            raise ValueError(f"Only 3D NIfTI volumes are supported, got ndim={ndim} for {path}")

        dtype = _dtype_from_nifti(datatype, endian)
        f.seek(vox_offset)
        count = int(np.prod(shape))
        volume = np.frombuffer(f.read(), dtype=dtype, count=count)
        if volume.size != count:
            raise ValueError(f"NIfTI payload truncated: {path}")

    volume = volume.reshape(shape, order="F").astype(np.float32, copy=False)
    if abs(scl_slope) > 1e-8:
        volume = volume * scl_slope + scl_inter
    return volume


def load_one_slice(path: Path, hu_min: float, hu_max: float) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required to read DICOM slices.")
        ds = pydicom.dcmread(str(path))
        image = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept
        return normalize_ct_image(image, hu_min=hu_min, hu_max=hu_max)

    image = Image.open(path).convert("L")
    image = np.asarray(image, dtype=np.float32)
    return normalize_standard_image(image)


class PatientCTDataset(Dataset):
    def __init__(
        self,
        df,
        base_dir: str | Path,
        image_size: int,
        num_slices_per_view: int,
        trim_edge_slices: int,
        hu_min: float,
        hu_max: float,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.base_dir = Path(base_dir)
        self.image_size = image_size
        self.num_slices_per_view = num_slices_per_view
        self.trim_edge_slices = trim_edge_slices
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_views = [self._load_view(row[col], view_index) for view_index, col in enumerate(VIEW_COLUMNS)]
        image_tensor = torch.stack(image_views, dim=0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "images": image_tensor,
            "label": label,
            "patient_id": str(row["patient_id"]),
        }

    def _load_view(self, view_dir_value: str, view_index: int) -> torch.Tensor:
        view_path = self._resolve_path(view_dir_value)
        if not view_path.exists():
            raise FileNotFoundError(f"View path not found: {view_path}")

        if view_path.is_file() and is_nifti_path(view_path):
            return self._load_nifti_view(view_path, view_index)

        if not view_path.is_dir():
            raise FileNotFoundError(f"Expected a view folder or NIfTI volume, got: {view_path}")

        all_paths = sorted(
            path for path in view_path.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not all_paths:
            raise FileNotFoundError(f"No supported slice files found in: {view_path}")

        sampled_paths = sample_slice_paths(
            all_paths,
            self.num_slices_per_view,
            trim_edge_slices=self.trim_edge_slices,
        )
        slices = []
        for path in sampled_paths:
            image = load_one_slice(path, hu_min=self.hu_min, hu_max=self.hu_max)
            image = resize_image(image, self.image_size)
            slices.append(torch.tensor(image, dtype=torch.float32))

        return torch.stack(slices, dim=0)

    def _load_nifti_view(self, volume_path: Path, view_index: int) -> torch.Tensor:
        volume = load_nifti_volume(str(volume_path.resolve()))
        axis = {0: 2, 1: 1, 2: 0}[view_index]
        sampled_indices = sample_slice_indices(
            volume.shape[axis],
            self.num_slices_per_view,
            trim_edge_slices=self.trim_edge_slices,
        )
        slices = []

        for slice_index in sampled_indices:
            if axis == 2:
                image = volume[:, :, slice_index]
            elif axis == 1:
                image = volume[:, slice_index, :]
            else:
                image = volume[slice_index, :, :]

            image = np.rot90(image)
            image = normalize_ct_image(image, hu_min=self.hu_min, hu_max=self.hu_max)
            image = resize_image(image, self.image_size)
            slices.append(torch.tensor(image, dtype=torch.float32))

        return torch.stack(slices, dim=0)

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()
