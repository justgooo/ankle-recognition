"""
dataset.py — CT 数据集加载与预处理
====================================
这个文件负责"怎么把CT图像从硬盘上读取出来，并处理成模型能吃的格式"。

主要功能：
    1. 从文件夹中读取 CT 切片图像（支持 PNG/JPG/DICOM/NIfTI 等格式）
    2. 对图像进行标准化处理（把像素值归一化到 0~1 之间）
    3. 把图像缩放到统一大小（比如 224×224）
    4. 从众多切片中均匀采样固定数量的切片（比如 16 张）
    5. 把 3 个视角（轴状面/冠状面/矢状面）的切片组合成一个完整的数据样本

核心类：
    - PatientCTDataset: 继承 PyTorch 的 Dataset，每次取一个病人的完整 CT 数据
"""

from __future__ import annotations

import gzip        # 用于解压 .gz 压缩文件（NIfTI 格式常用 .nii.gz）
import struct      # 用于解析二进制文件头（读取 NIfTI 文件时使用）
from functools import lru_cache  # 缓存装饰器：避免重复读取同一个文件
from pathlib import Path         # 路径操作工具
from typing import List

import numpy as np        # NumPy：数组和数值计算
import torch              # PyTorch：深度学习框架
from PIL import Image     # Pillow：图像读取和处理库
from torch.utils.data import Dataset  # PyTorch 的数据集基类

try:
    import pydicom  # pydicom：用来读取 DICOM 格式的医学影像文件
except ImportError:  # pragma: no cover - 如果没安装 pydicom 也不报错（非 DICOM 数据集不需要）
    pydicom = None


# ==================== 常量定义 ====================

# 支持的图像文件格式
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".dcm"}

# 三个视角对应的 CSV 列名
VIEW_COLUMNS = ["axial_dir", "coronal_dir", "sagittal_dir"]
# 分别代表：轴状面（横切面）、冠状面（从前往后切）、矢状面（从侧面切）

# 兼容旧版本的列名映射（view1 → axial, view2 → coronal, view3 → sagittal）
LEGACY_VIEW_COLUMN_ALIASES = {
    "view1_dir": "axial_dir",
    "view2_dir": "coronal_dir",
    "view3_dir": "sagittal_dir",
}

# 每个视角在 3D 体积数据中对应的切片轴
# 例如：axial（横切面）沿第 2 轴切，coronal 沿第 1 轴切，sagittal 沿第 0 轴切
VIEW_AXIS_BY_COLUMN = {
    "axial_dir": 2,
    "coronal_dir": 1,
    "sagittal_dir": 0,
}

# NIfTI 文件的后缀名（一种常见的医学影像 3D 格式）
NIFTI_SUFFIXES = (".nii", ".nii.gz", ".gz")


# ==================== 图像处理工具函数 ====================


def normalize_ct_image(image: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """
    CT 图像归一化 — 基于 HU（Hounsfield Unit）窗口。

    CT 图像的像素值是 HU 值，不同组织有不同的 HU 范围：
        - 空气: -1000 HU
        - 水:   0 HU
        - 骨骼: 400~1000 HU

    通过设置 hu_min 和 hu_max（窗口），我们只关注感兴趣的 HU 范围，
    并把它线性映射到 [0, 1]。

    参数：
        image:  原始 CT 图像（numpy 数组）
        hu_min: HU 窗口下限（低于此值的像素统一变为 0）
        hu_max: HU 窗口上限（高于此值的像素统一变为 1）

    返回：
        归一化后的图像，像素值在 [0, 1] 之间
    """
    image = np.clip(image.astype(np.float32), hu_min, hu_max)  # 把值限制在 [hu_min, hu_max]
    image = (image - hu_min) / max(hu_max - hu_min, 1e-6)      # 线性映射到 [0, 1]
    return image


def normalize_standard_image(image: np.ndarray) -> np.ndarray:
    """
    普通图像归一化 — 把像素值映射到 [0, 1]。

    适用于 PNG、JPG 等普通图像（不是 CT 的 HU 值）。
    使用 min-max 归一化：(像素 - 最小值) / (最大值 - 最小值)

    参数：
        image: 原始图像数组

    返回：
        归一化后的图像，像素值在 [0, 1] 之间
    """
    image = image.astype(np.float32)
    image_min = float(image.min())
    image_max = float(image.max())
    # 如果图像全黑或像素值都一样，直接返回全 0
    if image_max - image_min < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    把图像缩放到统一大小（size × size）。

    参数：
        image: 归一化后的灰度图（像素值 0~1）
        size:  目标大小，例如 224，则图像会变成 224×224

    返回：
        缩放后的图像数组
    """
    # 先把 [0, 1] 的浮点值转回 [0, 255] 的整数（PIL 需要 uint8 格式）
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L")  # "L" 表示灰度图
    pil_image = pil_image.resize((size, size), Image.BILINEAR)  # 双线性插值缩放
    return np.asarray(pil_image, dtype=np.float32) / 255.0  # 再转回 [0, 1]


def sample_slice_paths(paths: List[Path], num_slices: int, trim_edge_slices: int = 0) -> List[Path]:
    """
    从所有切片文件路径中均匀采样 num_slices 个。

    为什么要采样？
        - 每个病人的 CT 切片数量不一样（有的 100 张，有的 300 张）
        - 模型要求输入固定数量的切片（比如 16 张）
        - 所以用均匀采样把不同数量的切片统一到 num_slices 张

    参数：
        paths:            所有切片的文件路径列表（已排好顺序）
        num_slices:       目标采样数量（比如 16）
        trim_edge_slices: 去掉头尾各多少张（边缘切片经常没有用的信息）

    返回：
        采样后的文件路径列表
    """
    if len(paths) == 0:
        return []
    # 如果要去掉边缘，就多采一些，后面再裁掉头尾
    total_samples = num_slices + max(trim_edge_slices, 0) * 2
    # np.linspace 生成等间距的索引（从 0 到 len-1 之间均匀分布）
    indices = np.linspace(0, len(paths) - 1, total_samples)
    indices = np.round(indices).astype(int)  # 四舍五入到整数
    if trim_edge_slices > 0:
        indices = indices[trim_edge_slices:-trim_edge_slices]  # 去掉头尾
    return [paths[i] for i in indices]


def sample_slice_indices(length: int, num_slices: int, trim_edge_slices: int = 0) -> np.ndarray:
    """
    和 sample_slice_paths 类似，但返回的是索引数组而不是路径。
    用于 NIfTI 3D 体积数据（不是文件夹形式的切片）。

    参数：
        length:           切片轴的总长度（即总切片数）
        num_slices:       要采样的切片数
        trim_edge_slices: 去掉头尾各多少张

    返回：
        采样后的索引数组
    """
    if length <= 0:
        raise ValueError("Slice axis must be positive.")
    total_samples = num_slices + max(trim_edge_slices, 0) * 2
    indices = np.linspace(0, length - 1, total_samples)
    indices = np.round(indices).astype(int)
    if trim_edge_slices > 0:
        indices = indices[trim_edge_slices:-trim_edge_slices]
    return indices


def is_nifti_path(path: Path) -> bool:
    """判断文件路径是否是 NIfTI 格式（.nii 或 .nii.gz）"""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def canonicalize_view_columns(df):
    """
    把旧版列名（view1_dir, view2_dir, view3_dir）转换为新版列名（axial_dir, coronal_dir, sagittal_dir）。
    这样无论 CSV 用的是新版还是旧版列名，都能正常工作。
    """
    df = df.copy()
    for legacy_column, canonical_column in LEGACY_VIEW_COLUMN_ALIASES.items():
        if canonical_column not in df.columns and legacy_column in df.columns:
            df[canonical_column] = df[legacy_column]
    return df


# ==================== NIfTI 文件读取 ====================
# NIfTI 是一种 3D 医学影像格式，一个文件存储整个 3D 体积数据


def _dtype_from_nifti(datatype: int, endian: str):
    """
    根据 NIfTI 文件头中的 datatype 编号，确定数据的 numpy 数据类型。

    NIfTI 文件头会告诉我们数据是什么类型（整数、浮点数、几个字节等），
    这个函数负责把 NIfTI 的类型编号翻译成 numpy 能理解的数据类型。
    """
    dtype_map = {
        2: np.dtype(f"{endian}u1"),    # 无符号 8 位整数
        4: np.dtype(f"{endian}i2"),    # 有符号 16 位整数
        8: np.dtype(f"{endian}i4"),    # 有符号 32 位整数
        16: np.dtype(f"{endian}f4"),   # 32 位浮点数
        64: np.dtype(f"{endian}f8"),   # 64 位浮点数
        256: np.dtype(f"{endian}i1"),  # 有符号 8 位整数
        512: np.dtype(f"{endian}u2"),  # 无符号 16 位整数
        768: np.dtype(f"{endian}u4"),  # 无符号 32 位整数
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype: {datatype}")
    return dtype_map[datatype]


@lru_cache(maxsize=32)  # 缓存最近 32 个读取过的文件，避免重复读取
def load_nifti_volume(path_str: str) -> np.ndarray:
    """
    读取一个 NIfTI 格式的 3D 体积数据。

    NIfTI 文件结构：
        - 前 348 字节是文件头（header），包含数据的形状、类型等信息
        - 后面是实际的 3D 数据

    参数：
        path_str: NIfTI 文件的路径字符串

    返回：
        3D numpy 数组，形状为 (X, Y, Z)，代表 3D 体积数据
    """
    path = Path(path_str)
    # 如果文件名以 .gz 结尾，用 gzip 打开（压缩文件）；否则普通打开
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        # ---------- 读取文件头（348 字节） ----------
        header = f.read(348)
        if len(header) < 348:
            raise ValueError(f"Incomplete NIfTI header: {path}")

        # 检查字节序（大端 / 小端），NIfTI-1 的 sizeof_hdr 固定为 348
        sizeof_hdr = struct.unpack("<I", header[:4])[0]
        endian = "<"  # 默认小端
        if sizeof_hdr != 348:
            sizeof_hdr = struct.unpack(">I", header[:4])[0]
            if sizeof_hdr != 348:
                raise ValueError(f"Not a valid NIfTI-1 file: {path}")
            endian = ">"  # 大端

        # 解析文件头中的关键信息
        dim = struct.unpack(f"{endian}8h", header[40:56])  # 数据维度信息
        ndim = dim[0]                                       # 维度数量（应该是 3）
        shape = tuple(int(x) for x in dim[1 : ndim + 1])   # 各维度的大小 (X, Y, Z)
        datatype = struct.unpack(f"{endian}h", header[70:72])[0]  # 数据类型编号
        vox_offset = int(round(struct.unpack(f"{endian}f", header[108:112])[0]))  # 数据起始偏移量
        scl_slope = struct.unpack(f"{endian}f", header[112:116])[0]  # 缩放斜率
        scl_inter = struct.unpack(f"{endian}f", header[116:120])[0]  # 缩放截距

        if ndim != 3:
            raise ValueError(f"Only 3D NIfTI volumes are supported, got ndim={ndim} for {path}")

        # ---------- 读取实际的 3D 数据 ----------
        dtype = _dtype_from_nifti(datatype, endian)
        f.seek(vox_offset)  # 跳到数据开始的位置
        count = int(np.prod(shape))  # 总像素数 = X × Y × Z
        volume = np.frombuffer(f.read(), dtype=dtype, count=count)
        if volume.size != count:
            raise ValueError(f"NIfTI payload truncated: {path}")

    # 把 1D 数组重塑为 3D 数组（Fortran 列优先顺序，NIfTI 规范要求）
    volume = volume.reshape(shape, order="F").astype(np.float32, copy=False)
    # 应用缩放：real_value = raw_value × scl_slope + scl_inter
    if abs(scl_slope) > 1e-8:
        volume = volume * scl_slope + scl_inter
    return volume


def load_one_slice(path: Path, hu_min: float, hu_max: float) -> np.ndarray:
    """
    读取单张切片图像。

    支持两种格式：
        1. DICOM (.dcm) — 医院设备直接导出的格式，包含 HU 值
        2. 普通图像 (.png/.jpg/.bmp) — 已经转换好的图像

    参数：
        path:   切片文件路径
        hu_min: HU 窗口下限
        hu_max: HU 窗口上限

    返回：
        归一化后的 2D 图像数组，像素值在 [0, 1]
    """
    if path.suffix.lower() == ".dcm":
        # ---------- DICOM 格式 ----------
        if pydicom is None:
            raise ImportError("pydicom is required to read DICOM slices.")
        ds = pydicom.dcmread(str(path))          # 读取 DICOM 文件
        image = ds.pixel_array.astype(np.float32)  # 获取像素数据
        # DICOM 里的像素需要乘以 RescaleSlope 再加 RescaleIntercept 才能变成真正的 HU 值
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept
        return normalize_ct_image(image, hu_min=hu_min, hu_max=hu_max)

    # ---------- 普通图像格式（PNG/JPG 等） ----------
    image = Image.open(path).convert("L")  # 以灰度模式打开
    image = np.asarray(image, dtype=np.float32)
    return normalize_standard_image(image)


# ==================== 核心数据集类 ====================


class PatientCTDataset(Dataset):
    """
    病人 CT 数据集 — 继承自 PyTorch 的 Dataset。

    每次通过 __getitem__(index) 取出一个病人的数据，包含：
        - images: 形状为 (3, S, H, W) 的张量
            3 = 三个视角（轴状面/冠状面/矢状面）
            S = 每个视角的切片数量
            H, W = 图像的高度和宽度
        - label:  0 或 1 的标签（0=正常，1=异常）
        - patient_id: 病人 ID

    使用方式：
        dataset = PatientCTDataset(df, base_dir="data/", ...)
        sample = dataset[0]  # 取第 0 个病人的数据
        images = sample["images"]  # (3, 16, 224, 224)
        label = sample["label"]    # 0 或 1
    """

    def __init__(
        self,
        df,                           # 包含病人信息的 DataFrame（CSV 读取后的表格数据）
        base_dir: str | Path,         # 图像文件的根目录
        image_size: int,              # 图像缩放到的目标大小（如 224）
        num_slices_per_view: int,     # 每个视角采样多少张切片（如 16）
        trim_edge_slices: int,        # 去掉头尾各多少张边缘切片
        hu_min: float,               # HU 窗口下限
        hu_max: float,               # HU 窗口上限
    ) -> None:
        """
        初始化数据集。

        参数：
            df: Pandas DataFrame，必须包含以下列：
                - patient_id: 病人 ID
                - label: 标签（0 或 1）
                - axial_dir: 轴状面切片的目录路径
                - coronal_dir: 冠状面切片的目录路径
                - sagittal_dir: 矢状面切片的目录路径
            base_dir: 所有路径的根目录（如果 CSV 里的路径是相对路径，就相对于这个目录）
            image_size: 图像缩放目标大小
            num_slices_per_view: 每个视角取多少张切片
            trim_edge_slices: 去掉头尾边缘切片数
            hu_min: CT 窗口下限（HU 值）
            hu_max: CT 窗口上限（HU 值）
        """
        # canonicalize_view_columns: 把旧版列名转成标准列名
        # reset_index: 重置行号从 0 开始
        self.df = canonicalize_view_columns(df).reset_index(drop=True).copy()
        self.base_dir = Path(base_dir)
        self.image_size = image_size
        self.num_slices_per_view = num_slices_per_view
        self.trim_edge_slices = trim_edge_slices
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self) -> int:
        """返回数据集中有多少个病人"""
        return len(self.df)

    def __getitem__(self, index: int):
        """
        取出第 index 个病人的数据。

        PyTorch 的 DataLoader 会自动调用这个方法来获取训练数据。

        返回：
            一个字典，包含：
                - "images": (3, S, H, W) 的 float32 张量
                - "label":  0 或 1 的整数标签
                - "patient_id": 病人 ID 字符串
        """
        row = self.df.iloc[index]  # 取出第 index 行的数据
        # 分别加载 3 个视角的切片图像
        image_views = [self._load_view(row[col], col) for col in VIEW_COLUMNS]
        # 把 3 个视角的张量堆叠在一起，形成 (3, S, H, W) 的张量
        image_tensor = torch.stack(image_views, dim=0)
        # 把标签转成 PyTorch 的 long 类型张量
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "images": image_tensor,
            "label": label,
            "patient_id": str(row["patient_id"]),
        }

    def _load_view(self, view_dir_value: str, view_column: str) -> torch.Tensor:
        """
        加载一个视角的所有切片图像。

        参数：
            view_dir_value: 该视角切片所在的目录路径（从 CSV 读取）
            view_column:    视角列名（如 "axial_dir"）

        返回：
            (S, H, W) 的张量，S 张切片叠在一起
        """
        view_path = self._resolve_path(view_dir_value)
        if not view_path.exists():
            raise FileNotFoundError(f"View path not found: {view_path}")

        # 情况 1：路径指向一个 NIfTI 3D 体积文件
        if view_path.is_file() and is_nifti_path(view_path):
            return self._load_nifti_view(view_path, view_column)

        # 情况 2：路径指向一个文件夹，里面有很多切片图像
        if not view_path.is_dir():
            raise FileNotFoundError(f"Expected a view folder or NIfTI volume, got: {view_path}")

        # 找出文件夹中所有支持格式的图像文件，并按文件名排序
        all_paths = sorted(
            path for path in view_path.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not all_paths:
            raise FileNotFoundError(f"No supported slice files found in: {view_path}")

        # 从所有切片中均匀采样 num_slices_per_view 张
        sampled_paths = sample_slice_paths(
            all_paths,
            self.num_slices_per_view,
            trim_edge_slices=self.trim_edge_slices,
        )

        # 逐张读取、归一化、缩放
        slices = []
        for path in sampled_paths:
            image = load_one_slice(path, hu_min=self.hu_min, hu_max=self.hu_max)
            image = resize_image(image, self.image_size)
            slices.append(torch.tensor(image, dtype=torch.float32))

        # 把所有切片堆叠成 (S, H, W) 的张量
        return torch.stack(slices, dim=0)

    def _load_nifti_view(self, volume_path: Path, view_column: str) -> torch.Tensor:
        """
        从 NIfTI 3D 体积文件中加载指定视角的切片。

        NIfTI 文件存储的是完整的 3D 数据 (X, Y, Z)。
        不同的视角就是沿不同的轴切：
            - axial（轴状面）: 沿 Z 轴切 → volume[:, :, z]
            - coronal（冠状面）: 沿 Y 轴切 → volume[:, y, :]
            - sagittal（矢状面）: 沿 X 轴切 → volume[x, :, :]

        参数：
            volume_path: NIfTI 文件路径
            view_column: 视角列名，用来确定沿哪个轴切

        返回：
            (S, H, W) 的张量
        """
        volume = load_nifti_volume(str(volume_path.resolve()))  # 读取 3D 数据
        axis = VIEW_AXIS_BY_COLUMN[view_column]  # 确定切片轴

        # 在切片轴上均匀采样
        sampled_indices = sample_slice_indices(
            volume.shape[axis],
            self.num_slices_per_view,
            trim_edge_slices=self.trim_edge_slices,
        )

        slices = []
        for slice_index in sampled_indices:
            # 根据不同的轴，取出对应的 2D 切片
            if axis == 2:
                image = volume[:, :, slice_index]   # 轴状面
            elif axis == 1:
                image = volume[:, slice_index, :]   # 冠状面
            else:
                image = volume[slice_index, :, :]   # 矢状面

            image = np.rot90(image)  # 旋转 90 度，让图像方向正确
            image = normalize_ct_image(image, hu_min=self.hu_min, hu_max=self.hu_max)
            image = resize_image(image, self.image_size)
            slices.append(torch.tensor(image, dtype=torch.float32))

        return torch.stack(slices, dim=0)

    def _resolve_path(self, path_value: str) -> Path:
        """
        解析路径：如果是绝对路径就直接用，如果是相对路径就基于 base_dir 拼接。
        """
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()
