"""
model.py — 多视角 CT 分类模型
================================
这个文件定义了整个「足踝 CT 分类」项目的神经网络模型。

整体思路：
    1. 一张 CT 扫描有 3 个视角（轴状面 / 冠状面 / 矢状面），每个视角有多张切片图像。
    2. 用 encoder 把每张切片提取成一个 512 维的特征向量。
       支持五种 backbone：
       - ResNet18（经典分类 backbone）
       - ResUNet + Attention Gate（空间注意力增强，关注病灶区域）
       - ResNeXt（分组卷积 + 多分支聚合，通过 timm 的 resnext50_32x4d 实现）
       - SENet（通道注意力，Squeeze-and-Excitation，通过 timm 的 seresnet50 实现）
       - CSPNet（跨阶段部分连接，梯度复用更高效，通过 timm 的 cspresnet50 实现）
    3. 把同一个视角的所有切片特征聚合（mean pooling 或 AttentionPooling），得到该视角的代表特征。
    4. 多视角融合后分类，输出 2 个值：正常/异常

本文件包含以下类：
    - build_resnet18_encoder(): 构建 ResNet18 特征提取器
    - AttentionGate: Attention Gate 模块 (Oktay et al., 2018)
    - ResUNetEncoder: ResNet18-based UNet Encoder with Attention Gates
    - GenericTimmEncoder: 通用 timm backbone 包装器（支持 ResNeXt / SENet / CSPNet）
    - build_encoder(): 根据 backbone 类型构建 encoder
    - MultiViewEncoder: 多视角编码器（提取 3 个视角的特征，支持 mean pooling 或 AttentionPooling）
    - MultiViewCTClassifier: 特征融合分类器（拼接 3 个视角特征后分类）
    - MultiViewDecisionFusionClassifier: 决策融合分类器（每个视角单独分类后加权投票）
"""

from __future__ import annotations

import math
import os

import torch  # type: ignore[import-not-found]          # PyTorch：深度学习框架的核心库
import torch.nn as nn  # type: ignore[import-not-found]  # nn 模块：提供各种神经网络层（卷积、线性层等）
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-not-found]
# ↑ 从 torchvision 导入 ResNet18 预训练模型和对应的权重

from .attention_pooling import AttentionPooling  # 可学习注意力池化模块

# ==================== 方差控制：Backbone 冻结 ====================
# 冻结 ResNet18 的前 N 个 layer block，保留 ImageNet 预训练权重。
# 设为 0 表示不冻结（原始行为），设为 3 表示只训练 layer4 + 分类头。
# autoresearch Agent 通过修改此常量来实验不同冻结策略。
DEFAULT_FREEZE_LAYERS = 3  # Stage 10C VR-MS：冻结 conv1+layer1+layer2+layer3，只训练 layer4+head
# Keep the learned-weighting softmax untempered while isolating reliability-path
# module ablations. Ratio probes can override this constant in dedicated runs.
LEARNED_FUSION_TEMPERATURE = 1.0


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_float(name: str, default: float) -> float:
    if not math.isfinite(default) or default <= 0.0:
        raise ValueError(f"Default value for {name} must be finite and > 0, got {default}.")
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive float, got {raw!r}.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive float, got {raw!r}.")
    return value


def _env_unit_float(name: str, default: float) -> float:
    if not math.isfinite(default) or not (0.0 <= default <= 1.0):
        raise ValueError(f"Default value for {name} must be finite and in [0, 1], got {default}.")
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite float in [0, 1], got {raw!r}.") from exc
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a finite float in [0, 1], got {raw!r}.")
    return value


def _env_positive_int(name: str, default: int) -> int:
    if default <= 0:
        raise ValueError(f"Default value for {name} must be > 0, got {default}.")
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.")
    return value


def _env_view_mask(name: str) -> tuple[float, float, float] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated entries, got {raw!r}.")
    values: list[float] = []
    for part in parts:
        lowered = part.lower()
        if lowered in {"true", "yes", "on"}:
            value = 1.0
        elif lowered in {"false", "no", "off"}:
            value = 0.0
        else:
            try:
                value = float(part)
            except ValueError as exc:
                raise ValueError(f"{name} entries must be 0/1-like values, got {part!r}.") from exc
        if value not in {0.0, 1.0}:
            raise ValueError(f"{name} entries must be 0.0 or 1.0, got {part!r}.")
        values.append(value)
    if sum(values) <= 0.0:
        raise ValueError(f"{name} must keep at least one active view, got {raw!r}.")
    return values[0], values[1], values[2]


def build_resnet18_encoder(use_pretrained: bool, freeze_layers: int = DEFAULT_FREEZE_LAYERS) -> nn.Module:
    """
    构建一个 ResNet18 特征提取器（编码器）。

    为什么要改造 ResNet18？
        - 原版 ResNet18 接收 3 通道（RGB 彩色）图像，输出 1000 类分类结果。
        - 我们的 CT 图像是灰度图（1 个通道），而且只需要提取特征，不需要分类。
        - 所以需要：① 把第一层卷积从 3 通道改成 1 通道；② 把最后的分类层去掉。

    参数：
        use_pretrained: 是否使用在 ImageNet 上预训练好的权重。
            - True:  使用预训练权重（迁移学习，通常效果更好）
            - False: 随机初始化权重（从零开始学习）
        freeze_layers: 冻结前 N 个 layer block（0=不冻结，1=冻结到 layer1，
            2=冻结到 layer2，3=冻结到 layer3）。冻结的层不参与训练，
            保留 ImageNet 预训练权重，大幅减少可训练参数和 seed 方差。

    返回：
        改造后的 ResNet18 模型，输入灰度图，输出 512 维特征向量。
    """
    # ---------- 第 1 步：加载 ResNet18 模型 ----------
    # 如果 use_pretrained=True，就加载 ImageNet 预训练权重；否则随机初始化
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    backbone = resnet18(weights=weights)

    # ---------- 第 2 步：把第一层卷积从 3 通道改为 1 通道 ----------
    # 原始 ResNet18 的第一层卷积 conv1 接收 3 通道（RGB），但 CT 图是灰度的只有 1 通道
    old_conv = backbone.conv1  # 原来的 3 通道卷积层
    new_conv = nn.Conv2d(
        1,                           # 输入通道数改为 1（灰度图）
        old_conv.out_channels,       # 输出通道数保持不变（64）
        kernel_size=old_conv.kernel_size,  # 卷积核大小不变（7×7）
        stride=old_conv.stride,      # 步长不变（2）
        padding=old_conv.padding,    # 填充方式不变
        bias=False,                  # 不使用偏置项
    )
    # 如果加载了预训练权重，把原来 3 通道的权重"取平均"变成 1 通道的权重
    # 这样可以保留预训练的知识
    if weights is not None:
        with torch.no_grad():  # 不需要计算梯度（只是在复制权重）
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    backbone.conv1 = new_conv  # 用新的 1 通道卷积替换原来的

    # ---------- 第 3 步：去掉最后的分类层 ----------
    # 原来 ResNet18 最后有一个全连接层 fc，把 512 维特征映射到 1000 个类别
    # 我们只需要 512 维的特征，不需要分类，所以用 Identity() 替换（即什么都不做，直接输出）
    backbone.fc = nn.Identity()

    # ---------- 第 4 步：冻结前 N 个 layer block ----------
    # ResNet18 结构: conv1 → bn1 → layer1 → layer2 → layer3 → layer4 → avgpool → fc
    # 冻结目的：保留 ImageNet 预训练的通用视觉特征（边缘/纹理等），
    # 大幅减少可训练参数数量（从 ~11M/backbone 降到 ~2.6M），降低 seed 方差。
    if freeze_layers >= 1:
        for param in backbone.conv1.parameters():
            param.requires_grad = False
        for param in backbone.bn1.parameters():
            param.requires_grad = False
        for param in backbone.layer1.parameters():
            param.requires_grad = False
    if freeze_layers >= 2:
        for param in backbone.layer2.parameters():
            param.requires_grad = False
    if freeze_layers >= 3:
        for param in backbone.layer3.parameters():
            param.requires_grad = False
    # layer4 始终可训练（高级语义特征需要适配 CT 域）

    return backbone


# ==================== ResUNet + Attention Gate ====================


class AttentionGate(nn.Module):
    """Attention Gate (Oktay et al., 2018).

    用 gating signal ``g`` 对 skip connection 特征 ``x`` 做空间注意力加权。
    通过 1×1 卷积将 g 和 x 映射到共同的中间维度，
    经 ReLU + 1×1 Conv + Sigmoid 生成 attention map，
    对 x 的每个空间位置进行加权。

    参数：
        F_g: gating signal 的通道数（来自更深层 / bottleneck）
        F_l: skip connection 特征的通道数（来自 encoder 同层）
        F_int: 中间维度（一般取 F_l // 2 或 F_g // 2）
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            g: gating signal, (B, F_g, H_g, W_g) — 通常来自 bottleneck 或更深层
            x: skip connection, (B, F_l, H_x, W_x) — 来自 encoder 同层

        返回：
            attention-gated x, 形状与 x 相同 (B, F_l, H_x, W_x)
        """
        # 把 g 上采样到与 x 相同的空间尺寸
        g_up = nn.functional.interpolate(
            g, size=x.shape[2:], mode="bilinear", align_corners=False,
        )
        g1 = self.W_g(g_up)   # (B, F_int, H_x, W_x)
        x1 = self.W_x(x)      # (B, F_int, H_x, W_x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)    # (B, 1, H_x, W_x)  — spatial attention map
        return x * psi         # element-wise gating


class ResUNetEncoder(nn.Module):
    """ResNet18-based UNet Encoder with Attention Gates.

    复用 ResNet18 预训练权重作为 encoder，在 skip connection 上添加
    Attention Gate 做空间注意力加权。最终通过全局平均池化输出 512 维
    slice feature，与原 ResNet18 backbone 接口完全一致。

    结构：
        输入: (B, 1, H, W)

        Encoder (ResNet18):
            e1 = conv1 + bn1 + relu       → (B, 64, H/2, W/2)
            e2 = maxpool + layer1          → (B, 64, H/4, W/4)
            e3 = layer2                    → (B, 128, H/8, W/8)
            e4 = layer3                    → (B, 256, H/16, W/16)

        Bottleneck:
            b  = layer4                    → (B, 512, H/32, W/32)

        Attention Gates (用 b 作为 gating signal):
            a4 = AG(g=b, x=e4)            → (B, 256, H/16, W/16)
            a3 = AG(g=b, x=e3)            → (B, 128, H/8, W/8)
            a2 = AG(g=b, x=e2)            → (B, 64, H/4, W/4)

        Feature Fusion:
            GAP(b)                         → (B, 512)
            GAP(a4) + GAP(a3) + GAP(a2)   → 各自池化后经线性投影到 512D
            累加到 bottleneck feature 上

        输出: (B, 512)

    参数：
        use_pretrained: 是否使用 ImageNet 预训练权重
        freeze_layers: 冻结前 N 个 encoder stage（0-3），同 build_resnet18_encoder
    """

    def __init__(
        self,
        use_pretrained: bool = True,
        freeze_layers: int = DEFAULT_FREEZE_LAYERS,
    ) -> None:
        super().__init__()

        # ---------- 加载 ResNet18 ----------
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = resnet18(weights=weights)

        # 1 通道灰度输入
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if weights is not None:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # ---------- Encoder stages ----------
        # e1: conv1 + bn1 + relu  → (64, H/2, W/2)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        # e2: maxpool + layer1    → (64, H/4, W/4)
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        # e3: layer2              → (128, H/8, W/8)
        self.enc3 = backbone.layer2
        # e4: layer3              → (256, H/16, W/16)
        self.enc4 = backbone.layer3
        # bottleneck: layer4      → (512, H/32, W/32)
        self.bottleneck = backbone.layer4

        # ---------- Attention Gates ----------
        # AG4: gating=512, skip=256, int=128
        self.ag4 = AttentionGate(F_g=512, F_l=256, F_int=128)
        # AG3: gating=512, skip=128, int=64
        self.ag3 = AttentionGate(F_g=512, F_l=128, F_int=64)
        # AG2: gating=512, skip=64, int=32
        self.ag2 = AttentionGate(F_g=512, F_l=64, F_int=32)

        # ---------- 全局平均池化 ----------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---------- Skip feature 融合投影 ----------
        # 把 attention-gated 的多尺度特征池化后投影到 512D，累加到 bottleneck
        self.skip_proj = nn.Sequential(
            nn.Linear(256 + 128 + 64, 512),
            nn.ReLU(inplace=True),
        )

        # ---------- 冻结策略 ----------
        if freeze_layers >= 1:
            for param in self.enc1.parameters():
                param.requires_grad = False
            for param in self.enc2.parameters():
                param.requires_grad = False
        if freeze_layers >= 2:
            for param in self.enc3.parameters():
                param.requires_grad = False
        if freeze_layers >= 3:
            for param in self.enc4.parameters():
                param.requires_grad = False
        # bottleneck (layer4) 始终可训练
        # Attention Gate 模块始终可训练（无预训练权重）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (B, 1, H, W) 灰度 CT 切片

        返回：
            (B, 512) slice feature
        """
        # Encoder
        e1 = self.enc1(x)          # (B, 64, H/2, W/2)
        e2 = self.enc2(e1)         # (B, 64, H/4, W/4)
        e3 = self.enc3(e2)         # (B, 128, H/8, W/8)
        e4 = self.enc4(e3)         # (B, 256, H/16, W/16)
        b = self.bottleneck(e4)    # (B, 512, H/32, W/32)

        # Attention Gates — 用 bottleneck 作为 gating signal
        a4 = self.ag4(g=b, x=e4)  # (B, 256, H/16, W/16)
        a3 = self.ag3(g=b, x=e3)  # (B, 128, H/8, W/8)
        a2 = self.ag2(g=b, x=e2)  # (B, 64, H/4, W/4)

        # 全局池化 bottleneck → (B, 512)
        feat_b = self.gap(b).flatten(1)

        # 全局池化 attention-gated skip features → 投影并融合
        feat_a4 = self.gap(a4).flatten(1)  # (B, 256)
        feat_a3 = self.gap(a3).flatten(1)  # (B, 128)
        feat_a2 = self.gap(a2).flatten(1)  # (B, 64)
        feat_skip = torch.cat([feat_a4, feat_a3, feat_a2], dim=1)  # (B, 448)
        feat_skip = self.skip_proj(feat_skip)  # (B, 512)

        # 累加融合
        return feat_b + feat_skip  # (B, 512)


class GenericTimmEncoder(nn.Module):
    """
    通用 timm 编码器包装器，用于支持 ResNeXt, SENet, CSPNet 等。
    将任意维度的输出特征投影到 512 维，以兼容现有项目框架。
    """
    def __init__(
        self,
        model_name: str,
        use_pretrained: bool = True,
        freeze_layers: int = DEFAULT_FREEZE_LAYERS,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("请先安装 timm 库 (pip install timm) 以使用额外模型。")
            
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            in_chans=1,
            num_classes=0, # 全局池化后直接输出特征
        )
        
        # 获取 timm 模型的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.backbone(dummy_input)
            in_dim = dummy_output.shape[1]
            
        # 兼容性投影：统一为 512 维
        if in_dim != 512:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

        # 尽力而为的冻结策略：
        # - ResNeXt / SENet 这类 timm ResNet 家族暴露 conv1/layer1/layer2/layer3
        # - CSPNet 这类模型暴露 stem + stages[0/1/2/...]
        # 之前仅尝试访问 "stages_0" 这类不存在的属性，导致 cspnet 在
        # freeze_layers=3 时几乎没有真正冻结，比较结果不公平。
        self._apply_freeze_layers(freeze_layers)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_attr_if_present(self, name: str) -> bool:
        if not hasattr(self.backbone, name):
            return False
        self._freeze_module(getattr(self.backbone, name))
        return True

    def _freeze_stage_prefix(self, count: int) -> None:
        stages = getattr(self.backbone, "stages", None)
        if stages is None:
            return
        if not isinstance(stages, (nn.Sequential, nn.ModuleList, list, tuple)):
            return
        for stage in list(stages)[:count]:
            self._freeze_module(stage)

    def _apply_freeze_layers(self, freeze_layers: int) -> None:
        if freeze_layers >= 1:
            for name in ("conv1", "bn1", "layer1", "stem"):
                self._freeze_attr_if_present(name)
            self._freeze_stage_prefix(1)
        if freeze_layers >= 2:
            self._freeze_attr_if_present("layer2")
            self._freeze_stage_prefix(2)
        if freeze_layers >= 3:
            self._freeze_attr_if_present("layer3")
            self._freeze_stage_prefix(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.proj(feat)


def build_encoder(
    backbone: str = "resnet18",
    use_pretrained: bool = True,
    freeze_layers: int = DEFAULT_FREEZE_LAYERS,
) -> nn.Module:
    """根据 backbone 类型构建 encoder。

    参数：
        backbone: "resnet18" 或 "resunet" 或 "resnext" 或 "senet" 或 "cspnet"
        use_pretrained: 是否使用 ImageNet 预训练权重
        freeze_layers: 冻结前 N 个 encoder stage

    返回：
        encoder 模型，输出 (B, 512) 特征向量
    """
    if backbone == "resnet18":
        return build_resnet18_encoder(
            use_pretrained=use_pretrained,
            freeze_layers=freeze_layers,
        )
    if backbone == "resunet":
        return ResUNetEncoder(
            use_pretrained=use_pretrained,
            freeze_layers=freeze_layers,
        )
    if backbone == "resnext":
        return GenericTimmEncoder(
            model_name="resnext50_32x4d",
            use_pretrained=use_pretrained,
            freeze_layers=freeze_layers,
        )
    if backbone == "senet":
        return GenericTimmEncoder(
            model_name="seresnet50",
            use_pretrained=use_pretrained,
            freeze_layers=freeze_layers,
        )
    if backbone == "cspnet":
        return GenericTimmEncoder(
            model_name="cspresnet50",
            use_pretrained=use_pretrained,
            freeze_layers=freeze_layers,
        )
    raise ValueError(f"Unsupported backbone: {backbone!r}. Expected 'resnet18', 'resunet', 'resnext', 'senet', or 'cspnet'.")


class ResidualPerViewMLP(nn.Module):
    """Residual self-MLP used as a matched-capacity per-view control."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, view_feature: torch.Tensor) -> torch.Tensor:
        return self.norm(view_feature + self.mlp(view_feature))


class ViewFeatureRecalibration(nn.Module):
    """Identity-initialized per-view channel gate for feature-fusion models."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, view_feature: torch.Tensor) -> torch.Tensor:
        # Start from an exact identity map, then learn to up/down-weight
        # each pooled view channel before multi-view concatenation.
        gate = 2.0 * torch.sigmoid(self.gate(self.norm(view_feature)))
        return view_feature * gate


class FeaturePairwiseTokenResidual(nn.Module):
    """Low-capacity pairwise correction for feature-fusion view tokens."""

    def __init__(
        self,
        feature_dim: int = 512,
        pair_dim: int = 64,
        dropout: float = 0.05,
        residual_scale: float = 0.0625,
    ) -> None:
        super().__init__()
        if pair_dim <= 0:
            raise ValueError("pair_dim must be > 0.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0.")

        self.residual_scale = float(residual_scale)
        self.pair_norm = nn.LayerNorm(feature_dim * 3)
        self.token_norm = nn.LayerNorm(feature_dim + pair_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, pair_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim + pair_dim, feature_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        _, num_views, _ = view_features.shape
        if num_views <= 1:
            return view_features

        relative_features = view_features - view_features.mean(dim=1, keepdim=True)
        pair_contexts: list[torch.Tensor] = []
        for view_index in range(num_views):
            anchor = relative_features[:, view_index]
            encoded_pairs = []
            for other_index in range(num_views):
                if other_index == view_index:
                    continue
                other = relative_features[:, other_index]
                pair_input = torch.cat(
                    [
                        anchor,
                        anchor - other,
                        anchor * other,
                    ],
                    dim=-1,
                )
                encoded_pairs.append(self.pair_encoder(self.pair_norm(pair_input)))
            pair_contexts.append(torch.stack(encoded_pairs, dim=1).mean(dim=1))

        pair_context = torch.stack(pair_contexts, dim=1)
        residual_input = torch.cat([view_features, pair_context], dim=-1)
        residual = self.adapter(self.token_norm(residual_input))
        return view_features + self.residual_scale * residual


class FeatureViewTokenScalarGate(nn.Module):
    """Identity-initialized sample-wise scalar gate for mixed view tokens."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        scale = 2.0 * torch.sigmoid(self.gate(self.norm(view_features)))
        return view_features * scale


class FeatureViewTokenChannelGate(nn.Module):
    """Identity-initialized channel gate for mixed feature-fusion view tokens."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        scale = 2.0 * torch.sigmoid(self.gate(self.norm(view_features)))
        return view_features * scale


class FeatureCrossViewIdentitySkipGate(nn.Module):
    """Bounded residual gate from mixed view tokens back to their pre-mixer identities."""

    def __init__(
        self,
        feature_dim: int = 512,
        max_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if max_scale <= 0.0:
            raise ValueError("max_scale must be > 0.")
        self.max_scale = float(max_scale)
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        mixed_features: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        if mixed_features.ndim != 3 or skip_features.ndim != 3:
            raise ValueError("mixed_features and skip_features must have shape (batch, views, features).")
        if mixed_features.shape != skip_features.shape:
            raise ValueError("mixed_features and skip_features must have identical shapes.")
        skip_delta = skip_features - mixed_features
        gate = self.max_scale * torch.tanh(self.gate(self.norm(skip_delta)))
        return mixed_features + gate * skip_delta


class SharedLowRankReliabilityCalibrator(nn.Module):
    """Shared low-rank residual calibrator for per-view reliability logits."""

    def __init__(
        self,
        feature_dim: int = 512,
        bottleneck_dim: int = 32,
        scale_limit: float = 0.25,
        bias_limit: float = 0.15,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim * 2),
            nn.GLU(dim=1),
            nn.Linear(bottleneck_dim, 2),
        )
        self.scale_limit = float(scale_limit)
        self.bias_limit = float(bias_limit)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, view_feature: torch.Tensor, raw_logit: torch.Tensor) -> torch.Tensor:
        scale_delta, bias_delta = self.adapter(self.norm(view_feature)).chunk(2, dim=1)
        scale = 1.0 + self.scale_limit * torch.tanh(scale_delta)
        bias = self.bias_limit * torch.tanh(bias_delta)
        return raw_logit * scale + bias


class PerViewLogitTemperatureCalibrator(nn.Module):
    """Global per-view temperature scaling on classifier logits."""

    def __init__(
        self,
        num_views: int = 3,
        min_temperature: float = 0.5,
        max_temperature: float = 2.0,
    ) -> None:
        super().__init__()
        if min_temperature <= 0.0 or max_temperature <= 0.0:
            raise ValueError("Per-view temperatures must be > 0.")
        if min_temperature > max_temperature:
            raise ValueError("min_temperature must be <= max_temperature.")
        self.log_temperature = nn.Parameter(torch.zeros(num_views))
        self.min_log_temperature = float(math.log(min_temperature))
        self.max_log_temperature = float(math.log(max_temperature))

    def temperatures(self) -> torch.Tensor:
        return torch.exp(
            self.log_temperature.clamp(
                min=self.min_log_temperature,
                max=self.max_log_temperature,
            )
        )

    def forward(self, view_logits: torch.Tensor) -> torch.Tensor:
        temperatures = self.temperatures().to(
            device=view_logits.device,
            dtype=view_logits.dtype,
        ).view(1, -1, 1)
        centered_logits = view_logits - view_logits.mean(dim=-1, keepdim=True)
        return centered_logits / temperatures + view_logits.mean(dim=-1, keepdim=True)


class EvidenceAwareReliabilityGate(nn.Module):
    """Residual reliability-logit correction from detached per-view evidence."""

    def __init__(
        self,
        feature_dim: int = 512,
        evidence_dim: int = 8,
        bottleneck_dim: int = 64,
        residual_limit: float = 0.25,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.evidence_norm = nn.LayerNorm(evidence_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim + evidence_dim, bottleneck_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.residual_limit = float(residual_limit)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    @staticmethod
    def _evidence_from_logits(view_logits: torch.Tensor) -> torch.Tensor:
        detached_logits = view_logits.detach()
        probs = torch.softmax(detached_logits, dim=-1).clamp_min(1e-8)
        max_prob = probs.max(dim=-1).values
        margin = detached_logits.max(dim=-1).values - detached_logits.min(dim=-1).values
        entropy = -(probs * probs.log()).sum(dim=-1) / math.log(float(probs.shape[-1]))
        if probs.shape[-1] > 1:
            abnormal_prob = probs[..., 1]
        else:
            abnormal_prob = max_prob

        centered_max_prob = max_prob - max_prob.mean(dim=1, keepdim=True)
        centered_margin = margin - margin.mean(dim=1, keepdim=True)
        centered_entropy = entropy - entropy.mean(dim=1, keepdim=True)
        centered_abnormal_prob = abnormal_prob - abnormal_prob.mean(dim=1, keepdim=True)
        return torch.stack(
            [
                max_prob,
                margin,
                entropy,
                abnormal_prob,
                centered_max_prob,
                centered_margin,
                centered_entropy,
                centered_abnormal_prob,
            ],
            dim=-1,
        )

    def forward(
        self,
        view_features: torch.Tensor,
        view_logits: torch.Tensor,
    ) -> torch.Tensor:
        evidence = self._evidence_from_logits(view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        features = self.feature_norm(view_features)
        evidence = self.evidence_norm(evidence)
        residual_input = torch.cat([features, evidence], dim=-1)
        return self.residual_limit * torch.tanh(self.adapter(residual_input))


class CandidateViewReliabilityGate(nn.Module):
    """Candidate-limited reliability residual for evidence-supported non-axial views."""

    def __init__(
        self,
        feature_dim: int = 512,
        evidence_dim: int = 8,
        bottleneck_dim: int = 32,
        residual_limit: float = 0.35,
        margin_gap: float = 0.5,
        margin_window: float = 0.25,
        require_class_consensus: bool = False,
        consensus_margin: float = 0.0,
        consensus_window: float = 0.0,
        prob_advantage: float = 0.0,
        prob_window: float = 0.0,
        prototype_advantage: float = 0.0,
        prototype_window: float = 0.0,
        require_nonaxial_pair_consensus: bool = False,
        nonaxial_pair_consensus_margin: float = 0.2,
        nonaxial_pair_consensus_window: float = 0.2,
        detach_features: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")
        if margin_gap < 0.0:
            raise ValueError("margin_gap must be >= 0.")
        if margin_window < 0.0:
            raise ValueError("margin_window must be >= 0.")
        if consensus_margin < 0.0:
            raise ValueError("consensus_margin must be >= 0.")
        if consensus_window < 0.0:
            raise ValueError("consensus_window must be >= 0.")
        if prob_advantage < 0.0:
            raise ValueError("prob_advantage must be >= 0.")
        if prob_window < 0.0:
            raise ValueError("prob_window must be >= 0.")
        if prototype_advantage < 0.0:
            raise ValueError("prototype_advantage must be >= 0.")
        if prototype_window < 0.0:
            raise ValueError("prototype_window must be >= 0.")
        if nonaxial_pair_consensus_margin < 0.0:
            raise ValueError("nonaxial_pair_consensus_margin must be >= 0.")
        if nonaxial_pair_consensus_window < 0.0:
            raise ValueError("nonaxial_pair_consensus_window must be >= 0.")
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.evidence_norm = nn.LayerNorm(evidence_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim + evidence_dim, bottleneck_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.residual_limit = float(residual_limit)
        self.margin_gap = float(margin_gap)
        self.margin_window = float(margin_window)
        self.require_class_consensus = bool(require_class_consensus)
        self.consensus_margin = float(consensus_margin)
        self.consensus_window = float(consensus_window)
        self.prob_advantage = float(prob_advantage)
        self.prob_window = float(prob_window)
        self.prototype_advantage = float(prototype_advantage)
        self.prototype_window = float(prototype_window)
        self.require_nonaxial_pair_consensus = bool(require_nonaxial_pair_consensus)
        self.nonaxial_pair_consensus_margin = float(nonaxial_pair_consensus_margin)
        self.nonaxial_pair_consensus_window = float(nonaxial_pair_consensus_window)
        self.detach_features = bool(detach_features)
        self.class_prototypes = nn.Parameter(torch.empty(2, feature_dim))
        nn.init.normal_(self.class_prototypes, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def _candidate_scale(self, view_logits: torch.Tensor) -> torch.Tensor:
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        detached_logits = view_logits.detach()
        pred_margins = detached_logits.max(dim=-1).values - detached_logits.min(dim=-1).values
        axial_margin = pred_margins[:, :1]
        candidate_delta = pred_margins - axial_margin - self.margin_gap
        if self.margin_window > 0.0:
            candidate_scale = (candidate_delta / self.margin_window).clamp(0.0, 1.0)
        else:
            candidate_scale = candidate_delta.ge(0.0).to(dtype=view_logits.dtype)
        candidate_scale = candidate_scale.clone()
        candidate_scale[:, 0] = 0.0
        return candidate_scale

    def _class_consensus_scale(self, view_logits: torch.Tensor) -> torch.Tensor:
        if not self.require_class_consensus:
            return torch.ones(
                view_logits.shape[:2],
                device=view_logits.device,
                dtype=view_logits.dtype,
            )
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        detached_logits = view_logits.detach()
        if detached_logits.shape[-1] > 1:
            signed_margins = detached_logits[..., 1] - detached_logits[..., 0]
        else:
            signed_margins = detached_logits.squeeze(-1)
        candidate_direction = signed_margins.sign()
        support_values = []
        for view_index in range(3):
            other_support = [
                candidate_direction[:, view_index] * signed_margins[:, other_index]
                for other_index in range(3)
                if other_index != view_index
            ]
            support_values.append(torch.stack(other_support, dim=1).amax(dim=1))
        consensus_support = torch.stack(support_values, dim=1)
        consensus_delta = consensus_support - self.consensus_margin
        if self.consensus_window > 0.0:
            consensus_scale = (consensus_delta / self.consensus_window).clamp(0.0, 1.0)
        else:
            consensus_scale = consensus_delta.ge(0.0).to(dtype=view_logits.dtype)
        consensus_scale = consensus_scale * candidate_direction.ne(0.0).to(
            dtype=view_logits.dtype
        )
        consensus_scale = consensus_scale.clone()
        consensus_scale[:, 0] = 0.0
        return consensus_scale

    def _prob_advantage_scale(self, view_logits: torch.Tensor) -> torch.Tensor:
        if self.prob_advantage <= 0.0 and self.prob_window <= 0.0:
            return torch.ones(
                view_logits.shape[:2],
                device=view_logits.device,
                dtype=view_logits.dtype,
            )
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        detached_logits = view_logits.detach()
        if detached_logits.shape[-1] < 2:
            return torch.ones(
                view_logits.shape[:2],
                device=view_logits.device,
                dtype=view_logits.dtype,
            )

        probs = torch.softmax(detached_logits, dim=-1)
        abnormal_probs = probs[..., 1]
        normal_probs = probs[..., 0]
        signed_margins = detached_logits[..., 1] - detached_logits[..., 0]
        candidate_is_abnormal = signed_margins.ge(0.0)
        candidate_class_probs = torch.where(
            candidate_is_abnormal,
            abnormal_probs,
            normal_probs,
        )
        axial_same_class_probs = torch.where(
            candidate_is_abnormal,
            abnormal_probs[:, :1].expand_as(abnormal_probs),
            normal_probs[:, :1].expand_as(normal_probs),
        )
        prob_delta = candidate_class_probs - axial_same_class_probs - self.prob_advantage
        if self.prob_window > 0.0:
            prob_scale = (prob_delta / self.prob_window).clamp(0.0, 1.0)
        else:
            prob_scale = prob_delta.ge(0.0).to(dtype=view_logits.dtype)
        prob_scale = prob_scale * signed_margins.ne(0.0).to(dtype=view_logits.dtype)
        prob_scale = prob_scale.clone()
        prob_scale[:, 0] = 0.0
        return prob_scale

    def _prototype_alignment_scale(
        self,
        view_features: torch.Tensor,
        view_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.prototype_advantage <= 0.0 and self.prototype_window <= 0.0:
            return torch.ones(
                view_logits.shape[:2],
                device=view_logits.device,
                dtype=view_logits.dtype,
            )
        if view_features.ndim != 3 or view_logits.ndim != 3:
            raise ValueError("view_features and view_logits must have shape (batch, views, ...).")
        if view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        if view_logits.shape[-1] != self.class_prototypes.shape[0]:
            raise ValueError(
                "prototype-aligned candidate gate currently expects binary class logits."
            )

        normalized_features = nn.functional.normalize(view_features, dim=-1)
        normalized_prototypes = nn.functional.normalize(
            self.class_prototypes.to(
                device=view_features.device,
                dtype=view_features.dtype,
            ),
            dim=-1,
        )
        class_scores = torch.einsum(
            "bvf,cf->bvc",
            normalized_features,
            normalized_prototypes,
        )
        predicted_classes = view_logits.detach().argmax(dim=-1, keepdim=True)
        candidate_scores = class_scores.gather(dim=-1, index=predicted_classes).squeeze(-1)
        axial_same_class_scores = class_scores[:, :1, :].expand_as(class_scores)
        axial_same_class_scores = axial_same_class_scores.gather(
            dim=-1,
            index=predicted_classes,
        ).squeeze(-1)
        prototype_delta = (
            candidate_scores
            - axial_same_class_scores
            - self.prototype_advantage
        )
        if self.prototype_window > 0.0:
            prototype_scale = (prototype_delta / self.prototype_window).clamp(0.0, 1.0)
        else:
            prototype_scale = prototype_delta.ge(0.0).to(dtype=view_logits.dtype)
        prototype_scale = prototype_scale.clone()
        prototype_scale[:, 0] = 0.0
        return prototype_scale

    def _nonaxial_pair_consensus_scale(self, view_logits: torch.Tensor) -> torch.Tensor:
        if not self.require_nonaxial_pair_consensus:
            return torch.ones(
                view_logits.shape[:2],
                device=view_logits.device,
                dtype=view_logits.dtype,
            )
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        detached_logits = view_logits.detach()
        if detached_logits.shape[-1] > 1:
            signed_margins = detached_logits[..., 1] - detached_logits[..., 0]
        else:
            signed_margins = detached_logits.squeeze(-1)

        candidate_direction = signed_margins.sign()
        pair_support = torch.zeros_like(signed_margins)
        pair_support[:, 1] = candidate_direction[:, 1] * signed_margins[:, 2]
        pair_support[:, 2] = candidate_direction[:, 2] * signed_margins[:, 1]
        pair_delta = pair_support - self.nonaxial_pair_consensus_margin
        if self.nonaxial_pair_consensus_window > 0.0:
            pair_scale = (pair_delta / self.nonaxial_pair_consensus_window).clamp(0.0, 1.0)
        else:
            pair_scale = pair_delta.ge(0.0).to(dtype=view_logits.dtype)
        pair_scale = pair_scale * candidate_direction.ne(0.0).to(dtype=view_logits.dtype)
        pair_scale = pair_scale.clone()
        pair_scale[:, 0] = 0.0
        return pair_scale

    def prototype_logits(
        self,
        view_features: torch.Tensor,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        normalized_features = nn.functional.normalize(view_features, dim=-1)
        normalized_prototypes = nn.functional.normalize(
            self.class_prototypes.to(
                device=view_features.device,
                dtype=view_features.dtype,
            ),
            dim=-1,
        )
        class_scores = torch.einsum(
            "bvf,cf->bvc",
            normalized_features,
            normalized_prototypes,
        )
        return class_scores / float(temperature)

    def forward(
        self,
        view_features: torch.Tensor,
        view_logits: torch.Tensor,
    ) -> torch.Tensor:
        evidence = EvidenceAwareReliabilityGate._evidence_from_logits(view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        if self.detach_features:
            view_features = view_features.detach()
        features = self.feature_norm(view_features)
        evidence = self.evidence_norm(evidence)
        residual_input = torch.cat([features, evidence], dim=-1)
        residual = self.residual_limit * torch.tanh(self.adapter(residual_input))
        candidate_scale = self._candidate_scale(view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        consensus_scale = self._class_consensus_scale(view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        prob_advantage_scale = self._prob_advantage_scale(view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        prototype_scale = self._prototype_alignment_scale(view_features, view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        nonaxial_pair_consensus_scale = self._nonaxial_pair_consensus_scale(view_logits).to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        candidate_scale = (
            candidate_scale
            * consensus_scale
            * prob_advantage_scale
            * prototype_scale
            * nonaxial_pair_consensus_scale
        )
        return residual * candidate_scale.unsqueeze(-1)


class GateLogitRMSLimiter(nn.Module):
    """Bound decision-gate logit concentration while preserving view ranking."""

    def __init__(self, max_centered_rms: float = 1.3, eps: float = 1e-6) -> None:
        super().__init__()
        if max_centered_rms <= 0.0:
            raise ValueError("max_centered_rms must be > 0.")
        self.max_centered_rms = float(max_centered_rms)
        self.eps = float(eps)

    def forward(self, confidences: torch.Tensor) -> torch.Tensor:
        if confidences.ndim != 3 or confidences.shape[-1] != 1:
            raise ValueError("confidences must have shape (batch, views, 1).")
        mean_confidence = confidences.mean(dim=1, keepdim=True)
        centered = confidences - mean_confidence
        centered_rms = centered.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        max_rms = centered_rms.new_tensor(self.max_centered_rms)
        scale = (max_rms / centered_rms).clamp(max=1.0)
        return mean_confidence + centered * scale


class SagittalNormalRescueGate(nn.Module):
    """View-index constrained sagittal reliability boost for axial-FP rescue cases."""

    def __init__(
        self,
        residual_limit: float = 0.5,
        axial_abnormal_threshold: float = 0.65,
        sagittal_normal_threshold: float = 0.65,
        fused_abnormal_threshold: float = 0.55,
        window: float = 0.15,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")
        for name, value in (
            ("axial_abnormal_threshold", axial_abnormal_threshold),
            ("sagittal_normal_threshold", sagittal_normal_threshold),
            ("fused_abnormal_threshold", fused_abnormal_threshold),
            ("window", window),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        self.residual_limit = float(residual_limit)
        self.axial_abnormal_threshold = float(axial_abnormal_threshold)
        self.sagittal_normal_threshold = float(sagittal_normal_threshold)
        self.fused_abnormal_threshold = float(fused_abnormal_threshold)
        self.window = float(window)

    def _trigger_scale(self, values: torch.Tensor, threshold: float) -> torch.Tensor:
        if self.window > 0.0:
            return ((values - threshold) / self.window).clamp(0.0, 1.0)
        return values.ge(threshold).to(dtype=values.dtype)

    def forward(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor,
        fusion_temperature: float,
    ) -> torch.Tensor:
        if confidences.ndim != 3 or confidences.shape[1:] != (3, 1):
            raise ValueError("confidences must have shape (batch, 3, 1).")
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        if view_logits.shape[-1] != 2:
            raise ValueError("Sagittal normal rescue gate requires binary logits.")

        detached_logits = view_logits.detach()
        view_probs = torch.softmax(detached_logits, dim=-1)
        abnormal_probs = view_probs[..., 1]
        normal_probs = view_probs[..., 0]
        flat_confidences = confidences.detach().squeeze(-1)
        base_weights = torch.softmax(
            flat_confidences / max(float(fusion_temperature), 1e-6),
            dim=1,
        )
        fused_abnormal_prob = (base_weights * abnormal_probs).sum(dim=1)
        dominant_view = flat_confidences.argmax(dim=1)

        trigger = (
            self._trigger_scale(abnormal_probs[:, 0], self.axial_abnormal_threshold)
            * self._trigger_scale(normal_probs[:, 2], self.sagittal_normal_threshold)
            * self._trigger_scale(fused_abnormal_prob, self.fused_abnormal_threshold)
            * dominant_view.eq(0).to(dtype=flat_confidences.dtype)
        )
        residual = torch.zeros_like(confidences)
        residual[:, 2, 0] = self.residual_limit * trigger.to(dtype=residual.dtype)
        return residual


class PosthocFPRiskSagittalGate(nn.Module):
    """DFR-112 eval-side sagittal boost for high-precision axial-FP risk cases."""

    def __init__(
        self,
        residual: float = 5.0,
        axial_abnormal_min: float = 0.4,
        axial_abnormal_max: float = 0.88,
        sagittal_normal_min: float = 0.55,
        coronal_abnormal_max: float = 0.525,
        fused_abnormal_min: float = 0.8,
        confidence_gap_max: float = 5.0,
    ) -> None:
        super().__init__()
        if residual <= 0.0:
            raise ValueError("residual must be > 0.")
        for name, value in (
            ("axial_abnormal_min", axial_abnormal_min),
            ("axial_abnormal_max", axial_abnormal_max),
            ("sagittal_normal_min", sagittal_normal_min),
            ("coronal_abnormal_max", coronal_abnormal_max),
            ("fused_abnormal_min", fused_abnormal_min),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if axial_abnormal_min > axial_abnormal_max:
            raise ValueError("axial_abnormal_min must be <= axial_abnormal_max.")
        if confidence_gap_max <= 0.0:
            raise ValueError("confidence_gap_max must be > 0.")
        self.residual = float(residual)
        self.axial_abnormal_min = float(axial_abnormal_min)
        self.axial_abnormal_max = float(axial_abnormal_max)
        self.sagittal_normal_min = float(sagittal_normal_min)
        self.coronal_abnormal_max = float(coronal_abnormal_max)
        self.fused_abnormal_min = float(fused_abnormal_min)
        self.confidence_gap_max = float(confidence_gap_max)

    def forward(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor,
        fusion_temperature: float,
    ) -> torch.Tensor:
        if confidences.ndim != 3 or confidences.shape[1:] != (3, 1):
            raise ValueError("confidences must have shape (batch, 3, 1).")
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        if view_logits.shape[-1] != 2:
            raise ValueError("PosthocFPRiskSagittalGate requires binary logits.")

        detached_logits = view_logits.detach()
        detached_confidences = confidences.detach().squeeze(-1)
        scaled_confidences = detached_confidences / max(float(fusion_temperature), 1e-6)
        base_weights = torch.softmax(scaled_confidences, dim=1)
        fused_logits = (detached_logits * base_weights.unsqueeze(-1)).sum(dim=1)
        fused_abnormal = torch.softmax(fused_logits, dim=-1)[:, 1]
        view_probs = torch.softmax(detached_logits, dim=-1)
        abnormal_probs = view_probs[..., 1]
        normal_probs = view_probs[..., 0]
        confidence_gap = scaled_confidences[:, 0] - scaled_confidences[:, 2]

        trigger = (
            scaled_confidences.argmax(dim=1).eq(0)
            & fused_abnormal.ge(self.fused_abnormal_min)
            & abnormal_probs[:, 0].ge(self.axial_abnormal_min)
            & abnormal_probs[:, 0].le(self.axial_abnormal_max)
            & normal_probs[:, 2].ge(self.sagittal_normal_min)
            & abnormal_probs[:, 1].le(self.coronal_abnormal_max)
            & confidence_gap.le(self.confidence_gap_max)
        )
        residual = torch.zeros_like(confidences)
        residual[:, 2, 0] = self.residual * trigger.to(dtype=residual.dtype)
        return residual


class PosthocFNAbnormalRescueGate(nn.Module):
    """DFR-115 eval-side axial boost for high-precision positive-FN rescue cases."""

    def __init__(
        self,
        residual: float = 6.0,
        fused_abnormal_max: float = 0.2,
        axial_abnormal_min: float = 0.45,
        max_abnormal_min: float = 0.45,
        second_abnormal_min: float = 0.0,
        confidence_gap_max: float = 1.0,
        axial_abnormal_max: float = 0.65,
    ) -> None:
        super().__init__()
        if residual <= 0.0:
            raise ValueError("residual must be > 0.")
        for name, value in (
            ("fused_abnormal_max", fused_abnormal_max),
            ("axial_abnormal_min", axial_abnormal_min),
            ("max_abnormal_min", max_abnormal_min),
            ("second_abnormal_min", second_abnormal_min),
            ("axial_abnormal_max", axial_abnormal_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if axial_abnormal_min > axial_abnormal_max:
            raise ValueError("axial_abnormal_min must be <= axial_abnormal_max.")
        if confidence_gap_max <= 0.0:
            raise ValueError("confidence_gap_max must be > 0.")
        self.residual = float(residual)
        self.fused_abnormal_max = float(fused_abnormal_max)
        self.axial_abnormal_min = float(axial_abnormal_min)
        self.max_abnormal_min = float(max_abnormal_min)
        self.second_abnormal_min = float(second_abnormal_min)
        self.confidence_gap_max = float(confidence_gap_max)
        self.axial_abnormal_max = float(axial_abnormal_max)

    def forward(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor,
        fusion_temperature: float,
    ) -> torch.Tensor:
        if confidences.ndim != 3 or confidences.shape[1:] != (3, 1):
            raise ValueError("confidences must have shape (batch, 3, 1).")
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        if view_logits.shape[-1] != 2:
            raise ValueError("PosthocFNAbnormalRescueGate requires binary logits.")

        detached_logits = view_logits.detach()
        detached_confidences = confidences.detach().squeeze(-1)
        scaled_confidences = detached_confidences / max(float(fusion_temperature), 1e-6)
        base_weights = torch.softmax(scaled_confidences, dim=1)
        fused_logits = (detached_logits * base_weights.unsqueeze(-1)).sum(dim=1)
        fused_abnormal = torch.softmax(fused_logits, dim=-1)[:, 1]
        abnormal_probs = torch.softmax(detached_logits, dim=-1)[..., 1]
        max_abnormal = abnormal_probs.max(dim=1).values
        second_abnormal = abnormal_probs.topk(k=2, dim=1).values[:, 1]
        axial_confidence = scaled_confidences[:, 0]
        top_other_confidence = scaled_confidences[:, 1:].max(dim=1).values
        confidence_gap = top_other_confidence - axial_confidence

        trigger = (
            fused_abnormal.le(self.fused_abnormal_max)
            & abnormal_probs[:, 0].ge(self.axial_abnormal_min)
            & abnormal_probs[:, 0].le(self.axial_abnormal_max)
            & max_abnormal.ge(self.max_abnormal_min)
            & second_abnormal.ge(self.second_abnormal_min)
            & confidence_gap.le(self.confidence_gap_max)
        )
        residual = torch.zeros_like(confidences)
        residual[:, 0, 0] = self.residual * trigger.to(dtype=residual.dtype)
        return residual


class SagittalReliabilityCalibrator(nn.Module):
    """Learn low-capacity sagittal reliability residuals from detached evidence."""

    def __init__(
        self,
        hidden_dim: int = 32,
        residual_limit: float = 1.0,
        dropout: float = 0.05,
        coronal_suppression: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if not 0.0 <= coronal_suppression <= 1.0:
            raise ValueError("coronal_suppression must be in [0, 1].")

        self.residual_limit = float(residual_limit)
        self.coronal_suppression = float(coronal_suppression)
        input_dim = 28
        self.norm = nn.LayerNorm(input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor,
        fusion_temperature: float,
    ) -> torch.Tensor:
        if confidences.ndim != 3 or confidences.shape[1:] != (3, 1):
            raise ValueError("confidences must have shape (batch, 3, 1).")
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        if view_logits.shape[-1] != 2:
            raise ValueError("SagittalReliabilityCalibrator requires binary logits.")

        detached_confidences = confidences.detach().squeeze(-1)
        detached_logits = view_logits.detach()
        base_weights = torch.softmax(
            detached_confidences / max(float(fusion_temperature), 1e-6),
            dim=1,
        )
        view_probs = torch.softmax(detached_logits, dim=-1)
        abnormal_probs = view_probs[..., 1]
        view_margins = detached_logits[..., 1] - detached_logits[..., 0]
        centered_confidences = detached_confidences - detached_confidences.mean(
            dim=1,
            keepdim=True,
        )
        centered_abnormal = abnormal_probs - abnormal_probs.mean(dim=1, keepdim=True)
        centered_margins = view_margins - view_margins.mean(dim=1, keepdim=True)
        weighted_abnormal = base_weights * abnormal_probs
        fused_abnormal = weighted_abnormal.sum(dim=1, keepdim=True)
        fused_margin = (base_weights * view_margins).sum(dim=1, keepdim=True)
        safe_weights = base_weights.clamp_min(1e-8)
        weight_entropy = -(safe_weights * safe_weights.log()).sum(dim=1, keepdim=True)
        weight_entropy = weight_entropy / math.log(3.0)
        sagittal_context = torch.cat(
            [
                abnormal_probs[:, 2:3] - abnormal_probs[:, 0:1],
                view_margins[:, 2:3] - view_margins[:, 0:1],
                base_weights[:, 2:3] - base_weights[:, 0:1],
                centered_confidences[:, 2:3],
            ],
            dim=1,
        )
        evidence_features = torch.cat(
            [
                detached_confidences,
                base_weights,
                abnormal_probs,
                centered_abnormal,
                view_margins,
                centered_margins,
                weighted_abnormal,
                fused_abnormal,
                fused_margin,
                weight_entropy,
                sagittal_context,
            ],
            dim=1,
        )
        sagittal_residual = self.residual_limit * torch.tanh(
            self.adapter(self.norm(evidence_features))
        )
        residual = torch.zeros_like(confidences)
        residual[:, 2, 0] = sagittal_residual.squeeze(-1)
        if self.coronal_suppression > 0.0:
            positive_sagittal_residual = sagittal_residual.clamp_min(0.0).squeeze(-1)
            residual[:, 1, 0] = -self.coronal_suppression * positive_sagittal_residual
        return residual


class GateViewPriorDebiaser(nn.Module):
    """Remove persistent per-view reliability-logit offsets before softmax."""

    def __init__(
        self,
        strength: float = 1.0,
        momentum: float = 0.1,
        max_strength: float | None = None,
        evidence_close_gap: float = 0.0,
        evidence_close_window: float = 0.0,
        positive_safe: bool = False,
        positive_safe_abnormal_gap: float = 0.0,
        positive_safe_fused_floor: float = 0.6,
        positive_safe_window: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= strength <= 1.0):
            raise ValueError("strength must be in [0, 1].")
        if not (0.0 < momentum <= 1.0):
            raise ValueError("momentum must be in (0, 1].")
        if max_strength is None:
            max_strength = strength
        if not (0.0 <= max_strength <= 1.0):
            raise ValueError("max_strength must be in [0, 1].")
        if max_strength < strength:
            raise ValueError("max_strength must be >= strength.")
        if evidence_close_gap < 0.0:
            raise ValueError("evidence_close_gap must be >= 0.")
        if evidence_close_window < 0.0:
            raise ValueError("evidence_close_window must be >= 0.")
        if positive_safe_abnormal_gap < 0.0:
            raise ValueError("positive_safe_abnormal_gap must be >= 0.")
        if not (0.0 <= positive_safe_fused_floor <= 1.0):
            raise ValueError("positive_safe_fused_floor must be in [0, 1].")
        if positive_safe_window < 0.0:
            raise ValueError("positive_safe_window must be >= 0.")
        self.strength = float(strength)
        self.max_strength = float(max_strength)
        self.momentum = float(momentum)
        self.evidence_close_gap = float(evidence_close_gap)
        self.evidence_close_window = float(evidence_close_window)
        self.positive_safe = bool(positive_safe)
        self.positive_safe_abnormal_gap = float(positive_safe_abnormal_gap)
        self.positive_safe_fused_floor = float(positive_safe_fused_floor)
        self.positive_safe_window = float(positive_safe_window)
        self.register_buffer("running_centered_prior", torch.zeros(1, 3, 1))

    def _positive_safe_scale(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        if view_logits.shape[-1] < 2:
            return confidences.new_ones(confidences.shape[0], 1, 1)

        detached_logits = view_logits.detach()
        abnormal_probs = torch.softmax(detached_logits, dim=-1)[..., 1]
        axial_abnormal = abnormal_probs[:, :1]
        best_non_axial_abnormal = abnormal_probs[:, 1:].amax(dim=1, keepdim=True)
        abnormal_delta = (
            best_non_axial_abnormal
            - axial_abnormal
            + self.positive_safe_abnormal_gap
        )
        if self.positive_safe_window > 0.0:
            abnormal_safe = (
                abnormal_delta / self.positive_safe_window
            ).clamp(min=0.0, max=1.0)
        else:
            abnormal_safe = (abnormal_delta >= 0.0).to(dtype=confidences.dtype)

        max_debiased_confidences = confidences - self.max_strength * prior.to(
            device=confidences.device,
            dtype=confidences.dtype,
        )
        max_debiased_weights = torch.softmax(
            max_debiased_confidences.squeeze(-1),
            dim=1,
        )
        max_debiased_abnormal = (max_debiased_weights * abnormal_probs).sum(
            dim=1,
            keepdim=True,
        )
        floor_delta = max_debiased_abnormal - self.positive_safe_fused_floor
        if self.positive_safe_window > 0.0:
            fused_safe = (
                floor_delta / self.positive_safe_window
            ).clamp(min=0.0, max=1.0)
        else:
            fused_safe = (floor_delta >= 0.0).to(dtype=confidences.dtype)

        safe_scale = torch.maximum(abnormal_safe, fused_safe)
        return safe_scale.to(
            device=confidences.device,
            dtype=confidences.dtype,
        ).unsqueeze(-1)

    def _sample_strength(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor | None,
        prior: torch.Tensor,
    ) -> float | torch.Tensor:
        if self.max_strength <= self.strength or view_logits is None:
            return self.strength
        if view_logits.ndim != 3 or view_logits.shape[1] != 3:
            raise ValueError("view_logits must have shape (batch, 3, classes).")
        detached_logits = view_logits.detach()
        pred_margins = detached_logits.max(dim=-1).values - detached_logits.min(dim=-1).values
        axial_margin = pred_margins[:, :1]
        best_non_axial_margin = pred_margins[:, 1:].amax(dim=1, keepdim=True)
        close_delta = best_non_axial_margin - axial_margin + self.evidence_close_gap
        if self.evidence_close_window > 0.0:
            scale = (close_delta / self.evidence_close_window).clamp(min=0.0, max=1.0)
        else:
            scale = (close_delta >= 0.0).to(dtype=confidences.dtype)
        scale = scale.to(device=confidences.device, dtype=confidences.dtype)
        if self.positive_safe:
            scale = scale * self._positive_safe_scale(
                confidences,
                view_logits,
                prior,
            ).squeeze(-1)
        strength = self.strength + (self.max_strength - self.strength) * scale
        return strength.unsqueeze(-1)

    def forward(
        self,
        confidences: torch.Tensor,
        view_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if confidences.ndim != 3 or confidences.shape[-1] != 1:
            raise ValueError("confidences must have shape (batch, views, 1).")
        if confidences.shape[1] != self.running_centered_prior.shape[1]:
            raise ValueError("GateViewPriorDebiaser is configured for exactly 3 views.")

        if self.training:
            view_prior = confidences.detach().mean(dim=0, keepdim=True)
            centered_prior = view_prior - view_prior.mean(dim=1, keepdim=True)
            self.running_centered_prior.lerp_(
                centered_prior.to(
                    device=self.running_centered_prior.device,
                    dtype=self.running_centered_prior.dtype,
                ),
                self.momentum,
            )
            prior = centered_prior
        else:
            prior = self.running_centered_prior.to(
                device=confidences.device,
                dtype=confidences.dtype,
            )
        strength = self._sample_strength(confidences, view_logits, prior)
        return confidences - strength * prior


class RelativeViewReliabilityGate(nn.Module):
    """Residual reliability-logit correction from sample-wise relative view features."""

    def __init__(
        self,
        feature_dim: int = 512,
        bottleneck_dim: int = 32,
        residual_limit: float = 0.75,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")
        self.norm = nn.LayerNorm(feature_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.residual_limit = float(residual_limit)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        relative_features = view_features - view_features.mean(dim=1, keepdim=True)
        residual = self.adapter(self.norm(relative_features))
        return self.residual_limit * torch.tanh(residual)


class GateTransformerContextualizer(nn.Module):
    """Gate-only Transformer context over the three view tokens.

    The residual projection starts at zero, so enabling the module initially
    preserves the DFR-25 gate path and only learns a small cross-view correction
    for reliability scoring.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        attention_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.05,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0.")

        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, attention_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=num_heads,
            dim_feedforward=attention_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_proj = nn.Linear(attention_dim, feature_dim)
        self.residual_scale = float(residual_scale)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        relative_features = view_features - view_features.mean(dim=1, keepdim=True)
        tokens = self.input_proj(self.input_norm(relative_features))
        contextual_tokens = self.encoder(tokens)
        residual = self.output_proj(contextual_tokens)
        return view_features + self.residual_scale * residual


class PairwiseReliabilityGate(nn.Module):
    """Low-capacity pairwise view-context correction for reliability logits.

    This module keeps the late decision-fusion semantics intact.  It only adds a
    zero-initialized residual to each per-view reliability logit after comparing
    that view token with the other view tokens in the same sample.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pair_dim: int = 64,
        bottleneck_dim: int = 64,
        residual_limit: float = 0.25,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if pair_dim <= 0:
            raise ValueError("pair_dim must be > 0.")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be > 0.")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.pair_norm = nn.LayerNorm(feature_dim * 3)
        self.pair_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, pair_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim + pair_dim, bottleneck_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.residual_limit = float(residual_limit)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        batch_size, num_views, _ = view_features.shape
        if num_views <= 1:
            return view_features.new_zeros(batch_size, num_views, 1)

        relative_features = view_features - view_features.mean(dim=1, keepdim=True)
        pair_contexts = []
        for view_index in range(num_views):
            encoded_pairs = []
            anchor = relative_features[:, view_index]
            for other_index in range(num_views):
                if other_index == view_index:
                    continue
                other = relative_features[:, other_index]
                pair_input = torch.cat(
                    [
                        anchor,
                        anchor - other,
                        anchor * other,
                    ],
                    dim=-1,
                )
                encoded_pairs.append(self.pair_encoder(self.pair_norm(pair_input)))
            pair_contexts.append(torch.stack(encoded_pairs, dim=1).mean(dim=1))

        pair_context = torch.stack(pair_contexts, dim=1)
        features = self.feature_norm(view_features)
        residual_input = torch.cat([features, pair_context], dim=-1)
        residual = self.adapter(residual_input)
        return self.residual_limit * torch.tanh(residual)


class ViewRoleConfidenceScorer(nn.Module):
    """Shared confidence scorer from centered view features plus view-role tokens.

    The module follows the MVCNN/RotationNet lesson that view identity should be
    explicit, but it avoids feeding raw per-view offsets directly into the gate.
    Scores are bounded so the replacement gate cannot immediately hard-select a
    single anatomical plane.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_views: int = 3,
        role_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        logit_limit: float = 1.5,
    ) -> None:
        super().__init__()
        if num_views <= 0:
            raise ValueError("num_views must be > 0.")
        if role_dim <= 0:
            raise ValueError("role_dim must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if logit_limit <= 0.0:
            raise ValueError("logit_limit must be > 0.")

        self.num_views = int(num_views)
        self.logit_limit = float(logit_limit)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.role_embeddings = nn.Parameter(torch.empty(num_views, role_dim))
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim + role_dim),
            nn.Linear(feature_dim + role_dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.role_embeddings, mean=0.0, std=0.02)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape (batch, views, features).")
        batch_size, num_views, _ = view_features.shape
        if num_views > self.num_views:
            raise ValueError(
                f"ViewRoleConfidenceScorer configured for {self.num_views} views, got {num_views}."
            )

        relative_features = view_features - view_features.mean(dim=1, keepdim=True)
        normalized_features = self.feature_norm(relative_features)
        role_tokens = self.role_embeddings[:num_views].to(
            device=view_features.device,
            dtype=view_features.dtype,
        )
        role_tokens = role_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        score_input = torch.cat([normalized_features, role_tokens], dim=-1)
        raw_scores = self.scorer(score_input)
        return self.logit_limit * torch.tanh(raw_scores / self.logit_limit)


class EvidenceMarginResidualFusion(nn.Module):
    """Bounded residual on the binary fused margin from detached view evidence."""

    def __init__(
        self,
        num_views: int = 3,
        hidden_dim: int = 48,
        residual_limit: float = 1.5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_views <= 0:
            raise ValueError("num_views must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")

        self.num_views = int(num_views)
        self.residual_limit = float(residual_limit)
        input_dim = self.num_views * 6 + 7
        self.norm = nn.LayerNorm(input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        fused_logits: torch.Tensor,
        view_logits: torch.Tensor,
        fusion_weights: torch.Tensor,
    ) -> torch.Tensor:
        if fused_logits.shape[-1] != 2 or view_logits.shape[-1] != 2:
            return fused_logits
        if view_logits.ndim != 3 or fusion_weights.ndim != 3:
            raise ValueError("view_logits and fusion_weights must have shape (batch, views, ...).")
        if view_logits.shape[1] != self.num_views or fusion_weights.shape[1] != self.num_views:
            raise ValueError(
                f"EvidenceMarginResidualFusion configured for {self.num_views} views."
            )

        detached_view_logits = view_logits.detach()
        detached_weights = fusion_weights.detach().squeeze(-1)
        detached_fused_logits = fused_logits.detach()

        view_probs = torch.softmax(detached_view_logits, dim=-1)
        abnormal_probs = view_probs[..., 1]
        view_margins = detached_view_logits[..., 1] - detached_view_logits[..., 0]
        weighted_abnormal = detached_weights * abnormal_probs
        centered_abnormal = abnormal_probs - abnormal_probs.mean(dim=1, keepdim=True)
        centered_margins = view_margins - view_margins.mean(dim=1, keepdim=True)

        fused_probs = torch.softmax(detached_fused_logits, dim=-1)
        fused_abnormal = fused_probs[:, 1:2]
        fused_margin = detached_fused_logits[:, 1:2] - detached_fused_logits[:, 0:1]
        safe_weights = detached_weights.clamp_min(1e-8)
        weight_entropy = -(safe_weights * safe_weights.log()).sum(dim=1, keepdim=True)
        weight_entropy = weight_entropy / math.log(float(self.num_views))
        global_features = torch.cat(
            [
                fused_abnormal,
                fused_margin,
                abnormal_probs.amax(dim=1, keepdim=True),
                abnormal_probs.amin(dim=1, keepdim=True),
                view_margins.amax(dim=1, keepdim=True),
                view_margins.amin(dim=1, keepdim=True),
                weight_entropy,
            ],
            dim=1,
        )
        evidence_features = torch.cat(
            [
                detached_weights,
                abnormal_probs,
                centered_abnormal,
                view_margins,
                centered_margins,
                weighted_abnormal,
                global_features,
            ],
            dim=1,
        )
        raw_residual = self.adapter(self.norm(evidence_features))
        margin_residual = self.residual_limit * torch.tanh(raw_residual / self.residual_limit)

        center = fused_logits.mean(dim=-1, keepdim=True)
        margin = fused_logits[:, 1:2] - fused_logits[:, 0:1] + margin_residual
        return torch.cat(
            [
                center - 0.5 * margin,
                center + 0.5 * margin,
            ],
            dim=-1,
        )


class PositiveEvidenceFloorFusion(nn.Module):
    """One-way abnormal-margin boost from detached high-confidence view evidence."""

    def __init__(
        self,
        num_views: int = 3,
        hidden_dim: int = 48,
        residual_limit: float = 1.0,
        dropout: float = 0.05,
        evidence_threshold: float = 0.75,
        support_threshold: float = 0.2,
        fused_ceiling: float = 0.5,
        init_bias: float = -6.0,
    ) -> None:
        super().__init__()
        if num_views <= 0:
            raise ValueError("num_views must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be > 0.")
        for name, value in {
            "evidence_threshold": evidence_threshold,
            "support_threshold": support_threshold,
            "fused_ceiling": fused_ceiling,
        }.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")

        self.num_views = int(num_views)
        self.residual_limit = float(residual_limit)
        self.evidence_threshold = float(evidence_threshold)
        self.support_threshold = float(support_threshold)
        self.fused_ceiling = float(fused_ceiling)
        input_dim = self.num_views * 6 + 7
        self.norm = nn.LayerNorm(input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.constant_(self.adapter[-1].bias, float(init_bias))

    def forward(
        self,
        fused_logits: torch.Tensor,
        view_logits: torch.Tensor,
        fusion_weights: torch.Tensor,
    ) -> torch.Tensor:
        if fused_logits.shape[-1] != 2 or view_logits.shape[-1] != 2:
            return fused_logits
        if view_logits.ndim != 3 or fusion_weights.ndim != 3:
            raise ValueError("view_logits and fusion_weights must have shape (batch, views, ...).")
        if view_logits.shape[1] != self.num_views or fusion_weights.shape[1] != self.num_views:
            raise ValueError(
                f"PositiveEvidenceFloorFusion configured for {self.num_views} views."
            )

        detached_view_logits = view_logits.detach()
        detached_weights = fusion_weights.detach().squeeze(-1)
        detached_fused_logits = fused_logits.detach()

        view_probs = torch.softmax(detached_view_logits, dim=-1)
        abnormal_probs = view_probs[..., 1]
        view_margins = detached_view_logits[..., 1] - detached_view_logits[..., 0]
        weighted_abnormal = detached_weights * abnormal_probs
        centered_abnormal = abnormal_probs - abnormal_probs.mean(dim=1, keepdim=True)
        centered_margins = view_margins - view_margins.mean(dim=1, keepdim=True)

        fused_probs = torch.softmax(detached_fused_logits, dim=-1)
        fused_abnormal = fused_probs[:, 1:2]
        fused_margin = detached_fused_logits[:, 1:2] - detached_fused_logits[:, 0:1]
        top2_abnormal = abnormal_probs.topk(k=min(2, self.num_views), dim=1).values
        max_abnormal = top2_abnormal[:, :1]
        second_abnormal = (
            top2_abnormal[:, 1:2]
            if top2_abnormal.shape[1] > 1
            else top2_abnormal[:, :1]
        )
        safe_weights = detached_weights.clamp_min(1e-8)
        weight_entropy = -(safe_weights * safe_weights.log()).sum(dim=1, keepdim=True)
        weight_entropy = weight_entropy / math.log(float(self.num_views))
        global_features = torch.cat(
            [
                fused_abnormal,
                fused_margin,
                max_abnormal,
                second_abnormal,
                view_margins.amax(dim=1, keepdim=True),
                view_margins.amin(dim=1, keepdim=True),
                weight_entropy,
            ],
            dim=1,
        )
        evidence_features = torch.cat(
            [
                detached_weights,
                abnormal_probs,
                centered_abnormal,
                view_margins,
                centered_margins,
                weighted_abnormal,
                global_features,
            ],
            dim=1,
        )
        eligible = (
            max_abnormal.ge(self.evidence_threshold)
            & second_abnormal.ge(self.support_threshold)
            & fused_abnormal.lt(self.fused_ceiling)
        ).to(dtype=fused_logits.dtype)
        raw_boost = self.adapter(self.norm(evidence_features))
        margin_boost = self.residual_limit * torch.sigmoid(raw_boost) * eligible

        center = fused_logits.mean(dim=-1, keepdim=True)
        margin = fused_logits[:, 1:2] - fused_logits[:, 0:1] + margin_boost
        return torch.cat(
            [
                center - 0.5 * margin,
                center + 0.5 * margin,
            ],
            dim=-1,
        )


class MultiViewEncoder(nn.Module):
    """
    多视角编码器 - 把 3 个视角的 CT 切片图像分别提取成特征向量。

    CT 扫描通常有 3 个方向的视角：
        - 轴状面 (Axial)    - 从上往下看
        - 冠状面 (Coronal)  - 从前往后看
        - 矢状面 (Sagittal) - 从侧面看

    每个视角有多张切片（比如 16 张），这个类的作用就是：
        对每个视角 -> 把所有切片分别过 ResNet18 -> 池化聚合 -> 得到该视角的 512 维特征

    池化方式：
        - use_attention_pooling=False（默认）：简单 mean pooling（对所有切片取平均）
        - use_attention_pooling=True：AttentionPooling（用可学习注意力聚焦关键切片）
    """

    def __init__(
        self,
        share_backbone: bool = True,    # 3 个视角是否共享同一个 encoder
        use_pretrained: bool = False,    # 是否使用预训练权重
        use_attention_pooling: bool = False,  # 是否使用注意力池化替代 mean pooling
        freeze_layers: int = DEFAULT_FREEZE_LAYERS,  # 冻结 backbone 前 N 个 layer block
        backbone: str = "resnet18",      # backbone 类型: "resnet18" 或 "resunet"
    ) -> None:
        """
        参数：
            share_backbone: 是否让 3 个视角共用同一个 encoder？
                - True（推荐）：3 个视角用同一个 encoder，参数量小，不容易过拟合
                - False：每个视角有自己独立的 encoder（3 份参数），参数量更大
            use_pretrained: 是否使用 ImageNet 预训练权重
            use_attention_pooling: 是否使用 AttentionPooling？
                - True：用可学习注意力对切片加权聚合（能自动聚焦关键切片）
                - False（默认）：用简单 mean pooling（对所有切片取平均）
            freeze_layers: 冻结 backbone 前 N 个 layer block（0-3）
                - 0（默认）：所有层都可训练
                - 3（推荐小数据集）：只训练 layer4 + 分类头
            backbone: backbone 类型
                - "resnet18"：经典 ResNet18 分类 backbone
                - "resunet"：ResUNet + Attention Gate（空间注意力增强）
        """
        super().__init__()
        self.share_backbone = share_backbone
        self.use_attention_pooling = use_attention_pooling
        self.feature_dim = 512  # encoder 输出的特征维度固定为 512

        if share_backbone:
            # 共享模式：只创建一个 encoder，3 个视角都用它
            self.shared_encoder = build_encoder(
                backbone=backbone,
                use_pretrained=use_pretrained, freeze_layers=freeze_layers,
            )
        else:
            # 独立模式：创建 3 个独立的 encoder，每个视角用自己的
            self.view_encoders = nn.ModuleList(
                [build_encoder(
                    backbone=backbone,
                    use_pretrained=use_pretrained, freeze_layers=freeze_layers,
                ) for _ in range(3)]
            )

        # ---------- 切片池化方式 ----------
        if use_attention_pooling:
            if share_backbone:
                # 共享 backbone 时也共享 pooling 模块
                self.shared_pooling = AttentionPooling(self.feature_dim)
            else:
                # 独立模式：每个视角有自己的 AttentionPooling
                self.view_poolings = nn.ModuleList(
                    [AttentionPooling(self.feature_dim) for _ in range(3)]
                )

    def encode_views(self, images: torch.Tensor) -> list[torch.Tensor]:
        """
        对 3 个视角的图像分别提取特征。

        参数：
            images: 输入的 CT 图像张量
                形状为 (B, 3, S, H, W)，其中：
                    B = batch_size（一批有多少个病人）
                    3 = 视角数量（轴状、冠状、矢状）
                    S = num_slices（每个视角的切片数量，比如 16）
                    H = height（图像高度，比如 224）
                    W = width（图像宽度，比如 224）

        返回：
            一个列表，包含 3 个张量，每个形状为 (B, 512)
            分别代表 3 个视角各自的特征
        """
        # 从输入张量中提取各个维度的大小
        batch_size, num_views, num_slices, height, width = images.shape
        view_features = []  # 用来存放 3 个视角的特征

        for view_index in range(num_views):  # 循环处理每个视角（0, 1, 2）
            # 取出当前视角的所有切片图像
            # images[:, view_index, :, :, :] 的形状是 (B, S, H, W)
            # reshape 成 (B*S, 1, H, W)，把所有切片排成一批，方便 ResNet18 统一处理
            view_tensor = images[:, view_index, :, :, :].reshape(
                batch_size * num_slices, 1, height, width
            )

            # 用 ResNet18 提取每张切片的特征
            if self.share_backbone:
                slice_features = self.shared_encoder(view_tensor)     # 共享编码器
            else:
                slice_features = self.view_encoders[view_index](view_tensor)  # 该视角专用的编码器

            # slice_features 现在是 (B*S, 512)
            # 把它 reshape 回 (B, S, 512)，这样每个病人有 S 个切片各自的特征
            slice_features = slice_features.reshape(batch_size, num_slices, self.feature_dim)

            # 对所有切片的特征进行池化，得到该视角的代表性特征
            if self.use_attention_pooling:
                # AttentionPooling: 用可学习注意力对切片加权聚合
                # (B, S, 512) -> (B, 512)
                if self.share_backbone:
                    pooled_view_feature = self.shared_pooling(slice_features)
                else:
                    pooled_view_feature = self.view_poolings[view_index](slice_features)
            else:
                # Mean Pooling: 对所有切片取平均
                # (B, S, 512) -> mean(dim=1) -> (B, 512)
                pooled_view_feature = slice_features.mean(dim=1)
            view_features.append(pooled_view_feature)

        return view_features  # 返回 3 个 (B, 512) 的张量


class MultiViewCTClassifier(MultiViewEncoder):
    """
    特征融合分类器（Feature Fusion）。

    工作流程：
        1. 用父类 MultiViewEncoder 提取 3 个视角的特征，每个是 (B, 512)
        2. 把 3 个特征向量拼接在一起，得到 (B, 1536)
        3. 送进一个小型分类网络（MLP），输出 2 个类别的分数

    这是最简单直接的多视角融合方式，类似于"把 3 个视角的信息全部堆在一起，然后一起做判断"。
    """

    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
        fusion_hidden_dim: int = 256,    # 分类器隐藏层的维度
        dropout: float = 0.3,           # Dropout 比率（防止过拟合）
        use_attention_pooling: bool = False,  # 是否使用注意力池化
        freeze_layers: int = DEFAULT_FREEZE_LAYERS,  # 冻结 backbone 前 N 个 layer block
        backbone: str = "resnet18",      # backbone 类型
        minimal_fusion_baseline: bool = False,
        equal_weight_fusion: bool = False,
    ) -> None:
        _ = equal_weight_fusion  # build_model passes this shared key for decision fusion only.
        super().__init__(
            share_backbone=share_backbone,
            use_pretrained=use_pretrained,
            use_attention_pooling=use_attention_pooling,
            freeze_layers=freeze_layers,
            backbone=backbone,
        )
        # 3 个视角拼接后的总维度：512 * 3 = 1536
        fused_dim = self.feature_dim * 3
        self.minimal_fusion_baseline = minimal_fusion_baseline
        self.disable_feature_view_recalibration = _env_flag(
            "ANKLE_FEATURE_DISABLE_VIEW_RECALIBRATION"
        )
        self.disable_feature_cross_view_mixer = _env_flag(
            "ANKLE_FEATURE_DISABLE_CROSS_VIEW_MIXER"
        )
        self.disable_feature_glu_head = _env_flag("ANKLE_FEATURE_DISABLE_GLU_HEAD")
        self.enable_feature_view_role_embedding = _env_flag(
            "ANKLE_FEATURE_ENABLE_VIEW_ROLE_EMBEDDING"
        )
        self.enable_feature_pairwise_token_residual = _env_flag(
            "ANKLE_FEATURE_ENABLE_PAIRWISE_TOKEN_RESIDUAL"
        )
        self.enable_feature_view_token_scalar_gate = _env_flag(
            "ANKLE_FEATURE_ENABLE_VIEW_TOKEN_SCALAR_GATE"
        )
        self.enable_feature_view_token_channel_gate = _env_flag(
            "ANKLE_FEATURE_ENABLE_VIEW_TOKEN_CHANNEL_GATE"
        )
        self.enable_feature_xview_identity_skip_gate = _env_flag(
            "ANKLE_FEATURE_ENABLE_XVIEW_IDENTITY_SKIP_GATE"
        )

        if minimal_fusion_baseline:
            # 纯融合对比模式：不引入额外视角交互或门控模块，只保留最小 MLP 头。
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, 2),
            )
        else:
            if self.disable_feature_view_recalibration:
                self.view_recalibrators = nn.ModuleList()
            else:
                self.view_recalibrators = nn.ModuleList(
                    [ViewFeatureRecalibration(self.feature_dim) for _ in range(3)]
                )
            if self.disable_feature_cross_view_mixer:
                self.cross_view_mixer = nn.Identity()
            else:
                # 只在 pooled view token 之间加入极轻量的 cross-view context 交换，
                # 保持 256x8 mean-pooling 几何不变，检验“缺少显式视角交互”是否仍是瓶颈。
                from .cross_view_attention import CrossViewAttention

                self.cross_view_mixer = CrossViewAttention(
                    feature_dim=self.feature_dim,
                    attention_dim=_env_positive_int("ANKLE_FEATURE_XVIEW_ATTENTION_DIM", 256),
                    num_heads=_env_positive_int("ANKLE_FEATURE_XVIEW_NUM_HEADS", 4),
                    num_layers=_env_positive_int("ANKLE_FEATURE_XVIEW_NUM_LAYERS", 1),
                    dropout=_env_unit_float("ANKLE_FEATURE_XVIEW_DROPOUT", 0.1),
                    residual_scale=_env_positive_float("ANKLE_FEATURE_XVIEW_RESIDUAL_SCALE", 0.125),
                )
            if self.enable_feature_view_role_embedding:
                self.feature_view_role_embedding = nn.Parameter(
                    torch.zeros(3, self.feature_dim)
                )
            if self.enable_feature_pairwise_token_residual:
                self.feature_pairwise_token_residual = FeaturePairwiseTokenResidual(
                    feature_dim=self.feature_dim,
                    pair_dim=_env_positive_int(
                        "ANKLE_FEATURE_PAIRWISE_TOKEN_PAIR_DIM",
                        64,
                    ),
                    dropout=_env_unit_float(
                        "ANKLE_FEATURE_PAIRWISE_TOKEN_DROPOUT",
                        0.05,
                    ),
                    residual_scale=_env_positive_float(
                        "ANKLE_FEATURE_PAIRWISE_TOKEN_RESIDUAL_SCALE",
                        0.0625,
                    ),
                )
            if self.enable_feature_view_token_scalar_gate:
                self.feature_view_token_scalar_gate = FeatureViewTokenScalarGate(
                    feature_dim=self.feature_dim
                )
            if self.enable_feature_view_token_channel_gate:
                self.feature_view_token_channel_gate = FeatureViewTokenChannelGate(
                    feature_dim=self.feature_dim
                )
            if self.enable_feature_xview_identity_skip_gate:
                self.feature_xview_identity_skip_gate = FeatureCrossViewIdentitySkipGate(
                    feature_dim=self.feature_dim,
                    max_scale=_env_positive_float(
                        "ANKLE_FEATURE_XVIEW_IDENTITY_SKIP_MAX_SCALE",
                        0.5,
                    ),
                )

            if self.disable_feature_glu_head:
                self.classifier = nn.Sequential(
                    nn.LayerNorm(fused_dim),
                    nn.Linear(fused_dim, fusion_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_hidden_dim, 2),
                )
            else:
                # 分类器：保留现有 prenorm，但把 plain ReLU MLP 换成轻量 GLU 门控头，
                # 让 fused token 在不改 256x8 几何的前提下拥有更强的多视角交互表达力。
                self.classifier = nn.Sequential(
                    nn.LayerNorm(fused_dim),                 # 稳定跨视角拼接特征的尺度
                    nn.Linear(fused_dim, fusion_hidden_dim * 2),  # 为 GLU 同时生成 value / gate 分支
                    nn.GLU(dim=1),
                    nn.Dropout(dropout),                      # 随机丢弃 30% 的神经元（防止过拟合）
                    nn.Linear(fusion_hidden_dim, 2),          # 256 -> 2（输出 2 个类别的分数）
                )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播：从输入图像到分类结果。

        参数：
            images: (B, 3, S, H, W) 的 CT 图像张量

        返回：
            logits: (B, 2) 的张量，每个样本有 2 个分数（正常/异常）
                    分数越高表示模型越倾向于认为是该类别
        """
        view_features = self.encode_views(images)
        if self.minimal_fusion_baseline:
            image_feature = torch.cat(view_features, dim=1)
        else:
            # 第 1 步：提取 3 个视角的特征，并在拼接前做轻量 per-view 重标定
            # encode_views 返回 3 个 (B, 512)，之后再做一次轻量 cross-view mixing。
            if self.disable_feature_view_recalibration:
                recalibrated_features = view_features
            else:
                recalibrated_features = [
                    recalibrator(feature)
                    for recalibrator, feature in zip(self.view_recalibrators, view_features)
                ]
            stacked_features = torch.stack(recalibrated_features, dim=1)
            if self.enable_feature_view_role_embedding:
                stacked_features = stacked_features + self.feature_view_role_embedding.unsqueeze(0)
            skip_features = stacked_features
            mixed_features = self.cross_view_mixer(stacked_features)
            if self.enable_feature_xview_identity_skip_gate:
                mixed_features = self.feature_xview_identity_skip_gate(
                    mixed_features,
                    skip_features,
                )
            if self.enable_feature_pairwise_token_residual:
                mixed_features = self.feature_pairwise_token_residual(mixed_features)
            if self.enable_feature_view_token_scalar_gate:
                mixed_features = self.feature_view_token_scalar_gate(mixed_features)
            if self.enable_feature_view_token_channel_gate:
                mixed_features = self.feature_view_token_channel_gate(mixed_features)
            image_feature = mixed_features.reshape(mixed_features.shape[0], -1)
        # 第 2 步：送进分类器，得到分类结果
        return self.classifier(image_feature)


class MultiViewDecisionFusionClassifier(MultiViewEncoder):
    """
    决策融合分类器 - 视角可靠度门控版 (View Reliability Gating)。

    与原版 Decision Fusion 的关键区别：
        - 原版：3 个视角共用一组固定的全局权重（nn.Parameter(zeros(3))）
        - 本版：每个视角有一个 confidence head，根据当前样本的特征动态计算权重
          → 不同病人的融合权重不同，能适应"某个视角拍得不清楚"等个体差异
        - 控制变量模式：也支持固定等比例权重（每个视角 1/3），只改变 fusion weight 机制，
          其余 encoder / per-view classifier 保持不变，用来直接测 learned weighting 的净收益

    工作流程：
        1. 用父类 MultiViewEncoder 提取 3 个视角的特征
        2. 每个视角先经过 baseline plain classifier，避免继续扰动 classifier path
        3. reliability estimation 分支保留轻量 residual cross-view attention，
           再用共享、低秩、identity-init 的 residual calibrator 对 raw reliability logit 做小幅校准
        4. 3 个 reliability logits 经 softmax 归一化后作为融合权重
        5. 用动态权重对 3 个视角的分类结果加权平均
    """

    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        use_attention_pooling: bool = False,  # 是否使用注意力池化
        freeze_layers: int = DEFAULT_FREEZE_LAYERS,  # 冻结 backbone 前 N 个 layer block
        backbone: str = "resnet18",      # backbone 类型
        minimal_fusion_baseline: bool = False,
        equal_weight_fusion: bool = False,
    ) -> None:
        super().__init__(
            share_backbone=share_backbone,
            use_pretrained=use_pretrained,
            use_attention_pooling=use_attention_pooling,
            freeze_layers=freeze_layers,
            backbone=backbone,
        )
        self.minimal_fusion_baseline = minimal_fusion_baseline
        self.equal_weight_fusion = equal_weight_fusion
        self.disable_fusion_cross_view_mixer = _env_flag(
            "ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER"
        )
        self.disable_fusion_calibrator = _env_flag(
            "ANKLE_DISABLE_FUSION_CALIBRATOR"
        )
        self.enable_classifier_view_context = _env_flag(
            "ANKLE_DECISION_ENABLE_CLASSIFIER_VIEW_CONTEXT"
        )
        self.enable_aux_view_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS"
        )
        self.aux_view_loss_weight = _env_positive_float(
            "ANKLE_DECISION_AUX_VIEW_LOSS_WEIGHT",
            0.5,
        )
        self.enable_view_logit_temperature = _env_flag(
            "ANKLE_DECISION_ENABLE_VIEW_LOGIT_TEMPERATURE"
        )
        self.enable_evidence_aware_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_EVIDENCE_AWARE_GATE"
        )
        self.evidence_aware_gate_residual_limit = _env_positive_float(
            "ANKLE_DECISION_EVIDENCE_AWARE_GATE_RESIDUAL_LIMIT",
            0.25,
        )
        self.enable_candidate_view_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_CANDIDATE_VIEW_GATE"
        )
        self.candidate_view_gate_residual_limit = _env_positive_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_RESIDUAL_LIMIT",
            0.35,
        )
        self.candidate_view_gate_margin_gap = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_MARGIN_GAP",
            0.5,
        )
        self.candidate_view_gate_margin_window = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_MARGIN_WINDOW",
            0.25,
        )
        self.candidate_view_gate_require_class_consensus = _env_flag(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_REQUIRE_CLASS_CONSENSUS"
        )
        self.candidate_view_gate_detach_features = _env_flag(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_DETACH_FEATURES"
        )
        self.candidate_view_gate_consensus_margin = _env_positive_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_CONSENSUS_MARGIN",
            0.2,
        )
        self.candidate_view_gate_consensus_window = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_CONSENSUS_WINDOW",
            0.2,
        )
        self.candidate_view_gate_prob_advantage = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_PROB_ADVANTAGE",
            0.0,
        )
        self.candidate_view_gate_prob_window = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_PROB_WINDOW",
            0.0,
        )
        self.candidate_view_gate_prototype_advantage = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_PROTOTYPE_ADVANTAGE",
            0.0,
        )
        self.candidate_view_gate_prototype_window = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_PROTOTYPE_WINDOW",
            0.0,
        )
        self.candidate_view_gate_require_nonaxial_pair_consensus = _env_flag(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_REQUIRE_NONAXIAL_PAIR_CONSENSUS"
        )
        self.candidate_view_gate_nonaxial_pair_consensus_margin = _env_positive_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_NONAXIAL_PAIR_CONSENSUS_MARGIN",
            0.2,
        )
        self.candidate_view_gate_nonaxial_pair_consensus_window = _env_unit_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_GATE_NONAXIAL_PAIR_CONSENSUS_WINDOW",
            0.2,
        )
        self.enable_candidate_view_prototype_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_CANDIDATE_VIEW_PROTOTYPE_AUX_LOSS"
        )
        self.candidate_view_prototype_aux_weight = _env_positive_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_PROTOTYPE_AUX_WEIGHT",
            0.05,
        )
        self.candidate_view_prototype_aux_temperature = _env_positive_float(
            "ANKLE_DECISION_CANDIDATE_VIEW_PROTOTYPE_AUX_TEMPERATURE",
            0.2,
        )
        self.enable_gate_view_correctness_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_VIEW_CORRECTNESS_AUX_LOSS"
        )
        self.gate_view_correctness_aux_weight = _env_positive_float(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_WEIGHT",
            0.01,
        )
        self.gate_view_correctness_aux_base_scale = _env_positive_float(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_BASE_SCALE",
            0.5,
        )
        self.gate_view_correctness_aux_weight_scale = _env_positive_float(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_WEIGHT_SCALE",
            1.5,
        )
        self.gate_view_correctness_aux_require_disagreement = _env_flag(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_REQUIRE_DISAGREEMENT"
        )
        self.gate_view_correctness_aux_nonaxial_only = _env_flag(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_NONAXIAL_ONLY"
        )
        self.gate_view_correctness_aux_protect_strong_axial = _env_flag(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_PROTECT_STRONG_AXIAL"
        )
        self.gate_view_correctness_aux_axial_margin_cap = _env_positive_float(
            "ANKLE_DECISION_GATE_VIEW_CORRECTNESS_AUX_AXIAL_MARGIN_CAP",
            2.5,
        )
        self.enable_pairwise_selector_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_PAIRWISE_SELECTOR_AUX_LOSS"
        )
        self.pairwise_selector_aux_weight = _env_positive_float(
            "ANKLE_DECISION_PAIRWISE_SELECTOR_AUX_WEIGHT",
            0.005,
        )
        self.pairwise_selector_aux_require_disagreement = _env_flag(
            "ANKLE_DECISION_PAIRWISE_SELECTOR_AUX_REQUIRE_DISAGREEMENT"
        )
        self.enable_gate_pairwise_contrast_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_PAIRWISE_CONTRAST_AUX_LOSS"
        )
        self.gate_pairwise_contrast_aux_weight = _env_positive_float(
            "ANKLE_DECISION_GATE_PAIRWISE_CONTRAST_AUX_WEIGHT",
            0.0025,
        )
        self.gate_pairwise_contrast_aux_require_disagreement = _env_flag(
            "ANKLE_DECISION_GATE_PAIRWISE_CONTRAST_AUX_REQUIRE_DISAGREEMENT"
        )
        self.enable_gate_label_evidence_rank_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_LABEL_EVIDENCE_RANK_AUX_LOSS"
        )
        self.gate_label_evidence_rank_aux_weight = _env_positive_float(
            "ANKLE_DECISION_GATE_LABEL_EVIDENCE_RANK_AUX_WEIGHT",
            0.0025,
        )
        self.enable_gate_targeted_evidence_rank_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_TARGETED_EVIDENCE_RANK_AUX_LOSS"
        )
        self.gate_targeted_evidence_rank_aux_weight = _env_positive_float(
            "ANKLE_DECISION_GATE_TARGETED_EVIDENCE_RANK_AUX_WEIGHT",
            0.001,
        )
        self.gate_targeted_evidence_rank_aux_gap = _env_unit_float(
            "ANKLE_DECISION_GATE_TARGETED_EVIDENCE_RANK_AUX_GAP",
            0.05,
        )
        self.enable_gate_confirmed_target_rank_margin_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_CONFIRMED_TARGET_RANK_MARGIN_AUX_LOSS"
        )
        self.gate_confirmed_target_rank_margin_aux_weight = _env_positive_float(
            "ANKLE_DECISION_GATE_CONFIRMED_TARGET_RANK_MARGIN_AUX_WEIGHT",
            0.0025,
        )
        self.gate_confirmed_target_rank_margin_aux_gap = _env_unit_float(
            "ANKLE_DECISION_GATE_CONFIRMED_TARGET_RANK_MARGIN_AUX_GAP",
            0.02,
        )
        self.gate_confirmed_target_rank_margin_aux_margin = _env_positive_float(
            "ANKLE_DECISION_GATE_CONFIRMED_TARGET_RANK_MARGIN_AUX_MARGIN",
            0.25,
        )
        self.enable_axial_sagittal_rank_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_AXIAL_SAGITTAL_RANK_AUX_LOSS"
        )
        self.axial_sagittal_rank_aux_weight = _env_positive_float(
            "ANKLE_DECISION_AXIAL_SAGITTAL_RANK_AUX_WEIGHT",
            0.001,
        )
        self.enable_fp_risk_axial_sagittal_rank_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_LOSS"
        )
        self.fp_risk_axial_sagittal_rank_aux_weight = _env_positive_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_WEIGHT",
            0.001,
        )
        self.fp_risk_axial_sagittal_rank_aux_axial_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_AXIAL_ABNORMAL_MIN",
            0.82,
        )
        self.fp_risk_axial_sagittal_rank_aux_axial_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_AXIAL_ABNORMAL_MAX",
            0.93,
        )
        self.fp_risk_axial_sagittal_rank_aux_sagittal_normal_min = _env_unit_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_SAGITTAL_NORMAL_MIN",
            0.70,
        )
        self.fp_risk_axial_sagittal_rank_aux_fused_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_FUSED_ABNORMAL_MIN",
            0.75,
        )
        self.fp_risk_axial_sagittal_rank_aux_coronal_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_CORONAL_ABNORMAL_MAX",
            0.525,
        )
        self.fp_risk_axial_sagittal_rank_aux_margin = _env_positive_float(
            "ANKLE_DECISION_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_MARGIN",
            0.25,
        )
        self.enable_axial_fp_risk_normal_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_AXIAL_FP_RISK_NORMAL_AUX_LOSS"
        )
        self.axial_fp_risk_normal_aux_weight = _env_positive_float(
            "ANKLE_DECISION_AXIAL_FP_RISK_NORMAL_AUX_WEIGHT",
            0.001,
        )
        self.axial_fp_risk_normal_aux_axial_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_AXIAL_FP_RISK_NORMAL_AUX_AXIAL_ABNORMAL_MIN",
            0.925,
        )
        self.axial_fp_risk_normal_aux_sagittal_normal_min = _env_unit_float(
            "ANKLE_DECISION_AXIAL_FP_RISK_NORMAL_AUX_SAGITTAL_NORMAL_MIN",
            0.60,
        )
        self.axial_fp_risk_normal_aux_fused_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_AXIAL_FP_RISK_NORMAL_AUX_FUSED_ABNORMAL_MIN",
            0.75,
        )
        self.axial_fp_risk_normal_aux_coronal_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_AXIAL_FP_RISK_NORMAL_AUX_CORONAL_ABNORMAL_MAX",
            0.525,
        )
        self.enable_coronal_pair_abnormal_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_CORONAL_PAIR_ABNORMAL_AUX_LOSS"
        )
        self.coronal_pair_abnormal_aux_weight = _env_positive_float(
            "ANKLE_DECISION_CORONAL_PAIR_ABNORMAL_AUX_WEIGHT",
            0.005,
        )
        self.coronal_pair_abnormal_aux_require_disagreement = _env_flag(
            "ANKLE_DECISION_CORONAL_PAIR_ABNORMAL_AUX_REQUIRE_DISAGREEMENT"
        )
        self.enable_nonaxial_abnormal_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_NONAXIAL_ABNORMAL_AUX_LOSS"
        )
        self.nonaxial_abnormal_aux_weight = _env_positive_float(
            "ANKLE_DECISION_NONAXIAL_ABNORMAL_AUX_WEIGHT",
            0.025,
        )
        self.nonaxial_abnormal_aux_coronal_only = _env_flag(
            "ANKLE_DECISION_NONAXIAL_ABNORMAL_AUX_CORONAL_ONLY"
        )
        self.enable_sagittal_normal_rescue_aux_loss = _env_flag(
            "ANKLE_DECISION_ENABLE_SAGITTAL_NORMAL_RESCUE_AUX_LOSS"
        )
        self.sagittal_normal_rescue_aux_weight = _env_positive_float(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_AUX_WEIGHT",
            0.0025,
        )
        self.enable_sagittal_normal_rescue_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_SAGITTAL_NORMAL_RESCUE_GATE"
        )
        self.sagittal_normal_rescue_gate_eval_only = _env_flag(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_GATE_EVAL_ONLY"
        )
        self.sagittal_normal_rescue_gate_residual_limit = _env_positive_float(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_GATE_RESIDUAL_LIMIT",
            0.5,
        )
        self.sagittal_normal_rescue_gate_axial_abnormal_threshold = _env_unit_float(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_GATE_AXIAL_ABNORMAL_THRESHOLD",
            0.65,
        )
        self.sagittal_normal_rescue_gate_sagittal_normal_threshold = _env_unit_float(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_GATE_SAGITTAL_NORMAL_THRESHOLD",
            0.65,
        )
        self.sagittal_normal_rescue_gate_fused_abnormal_threshold = _env_unit_float(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_GATE_FUSED_ABNORMAL_THRESHOLD",
            0.55,
        )
        self.sagittal_normal_rescue_gate_window = _env_unit_float(
            "ANKLE_DECISION_SAGITTAL_NORMAL_RESCUE_GATE_WINDOW",
            0.15,
        )
        self.enable_posthoc_fp_risk_sagittal_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_POSTHOC_FP_RISK_SAGITTAL_GATE"
        )
        self.posthoc_fp_risk_sagittal_gate_residual = _env_positive_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_RESIDUAL",
            5.0,
        )
        self.posthoc_fp_risk_sagittal_gate_axial_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_AXIAL_ABNORMAL_MIN",
            0.4,
        )
        self.posthoc_fp_risk_sagittal_gate_axial_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_AXIAL_ABNORMAL_MAX",
            0.88,
        )
        self.posthoc_fp_risk_sagittal_gate_sagittal_normal_min = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_SAGITTAL_NORMAL_MIN",
            0.55,
        )
        self.posthoc_fp_risk_sagittal_gate_coronal_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_CORONAL_ABNORMAL_MAX",
            0.525,
        )
        self.posthoc_fp_risk_sagittal_gate_fused_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_FUSED_ABNORMAL_MIN",
            0.8,
        )
        self.posthoc_fp_risk_sagittal_gate_confidence_gap_max = _env_positive_float(
            "ANKLE_DECISION_POSTHOC_FP_RISK_SAGITTAL_GATE_CONFIDENCE_GAP_MAX",
            5.0,
        )
        self.enable_posthoc_fn_abnormal_rescue_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_POSTHOC_FN_ABNORMAL_RESCUE_GATE"
        )
        self.posthoc_fn_abnormal_rescue_gate_residual = _env_positive_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_RESIDUAL",
            6.0,
        )
        self.posthoc_fn_abnormal_rescue_gate_fused_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_FUSED_ABNORMAL_MAX",
            0.2,
        )
        self.posthoc_fn_abnormal_rescue_gate_axial_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_AXIAL_ABNORMAL_MIN",
            0.45,
        )
        self.posthoc_fn_abnormal_rescue_gate_max_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_MAX_ABNORMAL_MIN",
            0.45,
        )
        self.posthoc_fn_abnormal_rescue_gate_second_abnormal_min = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_SECOND_ABNORMAL_MIN",
            0.0,
        )
        self.posthoc_fn_abnormal_rescue_gate_confidence_gap_max = _env_positive_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_CONFIDENCE_GAP_MAX",
            1.0,
        )
        self.posthoc_fn_abnormal_rescue_gate_axial_abnormal_max = _env_unit_float(
            "ANKLE_DECISION_POSTHOC_FN_ABNORMAL_RESCUE_GATE_AXIAL_ABNORMAL_MAX",
            0.65,
        )
        self.enable_sagittal_reliability_calibrator = _env_flag(
            "ANKLE_DECISION_ENABLE_SAGITTAL_RELIABILITY_CALIBRATOR"
        )
        self.sagittal_reliability_calibrator_hidden_dim = _env_positive_int(
            "ANKLE_DECISION_SAGITTAL_RELIABILITY_CALIBRATOR_HIDDEN_DIM",
            32,
        )
        self.sagittal_reliability_calibrator_residual_limit = _env_positive_float(
            "ANKLE_DECISION_SAGITTAL_RELIABILITY_CALIBRATOR_RESIDUAL_LIMIT",
            1.0,
        )
        self.sagittal_reliability_calibrator_dropout = _env_unit_float(
            "ANKLE_DECISION_SAGITTAL_RELIABILITY_CALIBRATOR_DROPOUT",
            0.05,
        )
        self.sagittal_reliability_calibrator_coronal_suppression = _env_unit_float(
            "ANKLE_DECISION_SAGITTAL_RELIABILITY_CALIBRATOR_CORONAL_SUPPRESSION",
            0.0,
        )
        self.enable_gate_logit_rms_limit = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_LOGIT_RMS_LIMIT"
        )
        self.gate_logit_max_centered_rms = _env_positive_float(
            "ANKLE_DECISION_GATE_LOGIT_MAX_CENTERED_RMS",
            1.3,
        )
        self.enable_gate_view_prior_debias = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_VIEW_PRIOR_DEBIAS"
        )
        self.gate_view_prior_debias_strength = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_STRENGTH",
            1.0,
        )
        self.gate_view_prior_debias_max_strength = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MAX_STRENGTH",
            self.gate_view_prior_debias_strength,
        )
        self.gate_view_prior_debias_momentum = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_MOMENTUM",
            0.1,
        )
        self.gate_view_prior_debias_evidence_close_gap = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_EVIDENCE_CLOSE_GAP",
            0.0,
        )
        self.gate_view_prior_debias_evidence_close_window = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_EVIDENCE_CLOSE_WINDOW",
            0.0,
        )
        self.gate_view_prior_debias_positive_safe = _env_flag(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_POSITIVE_SAFE"
        )
        self.gate_view_prior_debias_positive_safe_abnormal_gap = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_POSITIVE_SAFE_ABNORMAL_GAP",
            0.0,
        )
        self.gate_view_prior_debias_positive_safe_fused_floor = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_POSITIVE_SAFE_FUSED_FLOOR",
            0.6,
        )
        self.gate_view_prior_debias_positive_safe_window = _env_unit_float(
            "ANKLE_DECISION_GATE_VIEW_PRIOR_DEBIAS_POSITIVE_SAFE_WINDOW",
            0.0,
        )
        self.use_shared_confidence_head = _env_flag(
            "ANKLE_DECISION_USE_SHARED_CONFIDENCE_HEAD"
        )
        self.use_relative_confidence_features = _env_flag(
            "ANKLE_DECISION_USE_RELATIVE_CONFIDENCE_FEATURES"
        )
        self.use_view_role_confidence_head = _env_flag(
            "ANKLE_DECISION_USE_VIEW_ROLE_CONFIDENCE_HEAD"
        )
        self.detach_view_role_gate_features = _env_flag(
            "ANKLE_DECISION_DETACH_VIEW_ROLE_GATE_FEATURES"
        )
        self.decouple_role_gate_classifier_gradient = _env_flag(
            "ANKLE_DECISION_DECOUPLE_ROLE_GATE_CLASSIFIER_GRADIENT"
        )
        self.view_role_confidence_role_dim = _env_positive_int(
            "ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_ROLE_DIM",
            32,
        )
        self.view_role_confidence_hidden_dim = _env_positive_int(
            "ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_HIDDEN_DIM",
            128,
        )
        self.view_role_confidence_dropout = _env_unit_float(
            "ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_DROPOUT",
            0.05,
        )
        self.view_role_confidence_logit_limit = _env_positive_float(
            "ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_LOGIT_LIMIT",
            1.5,
        )
        self.view_role_confidence_blend = _env_unit_float(
            "ANKLE_DECISION_VIEW_ROLE_CONFIDENCE_BLEND",
            1.0,
        )
        self.enable_conditional_view_role_residual = _env_flag(
            "ANKLE_DECISION_ENABLE_CONDITIONAL_VIEW_ROLE_RESIDUAL"
        )
        self.conditional_view_role_residual_scale = _env_unit_float(
            "ANKLE_DECISION_CONDITIONAL_VIEW_ROLE_RESIDUAL_SCALE",
            0.25,
        )
        self.conditional_view_role_residual_gate_gap = _env_positive_float(
            "ANKLE_DECISION_CONDITIONAL_VIEW_ROLE_RESIDUAL_GATE_GAP",
            0.5,
        )
        self.conditional_view_role_residual_fp_prob_threshold = _env_unit_float(
            "ANKLE_DECISION_CONDITIONAL_VIEW_ROLE_RESIDUAL_FP_PROB_THRESHOLD",
            0.7,
        )
        self.conditional_view_role_residual_coronal_low_prob = _env_unit_float(
            "ANKLE_DECISION_CONDITIONAL_VIEW_ROLE_RESIDUAL_CORONAL_LOW_PROB",
            0.35,
        )
        self.conditional_view_role_residual_axial_high_prob = _env_unit_float(
            "ANKLE_DECISION_CONDITIONAL_VIEW_ROLE_RESIDUAL_AXIAL_HIGH_PROB",
            0.65,
        )
        self.conditional_view_role_residual_positive_floor = _env_unit_float(
            "ANKLE_DECISION_CONDITIONAL_VIEW_ROLE_RESIDUAL_POSITIVE_FLOOR",
            0.7,
        )
        self.enable_evidence_margin_residual_fusion = _env_flag(
            "ANKLE_DECISION_ENABLE_EVIDENCE_MARGIN_RESIDUAL_FUSION"
        )
        self.evidence_margin_residual_hidden_dim = _env_positive_int(
            "ANKLE_DECISION_EVIDENCE_MARGIN_RESIDUAL_HIDDEN_DIM",
            48,
        )
        self.evidence_margin_residual_limit = _env_positive_float(
            "ANKLE_DECISION_EVIDENCE_MARGIN_RESIDUAL_LIMIT",
            1.5,
        )
        self.evidence_margin_residual_dropout = _env_unit_float(
            "ANKLE_DECISION_EVIDENCE_MARGIN_RESIDUAL_DROPOUT",
            0.05,
        )
        self.enable_positive_evidence_floor_fusion = _env_flag(
            "ANKLE_DECISION_ENABLE_POSITIVE_EVIDENCE_FLOOR_FUSION"
        )
        self.positive_evidence_floor_hidden_dim = _env_positive_int(
            "ANKLE_DECISION_POSITIVE_EVIDENCE_FLOOR_HIDDEN_DIM",
            48,
        )
        self.positive_evidence_floor_residual_limit = _env_positive_float(
            "ANKLE_DECISION_POSITIVE_EVIDENCE_FLOOR_RESIDUAL_LIMIT",
            1.0,
        )
        self.positive_evidence_floor_dropout = _env_unit_float(
            "ANKLE_DECISION_POSITIVE_EVIDENCE_FLOOR_DROPOUT",
            0.05,
        )
        self.positive_evidence_floor_evidence_threshold = _env_unit_float(
            "ANKLE_DECISION_POSITIVE_EVIDENCE_FLOOR_EVIDENCE_THRESHOLD",
            0.75,
        )
        self.positive_evidence_floor_support_threshold = _env_unit_float(
            "ANKLE_DECISION_POSITIVE_EVIDENCE_FLOOR_SUPPORT_THRESHOLD",
            0.2,
        )
        self.positive_evidence_floor_fused_ceiling = _env_unit_float(
            "ANKLE_DECISION_POSITIVE_EVIDENCE_FLOOR_FUSED_CEILING",
            0.5,
        )
        self.enable_relative_view_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_RELATIVE_VIEW_GATE"
        )
        self.relative_view_gate_residual_limit = _env_positive_float(
            "ANKLE_DECISION_RELATIVE_VIEW_GATE_RESIDUAL_LIMIT",
            0.75,
        )
        self.enable_gate_transformer_context = _env_flag(
            "ANKLE_DECISION_ENABLE_GATE_TRANSFORMER_CONTEXT"
        )
        self.enable_pairwise_reliability_gate = _env_flag(
            "ANKLE_DECISION_ENABLE_PAIRWISE_RELIABILITY_GATE"
        )
        self.pairwise_reliability_pair_dim = _env_positive_int(
            "ANKLE_DECISION_PAIRWISE_RELIABILITY_PAIR_DIM",
            64,
        )
        self.pairwise_reliability_bottleneck_dim = _env_positive_int(
            "ANKLE_DECISION_PAIRWISE_RELIABILITY_BOTTLENECK_DIM",
            64,
        )
        self.pairwise_reliability_residual_limit = _env_positive_float(
            "ANKLE_DECISION_PAIRWISE_RELIABILITY_RESIDUAL_LIMIT",
            0.25,
        )
        self.pairwise_reliability_dropout = _env_unit_float(
            "ANKLE_DECISION_PAIRWISE_RELIABILITY_DROPOUT",
            0.05,
        )
        self.gate_transformer_attention_dim = _env_positive_int(
            "ANKLE_DECISION_GATE_TRANSFORMER_ATTENTION_DIM",
            128,
        )
        self.gate_transformer_num_heads = _env_positive_int(
            "ANKLE_DECISION_GATE_TRANSFORMER_NUM_HEADS",
            4,
        )
        self.gate_transformer_num_layers = _env_positive_int(
            "ANKLE_DECISION_GATE_TRANSFORMER_NUM_LAYERS",
            1,
        )
        self.gate_transformer_dropout = _env_unit_float(
            "ANKLE_DECISION_GATE_TRANSFORMER_DROPOUT",
            0.05,
        )
        self.gate_transformer_residual_scale = _env_positive_float(
            "ANKLE_DECISION_GATE_TRANSFORMER_RESIDUAL_SCALE",
            0.2,
        )
        self.gate_teacher_blend = _env_unit_float(
            "ANKLE_DECISION_GATE_TEACHER_BLEND",
            0.0,
        )
        self.gate_teacher_blend_eval_only = _env_flag(
            "ANKLE_DECISION_GATE_TEACHER_BLEND_EVAL_ONLY"
        )
        self.gate_teacher_temperature = _env_positive_float(
            "ANKLE_DECISION_GATE_TEACHER_TEMPERATURE",
            1.0,
        )
        self.fusion_weight_floor = _env_unit_float(
            "ANKLE_DECISION_FUSION_WEIGHT_FLOOR",
            0.0,
        )
        self.train_view_dropout_prob = _env_unit_float(
            "ANKLE_DECISION_TRAIN_VIEW_DROPOUT_PROB",
            0.0,
        )
        self.train_axial_blur_prob = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_BLUR_PROB",
            0.0,
        )
        self.train_axial_blur_kernel = _env_positive_int(
            "ANKLE_DECISION_TRAIN_AXIAL_BLUR_KERNEL",
            9,
        )
        self.train_dominant_gate_dropout_prob = _env_unit_float(
            "ANKLE_DECISION_TRAIN_DOMINANT_GATE_DROPOUT_PROB",
            0.0,
        )
        self.train_axial_teacher_misalignment_scale_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_SCALE_THRESHOLD",
            0.0,
        )
        self.train_axial_teacher_misalignment_excess_scale_window = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW",
            0.0,
        )
        self.train_axial_teacher_misalignment_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD",
            0.0,
        )
        self.train_axial_teacher_misalignment_dropout_boost = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_DROPOUT_BOOST",
            0.0,
        )
        self.train_teacher_non_axial_competitive_gap = _env_unit_float(
            "ANKLE_DECISION_TRAIN_TEACHER_NONAXIAL_COMPETITIVE_GAP",
            0.0,
        )
        self.train_require_teacher_non_axial_top = _env_flag(
            "ANKLE_DECISION_TRAIN_REQUIRE_TEACHER_NONAXIAL_TOP"
        )
        self.train_non_dominant_weight_floor = _env_unit_float(
            "ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR",
            0.0,
        )
        self.train_target_top_non_dominant_rescue = _env_flag(
            "ANKLE_DECISION_TRAIN_TARGET_TOP_NONDOMINANT_RESCUE"
        )
        self.train_target_teacher_non_dominant_rescue = _env_flag(
            "ANKLE_DECISION_TRAIN_TARGET_TEACHER_NONDOMINANT_RESCUE"
        )
        self.train_target_teacher_dropout_redistribution = _env_flag(
            "ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_REDISTRIBUTION"
        )
        self.train_target_teacher_dropout_only_on_fallback_disagreement = _env_flag(
            "ANKLE_DECISION_TRAIN_TARGET_TEACHER_DROPOUT_ONLY_ON_FALLBACK_DISAGREEMENT"
        )
        self.train_non_dominant_rescue_scale = _env_unit_float(
            "ANKLE_DECISION_TRAIN_NONDOMINANT_RESCUE_SCALE",
            1.0,
        )
        self.train_axial_entropy_scale_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_SCALE_THRESHOLD",
            0.0,
        )
        self.train_axial_entropy_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_ENTROPY_THRESHOLD",
            0.0,
        )
        self.train_axial_top12_margin_scale_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_SCALE_THRESHOLD",
            0.0,
        )
        self.train_axial_top12_margin_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_TOP12_MARGIN_THRESHOLD",
            0.0,
        )
        self.train_axial_dominance_threshold = _env_unit_float(
            "ANKLE_DECISION_TRAIN_AXIAL_DOMINANCE_THRESHOLD",
            0.0,
        )
        self.forced_active_view_mask = _env_view_mask(
            "ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK"
        )
        if self.train_axial_blur_kernel % 2 == 0:
            raise ValueError("ANKLE_DECISION_TRAIN_AXIAL_BLUR_KERNEL must be odd.")
        self.fusion_temperature = _env_positive_float(
            "ANKLE_LEARNED_FUSION_TEMPERATURE",
            LEARNED_FUSION_TEMPERATURE,
        )
        if self.fusion_weight_floor * 3.0 >= 1.0:
            raise ValueError(
                "ANKLE_DECISION_FUSION_WEIGHT_FLOOR must keep positive residual mass for 3 views."
            )
        if (
            self.train_axial_teacher_misalignment_scale_threshold > 0.0
            and self.train_axial_teacher_misalignment_excess_scale_window > 0.0
        ):
            raise ValueError(
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_SCALE_THRESHOLD and "
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW "
                "are mutually exclusive."
            )
        if (
            self.train_axial_teacher_misalignment_excess_scale_window > 0.0
            and self.train_axial_teacher_misalignment_threshold <= 0.0
        ):
            raise ValueError(
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW "
                "requires ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD."
            )
        if self.enable_candidate_view_prototype_aux_loss and not self.enable_candidate_view_gate:
            raise ValueError(
                "ANKLE_DECISION_ENABLE_CANDIDATE_VIEW_PROTOTYPE_AUX_LOSS requires "
                "ANKLE_DECISION_ENABLE_CANDIDATE_VIEW_GATE."
            )
        enabled_aux_modes = [
            self.enable_aux_view_loss,
            self.enable_candidate_view_prototype_aux_loss,
            self.enable_gate_view_correctness_aux_loss,
            self.enable_pairwise_selector_aux_loss,
            self.enable_gate_pairwise_contrast_aux_loss,
            self.enable_gate_label_evidence_rank_aux_loss,
            self.enable_gate_targeted_evidence_rank_aux_loss,
            self.enable_gate_confirmed_target_rank_margin_aux_loss,
            self.enable_axial_sagittal_rank_aux_loss,
            self.enable_fp_risk_axial_sagittal_rank_aux_loss,
            self.enable_axial_fp_risk_normal_aux_loss,
            self.enable_coronal_pair_abnormal_aux_loss,
            self.enable_nonaxial_abnormal_aux_loss,
            self.enable_sagittal_normal_rescue_aux_loss,
        ]
        if sum(bool(flag) for flag in enabled_aux_modes) > 1:
            raise ValueError(
                "ANKLE_DECISION_ENABLE_AUX_VIEW_LOSS, "
                "ANKLE_DECISION_ENABLE_CANDIDATE_VIEW_PROTOTYPE_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_GATE_VIEW_CORRECTNESS_AUX_LOSS, "
                "ANKLE_DECISION_ENABLE_PAIRWISE_SELECTOR_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_GATE_PAIRWISE_CONTRAST_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_GATE_LABEL_EVIDENCE_RANK_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_GATE_TARGETED_EVIDENCE_RANK_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_GATE_CONFIRMED_TARGET_RANK_MARGIN_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_AXIAL_SAGITTAL_RANK_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_FP_RISK_AXIAL_SAGITTAL_RANK_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_AXIAL_FP_RISK_NORMAL_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_CORONAL_PAIR_ABNORMAL_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_NONAXIAL_ABNORMAL_AUX_LOSS, and "
                "ANKLE_DECISION_ENABLE_SAGITTAL_NORMAL_RESCUE_AUX_LOSS are mutually exclusive."
            )
        if self.minimal_fusion_baseline and self.equal_weight_fusion:
            raise ValueError(
                "minimal_fusion_baseline and equal_weight_fusion are mutually exclusive."
            )
        if minimal_fusion_baseline:
            # 纯融合对比模式：每个视角只保留最基础分类器和 raw-logit reliability head。
            self.view_classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.feature_dim, fusion_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(fusion_hidden_dim, 2),
                    )
                    for _ in range(3)
                ]
            )
            self.confidence_heads = nn.ModuleList(
                [nn.Linear(self.feature_dim, 1) for _ in range(3)]
            )
        else:
            # 每个视角有一个独立的分类器（共 3 个）
            # LayerNorm 在分类器前面稳定特征分布，减少不同 seed 之间的输出尺度差异
            self.view_classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.feature_dim),                  # 稳定特征分布
                        nn.Linear(self.feature_dim, fusion_hidden_dim),  # 512 -> 256
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(fusion_hidden_dim, 2),  # 256 -> 2（正常/异常）
                    )
                    for _ in range(3)  # 创建 3 个分类器
                ]
            )
            if self.enable_classifier_view_context:
                self.classifier_view_recalibrators = nn.ModuleList(
                    [ViewFeatureRecalibration(self.feature_dim) for _ in range(3)]
                )
                from .cross_view_attention import CrossViewAttention

                self.classifier_cross_view_mixer = CrossViewAttention(
                    feature_dim=self.feature_dim,
                    attention_dim=256,
                    num_heads=4,
                    num_layers=1,
                    dropout=0.1,
                    residual_scale=0.125,
                )
            if not self.equal_weight_fusion:
                # 默认 learned path：先做轻量 cross-view token interaction，再叠加共享的
                # 低秩 residual calibrator。需要做单模块 ablation 时，可通过环境变量
                # 关闭 mixer / calibrator，或直接覆写 softmax temperature。
                if not self.disable_fusion_cross_view_mixer:
                    from .cross_view_attention import CrossViewAttention

                    self.cross_view_mixer = CrossViewAttention(
                        feature_dim=self.feature_dim,
                        num_heads=4,
                        num_layers=1,
                        dropout=0.1,
                        residual_scale=0.125,
                    )
                def make_confidence_head() -> nn.Sequential:
                    return nn.Sequential(
                        nn.LayerNorm(self.feature_dim),
                        nn.Linear(self.feature_dim, 1),  # 512 → 1 reliability logit
                    )

                if self.use_view_role_confidence_head:
                    self.view_role_confidence_head = ViewRoleConfidenceScorer(
                        feature_dim=self.feature_dim,
                        num_views=3,
                        role_dim=self.view_role_confidence_role_dim,
                        hidden_dim=self.view_role_confidence_hidden_dim,
                        dropout=self.view_role_confidence_dropout,
                        logit_limit=self.view_role_confidence_logit_limit,
                    )
                needs_base_confidence_head = (
                    not self.use_view_role_confidence_head
                    or self.view_role_confidence_blend < 1.0
                    or self.enable_conditional_view_role_residual
                    or self.decouple_role_gate_classifier_gradient
                )
                if needs_base_confidence_head:
                    if self.use_shared_confidence_head:
                        self.shared_confidence_head = make_confidence_head()
                    else:
                        self.confidence_heads = nn.ModuleList(
                            [make_confidence_head() for _ in range(3)]
                        )
                if not self.disable_fusion_calibrator:
                    self.confidence_calibrator = SharedLowRankReliabilityCalibrator(
                        feature_dim=self.feature_dim,
                        bottleneck_dim=32,
                        scale_limit=0.25,
                        bias_limit=0.15,
                    )
                if self.enable_evidence_aware_gate:
                    self.evidence_aware_gate = EvidenceAwareReliabilityGate(
                        feature_dim=self.feature_dim,
                        residual_limit=self.evidence_aware_gate_residual_limit,
                    )
                if self.enable_candidate_view_gate:
                    self.candidate_view_gate = CandidateViewReliabilityGate(
                        feature_dim=self.feature_dim,
                        residual_limit=self.candidate_view_gate_residual_limit,
                        margin_gap=self.candidate_view_gate_margin_gap,
                        margin_window=self.candidate_view_gate_margin_window,
                        require_class_consensus=self.candidate_view_gate_require_class_consensus,
                        consensus_margin=self.candidate_view_gate_consensus_margin,
                        consensus_window=self.candidate_view_gate_consensus_window,
                        prob_advantage=self.candidate_view_gate_prob_advantage,
                        prob_window=self.candidate_view_gate_prob_window,
                        prototype_advantage=self.candidate_view_gate_prototype_advantage,
                        prototype_window=self.candidate_view_gate_prototype_window,
                        require_nonaxial_pair_consensus=(
                            self.candidate_view_gate_require_nonaxial_pair_consensus
                        ),
                        nonaxial_pair_consensus_margin=(
                            self.candidate_view_gate_nonaxial_pair_consensus_margin
                        ),
                        nonaxial_pair_consensus_window=(
                            self.candidate_view_gate_nonaxial_pair_consensus_window
                        ),
                        detach_features=self.candidate_view_gate_detach_features,
                    )
                if self.enable_gate_logit_rms_limit:
                    self.gate_logit_rms_limiter = GateLogitRMSLimiter(
                        max_centered_rms=self.gate_logit_max_centered_rms,
                    )
                if self.enable_sagittal_normal_rescue_gate:
                    self.sagittal_normal_rescue_gate = SagittalNormalRescueGate(
                        residual_limit=self.sagittal_normal_rescue_gate_residual_limit,
                        axial_abnormal_threshold=(
                            self.sagittal_normal_rescue_gate_axial_abnormal_threshold
                        ),
                        sagittal_normal_threshold=(
                            self.sagittal_normal_rescue_gate_sagittal_normal_threshold
                        ),
                        fused_abnormal_threshold=(
                            self.sagittal_normal_rescue_gate_fused_abnormal_threshold
                        ),
                        window=self.sagittal_normal_rescue_gate_window,
                    )
                if self.enable_posthoc_fp_risk_sagittal_gate:
                    self.posthoc_fp_risk_sagittal_gate = PosthocFPRiskSagittalGate(
                        residual=self.posthoc_fp_risk_sagittal_gate_residual,
                        axial_abnormal_min=(
                            self.posthoc_fp_risk_sagittal_gate_axial_abnormal_min
                        ),
                        axial_abnormal_max=(
                            self.posthoc_fp_risk_sagittal_gate_axial_abnormal_max
                        ),
                        sagittal_normal_min=(
                            self.posthoc_fp_risk_sagittal_gate_sagittal_normal_min
                        ),
                        coronal_abnormal_max=(
                            self.posthoc_fp_risk_sagittal_gate_coronal_abnormal_max
                        ),
                        fused_abnormal_min=(
                            self.posthoc_fp_risk_sagittal_gate_fused_abnormal_min
                        ),
                        confidence_gap_max=(
                            self.posthoc_fp_risk_sagittal_gate_confidence_gap_max
                        ),
                    )
                if self.enable_posthoc_fn_abnormal_rescue_gate:
                    self.posthoc_fn_abnormal_rescue_gate = PosthocFNAbnormalRescueGate(
                        residual=self.posthoc_fn_abnormal_rescue_gate_residual,
                        fused_abnormal_max=(
                            self.posthoc_fn_abnormal_rescue_gate_fused_abnormal_max
                        ),
                        axial_abnormal_min=(
                            self.posthoc_fn_abnormal_rescue_gate_axial_abnormal_min
                        ),
                        max_abnormal_min=(
                            self.posthoc_fn_abnormal_rescue_gate_max_abnormal_min
                        ),
                        second_abnormal_min=(
                            self.posthoc_fn_abnormal_rescue_gate_second_abnormal_min
                        ),
                        confidence_gap_max=(
                            self.posthoc_fn_abnormal_rescue_gate_confidence_gap_max
                        ),
                        axial_abnormal_max=(
                            self.posthoc_fn_abnormal_rescue_gate_axial_abnormal_max
                        ),
                    )
                if self.enable_sagittal_reliability_calibrator:
                    self.sagittal_reliability_calibrator = SagittalReliabilityCalibrator(
                        hidden_dim=self.sagittal_reliability_calibrator_hidden_dim,
                        residual_limit=(
                            self.sagittal_reliability_calibrator_residual_limit
                        ),
                        dropout=self.sagittal_reliability_calibrator_dropout,
                        coronal_suppression=(
                            self.sagittal_reliability_calibrator_coronal_suppression
                        ),
                    )
                if self.enable_gate_view_prior_debias:
                    self.gate_view_prior_debiaser = GateViewPriorDebiaser(
                        strength=self.gate_view_prior_debias_strength,
                        momentum=self.gate_view_prior_debias_momentum,
                        max_strength=self.gate_view_prior_debias_max_strength,
                        evidence_close_gap=self.gate_view_prior_debias_evidence_close_gap,
                        evidence_close_window=self.gate_view_prior_debias_evidence_close_window,
                        positive_safe=self.gate_view_prior_debias_positive_safe,
                        positive_safe_abnormal_gap=self.gate_view_prior_debias_positive_safe_abnormal_gap,
                        positive_safe_fused_floor=self.gate_view_prior_debias_positive_safe_fused_floor,
                        positive_safe_window=self.gate_view_prior_debias_positive_safe_window,
                    )
                if self.enable_relative_view_gate:
                    self.relative_view_gate = RelativeViewReliabilityGate(
                        feature_dim=self.feature_dim,
                        residual_limit=self.relative_view_gate_residual_limit,
                    )
                if self.enable_gate_transformer_context:
                    self.gate_transformer_contextualizer = GateTransformerContextualizer(
                        feature_dim=self.feature_dim,
                        attention_dim=self.gate_transformer_attention_dim,
                        num_heads=self.gate_transformer_num_heads,
                        num_layers=self.gate_transformer_num_layers,
                        dropout=self.gate_transformer_dropout,
                        residual_scale=self.gate_transformer_residual_scale,
                    )
                if self.enable_pairwise_reliability_gate:
                    self.pairwise_reliability_gate = PairwiseReliabilityGate(
                        feature_dim=self.feature_dim,
                        pair_dim=self.pairwise_reliability_pair_dim,
                        bottleneck_dim=self.pairwise_reliability_bottleneck_dim,
                        residual_limit=self.pairwise_reliability_residual_limit,
                        dropout=self.pairwise_reliability_dropout,
                    )
                if self.enable_evidence_margin_residual_fusion:
                    self.evidence_margin_residual_fusion = EvidenceMarginResidualFusion(
                        num_views=3,
                        hidden_dim=self.evidence_margin_residual_hidden_dim,
                        residual_limit=self.evidence_margin_residual_limit,
                        dropout=self.evidence_margin_residual_dropout,
                    )
                if self.enable_positive_evidence_floor_fusion:
                    self.positive_evidence_floor_fusion = PositiveEvidenceFloorFusion(
                        num_views=3,
                        hidden_dim=self.positive_evidence_floor_hidden_dim,
                        residual_limit=self.positive_evidence_floor_residual_limit,
                        dropout=self.positive_evidence_floor_dropout,
                        evidence_threshold=self.positive_evidence_floor_evidence_threshold,
                        support_threshold=self.positive_evidence_floor_support_threshold,
                        fused_ceiling=self.positive_evidence_floor_fused_ceiling,
                    )
        if self.enable_view_logit_temperature:
            self.view_logit_temperature_calibrator = PerViewLogitTemperatureCalibrator(
                num_views=3,
                min_temperature=0.5,
                max_temperature=2.0,
            )

    def _maybe_calibrate_view_logits(self, view_logits: torch.Tensor) -> torch.Tensor:
        if not self.enable_view_logit_temperature:
            return view_logits
        return self.view_logit_temperature_calibrator(view_logits)

    def _compute_gate_teacher_weights(self, view_logits: torch.Tensor) -> torch.Tensor:
        teacher_scores = view_logits.detach().amax(dim=-1) - view_logits.detach().amin(dim=-1)
        teacher_weights = torch.softmax(
            teacher_scores / self.gate_teacher_temperature,
            dim=1,
        )
        return teacher_weights.unsqueeze(-1)

    def _apply_conditional_view_role_residual(
        self,
        base_confidences: torch.Tensor,
        role_confidences: torch.Tensor,
        view_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Use role-aware routing only on FP-risk or low-gap cases, while protecting axial-positive evidence."""
        if (
            not self.enable_conditional_view_role_residual
            or role_confidences is None
            or self.conditional_view_role_residual_scale <= 0.0
        ):
            return base_confidences

        base_scores = base_confidences.squeeze(-1)
        sorted_scores = torch.sort(base_scores.detach(), dim=1, descending=True).values
        top12_gap = (
            sorted_scores[:, 0] - sorted_scores[:, 1]
            if sorted_scores.shape[1] > 1
            else sorted_scores[:, 0].new_zeros(sorted_scores.shape[0])
        )
        low_gap = top12_gap <= self.conditional_view_role_residual_gate_gap

        view_probs = torch.softmax(view_logits.detach(), dim=-1)[..., 1]
        base_weights = torch.softmax(base_scores.detach() / self.fusion_temperature, dim=1)
        fused_abnormal_prob = (base_weights * view_probs).sum(dim=1)
        axial_prob = view_probs[:, 0]
        coronal_prob = view_probs[:, 1] if view_probs.shape[1] > 1 else axial_prob
        fp_risk = (
            fused_abnormal_prob >= self.conditional_view_role_residual_fp_prob_threshold
        ) & (
            axial_prob >= self.conditional_view_role_residual_axial_high_prob
        ) & (
            coronal_prob <= self.conditional_view_role_residual_coronal_low_prob
        )
        positive_guard = axial_prob >= self.conditional_view_role_residual_positive_floor
        apply_mask = (low_gap | fp_risk) & (~positive_guard | fp_risk)
        apply_mask = apply_mask.to(dtype=base_confidences.dtype, device=base_confidences.device)
        residual = (role_confidences - base_confidences).clamp(
            min=-self.view_role_confidence_logit_limit,
            max=self.view_role_confidence_logit_limit,
        )
        return base_confidences + (
            self.conditional_view_role_residual_scale
            * apply_mask.view(-1, 1, 1)
            * residual
        )

    def _compute_classifier_features(
        self,
        view_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Optionally contextualize per-view classifier features while keeping late fusion."""
        if self.minimal_fusion_baseline or not self.enable_classifier_view_context:
            return view_features
        recalibrated_features = [
            recalibrator(feature)
            for recalibrator, feature in zip(self.classifier_view_recalibrators, view_features)
        ]
        stacked_features = torch.stack(recalibrated_features, dim=1)
        contextualized = self.classifier_cross_view_mixer(stacked_features)
        return list(contextualized.unbind(dim=1))

    def _apply_train_view_robustness(self, images: torch.Tensor) -> torch.Tensor:
        """Apply lightweight train-time view corruptions without changing late-fusion semantics."""
        if not self.training:
            return images
        if self.train_view_dropout_prob <= 0.0 and self.train_axial_blur_prob <= 0.0:
            return images

        augmented = images.clone()
        batch_size, num_views, num_slices, height, width = augmented.shape

        if self.train_view_dropout_prob > 0.0:
            drop_mask = torch.rand(batch_size, device=augmented.device) < self.train_view_dropout_prob
            if torch.any(drop_mask):
                view_indices = torch.randint(
                    low=0,
                    high=num_views,
                    size=(batch_size,),
                    device=augmented.device,
                )
                sample_indices = drop_mask.nonzero(as_tuple=False).squeeze(1)
                augmented[sample_indices, view_indices[sample_indices]] = 0.0

        if self.train_axial_blur_prob > 0.0:
            blur_mask = torch.rand(batch_size, device=augmented.device) < self.train_axial_blur_prob
            if torch.any(blur_mask):
                sample_indices = blur_mask.nonzero(as_tuple=False).squeeze(1)
                axial_tensor = augmented[sample_indices, 0].reshape(-1, 1, height, width)
                blurred_axial = nn.functional.avg_pool2d(
                    axial_tensor,
                    kernel_size=self.train_axial_blur_kernel,
                    stride=1,
                    padding=self.train_axial_blur_kernel // 2,
                )
                augmented[sample_indices, 0] = blurred_axial.reshape(
                    sample_indices.shape[0],
                    num_slices,
                    height,
                    width,
                )

        return augmented

    def _apply_train_dominant_gate_dropout(
        self,
        confidences: torch.Tensor,
        drop_mask: torch.Tensor | None = None,
        drop_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """During training, flatten the dominant gate on some samples so weak views receive fusion gradients."""
        if not self.training or self.train_dominant_gate_dropout_prob <= 0.0:
            return confidences

        batch_size, num_views, _ = confidences.shape
        if drop_mask is None:
            drop_mask = self._sample_train_dominant_gate_dropout_mask(
                batch_size,
                confidences.device,
            )
        if not torch.any(drop_mask):
            return confidences
        if drop_strength is None:
            drop_strength = drop_mask.to(dtype=confidences.dtype, device=confidences.device)
        elif drop_strength.ndim != 1 or drop_strength.shape[0] != batch_size:
            raise ValueError("drop_strength must be a (batch_size,) tensor.")
        if not torch.any(drop_strength.gt(0.0)):
            return confidences

        flat_confidences = confidences.squeeze(-1)
        dominant_view = flat_confidences.detach().argmax(dim=1)
        dominant_mask = nn.functional.one_hot(
            dominant_view,
            num_classes=num_views,
        ).to(dtype=flat_confidences.dtype, device=flat_confidences.device)
        dominant_confidence = (flat_confidences * dominant_mask).sum(dim=1, keepdim=True)
        replacement = (flat_confidences.sum(dim=1, keepdim=True) - dominant_confidence) / max(num_views - 1, 1)
        dropped_confidences = (
            flat_confidences * (1.0 - dominant_mask)
            + replacement * dominant_mask
        )
        apply_mask = drop_strength.to(dtype=flat_confidences.dtype).unsqueeze(1)
        adjusted = flat_confidences * (1.0 - apply_mask) + dropped_confidences * apply_mask
        return adjusted.unsqueeze(-1)

    def _sample_train_dominant_gate_dropout_mask(
        self,
        batch_size: int,
        device: torch.device,
        fusion_weights: torch.Tensor | None = None,
        teacher_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.training or self.train_dominant_gate_dropout_prob <= 0.0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        sample_probs = torch.full(
            (batch_size,),
            self.train_dominant_gate_dropout_prob,
            device=device,
        )
        if (
            self.train_axial_teacher_misalignment_scale_threshold <= 0.0
            and self.train_axial_teacher_misalignment_threshold <= 0.0
            and self.train_axial_teacher_misalignment_dropout_boost <= 0.0
            and self.train_teacher_non_axial_competitive_gap <= 0.0
            and not self.train_require_teacher_non_axial_top
        ):
            return torch.rand(batch_size, device=device) < sample_probs
        if fusion_weights is None or teacher_weights is None:
            raise ValueError(
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_THRESHOLD / "
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_SCALE_THRESHOLD / "
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_DROPOUT_BOOST / "
                "ANKLE_DECISION_TRAIN_TEACHER_NONAXIAL_COMPETITIVE_GAP / "
                "ANKLE_DECISION_TRAIN_REQUIRE_TEACHER_NONAXIAL_TOP "
                "requires gate and teacher weights."
            )

        flat_gate_weights = fusion_weights.detach().squeeze(-1)
        flat_teacher_weights = teacher_weights.detach().squeeze(-1)
        dominant_view = flat_gate_weights.argmax(dim=1)
        teacher_top_view = flat_teacher_weights.argmax(dim=1)
        strongest_non_axial_teacher = flat_teacher_weights[:, 1:].amax(dim=1)
        teacher_axial_non_axial_gap = (
            flat_teacher_weights[:, 0] - strongest_non_axial_teacher
        )
        axial_misalignment = (
            flat_gate_weights[:, 0] - flat_teacher_weights[:, 0]
        ).clamp_min(0.0)
        if self.train_axial_teacher_misalignment_dropout_boost > 0.0:
            # Preserve the base DFR-25 random dropout coverage, but add extra
            # correction on samples where gate routing over-trusts axial while
            # the detached view-logit teacher still sees non-axial evidence.
            boost_mask = dominant_view.eq(0)
            if self.train_require_teacher_non_axial_top:
                boost_mask = boost_mask & teacher_top_view.ne(0)
            if self.train_teacher_non_axial_competitive_gap > 0.0:
                boost_mask = boost_mask & teacher_axial_non_axial_gap.le(
                    self.train_teacher_non_axial_competitive_gap
                )
            if self.train_axial_teacher_misalignment_threshold > 0.0:
                boost_mask = boost_mask & axial_misalignment.ge(
                    self.train_axial_teacher_misalignment_threshold
                )
            if self.train_axial_teacher_misalignment_scale_threshold > 0.0:
                boost_values = (
                    axial_misalignment
                    / max(self.train_axial_teacher_misalignment_scale_threshold, 1e-6)
                ).clamp(0.0, 1.0)
            else:
                boost_values = torch.ones_like(sample_probs)
            boost_values = boost_values * boost_mask.to(dtype=sample_probs.dtype)
            sample_probs = sample_probs + (
                self.train_axial_teacher_misalignment_dropout_boost * boost_values
            )
            return torch.rand(batch_size, device=device) < sample_probs.clamp(0.0, 1.0)

        sample_probs = sample_probs * dominant_view.eq(0).to(dtype=sample_probs.dtype)
        if self.train_require_teacher_non_axial_top:
            sample_probs = sample_probs * teacher_top_view.ne(0).to(dtype=sample_probs.dtype)
        if self.train_teacher_non_axial_competitive_gap > 0.0:
            # Keep the DFR-37 hard mismatch base, but only intervene when the
            # detached teacher still sees at least one non-axial view as
            # genuinely competitive with axial rather than requiring a full
            # top-view flip like DFR-40.
            sample_probs = sample_probs * teacher_axial_non_axial_gap.le(
                self.train_teacher_non_axial_competitive_gap
            ).to(dtype=sample_probs.dtype)
        if self.train_axial_teacher_misalignment_scale_threshold > 0.0:
            # Ramp dropout coverage smoothly from aligned cases up to the old
            # hard-threshold mismatch point so near-mismatch samples get some
            # correction without fully widening the intervention again.
            misalignment_scale = (
                axial_misalignment
                / max(self.train_axial_teacher_misalignment_scale_threshold, 1e-6)
            ).clamp(0.0, 1.0)
            sample_probs = sample_probs * misalignment_scale.to(dtype=sample_probs.dtype)
        elif self.train_axial_teacher_misalignment_threshold > 0.0:
            sample_probs = sample_probs * axial_misalignment.ge(
                self.train_axial_teacher_misalignment_threshold
            ).to(dtype=sample_probs.dtype)
        return torch.rand(batch_size, device=device) < sample_probs.clamp(0.0, 1.0)

    def _compute_train_dominant_gate_dropout_strength(
        self,
        drop_mask: torch.Tensor,
        fusion_weights: torch.Tensor | None = None,
        teacher_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.training or self.train_dominant_gate_dropout_prob <= 0.0:
            return torch.zeros_like(drop_mask, dtype=torch.float32)
        drop_strength = drop_mask.to(dtype=torch.float32)
        if self.train_axial_teacher_misalignment_excess_scale_window <= 0.0:
            return drop_strength
        if fusion_weights is None or teacher_weights is None:
            raise ValueError(
                "ANKLE_DECISION_TRAIN_AXIAL_TEACHER_MISALIGNMENT_EXCESS_SCALE_WINDOW "
                "requires gate and teacher weights."
            )

        flat_gate_weights = fusion_weights.detach().squeeze(-1)
        flat_teacher_weights = teacher_weights.detach().squeeze(-1)
        dominant_view = flat_gate_weights.argmax(dim=1)
        axial_misalignment = (
            flat_gate_weights[:, 0] - flat_teacher_weights[:, 0]
        ).clamp_min(0.0)
        misalignment_excess = (
            axial_misalignment - self.train_axial_teacher_misalignment_threshold
        ) / max(self.train_axial_teacher_misalignment_excess_scale_window, 1e-6)
        strength_scale = misalignment_excess.clamp(0.0, 1.0).to(dtype=drop_strength.dtype)
        strength_scale = strength_scale * dominant_view.eq(0).to(dtype=drop_strength.dtype)
        return drop_strength * strength_scale

    def _apply_train_teacher_targeted_dominant_gate_dropout_redistribution(
        self,
        fusion_weights: torch.Tensor,
        target_weights: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Optionally redirect dropped dominant gate mass toward the teacher-selected fallback view."""
        if (
            not self.training
            or not self.train_target_teacher_dropout_redistribution
            or self.train_dominant_gate_dropout_prob <= 0.0
        ):
            return fusion_weights

        batch_size, num_views, _ = fusion_weights.shape
        if num_views <= 1:
            return fusion_weights
        if drop_mask is None:
            drop_mask = self._sample_train_dominant_gate_dropout_mask(
                batch_size,
                fusion_weights.device,
            )
        if not torch.any(drop_mask):
            return fusion_weights

        flat_weights = fusion_weights.squeeze(-1)
        dominant_view = flat_weights.detach().argmax(dim=1)
        dominant_mask = nn.functional.one_hot(
            dominant_view,
            num_classes=num_views,
        ).to(dtype=flat_weights.dtype, device=flat_weights.device)
        gate_fallback_view = flat_weights.detach().masked_fill(
            dominant_mask.bool(),
            -1.0,
        ).argmax(dim=1)
        target_scores = flat_weights.detach()
        if target_weights is not None:
            target_scores = target_weights.detach().squeeze(-1)
        target_fallback_view = target_scores.masked_fill(
            dominant_mask.bool(),
            -1.0,
        ).argmax(dim=1)
        fallback_mask = nn.functional.one_hot(
            target_fallback_view,
            num_classes=num_views,
        ).to(dtype=flat_weights.dtype, device=flat_weights.device)

        dominant_weight = (flat_weights * dominant_mask).sum(dim=1, keepdim=True)
        non_dominant_mean = (
            (flat_weights * (1.0 - dominant_mask)).sum(dim=1, keepdim=True)
            / max(num_views - 1, 1)
        )
        redistributed_mass = (dominant_weight - non_dominant_mean).clamp_min(0.0)
        flattened = flat_weights * (1.0 - dominant_mask)
        flattened = flattened + dominant_mask * non_dominant_mean
        flattened = flattened / flattened.sum(dim=1, keepdim=True).clamp_min(1e-12)

        targeted = flattened + fallback_mask * redistributed_mass
        targeted = targeted / targeted.sum(dim=1, keepdim=True).clamp_min(1e-12)

        target_mask = drop_mask
        if self.train_target_teacher_dropout_only_on_fallback_disagreement:
            # Keep the base DFR-25/37 flattening on mismatch samples where gate
            # and teacher already agree on the fallback view; only redirect the
            # dropped dominant mass when the fallback route itself is misaligned.
            target_mask = drop_mask & target_fallback_view.ne(gate_fallback_view)
        target_apply_mask = target_mask.to(dtype=flat_weights.dtype).unsqueeze(1)
        flatten_apply_mask = (
            (drop_mask & ~target_mask).to(dtype=flat_weights.dtype).unsqueeze(1)
        )
        base_apply_mask = drop_mask.to(dtype=flat_weights.dtype).unsqueeze(1)
        blended = flat_weights * (1.0 - base_apply_mask)
        blended = blended + flattened * flatten_apply_mask + targeted * target_apply_mask
        return blended.unsqueeze(-1)

    def _apply_train_non_dominant_weight_floor(
        self,
        fusion_weights: torch.Tensor,
        rescue_target_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """During training, reserve a small mass for weak views so their experts keep learning."""
        if (
            not self.training
            or self.train_non_dominant_weight_floor <= 0.0
            or self.train_non_dominant_rescue_scale <= 0.0
        ):
            return fusion_weights

        batch_size, num_views, _ = fusion_weights.shape
        if num_views <= 1:
            return fusion_weights

        reserved_mass = self.train_non_dominant_weight_floor * (num_views - 1)
        if reserved_mass >= 1.0:
            raise ValueError(
                "ANKLE_DECISION_TRAIN_NONDOMINANT_WEIGHT_FLOOR leaves no residual mass for the dominant view."
            )

        flat_weights = fusion_weights.squeeze(-1)
        dominant_view = flat_weights.detach().argmax(dim=1)
        dominant_mask = nn.functional.one_hot(
            dominant_view,
            num_classes=num_views,
        ).to(dtype=flat_weights.dtype, device=flat_weights.device)
        dominant_weight = (flat_weights * dominant_mask).sum(dim=1, keepdim=True)
        detached_weights = flat_weights.detach().clamp_min(1e-12)
        if self.train_axial_entropy_scale_threshold > 0.0:
            # Scale rescue with how far the whole gate distribution has collapsed
            # below the entropy threshold so near-collapse cases still feed weak views.
            normalized_entropy = -(
                detached_weights * detached_weights.log()
            ).sum(dim=1) / math.log(float(num_views))
            entropy_deficit = (
                self.train_axial_entropy_scale_threshold - normalized_entropy
            ) / max(self.train_axial_entropy_scale_threshold, 1e-6)
            apply_mask = (
                dominant_view.eq(0).to(dtype=flat_weights.dtype, device=flat_weights.device)
                * entropy_deficit.clamp(0.0, 1.0).to(
                    dtype=flat_weights.dtype,
                    device=flat_weights.device,
                )
            ).unsqueeze(1)
            if not torch.any(apply_mask.gt(0.0)):
                return fusion_weights
        elif self.train_axial_entropy_threshold > 0.0:
            # Detect collapse from the whole gate distribution so we only rescue
            # truly low-entropy axial lock-in, not healthy axial-sagittal sharing.
            normalized_entropy = -(
                detached_weights * detached_weights.log()
            ).sum(dim=1) / math.log(float(num_views))
            apply_mask = (
                dominant_view.eq(0).to(dtype=flat_weights.dtype, device=flat_weights.device)
                * normalized_entropy.le(self.train_axial_entropy_threshold).to(
                    dtype=flat_weights.dtype,
                    device=flat_weights.device,
                )
            ).unsqueeze(1)
            if not torch.any(apply_mask.gt(0.0)):
                return fusion_weights
        elif self.train_axial_top12_margin_scale_threshold > 0.0:
            # Scale rescue with the axial top1-top2 gate gap so near-collapse
            # samples still feed weak-view experts without flattening healthy routing.
            top2_weights = flat_weights.detach().topk(k=min(2, num_views), dim=1).values
            if top2_weights.shape[1] < 2:
                return fusion_weights
            top12_margin = top2_weights[:, 0] - top2_weights[:, 1]
            margin_window = max(
                1.0 - self.train_axial_top12_margin_scale_threshold,
                1e-6,
            )
            margin_excess = (
                top12_margin - self.train_axial_top12_margin_scale_threshold
            ) / margin_window
            apply_mask = (
                dominant_view.eq(0).to(dtype=flat_weights.dtype, device=flat_weights.device)
                * margin_excess.clamp(0.0, 1.0).to(
                    dtype=flat_weights.dtype,
                    device=flat_weights.device,
                )
            ).unsqueeze(1)
            if not torch.any(apply_mask.gt(0.0)):
                return fusion_weights
        elif self.train_axial_top12_margin_threshold > 0.0:
            # Use the top1-top2 gate gap as a hard collapse detector so rescue
            # only fires when axial fully separates from the fallback view.
            top2_weights = flat_weights.detach().topk(k=min(2, num_views), dim=1).values
            if top2_weights.shape[1] < 2:
                return fusion_weights
            top12_margin = top2_weights[:, 0] - top2_weights[:, 1]
            apply_mask = (
                dominant_view.eq(0).to(dtype=flat_weights.dtype, device=flat_weights.device)
                * top12_margin.ge(self.train_axial_top12_margin_threshold).to(
                    dtype=flat_weights.dtype,
                    device=flat_weights.device,
                )
            ).unsqueeze(1)
            if not torch.any(apply_mask.gt(0.0)):
                return fusion_weights
        elif self.train_axial_dominance_threshold > 0.0:
            # Scale the rescue smoothly with axial dominance excess so mildly
            # concentrated samples keep their learned routing freedom.
            dominance_window = max(1.0 - self.train_axial_dominance_threshold, 1e-6)
            dominance_excess = (
                dominant_weight.squeeze(1) - self.train_axial_dominance_threshold
            ) / dominance_window
            apply_mask = (
                dominant_view.eq(0).to(dtype=flat_weights.dtype, device=flat_weights.device)
                * dominance_excess.clamp(0.0, 1.0).to(
                    dtype=flat_weights.dtype,
                    device=flat_weights.device,
                )
            ).unsqueeze(1)
            if not torch.any(apply_mask.gt(0.0)):
                return fusion_weights
        else:
            apply_mask = torch.ones(
                batch_size,
                1,
                device=flat_weights.device,
                dtype=flat_weights.dtype,
            )
        apply_mask = apply_mask * self.train_non_dominant_rescue_scale
        non_dominant_mask = 1.0 - dominant_mask
        adjusted = flat_weights * (1.0 - reserved_mass)
        if self.train_target_teacher_non_dominant_rescue:
            # Route the full rescue mass toward the fallback view with the
            # strongest detached classifier evidence instead of trusting the
            # already-collapsed gate ranking to choose who gets the gradient.
            target_scores = flat_weights.detach()
            if rescue_target_weights is not None:
                target_scores = rescue_target_weights.detach().squeeze(-1)
            fallback_view = target_scores.masked_fill(
                dominant_mask.bool(),
                -1.0,
            ).argmax(dim=1)
            fallback_mask = nn.functional.one_hot(
                fallback_view,
                num_classes=num_views,
            ).to(dtype=flat_weights.dtype, device=flat_weights.device)
            adjusted = adjusted + fallback_mask * reserved_mass
        elif self.train_target_top_non_dominant_rescue:
            # Concentrate the same rescue mass on the strongest fallback view so
            # it can become a viable alternative instead of splitting mass
            # equally across two still-starving non-dominant experts.
            fallback_view = flat_weights.detach().masked_fill(
                dominant_mask.bool(),
                -1.0,
            ).argmax(dim=1)
            fallback_mask = nn.functional.one_hot(
                fallback_view,
                num_classes=num_views,
            ).to(dtype=flat_weights.dtype, device=flat_weights.device)
            adjusted = adjusted + fallback_mask * reserved_mass
        else:
            adjusted = adjusted + non_dominant_mask * self.train_non_dominant_weight_floor
        adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-12)
        blended = flat_weights * (1.0 - apply_mask) + adjusted * apply_mask
        return blended.unsqueeze(-1)

    def _compute_decision_outputs(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-view logits plus learned fusion weights for analysis/control runs."""
        images = self._apply_train_view_robustness(images)
        view_features = self.encode_views(images)  # 3 个 (B, 512) 的列表
        classifier_features = self._compute_classifier_features(view_features)

        view_logits = torch.stack(
            [
                classifier(feature)
                for classifier, feature in zip(self.view_classifiers, classifier_features)
            ],
            dim=1,
        )  # (B, 3, 2)
        view_logits = self._maybe_calibrate_view_logits(view_logits)

        if self.equal_weight_fusion:
            batch_size = view_logits.shape[0]
            confidences = torch.zeros(
                batch_size,
                view_logits.shape[1],
                1,
                device=view_logits.device,
                dtype=view_logits.dtype,
            )
            fusion_weights = torch.full_like(confidences, 1.0 / view_logits.shape[1])
            return {
                "view_logits": view_logits,
                "confidences": confidences,
                "scaled_confidences": confidences,
                "fusion_weights": fusion_weights,
            }

        dominant_gate_dropout_mask = torch.zeros(
            view_logits.shape[0],
            device=view_logits.device,
            dtype=torch.bool,
        )
        dominant_gate_dropout_strength = dominant_gate_dropout_mask.to(dtype=view_logits.dtype)
        teacher_fusion_weights = None
        teacher_rescue_weights = None
        role_confidences = None
        base_confidences = None
        candidate_prototype_logits = None

        if self.minimal_fusion_baseline:
            confidences = torch.stack(
                [
                    head(feature)
                    for head, feature in zip(self.confidence_heads, view_features)
                ],
                dim=1,
            )  # (B, 3, 1)
        else:
            if self.disable_fusion_cross_view_mixer:
                gating_features = view_features
            else:
                stacked_features = torch.stack(view_features, dim=1)  # (B, 3, 512)
                gating_features = list(self.cross_view_mixer(stacked_features).unbind(dim=1))
            if self.enable_gate_transformer_context:
                stacked_gate_features = torch.stack(gating_features, dim=1)
                gating_features = list(
                    self.gate_transformer_contextualizer(stacked_gate_features).unbind(dim=1)
                )
            confidence_features = gating_features
            if self.use_relative_confidence_features:
                stacked_confidence_features = torch.stack(gating_features, dim=1)
                relative_confidence_features = (
                    stacked_confidence_features
                    - stacked_confidence_features.mean(dim=1, keepdim=True)
                )
                confidence_features = list(relative_confidence_features.unbind(dim=1))
            if self.use_view_role_confidence_head:
                role_gate_features = torch.stack(gating_features, dim=1)
                if self.detach_view_role_gate_features:
                    role_gate_features = role_gate_features.detach()
                role_confidences = self.view_role_confidence_head(role_gate_features)
            if (
                role_confidences is None
                or self.view_role_confidence_blend < 1.0
                or self.enable_conditional_view_role_residual
                or self.decouple_role_gate_classifier_gradient
            ):
                calibrated_confidences = []
                if self.use_shared_confidence_head:
                    confidence_heads = [self.shared_confidence_head for _ in confidence_features]
                else:
                    confidence_heads = self.confidence_heads
                for head, feature in zip(confidence_heads, confidence_features):
                    raw_confidence = head(feature)
                    if self.disable_fusion_calibrator:
                        calibrated_confidences.append(raw_confidence)
                    else:
                        calibrated_confidences.append(
                            self.confidence_calibrator(feature, raw_confidence)
                        )
                base_confidences = torch.stack(calibrated_confidences, dim=1)  # (B, 3, 1)
            if role_confidences is None:
                confidences = base_confidences
            elif self.enable_conditional_view_role_residual:
                confidences = self._apply_conditional_view_role_residual(
                    base_confidences,
                    role_confidences,
                    view_logits,
                )
            elif self.view_role_confidence_blend >= 1.0:
                confidences = role_confidences
            else:
                blend = self.view_role_confidence_blend
                confidences = (1.0 - blend) * base_confidences + blend * role_confidences
            if self.enable_evidence_aware_gate:
                evidence_gate_features = torch.stack(gating_features, dim=1)
                confidences = confidences + self.evidence_aware_gate(
                    evidence_gate_features,
                    view_logits,
                )
            if self.enable_candidate_view_gate:
                candidate_gate_features = torch.stack(gating_features, dim=1)
                confidences = confidences + self.candidate_view_gate(
                    candidate_gate_features,
                    view_logits,
                )
                if self.enable_candidate_view_prototype_aux_loss:
                    candidate_prototype_logits = self.candidate_view_gate.prototype_logits(
                        candidate_gate_features,
                        temperature=self.candidate_view_prototype_aux_temperature,
                    )
            if self.enable_relative_view_gate:
                relative_gate_features = torch.stack(gating_features, dim=1)
                confidences = confidences + self.relative_view_gate(
                    relative_gate_features
                )
            if self.enable_pairwise_reliability_gate:
                pairwise_gate_features = torch.stack(gating_features, dim=1)
                confidences = confidences + self.pairwise_reliability_gate(
                    pairwise_gate_features
                )
            if self.enable_sagittal_normal_rescue_gate and not (
                self.training and self.sagittal_normal_rescue_gate_eval_only
            ):
                confidences = confidences + self.sagittal_normal_rescue_gate(
                    confidences,
                    view_logits,
                    self.fusion_temperature,
                )
            posthoc_base_confidences = confidences
            if self.enable_posthoc_fp_risk_sagittal_gate and not self.training:
                confidences = confidences + self.posthoc_fp_risk_sagittal_gate(
                    posthoc_base_confidences,
                    view_logits,
                    self.fusion_temperature,
                )
            if self.enable_posthoc_fn_abnormal_rescue_gate and not self.training:
                confidences = confidences + self.posthoc_fn_abnormal_rescue_gate(
                    posthoc_base_confidences,
                    view_logits,
                    self.fusion_temperature,
                )
            if self.enable_sagittal_reliability_calibrator:
                confidences = confidences + self.sagittal_reliability_calibrator(
                    confidences,
                    view_logits,
                    self.fusion_temperature,
                )
            if self.enable_gate_logit_rms_limit:
                confidences = self.gate_logit_rms_limiter(confidences)
            if self.enable_gate_view_prior_debias:
                confidences = self.gate_view_prior_debiaser(confidences, view_logits)
        scaled_confidences = confidences / self.fusion_temperature
        raw_fusion_weights = torch.softmax(scaled_confidences, dim=1)  # (B, 3, 1)
        classifier_gradient_confidences = None
        classifier_gradient_fusion_weights = None
        if (
            self.training
            and self.decouple_role_gate_classifier_gradient
            and role_confidences is not None
            and base_confidences is not None
        ):
            classifier_gradient_confidences = base_confidences
            classifier_gradient_fusion_weights = torch.softmax(
                classifier_gradient_confidences / self.fusion_temperature,
                dim=1,
            )
        apply_gate_teacher_blend = self.gate_teacher_blend > 0.0 and not (
            self.training and self.gate_teacher_blend_eval_only
        )
        if (
            apply_gate_teacher_blend
            or self.train_target_teacher_non_dominant_rescue
            or self.train_target_teacher_dropout_redistribution
            or self.train_axial_teacher_misalignment_scale_threshold > 0.0
            or self.train_axial_teacher_misalignment_threshold > 0.0
            or self.train_axial_teacher_misalignment_dropout_boost > 0.0
            or self.train_teacher_non_axial_competitive_gap > 0.0
            or self.train_require_teacher_non_axial_top
        ):
            teacher_fusion_weights = self._compute_gate_teacher_weights(view_logits)
        if self.train_dominant_gate_dropout_prob > 0.0 and not self.minimal_fusion_baseline:
            dominant_gate_dropout_mask = self._sample_train_dominant_gate_dropout_mask(
                view_logits.shape[0],
                view_logits.device,
                fusion_weights=raw_fusion_weights,
                teacher_weights=teacher_fusion_weights,
            )
            dominant_gate_dropout_strength = self._compute_train_dominant_gate_dropout_strength(
                dominant_gate_dropout_mask,
                fusion_weights=raw_fusion_weights,
                teacher_weights=teacher_fusion_weights,
            ).to(dtype=view_logits.dtype, device=view_logits.device)
            if self.train_target_teacher_dropout_redistribution:
                pass
            else:
                confidences = self._apply_train_dominant_gate_dropout(
                    confidences,
                    drop_mask=dominant_gate_dropout_mask,
                    drop_strength=dominant_gate_dropout_strength,
                )
                if classifier_gradient_confidences is not None:
                    classifier_gradient_confidences = self._apply_train_dominant_gate_dropout(
                        classifier_gradient_confidences,
                        drop_mask=dominant_gate_dropout_mask,
                        drop_strength=dominant_gate_dropout_strength,
                    )
                scaled_confidences = confidences / self.fusion_temperature
                raw_fusion_weights = torch.softmax(scaled_confidences, dim=1)
                if classifier_gradient_confidences is not None:
                    classifier_gradient_fusion_weights = torch.softmax(
                        classifier_gradient_confidences / self.fusion_temperature,
                        dim=1,
                    )

        blended_fusion_weights = raw_fusion_weights
        if apply_gate_teacher_blend:
            blended_fusion_weights = (
                (1.0 - self.gate_teacher_blend) * raw_fusion_weights
                + self.gate_teacher_blend * teacher_fusion_weights
            )
            blended_fusion_weights = blended_fusion_weights / blended_fusion_weights.sum(
                dim=1,
                keepdim=True,
            )
        blended_fusion_weights = self._apply_train_teacher_targeted_dominant_gate_dropout_redistribution(
            blended_fusion_weights,
            target_weights=teacher_fusion_weights,
            drop_mask=dominant_gate_dropout_mask,
        )
        if self.train_target_teacher_non_dominant_rescue:
            teacher_rescue_weights = teacher_fusion_weights
            if teacher_rescue_weights is None:
                teacher_rescue_weights = self._compute_gate_teacher_weights(view_logits)
        blended_fusion_weights = self._apply_train_non_dominant_weight_floor(
            blended_fusion_weights,
            rescue_target_weights=teacher_rescue_weights,
        )
        fusion_weights = self._apply_fusion_weight_floor(blended_fusion_weights)
        if classifier_gradient_fusion_weights is not None:
            classifier_gradient_fusion_weights = self._apply_fusion_weight_floor(
                classifier_gradient_fusion_weights
            )
        view_logit_temperatures = None
        if self.enable_view_logit_temperature:
            view_logit_temperatures = self.view_logit_temperature_calibrator.temperatures().detach()
        return {
            "view_logits": view_logits,
            "confidences": confidences,
            "scaled_confidences": scaled_confidences,
            "raw_fusion_weights": raw_fusion_weights,
            "teacher_fusion_weights": teacher_fusion_weights,
            "blended_fusion_weights": blended_fusion_weights,
            "fusion_weights": fusion_weights,
            "classifier_gradient_fusion_weights": classifier_gradient_fusion_weights,
            "view_logit_temperatures": view_logit_temperatures,
            "candidate_prototype_logits": candidate_prototype_logits,
        }

    def _set_aux_view_loss_state(
        self,
        view_logits: torch.Tensor,
        fusion_weights: torch.Tensor | None = None,
        candidate_prototype_logits: torch.Tensor | None = None,
    ) -> None:
        """Expose optional per-view auxiliary logits through the existing train.py hook."""
        aux_logits = None
        aux_weight = self.aux_view_loss_weight
        if self.enable_candidate_view_prototype_aux_loss:
            if candidate_prototype_logits is None:
                raise RuntimeError(
                    "Candidate prototype auxiliary loss is enabled but prototype logits "
                    "were not produced."
                )
            aux_logits = candidate_prototype_logits
            aux_weight = self.candidate_view_prototype_aux_weight
        elif self.enable_gate_view_correctness_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Gate view-correctness auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Gate view-correctness auxiliary loss expects fusion weights with "
                    "shape (batch, views, 1) matching view logits."
                )
            aux_fusion_weights = fusion_weights
            if self.gate_view_correctness_aux_require_disagreement:
                detached_predictions = view_logits.detach().argmax(dim=-1)
                disagreement_mask = detached_predictions.ne(
                    detached_predictions[:, :1]
                ).any(dim=1, keepdim=True)
                aux_fusion_weights = fusion_weights * disagreement_mask.to(
                    device=fusion_weights.device,
                    dtype=fusion_weights.dtype,
                ).unsqueeze(-1)
            if self.gate_view_correctness_aux_nonaxial_only:
                view_mask = torch.ones_like(aux_fusion_weights)
                view_mask[:, :1] = 0.0
                aux_fusion_weights = aux_fusion_weights * view_mask
            if self.gate_view_correctness_aux_protect_strong_axial:
                axial_logits = view_logits.detach()[:, 0]
                axial_margin = axial_logits.amax(dim=-1) - axial_logits.amin(dim=-1)
                weak_axial_mask = axial_margin.le(
                    self.gate_view_correctness_aux_axial_margin_cap
                )
                nonaxial_mask = torch.ones_like(aux_fusion_weights)
                nonaxial_mask[:, :1] = 0.0
                protected_nonaxial = aux_fusion_weights * nonaxial_mask
                preserved_axial = aux_fusion_weights * (1.0 - nonaxial_mask)
                aux_fusion_weights = preserved_axial + protected_nonaxial * weak_axial_mask.to(
                    device=fusion_weights.device,
                    dtype=fusion_weights.dtype,
                ).view(-1, 1, 1)
            view_scales = (
                self.gate_view_correctness_aux_base_scale
                + self.gate_view_correctness_aux_weight_scale * aux_fusion_weights
            )
            aux_logits = view_logits.detach() * view_scales
            aux_weight = self.gate_view_correctness_aux_weight
        elif self.enable_pairwise_selector_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Pairwise selector auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Pairwise selector auxiliary loss expects view logits with shape "
                    "(batch, 3, classes)."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Pairwise selector auxiliary loss expects fusion weights with "
                    "shape (batch, 3, 1) matching view logits."
                )
            flat_weights = fusion_weights.squeeze(-1)
            pair_logits = []
            for nonaxial_index in (1, 2):
                pair_indices = [0, nonaxial_index]
                pair_weights = flat_weights[:, pair_indices]
                pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                candidate_logits = (
                    view_logits.detach()[:, pair_indices, :]
                    * pair_weights.unsqueeze(-1)
                ).sum(dim=1)
                if self.pairwise_selector_aux_require_disagreement:
                    detached_predictions = view_logits.detach().argmax(dim=-1)
                    pair_disagreement = detached_predictions[:, nonaxial_index].ne(
                        detached_predictions[:, 0]
                    )
                    candidate_logits = torch.where(
                        pair_disagreement.view(-1, 1),
                        candidate_logits,
                        candidate_logits.detach(),
                    )
                pair_logits.append(candidate_logits)
            aux_logits = torch.stack(pair_logits, dim=1)
            aux_weight = self.pairwise_selector_aux_weight
        elif self.enable_gate_pairwise_contrast_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Gate pairwise contrast auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Gate pairwise contrast auxiliary loss expects view logits with shape "
                    "(batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "Gate pairwise contrast auxiliary loss requires binary logits."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Gate pairwise contrast auxiliary loss expects fusion weights with "
                    "shape (batch, 3, 1) matching view logits."
                )
            flat_weights = fusion_weights.squeeze(-1)
            detached_logits = view_logits.detach()
            detached_predictions = detached_logits.argmax(dim=-1)
            pair_logits = []
            for pair_indices in ([0, 1], [0, 2], [1, 2]):
                pair_weights = flat_weights[:, pair_indices]
                pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                candidate_logits = (
                    detached_logits[:, pair_indices, :] * pair_weights.unsqueeze(-1)
                ).sum(dim=1)
                if self.gate_pairwise_contrast_aux_require_disagreement:
                    pair_disagreement = detached_predictions[:, pair_indices[0]].ne(
                        detached_predictions[:, pair_indices[1]]
                    )
                    candidate_logits = torch.where(
                        pair_disagreement.view(-1, 1),
                        candidate_logits,
                        candidate_logits.detach(),
                    )
                pair_logits.append(candidate_logits)
            aux_logits = torch.stack(pair_logits, dim=1)
            aux_weight = self.gate_pairwise_contrast_aux_weight
        elif self.enable_gate_label_evidence_rank_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Gate label-evidence rank auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3:
                raise RuntimeError(
                    "Gate label-evidence rank auxiliary loss expects view logits with shape "
                    "(batch, views, classes)."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Gate label-evidence rank auxiliary loss expects fusion weights with "
                    "shape (batch, views, 1) matching view logits."
                )
            detached_log_probs = torch.log_softmax(view_logits.detach(), dim=-1)
            log_weights = fusion_weights.squeeze(-1).clamp_min(1e-8).log()
            mixture_log_probs = torch.logsumexp(
                log_weights.unsqueeze(-1) + detached_log_probs,
                dim=1,
            )
            aux_logits = mixture_log_probs.unsqueeze(1)
            aux_weight = self.gate_label_evidence_rank_aux_weight
        elif self.enable_gate_targeted_evidence_rank_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Gate targeted evidence-rank auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3:
                raise RuntimeError(
                    "Gate targeted evidence-rank auxiliary loss expects view logits with shape "
                    "(batch, views, classes)."
                )
            if view_logits.shape[1] < 2:
                raise RuntimeError(
                    "Gate targeted evidence-rank auxiliary loss requires at least two views."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Gate targeted evidence-rank auxiliary loss expects fusion weights with "
                    "shape (batch, views, 1) matching view logits."
                )
            detached_log_probs = torch.log_softmax(view_logits.detach(), dim=-1)
            axial_log_probs = detached_log_probs[:, :1, :]
            nonaxial_log_probs = detached_log_probs[:, 1:, :]
            best_advantage, best_relative_view = (
                nonaxial_log_probs - axial_log_probs
            ).max(dim=1)
            has_nonaxial_target = best_advantage.ge(
                self.gate_targeted_evidence_rank_aux_gap
            )
            target_view = best_relative_view + 1
            flat_weights = fusion_weights.squeeze(-1).clamp_min(1e-8)
            target_weights = flat_weights.gather(dim=1, index=target_view)
            axial_weights = flat_weights[:, :1].expand_as(target_weights)
            pair_normalizer = (target_weights + axial_weights).clamp_min(1e-8)
            target_pair_log_weights = (target_weights / pair_normalizer).clamp_min(1e-8).log()
            axial_pair_log_weights = (axial_weights / pair_normalizer).clamp_min(1e-8).log()
            aux_logits = torch.where(
                has_nonaxial_target,
                target_pair_log_weights,
                axial_pair_log_weights.detach(),
            ).unsqueeze(1)
            aux_weight = self.gate_targeted_evidence_rank_aux_weight
        elif self.enable_gate_confirmed_target_rank_margin_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Gate confirmed-target rank-margin auxiliary loss is enabled but "
                    "fusion weights were not produced."
                )
            if view_logits.ndim != 3:
                raise RuntimeError(
                    "Gate confirmed-target rank-margin auxiliary loss expects view logits "
                    "with shape (batch, views, classes)."
                )
            if view_logits.shape[1] < 2 or view_logits.shape[-1] < 2:
                raise RuntimeError(
                    "Gate confirmed-target rank-margin auxiliary loss requires at least "
                    "two views and two classes."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Gate confirmed-target rank-margin auxiliary loss expects fusion weights "
                    "with shape (batch, views, 1) matching view logits."
                )

            detached_logits = view_logits.detach()
            detached_log_probs = torch.log_softmax(detached_logits, dim=-1)
            axial_log_probs = detached_log_probs[:, :1, :]
            nonaxial_log_probs = detached_log_probs[:, 1:, :]
            target_advantages = nonaxial_log_probs - axial_log_probs

            nonaxial_logits = detached_logits[:, 1:, :]
            num_classes = nonaxial_logits.shape[-1]
            class_mask = torch.eye(
                num_classes,
                device=nonaxial_logits.device,
                dtype=torch.bool,
            ).view(1, 1, num_classes, num_classes)
            other_class_logits = nonaxial_logits.unsqueeze(-2).masked_fill(
                class_mask,
                -torch.inf,
            )
            nonaxial_class_margins = nonaxial_logits - other_class_logits.max(dim=-1).values
            target_advantages = target_advantages.masked_fill(
                nonaxial_class_margins.lt(0.0),
                -torch.inf,
            )
            best_advantage, best_relative_view = target_advantages.max(dim=1)
            has_nonaxial_target = best_advantage.ge(
                self.gate_confirmed_target_rank_margin_aux_gap
            )

            target_view = best_relative_view + 1
            flat_log_weights = fusion_weights.squeeze(-1).clamp_min(1e-8).log()
            target_log_weights = flat_log_weights.gather(dim=1, index=target_view)
            axial_log_weights = flat_log_weights[:, :1].expand_as(target_log_weights)
            rank_margin_logits = (
                target_log_weights
                - axial_log_weights
                - self.gate_confirmed_target_rank_margin_aux_margin
            )
            baseline_logits = torch.zeros_like(rank_margin_logits).detach()
            aux_logits = torch.where(
                has_nonaxial_target,
                rank_margin_logits,
                baseline_logits,
            ).unsqueeze(1)
            aux_weight = self.gate_confirmed_target_rank_margin_aux_weight
        elif self.enable_axial_sagittal_rank_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Axial-sagittal rank auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Axial-sagittal rank auxiliary loss expects view logits with shape "
                    "(batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "Axial-sagittal rank auxiliary loss requires binary logits."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Axial-sagittal rank auxiliary loss expects fusion weights with "
                    "shape (batch, 3, 1) matching view logits."
                )
            detached_predictions = view_logits.detach().argmax(dim=-1)
            rank_mask = detached_predictions[:, 0].eq(1) & detached_predictions[:, 2].eq(0)
            pair_weights = fusion_weights.squeeze(-1)[:, [0, 2]]
            pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pair_log_weights = pair_weights.clamp_min(1e-8).log()
            # Class 0 should choose sagittal-normal reliability; class 1 should preserve axial-abnormal reliability.
            rank_logits = torch.stack(
                [pair_log_weights[:, 1], pair_log_weights[:, 0]],
                dim=1,
            )
            aux_logits = torch.where(
                rank_mask.view(-1, 1),
                rank_logits,
                rank_logits.detach(),
            ).unsqueeze(1)
            aux_weight = self.axial_sagittal_rank_aux_weight
        elif self.enable_fp_risk_axial_sagittal_rank_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "FP-risk axial-sagittal rank auxiliary loss is enabled but fusion "
                    "weights were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "FP-risk axial-sagittal rank auxiliary loss expects view logits "
                    "with shape (batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "FP-risk axial-sagittal rank auxiliary loss requires binary logits."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "FP-risk axial-sagittal rank auxiliary loss expects fusion weights "
                    "with shape (batch, 3, 1) matching view logits."
                )

            detached_probs = torch.softmax(view_logits.detach(), dim=-1)
            abnormal_probs = detached_probs[..., 1]
            axial_abnormal = abnormal_probs[:, 0]
            coronal_abnormal = abnormal_probs[:, 1]
            sagittal_normal = detached_probs[:, 2, 0]
            detached_weights = fusion_weights.detach()
            detached_fused_logits = (view_logits.detach() * detached_weights).sum(dim=1)
            fused_abnormal = torch.softmax(detached_fused_logits, dim=-1)[:, 1]

            rank_mask = (
                fused_abnormal.ge(self.fp_risk_axial_sagittal_rank_aux_fused_abnormal_min)
                & axial_abnormal.ge(
                    self.fp_risk_axial_sagittal_rank_aux_axial_abnormal_min
                )
                & axial_abnormal.le(
                    self.fp_risk_axial_sagittal_rank_aux_axial_abnormal_max
                )
                & sagittal_normal.ge(
                    self.fp_risk_axial_sagittal_rank_aux_sagittal_normal_min
                )
                & coronal_abnormal.le(
                    self.fp_risk_axial_sagittal_rank_aux_coronal_abnormal_max
                )
            )
            pair_log_weights = fusion_weights.squeeze(-1)[:, [0, 2]].clamp_min(1e-8).log()
            margin = self.fp_risk_axial_sagittal_rank_aux_margin
            # class 0 promotes sagittal normal rescue; class 1 preserves axial abnormal evidence.
            rank_logits = torch.stack(
                [
                    pair_log_weights[:, 1] - pair_log_weights[:, 0] - margin,
                    pair_log_weights[:, 0] - pair_log_weights[:, 1] - margin,
                ],
                dim=1,
            )
            aux_logits = torch.where(
                rank_mask.view(-1, 1),
                rank_logits,
                rank_logits.detach(),
            ).unsqueeze(1)
            aux_weight = self.fp_risk_axial_sagittal_rank_aux_weight
        elif self.enable_axial_fp_risk_normal_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Axial FP-risk normal auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Axial FP-risk normal auxiliary loss expects view logits with "
                    "shape (batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "Axial FP-risk normal auxiliary loss requires binary logits."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Axial FP-risk normal auxiliary loss expects fusion weights with "
                    "shape (batch, 3, 1) matching view logits."
                )

            detached_probs = torch.softmax(view_logits.detach(), dim=-1)
            abnormal_probs = detached_probs[..., 1]
            axial_abnormal = abnormal_probs[:, 0]
            coronal_abnormal = abnormal_probs[:, 1]
            sagittal_normal = detached_probs[:, 2, 0]
            detached_weights = fusion_weights.detach()
            detached_fused_logits = (view_logits.detach() * detached_weights).sum(dim=1)
            fused_abnormal = torch.softmax(detached_fused_logits, dim=-1)[:, 1]

            risk_mask = (
                fused_abnormal.ge(self.axial_fp_risk_normal_aux_fused_abnormal_min)
                & axial_abnormal.ge(self.axial_fp_risk_normal_aux_axial_abnormal_min)
                & sagittal_normal.ge(self.axial_fp_risk_normal_aux_sagittal_normal_min)
                & coronal_abnormal.le(
                    self.axial_fp_risk_normal_aux_coronal_abnormal_max
                )
            )
            axial_normal_logits = view_logits[:, 0:1, 0:1]
            axial_abnormal_logits = view_logits[:, 0:1, 1:2].detach()
            active_aux_logits = torch.cat(
                [axial_normal_logits, axial_abnormal_logits],
                dim=-1,
            )
            aux_logits = torch.where(
                risk_mask.view(-1, 1, 1),
                active_aux_logits,
                active_aux_logits.detach(),
            )
            aux_weight = self.axial_fp_risk_normal_aux_weight
        elif self.enable_coronal_pair_abnormal_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Coronal pair abnormal auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Coronal pair abnormal auxiliary loss expects view logits with shape "
                    "(batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "Coronal pair abnormal auxiliary loss requires binary logits."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Coronal pair abnormal auxiliary loss expects fusion weights with "
                    "shape (batch, 3, 1) matching view logits."
                )
            coronal_normal_logits = view_logits[:, 1:2, :1].detach()
            coronal_abnormal_logits = view_logits[:, 1:2, 1:2]
            coronal_abnormal_aux = torch.cat(
                [coronal_normal_logits, coronal_abnormal_logits],
                dim=-1,
            ).squeeze(1)
            pair_indices = [0, 1]
            pair_weights = fusion_weights.squeeze(-1)[:, pair_indices]
            pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pair_selector_aux = (
                view_logits.detach()[:, pair_indices, :] * pair_weights.unsqueeze(-1)
            ).sum(dim=1)
            if self.coronal_pair_abnormal_aux_require_disagreement:
                detached_predictions = view_logits.detach().argmax(dim=-1)
                pair_disagreement = detached_predictions[:, 1].ne(detached_predictions[:, 0])
                pair_selector_aux = torch.where(
                    pair_disagreement.view(-1, 1),
                    pair_selector_aux,
                    pair_selector_aux.detach(),
                )
            aux_logits = torch.stack([coronal_abnormal_aux, pair_selector_aux], dim=1)
            aux_weight = self.coronal_pair_abnormal_aux_weight
        elif self.enable_nonaxial_abnormal_aux_loss:
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Non-axial abnormal auxiliary loss expects view logits with shape "
                    "(batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "Non-axial abnormal auxiliary loss requires binary logits."
                )
            aux_view_logits = (
                view_logits[:, 1:2]
                if self.nonaxial_abnormal_aux_coronal_only
                else view_logits[:, 1:]
            )
            normal_logits = aux_view_logits[..., :1].detach()
            abnormal_logits = aux_view_logits[..., 1:2]
            aux_logits = torch.cat([normal_logits, abnormal_logits], dim=-1)
            aux_weight = self.nonaxial_abnormal_aux_weight
        elif self.enable_sagittal_normal_rescue_aux_loss:
            if fusion_weights is None:
                raise RuntimeError(
                    "Sagittal normal rescue auxiliary loss is enabled but fusion weights "
                    "were not produced."
                )
            if view_logits.ndim != 3 or view_logits.shape[1] != 3:
                raise RuntimeError(
                    "Sagittal normal rescue auxiliary loss expects view logits with shape "
                    "(batch, 3, classes)."
                )
            if view_logits.shape[-1] != 2:
                raise RuntimeError(
                    "Sagittal normal rescue auxiliary loss requires binary logits."
                )
            if fusion_weights.ndim != 3 or fusion_weights.shape[:2] != view_logits.shape[:2]:
                raise RuntimeError(
                    "Sagittal normal rescue auxiliary loss expects fusion weights with "
                    "shape (batch, 3, 1) matching view logits."
                )
            detached_predictions = view_logits.detach().argmax(dim=-1)
            rescue_mask = detached_predictions[:, 0].eq(1) & detached_predictions[:, 2].eq(0)
            pair_indices = [0, 2]
            pair_weights = fusion_weights.squeeze(-1)[:, pair_indices]
            pair_weights = pair_weights / pair_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pair_logits = (
                view_logits.detach()[:, pair_indices, :] * pair_weights.unsqueeze(-1)
            ).sum(dim=1)
            aux_logits = torch.where(
                rescue_mask.view(-1, 1),
                pair_logits,
                pair_logits.detach(),
            ).unsqueeze(1)
            aux_weight = self.sagittal_normal_rescue_aux_weight
        elif self.enable_aux_view_loss:
            aux_logits = view_logits

        if aux_logits is not None:
            self._view_logits = aux_logits
            log_var_value = -math.log(aux_weight)
            self._log_vars = torch.full(
                (aux_logits.shape[0], aux_logits.shape[1], 1),
                log_var_value,
                device=aux_logits.device,
                dtype=aux_logits.dtype,
            )
        else:
            self._view_logits = None
            self._log_vars = None

    def _apply_fusion_weight_floor(
        self,
        fusion_weights: torch.Tensor,
        active_view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bound late-fusion collapse by reserving a minimum mass for every active view."""
        if self.fusion_weight_floor <= 0.0:
            return fusion_weights
        if active_view_mask is None:
            active_mask = torch.ones_like(fusion_weights)
        else:
            active_mask = active_view_mask.to(
                device=fusion_weights.device,
                dtype=fusion_weights.dtype,
            ).unsqueeze(-1)
        active_count = active_mask.sum(dim=1, keepdim=True)
        residual_mass = 1.0 - self.fusion_weight_floor * active_count
        if torch.any(residual_mass <= 0):
            raise ValueError(
                "ANKLE_DECISION_FUSION_WEIGHT_FLOOR leaves no residual mass for active views."
            )
        return fusion_weights * residual_mass + active_mask * self.fusion_weight_floor

    def fuse_decisions(
        self,
        view_logits: torch.Tensor,
        fusion_weights: torch.Tensor,
        active_view_mask: torch.Tensor | None = None,
        classifier_gradient_fusion_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse per-view logits, optionally renormalizing over an active view subset."""
        if active_view_mask is not None:
            if active_view_mask.ndim == 1:
                active_view_mask = active_view_mask.unsqueeze(0)
            if active_view_mask.ndim != 2:
                raise ValueError("active_view_mask must have shape (3,) or (B, 3).")
            if active_view_mask.shape[-1] != view_logits.shape[1]:
                raise ValueError(
                    "active_view_mask last dimension must match the number of views."
                )
            if active_view_mask.shape[0] not in {1, view_logits.shape[0]}:
                raise ValueError(
                    "active_view_mask batch dimension must be 1 or match view_logits."
                )
            if active_view_mask.shape[0] == 1 and view_logits.shape[0] != 1:
                active_view_mask = active_view_mask.expand(view_logits.shape[0], -1)
            mask = active_view_mask.to(device=view_logits.device, dtype=view_logits.dtype).unsqueeze(-1)
            masked_weights = fusion_weights * mask
            normalizer = masked_weights.sum(dim=1, keepdim=True)
            if torch.any(normalizer <= 0):
                raise ValueError("active_view_mask must keep at least one view per sample.")
            fusion_weights = masked_weights / normalizer

        if active_view_mask is not None:
            fusion_weights = self._apply_fusion_weight_floor(fusion_weights, active_view_mask)
        if (
            self.training
            and classifier_gradient_fusion_weights is not None
            and active_view_mask is None
        ):
            if classifier_gradient_fusion_weights.shape != fusion_weights.shape:
                raise ValueError(
                    "classifier_gradient_fusion_weights must match fusion_weights shape."
                )
            fused_logits = (view_logits * classifier_gradient_fusion_weights).sum(dim=1)
            role_residual = (
                view_logits.detach()
                * (fusion_weights - classifier_gradient_fusion_weights.detach())
            ).sum(dim=1)
            fused_logits = fused_logits + role_residual
        else:
            fused_logits = (view_logits * fusion_weights).sum(dim=1)
        if (
            self.enable_evidence_margin_residual_fusion
            and hasattr(self, "evidence_margin_residual_fusion")
            and active_view_mask is None
        ):
            fused_logits = self.evidence_margin_residual_fusion(
                fused_logits,
                view_logits,
                fusion_weights,
            )
        if (
            self.enable_positive_evidence_floor_fusion
            and hasattr(self, "positive_evidence_floor_fusion")
            and active_view_mask is None
        ):
            fused_logits = self.positive_evidence_floor_fusion(
                fused_logits,
                view_logits,
                fusion_weights,
            )
        return fused_logits

    def forward_with_decision_info(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the usual fused logits plus intermediate tensors for matched controls."""
        decision_outputs = self._compute_decision_outputs(images)
        self._set_aux_view_loss_state(
            decision_outputs["view_logits"],
            fusion_weights=decision_outputs.get(
                "raw_fusion_weights",
                decision_outputs["fusion_weights"],
            ),
            candidate_prototype_logits=decision_outputs.get("candidate_prototype_logits"),
        )
        active_view_mask = None
        if self.forced_active_view_mask is not None:
            active_view_mask = decision_outputs["view_logits"].new_tensor(self.forced_active_view_mask)
        logits = self.fuse_decisions(
            decision_outputs["view_logits"],
            decision_outputs["fusion_weights"],
            active_view_mask=active_view_mask,
            classifier_gradient_fusion_weights=decision_outputs.get(
                "classifier_gradient_fusion_weights"
            ),
        )
        if active_view_mask is not None:
            decision_outputs["active_view_mask"] = active_view_mask
        return logits, decision_outputs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播：每个视角独立分类，用动态可信度权重融合。

        参数：
            images: (B, 3, S, H, W) 的 CT 图像张量

        返回：
            logits: (B, 2) 的张量，3 个视角动态加权投票后的分类结果
        """
        logits, _ = self.forward_with_decision_info(images)
        return logits


class MultiViewAttentionClassifier(nn.Module):
    """
    注意力融合分类器（Attention Fusion）。

    相比 MultiViewCTClassifier（特征融合）的三大改进：
        1. Attention Pooling：不再对切片取简单平均，而是用可学习注意力聚焦关键切片
        2. Cross-View Attention：3 个视角通过 Transformer 自注意力交换信息
        3. 更强的分类头：LayerNorm + 两层 MLP

    工作流程：
        ResNet18 逐张切片提特征 (B*S, 512)
        -> reshape 成 (B, S, 512)
        -> Attention Pooling: (B, S, 512) -> (B, 512)  <- 替换 mean pooling
        -> stack 3 个视角: (B, 3, 512)
        -> Cross-View Attention: (B, 3, 512) -> (B, 3, 512)
        -> reshape: (B, 1536)
        -> MLP 分类器 -> (B, 2)
    """

    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
        freeze_layers: int = DEFAULT_FREEZE_LAYERS,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        cross_view_heads: int = 8,
        cross_view_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_dim = 512
        self.share_backbone = share_backbone

        # ---------- Backbone ----------
        if share_backbone:
            self.shared_encoder = build_resnet18_encoder(
                use_pretrained=use_pretrained,
                freeze_layers=freeze_layers,
            )
        else:
            self.view_encoders = nn.ModuleList(
                [
                    build_resnet18_encoder(
                        use_pretrained=use_pretrained,
                        freeze_layers=freeze_layers,
                    )
                    for _ in range(3)
                ]
            )

        # ---------- Attention Pooling（逐视角） ----------
        if share_backbone:
            # 共享 backbone 时也共享 pooling
            self.shared_pooling = AttentionPooling(self.feature_dim)
        else:
            self.view_poolings = nn.ModuleList(
                [AttentionPooling(self.feature_dim) for _ in range(3)]
            )

        # ---------- Cross-View Attention ----------
        from .cross_view_attention import CrossViewAttention

        self.cross_view_attention = CrossViewAttention(
            feature_dim=self.feature_dim,
            num_heads=cross_view_heads,
            num_layers=cross_view_layers,
            dropout=dropout,
        )

        # ---------- 分类头 ----------
        fused_dim = self.feature_dim * 3  # 512 * 3 = 1536
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 2),
        )

    def _forward_impl(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, num_views, num_slices, height, width = images.shape
        view_features = []
        slice_attention_weights = []

        for view_index in range(num_views):
            # (B, S, H, W) -> (B*S, 1, H, W)
            view_tensor = images[:, view_index, :, :, :].reshape(
                batch_size * num_slices, 1, height, width
            )

            # ResNet18 slice features: (B*S, 512)
            if self.share_backbone:
                slice_features = self.shared_encoder(view_tensor)
            else:
                slice_features = self.view_encoders[view_index](view_tensor)

            # (B*S, 512) -> (B, S, 512)
            slice_features = slice_features.reshape(batch_size, num_slices, self.feature_dim)

            # Attention Pooling: (B, S, 512) -> (B, 512)
            if self.share_backbone:
                pooled, weights = self.shared_pooling(slice_features, return_weights=True)
            else:
                pooled, weights = self.view_poolings[view_index](
                    slice_features,
                    return_weights=True,
                )

            view_features.append(pooled)
            slice_attention_weights.append(weights)

        # Stack view features: 3 x (B, 512) -> (B, 3, 512)
        stacked = torch.stack(view_features, dim=1)

        # Cross-View Attention: (B, 3, 512) -> (B, 3, 512)
        enhanced = self.cross_view_attention(stacked)

        # Flatten: (B, 3, 512) -> (B, 1536)
        fused = enhanced.reshape(batch_size, -1)

        # Classification
        logits = self.classifier(fused)
        return logits, {
            "slice_attention_weights": torch.stack(slice_attention_weights, dim=1),
        }

    def forward_with_attention(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward pass and return slice-level attention weights."""
        return self._forward_impl(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        logits, _ = self._forward_impl(images)
        return logits
