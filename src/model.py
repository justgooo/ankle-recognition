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
    ) -> None:
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

        if minimal_fusion_baseline:
            # 纯融合对比模式：不引入额外视角交互或门控模块，只保留最小 MLP 头。
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, 2),
            )
        else:
            self.view_recalibrators = nn.ModuleList(
                [ViewFeatureRecalibration(self.feature_dim) for _ in range(3)]
            )
            # 只在 pooled view token 之间加入极轻量的 cross-view context 交换，
            # 保持 256x8 mean-pooling 几何不变，检验“缺少显式视角交互”是否仍是瓶颈。
            from .cross_view_attention import CrossViewAttention

            self.cross_view_mixer = CrossViewAttention(
                feature_dim=self.feature_dim,
                attention_dim=256,
                num_heads=4,
                num_layers=1,
                dropout=0.1,
                residual_scale=0.125,
            )

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
            recalibrated_features = [
                recalibrator(feature)
                for recalibrator, feature in zip(self.view_recalibrators, view_features)
            ]
            mixed_features = self.cross_view_mixer(torch.stack(recalibrated_features, dim=1))
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
        self.gate_teacher_blend = _env_unit_float(
            "ANKLE_DECISION_GATE_TEACHER_BLEND",
            0.0,
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
                self.confidence_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(self.feature_dim),
                            nn.Linear(self.feature_dim, 1),  # 512 → 1 reliability logit
                        )
                        for _ in range(3)
                    ]
                )
                if not self.disable_fusion_calibrator:
                    self.confidence_calibrator = SharedLowRankReliabilityCalibrator(
                        feature_dim=self.feature_dim,
                        bottleneck_dim=32,
                        scale_limit=0.25,
                        bias_limit=0.15,
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
            calibrated_confidences = []
            for head, feature in zip(self.confidence_heads, gating_features):
                raw_confidence = head(feature)
                if self.disable_fusion_calibrator:
                    calibrated_confidences.append(raw_confidence)
                else:
                    calibrated_confidences.append(
                        self.confidence_calibrator(feature, raw_confidence)
                    )
            confidences = torch.stack(calibrated_confidences, dim=1)  # (B, 3, 1)

        scaled_confidences = confidences / self.fusion_temperature
        raw_fusion_weights = torch.softmax(scaled_confidences, dim=1)  # (B, 3, 1)
        teacher_fusion_weights = None
        blended_fusion_weights = raw_fusion_weights
        if self.gate_teacher_blend > 0.0:
            teacher_fusion_weights = self._compute_gate_teacher_weights(view_logits)
            blended_fusion_weights = (
                (1.0 - self.gate_teacher_blend) * raw_fusion_weights
                + self.gate_teacher_blend * teacher_fusion_weights
            )
            blended_fusion_weights = blended_fusion_weights / blended_fusion_weights.sum(
                dim=1,
                keepdim=True,
            )
        fusion_weights = self._apply_fusion_weight_floor(blended_fusion_weights)
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
            "view_logit_temperatures": view_logit_temperatures,
        }

    def _set_aux_view_loss_state(self, view_logits: torch.Tensor) -> None:
        """Expose per-view logits through the existing train.py UWDF hook."""
        if self.enable_aux_view_loss:
            self._view_logits = view_logits
            log_var_value = -math.log(self.aux_view_loss_weight)
            self._log_vars = torch.full(
                (view_logits.shape[0], view_logits.shape[1], 1),
                log_var_value,
                device=view_logits.device,
                dtype=view_logits.dtype,
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
        return (view_logits * fusion_weights).sum(dim=1)

    def forward_with_decision_info(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the usual fused logits plus intermediate tensors for matched controls."""
        decision_outputs = self._compute_decision_outputs(images)
        self._set_aux_view_loss_state(decision_outputs["view_logits"])
        logits = self.fuse_decisions(
            decision_outputs["view_logits"],
            decision_outputs["fusion_weights"],
        )
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
