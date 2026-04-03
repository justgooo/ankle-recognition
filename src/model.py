"""
model.py — 多视角 CT 分类模型
================================
这个文件定义了整个「足踝 CT 分类」项目的神经网络模型。

整体思路：
    1. 一张 CT 扫描有 3 个视角（轴状面 / 冠状面 / 矢状面），每个视角有多张切片图像。
    2. 用 ResNet18（一种经典的图像识别网络）把每张切片提取成一个 512 维的特征向量。
    3. 把同一个视角的所有切片特征取"平均"，得到该视角的代表特征。
    4. 把 3 个视角的特征拼起来，送进分类器（一个小的全连接网络），输出 2 个值：
       - 第 0 个值代表"正常"的概率
       - 第 1 个值代表"异常"的概率

本文件包含以下类：
    - build_resnet18_encoder(): 构建 ResNet18 特征提取器
    - MultiViewEncoder: 多视角编码器（提取 3 个视角的特征）
    - MultiViewCTClassifier: 特征融合分类器（拼接 3 个视角特征后分类）
    - MultiViewDecisionFusionClassifier: 决策融合分类器（每个视角单独分类后加权投票）
"""

from __future__ import annotations

import torch  # type: ignore[import-not-found]          # PyTorch：深度学习框架的核心库
import torch.nn as nn  # type: ignore[import-not-found]  # nn 模块：提供各种神经网络层（卷积、线性层等）
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-not-found]
# ↑ 从 torchvision 导入 ResNet18 预训练模型和对应的权重


def build_resnet18_encoder(use_pretrained: bool) -> nn.Module:
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
    return backbone


class MultiViewEncoder(nn.Module):
    """
    多视角编码器 — 把 3 个视角的 CT 切片图像分别提取成特征向量。

    CT 扫描通常有 3 个方向的视角：
        - 轴状面 (Axial)    — 从上往下看
        - 冠状面 (Coronal)  — 从前往后看
        - 矢状面 (Sagittal) — 从侧面看

    每个视角有多张切片（比如 16 张），这个类的作用就是：
        对每个视角 → 把所有切片分别过 ResNet18 → 取平均 → 得到该视角的 512 维特征
    """

    def __init__(
        self,
        share_backbone: bool = True,    # 3 个视角是否共享同一个 ResNet18
        use_pretrained: bool = False,    # 是否使用预训练权重
    ) -> None:
        """
        参数：
            share_backbone: 是否让 3 个视角共用同一个 ResNet18？
                - True（推荐）：3 个视角用同一个 ResNet18，参数量小，不容易过拟合
                - False：每个视角有自己独立的 ResNet18（3 份参数），参数量更大
            use_pretrained: 是否使用 ImageNet 预训练权重
        """
        super().__init__()
        self.share_backbone = share_backbone
        self.feature_dim = 512  # ResNet18 输出的特征维度固定为 512

        if share_backbone:
            # 共享模式：只创建一个 ResNet18，3 个视角都用它
            self.shared_encoder = build_resnet18_encoder(use_pretrained=use_pretrained)
        else:
            # 独立模式：创建 3 个独立的 ResNet18，每个视角用自己的
            self.view_encoders = nn.ModuleList(
                [build_resnet18_encoder(use_pretrained=use_pretrained) for _ in range(3)]
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
            分别代表 3 个视角各自的平均特征
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

            # 对所有切片的特征取平均，得到该视角的代表性特征
            # (B, S, 512) → mean(dim=1) → (B, 512)
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
    ) -> None:
        super().__init__(
            share_backbone=share_backbone,
            use_pretrained=use_pretrained,
        )
        # 3 个视角拼接后的总维度：512 × 3 = 1536
        fused_dim = self.feature_dim * 3

        # 分类器：一个两层的全连接网络（MLP）
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),  # 1536 → 256（降维）
            nn.ReLU(inplace=True),                    # ReLU 激活函数（引入非线性）
            nn.Dropout(dropout),                      # 随机丢弃 30% 的神经元（防止过拟合）
            nn.Linear(fusion_hidden_dim, 2),          # 256 → 2（输出 2 个类别的分数）
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
        # 第 1 步：提取 3 个视角的特征并拼接
        # encode_views 返回 3 个 (B, 512) → cat 后变成 (B, 1536)
        image_feature = torch.cat(self.encode_views(images), dim=1)
        # 第 2 步：送进分类器，得到分类结果
        return self.classifier(image_feature)


class MultiViewDecisionFusionClassifier(MultiViewEncoder):
    """
    决策融合分类器 — 非对称安全融合版 (Asymmetric Safety-Biased Fusion)。

    与原版 Decision Fusion 的关键区别：
        - 原版：对正常和异常 logit 使用相同的对称加权平均
        - 本版：非对称融合——
          · 正常 logit：加权平均（保守估计正常概率）
          · 异常 logit：temperature-scaled logsumexp（任一视角认为异常就倾向异常）

    临床意义：
        "宁可误诊，不可漏诊" — 只要任何一个视角发现异常信号，
        融合结果就会倾向于判定为异常，从而降低漏诊风险。

    temperature 参数控制保守程度：
        - temperature → 0: 接近 hard max（最保守，最不容易漏诊）
        - temperature → ∞: 接近均匀平均（退化为原版 Decision Fusion）
    """

    # temperature 默认值，每次实验可在此处直接修改
    DEFAULT_TEMPERATURE = 0.5

    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            share_backbone=share_backbone,
            use_pretrained=use_pretrained,
        )
        # 每个视角有一个独立的分类器（共 3 个）
        self.view_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim, fusion_hidden_dim),  # 512 → 256
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_hidden_dim, 2),  # 256 → 2（正常/异常）
                )
                for _ in range(3)  # 创建 3 个分类器
            ]
        )
        # 可学习的视角权重（用于正常 logit 的加权平均）
        self.view_weight_logits = nn.Parameter(torch.zeros(3))
        # 非对称融合温度参数
        self.temperature = self.DEFAULT_TEMPERATURE

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播：每个视角独立分类，非对称融合。

        正常 logit = 加权平均（保守估计"正常"）
        异常 logit = temperature-scaled logsumexp（放大异常信号）

        参数：
            images: (B, 3, S, H, W) 的 CT 图像张量

        返回：
            logits: (B, 2) 的张量，非对称融合后的分类结果
        """
        # 第 1 步：提取 3 个视角的特征
        view_features = self.encode_views(images)  # 3 个 (B, 512) 的列表

        # 第 2 步：每个视角分别做分类
        view_logits = torch.stack(
            [classifier(feature) for classifier, feature in zip(self.view_classifiers, view_features)],
            dim=1,
        )  # (B, 3, 2)

        # 第 3 步：计算正常 logit 的融合权重（固定的全局权重）
        fusion_weights = torch.softmax(self.view_weight_logits, dim=0)  # (3,)

        # 第 4 步：非对称融合
        # 正常 logit (index=0)：加权平均
        normal_logits = view_logits[:, :, 0]  # (B, 3)
        fused_normal = (normal_logits * fusion_weights.unsqueeze(0)).sum(dim=1)  # (B,)

        # 异常 logit (index=1)：temperature-scaled logsumexp
        # logsumexp(x/T) * T ≈ max(x) when T→0, ≈ mean(x)+log(3)*T when T→∞
        abnormal_logits = view_logits[:, :, 1]  # (B, 3)
        fused_abnormal = self.temperature * torch.logsumexp(
            abnormal_logits / self.temperature, dim=1
        )  # (B,)

        # 拼接为 (B, 2)
        return torch.stack([fused_normal, fused_abnormal], dim=1)


class MultiViewAttentionClassifier(nn.Module):
    """
    注意力融合分类器（Attention Fusion）。

    相比 MultiViewCTClassifier（特征融合）的三大改进：
        1. Attention Pooling：不再对切片取简单平均，而是用可学习注意力聚焦关键切片
        2. Cross-View Attention：3 个视角通过 Transformer 自注意力交换信息
        3. 更强的分类头：LayerNorm + 两层 MLP

    工作流程：
        ResNet18 逐张切片提特征 (B*S, 512)
        → reshape 成 (B, S, 512)
        → Attention Pooling: (B, S, 512) → (B, 512)  ← 替换 mean pooling
        → stack 3 个视角: (B, 3, 512)
        → Cross-View Attention: (B, 3, 512) → (B, 3, 512)
        → reshape: (B, 1536)
        → MLP 分类器 → (B, 2)
    """

    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
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
            self.shared_encoder = build_resnet18_encoder(use_pretrained=use_pretrained)
        else:
            self.view_encoders = nn.ModuleList(
                [build_resnet18_encoder(use_pretrained=use_pretrained) for _ in range(3)]
            )

        # ---------- Attention Pooling（逐视角） ----------
        from .attention_pooling import AttentionPooling

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
        fused_dim = self.feature_dim * 3  # 512 × 3 = 1536
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
