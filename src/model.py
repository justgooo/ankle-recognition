"""
model.py — 多视角 CT 分类模型
================================
这个文件定义了整个「足踝 CT 分类」项目的神经网络模型。

整体思路：
    1. 一张 CT 扫描有 3 个视角（轴状面 / 冠状面 / 矢状面），每个视角有多张切片图像。
    2. 用 ResNet18（一种经典的图像识别网络）把每张切片提取成一个 512 维的特征向量。
    3. 把同一个视角的所有切片特征聚合（mean pooling 或 AttentionPooling），得到该视角的代表特征。
    4. 把 3 个视角的特征拼起来，送进分类器（一个小的全连接网络），输出 2 个值：
       - 第 0 个值代表"正常"的概率
       - 第 1 个值代表"异常"的概率

本文件包含以下类：
    - build_resnet18_encoder(): 构建 ResNet18 特征提取器
    - MultiViewEncoder: 多视角编码器（提取 3 个视角的特征，支持 mean pooling 或 AttentionPooling）
    - MultiViewCTClassifier: 特征融合分类器（拼接 3 个视角特征后分类）
    - MultiViewDecisionFusionClassifier: 决策融合分类器（每个视角单独分类后加权投票）
"""

from __future__ import annotations

import torch  # type: ignore[import-not-found]          # PyTorch：深度学习框架的核心库
import torch.nn as nn  # type: ignore[import-not-found]  # nn 模块：提供各种神经网络层（卷积、线性层等）
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-not-found]
# ↑ 从 torchvision 导入 ResNet18 预训练模型和对应的权重

from .attention_pooling import AttentionPooling  # 可学习注意力池化模块


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
        share_backbone: bool = True,    # 3 个视角是否共享同一个 ResNet18
        use_pretrained: bool = False,    # 是否使用预训练权重
        use_attention_pooling: bool = False,  # 是否使用注意力池化替代 mean pooling
    ) -> None:
        """
        参数：
            share_backbone: 是否让 3 个视角共用同一个 ResNet18？
                - True（推荐）：3 个视角用同一个 ResNet18，参数量小，不容易过拟合
                - False：每个视角有自己独立的 ResNet18（3 份参数），参数量更大
            use_pretrained: 是否使用 ImageNet 预训练权重
            use_attention_pooling: 是否使用 AttentionPooling？
                - True：用可学习注意力对切片加权聚合（能自动聚焦关键切片）
                - False（默认）：用简单 mean pooling（对所有切片取平均）
        """
        super().__init__()
        self.share_backbone = share_backbone
        self.use_attention_pooling = use_attention_pooling
        self.feature_dim = 512  # ResNet18 输出的特征维度固定为 512

        if share_backbone:
            # 共享模式：只创建一个 ResNet18，3 个视角都用它
            self.shared_encoder = build_resnet18_encoder(use_pretrained=use_pretrained)
        else:
            # 独立模式：创建 3 个独立的 ResNet18，每个视角用自己的
            self.view_encoders = nn.ModuleList(
                [build_resnet18_encoder(use_pretrained=use_pretrained) for _ in range(3)]
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
    ) -> None:
        super().__init__(
            share_backbone=share_backbone,
            use_pretrained=use_pretrained,
            use_attention_pooling=use_attention_pooling,
        )
        # 3 个视角拼接后的总维度：512 * 3 = 1536
        fused_dim = self.feature_dim * 3

        # 分类器：一个两层的全连接网络（MLP）
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),  # 1536 -> 256（降维）
            nn.ReLU(inplace=True),                    # ReLU 激活函数（引入非线性）
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
        # 第 1 步：提取 3 个视角的特征并拼接
        # encode_views 返回 3 个 (B, 512) -> cat 后变成 (B, 1536)
        image_feature = torch.cat(self.encode_views(images), dim=1)
        # 第 2 步：送进分类器，得到分类结果
        return self.classifier(image_feature)


class UncertaintyHead(nn.Module):
    """
    不确定性双头分类器 —— 同时输出分类 logits 和预测不确定性。

    每个视角各自使用一个 UncertaintyHead，输出：
        - logits  (B, 2)：该视角对"正常/异常"的分类分数
        - log_var (B, 1)：该视角的预测不确定性（对数方差 log σ²）

    log_var 的含义：
        - log_var 大（如 +3）→ 方差大 → 该视角不确定 → 在融合时权重低
        - log_var 小（如 -3）→ 方差小 → 该视角有信心 → 在融合时权重高
        - 初始化为 0 → 初始方差 = 1 → 三个视角等权起步
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # 共享的特征变换层
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # 分类头：输出 logits
        self.logits_head = nn.Linear(hidden_dim, num_classes)
        # 不确定性头：输出 log(σ²)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        # 初始化：log_var ≈ 0 → σ² ≈ 1 → 三个视角初始等权
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            x: (B, feature_dim) 的视角特征

        返回：
            logits:  (B, 2) 分类分数
            log_var: (B, 1) 预测不确定性
        """
        h = self.shared(x)
        logits = self.logits_head(h)
        log_var = self.log_var_head(h)
        # 限制 log_var 范围，防止数值不稳定
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        return logits, log_var


class MultiViewDecisionFusionClassifier(MultiViewEncoder):
    """
    不确定性加权决策融合 (Uncertainty-Weighted Decision Fusion, UWDF)。

    核心思想：
        每个视角不仅输出分类结果，还输出"我对这个判断有多不确定"。
        不确定性高的视角在融合时权重自动降低。

    数学原理（Heteroscedastic Uncertainty, Kendall & Gal 2017）：
        每个视角 v 输出：
            - logits_v  (B, 2)：分类分数
            - log_var_v (B, 1)：预测不确定性（对数方差 log σ²）

        融合权重 = softmax(precision)，其中 precision_v = exp(-log_var_v) = 1/σ²_v
        融合结果 = Σ_v weight_v * logits_v

        训练损失（在 train.py 中计算）：
            L = CE(fused_logits, label)
              + (1/V) * Σ_v [exp(-s_v) * CE(logits_v, label) + s_v]
            其中 s_v = log_var_v

        第一项：全局分类损失（标准交叉熵）
        第二项：不确定性校准辅助损失
            - exp(-s_v) * CE_v：不确定性高时降低该视角的损失贡献
            - s_v：防止不确定性无限增大（正则项）

    与传统 Decision Fusion 的区别：
        - 传统：视角权重是全局固定的（所有样本共用一组权重）
        - UWDF：视角权重是样本自适应的（每个样本、每个视角的不确定性不同）

    与 View Reliability Gating (VRG) 的区别：
        - VRG：用 sigmoid(MLP(feature)) 做二值化门控，学"重要性"
        - UWDF：用 softmax(1/σ²) 做概率化精度加权，学"不确定性"，有明确概率论含义
    """

    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        use_attention_pooling: bool = False,
    ) -> None:
        super().__init__(
            share_backbone=share_backbone,
            use_pretrained=use_pretrained,
            use_attention_pooling=use_attention_pooling,
        )
        # 3 个视角各自的不确定性双头分类器
        self.view_heads = nn.ModuleList(
            [
                UncertaintyHead(
                    self.feature_dim, fusion_hidden_dim, num_classes=2, dropout=dropout
                )
                for _ in range(3)
            ]
        )
        # 暴露给 train.py 的辅助信息（每次 forward 后更新）
        self._view_logits: torch.Tensor | None = None
        self._log_vars: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播：提取 3 个视角特征，各自预测 logits 和不确定性，精度加权融合。

        参数：
            images: (B, 3, S, H, W) 的 CT 图像张量

        返回：
            fused_logits: (B, 2) 的张量，精度加权融合后的分类结果
        """
        # 第 1 步：提取 3 个视角的特征
        view_features = self.encode_views(images)

        # 第 2 步：每个视角输出 logits 和不确定性
        logits_list: list[torch.Tensor] = []
        log_var_list: list[torch.Tensor] = []
        for i, feat in enumerate(view_features):
            logits_i, log_var_i = self.view_heads[i](feat)
            logits_list.append(logits_i)
            log_var_list.append(log_var_i)

        all_logits = torch.stack(logits_list, dim=1)    # (B, 3, 2)
        all_log_vars = torch.stack(log_var_list, dim=1)  # (B, 3, 1)

        # 暴露给 train.py 用于计算辅助损失
        self._view_logits = all_logits
        self._log_vars = all_log_vars

        # 第 3 步：精度加权融合 (Precision-Weighted Fusion)
        # precision = 1/σ² = exp(-log_var)
        precision = torch.exp(-all_log_vars)         # (B, 3, 1)
        weights = torch.softmax(precision, dim=1)     # (B, 3, 1) 归一化
        fused_logits = (all_logits * weights).sum(dim=1)  # (B, 2)

        return fused_logits


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
