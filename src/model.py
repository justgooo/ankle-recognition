from __future__ import annotations

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-not-found]


def build_resnet18_encoder(use_pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    backbone = resnet18(weights=weights)

    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if weights is not None:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    backbone.conv1 = new_conv
    backbone.fc = nn.Identity()
    return backbone


class MultiViewCTClassifier(nn.Module):
    def __init__(
        self,
        share_backbone: bool = True,
        use_pretrained: bool = False,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.share_backbone = share_backbone
        self.feature_dim = 512

        if share_backbone:
            self.shared_encoder = build_resnet18_encoder(use_pretrained=use_pretrained)
        else:
            self.view_encoders = nn.ModuleList(
                [build_resnet18_encoder(use_pretrained=use_pretrained) for _ in range(3)]
            )

        fused_dim = self.feature_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, num_slices, height, width = images.shape
        view_features = []

        for view_index in range(num_views):
            view_tensor = images[:, view_index, :, :, :].reshape(
                batch_size * num_slices, 1, height, width
            )

            if self.share_backbone:
                slice_features = self.shared_encoder(view_tensor)
            else:
                slice_features = self.view_encoders[view_index](view_tensor)

            slice_features = slice_features.reshape(batch_size, num_slices, self.feature_dim)
            pooled_view_feature = slice_features.mean(dim=1)
            view_features.append(pooled_view_feature)

        image_feature = torch.cat(view_features, dim=1)
        return self.classifier(image_feature)
