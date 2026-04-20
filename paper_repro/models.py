from __future__ import annotations

import math
from typing import Iterable, Sequence

import timm
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


def center_slice(view: torch.Tensor) -> torch.Tensor:
    index = view.shape[1] // 2
    return view[:, index : index + 1]


def mean_projection(view: torch.Tensor) -> torch.Tensor:
    return view.mean(dim=1, keepdim=True)


def max_projection(view: torch.Tensor) -> torch.Tensor:
    return view.amax(dim=1, keepdim=True)


def min_projection(view: torch.Tensor) -> torch.Tensor:
    return view.amin(dim=1, keepdim=True)


def std_projection(view: torch.Tensor) -> torch.Tensor:
    return view.std(dim=1, keepdim=True)


def stack_triplets(view: torch.Tensor) -> torch.Tensor:
    prev_idx = [0, *range(view.shape[1] - 1)]
    next_idx = [*range(1, view.shape[1]), view.shape[1] - 1]
    prev_slice = view[:, prev_idx]
    next_slice = view[:, next_idx]
    return torch.stack([prev_slice, view, next_slice], dim=2)


def sliding_window_positions(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def compute_3d_bbox(mask: torch.Tensor, margin: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    coords = torch.nonzero(mask, as_tuple=False)
    depth, height, width = mask.shape
    if coords.numel() == 0:
        return 0, depth, 0, height, 0, width
    z0 = max(0, int(coords[:, 0].min().item()) - int(margin[0]))
    z1 = min(depth, int(coords[:, 0].max().item()) + 1 + int(margin[0]))
    y0 = max(0, int(coords[:, 1].min().item()) - int(margin[1]))
    y1 = min(height, int(coords[:, 1].max().item()) + 1 + int(margin[1]))
    x0 = max(0, int(coords[:, 2].min().item()) - int(margin[2]))
    x1 = min(width, int(coords[:, 2].max().item()) + 1 + int(margin[2]))
    return z0, z1, y0, y1, x0, x1


def extract_candidate_volume_patches(
    images: torch.Tensor,
    *,
    patch_size: Sequence[int],
    patch_stride: Sequence[int],
    roi_threshold: float,
    roi_margin: Sequence[int],
    max_patches_per_view: int,
) -> tuple[torch.Tensor, list[int]]:
    patch_depth, patch_height, patch_width = [int(item) for item in patch_size]
    stride_depth, stride_height, stride_width = [int(item) for item in patch_stride]
    patches: list[torch.Tensor] = []
    sample_patch_counts: list[int] = []

    for sample_index in range(images.shape[0]):
        sample_count = 0
        for view_index in range(images.shape[1]):
            volume = images[sample_index, view_index]
            mask = volume > roi_threshold
            z0, z1, y0, y1, x0, x1 = compute_3d_bbox(mask, margin=roi_margin)
            cropped = volume[z0:z1, y0:y1, x0:x1]
            depth, height, width = cropped.shape
            if depth < patch_depth or height < patch_height or width < patch_width:
                padded = torch.zeros(
                    max(depth, patch_depth),
                    max(height, patch_height),
                    max(width, patch_width),
                    device=images.device,
                    dtype=images.dtype,
                )
                padded[:depth, :height, :width] = cropped
                cropped = padded
                depth, height, width = cropped.shape

            candidates: list[tuple[float, torch.Tensor]] = []
            depth_positions = sliding_window_positions(depth, patch_depth, stride_depth)
            height_positions = sliding_window_positions(height, patch_height, stride_height)
            width_positions = sliding_window_positions(width, patch_width, stride_width)
            for start_d in depth_positions:
                for start_h in height_positions:
                    for start_w in width_positions:
                        patch = cropped[
                            start_d : start_d + patch_depth,
                            start_h : start_h + patch_height,
                            start_w : start_w + patch_width,
                        ]
                        heuristic = float(
                            patch.std().item()
                            + 0.5 * (patch > roi_threshold).float().mean().item()
                            + 0.25 * patch.mean().item()
                        )
                        candidates.append((heuristic, patch))

            candidates.sort(key=lambda item: item[0], reverse=True)
            selected = candidates[: max(1, int(max_patches_per_view))]
            for _, patch in selected:
                patches.append(patch)
                sample_count += 1
        sample_patch_counts.append(sample_count)

    if not patches:
        fallback = images[:, 0]
        return fallback, [1 for _ in range(images.shape[0])]
    patch_batch = torch.stack(patches, dim=0)
    return patch_batch, sample_patch_counts


def crop_batch_2d(images: torch.Tensor, threshold: float = 0.55, scale: float = 1.2) -> torch.Tensor:
    batch, channels, height, width = images.shape
    cropped = []
    for sample in images:
        mask = sample.mean(dim=0) > threshold
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            y0 = max(0, height // 4)
            y1 = min(height, height - y0)
            x0 = max(0, width // 4)
            x1 = min(width, width - x0)
        else:
            y0 = int(coords[:, 0].min().item())
            y1 = int(coords[:, 0].max().item()) + 1
            x0 = int(coords[:, 1].min().item())
            x1 = int(coords[:, 1].max().item()) + 1
            cy = 0.5 * (y0 + y1)
            cx = 0.5 * (x0 + x1)
            hh = max(8.0, (y1 - y0) * scale)
            ww = max(8.0, (x1 - x0) * scale)
            y0 = max(0, int(cy - hh / 2))
            y1 = min(height, int(cy + hh / 2))
            x0 = max(0, int(cx - ww / 2))
            x1 = min(width, int(cx + ww / 2))
        crop = sample[:, y0:y1, x0:x1].unsqueeze(0)
        crop = torch.nn.functional.interpolate(
            crop,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        cropped.append(crop.squeeze(0))
    return torch.stack(cropped, dim=0)


def roi_project_views(images: torch.Tensor, projection_fn) -> torch.Tensor:
    projections = []
    for view_index in range(images.shape[1]):
        projected = projection_fn(images[:, view_index])
        projections.append(crop_batch_2d(projected))
    return torch.stack(projections, dim=1)


def handcrafted_features(images: torch.Tensor) -> torch.Tensor:
    feats = []
    for view_index in range(images.shape[1]):
        view = images[:, view_index]
        mean_img = mean_projection(view)
        max_img = max_projection(view)
        bone_ratio = (view > 0.6).float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        mean_intensity = view.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        std_intensity = view.std(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        max_intensity = view.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=images.device,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=images.device,
        ).unsqueeze(0)
        edge_x = torch.nn.functional.conv2d(mean_img, sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(mean_img, sobel_y, padding=1)
        edge_energy = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        contrast = (max_img - mean_img).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        feats.append(torch.cat([mean_intensity, std_intensity, max_intensity, bone_ratio, edge_energy, contrast], dim=1))
    return torch.cat(feats, dim=1)


class Timm2DEncoder(nn.Module):
    def __init__(self, model_name: str, in_chans: int = 1, pretrained: bool = False) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = int(getattr(self.model, "num_features"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled, weights


class SliceSetEncoder(nn.Module):
    def __init__(self, encoder: Timm2DEncoder, pool: str = "mean") -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.attention = AttentionPool(encoder.feature_dim) if pool == "attention" else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, views, slices, height, width = images.shape
        flat = images.reshape(batch * views * slices, 1, height, width)
        features = self.encoder(flat).reshape(batch, views, slices, -1)
        if self.pool == "mean":
            return features.mean(dim=2)
        if self.pool == "max":
            return features.amax(dim=2)
        if self.pool == "attention":
            pooled = []
            for view_index in range(views):
                pooled_view, _ = self.attention(features[:, view_index])
                pooled.append(pooled_view)
            return torch.stack(pooled, dim=1)
        raise ValueError(f"Unsupported pool: {self.pool}")


class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class MBConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand: int = 4, stride: tuple[int, int, int] = (1, 1, 1)) -> None:
        super().__init__()
        mid = in_ch * expand
        self.use_residual = stride == (1, 1, 1) and in_ch == out_ch
        layers: list[nn.Module] = []
        if expand != 1:
            layers.extend([nn.Conv3d(in_ch, mid, kernel_size=1, bias=False), nn.BatchNorm3d(mid), nn.SiLU()])
        else:
            mid = in_ch
        layers.extend(
            [
                nn.Conv3d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=mid, bias=False),
                nn.BatchNorm3d(mid),
                nn.SiLU(),
                SE3D(mid),
                nn.Conv3d(mid, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNet3DEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, base_channels, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            MBConv3d(base_channels, base_channels, expand=1),
            MBConv3d(base_channels, base_channels * 2, stride=(1, 2, 2)),
            MBConv3d(base_channels * 2, base_channels * 2),
            MBConv3d(base_channels * 2, base_channels * 4, stride=(2, 2, 2)),
            MBConv3d(base_channels * 4, base_channels * 4),
            MBConv3d(base_channels * 4, base_channels * 6, stride=(2, 2, 2)),
        )
        self.head = nn.Sequential(
            nn.Conv3d(base_channels * 6, base_channels * 8, kernel_size=1, bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.feature_dim = base_channels * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.flatten(1)


class R3D18Encoder(nn.Module):
    def __init__(self, in_chans: int = 1) -> None:
        super().__init__()
        self.model = r3d_18(weights=None)
        old_conv = self.model.stem[0]
        new_conv = nn.Conv3d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if old_conv.weight.shape[1] >= 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.model.stem[0] = new_conv
        self.model.fc = nn.Identity()
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_3d_encoder(name: str) -> nn.Module:
    if name == "efficient3d":
        return EfficientNet3DEncoder()
    if name == "r3d18":
        return R3D18Encoder()
    raise ValueError(f"Unsupported 3D encoder: {name}")


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 2, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out_conv = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class CrossViewSwapBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        global_context = normalized.mean(dim=1, keepdim=True).expand_as(normalized)
        gates = self.gate(torch.cat([normalized, global_context], dim=-1))
        swapped = gates * global_context + (1 - gates) * normalized
        return tokens + self.proj(swapped)


class MambaLiteBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        x = self.norm(tokens)
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        value = self.dwconv(value.transpose(1, 2)).transpose(1, 2)
        state = []
        prev = torch.zeros_like(value[:, 0])
        for index in range(value.shape[1]):
            update = torch.tanh(gate[:, index]) * value[:, index] + 0.5 * prev
            state.append(update)
            prev = update
        stacked = torch.stack(state, dim=1)
        return residual + self.out_proj(stacked)


class PrototypeHead(nn.Module):
    def __init__(self, dim: int, num_classes: int = 2, temperature: float = 0.07) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=-1)
        prototypes = torch.nn.functional.normalize(self.prototypes, dim=-1)
        return x @ prototypes.t() / self.temperature


class TriViewVolumeClassifier(nn.Module):
    def __init__(self, encoder_name: str, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = build_3d_encoder(encoder_name)
        feature_dim = int(getattr(self.encoder, "feature_dim"))
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        for view_index in range(images.shape[1]):
            volume = images[:, view_index].unsqueeze(1)
            features.append(self.encoder(volume))
        merged = torch.cat(features, dim=1)
        return self.head(merged)


class SeResNet50TripViewClassifier(nn.Module):
    def __init__(self, encoder_name: str = "seresnet50", pool: str = "attention", hidden_dim: int = 256, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.slice_encoder = SliceSetEncoder(Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained), pool=pool)
        feature_dim = self.slice_encoder.encoder.feature_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.slice_encoder(images)
        return self.head(features.reshape(features.shape[0], -1))


class XFMambaLiteClassifier(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", pool: str = "attention", hidden_dim: int = 256, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.slice_encoder = SliceSetEncoder(Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained), pool=pool)
        dim = self.slice_encoder.encoder.feature_dim
        self.swap = CrossViewSwapBlock(dim)
        self.deep = MambaLiteBlock(dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.slice_encoder(images)
        tokens = self.swap(tokens)
        tokens = self.deep(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens.reshape(tokens.shape[0], -1))


class DecisionFusionSingleExpertClassifier(nn.Module):
    def __init__(
        self,
        view_index: int,
        encoder_name: str = "densenet121",
        dropout: float = 0.3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.view_index = view_index
        self.encoder = Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained)
        feature_dim = self.encoder.feature_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projected = mean_projection(images[:, self.view_index])
        features = self.encoder(crop_batch_2d(projected))
        return self.head(features)


class DecisionFusionExpertsClassifier(nn.Module):
    def __init__(self, encoder_name: str = "densenet121", weights: Iterable[float] | None = None, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained) for _ in range(3)])
        feature_dim = self.encoders[0].feature_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim // 2, 2),
                )
                for _ in range(3)
            ]
        )
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        weight_tensor = torch.tensor(list(weights), dtype=torch.float32)
        self.register_buffer("weights", weight_tensor / weight_tensor.sum())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = []
        for view_index, encoder in enumerate(self.encoders):
            projected = mean_projection(images[:, view_index])
            features = encoder(crop_batch_2d(projected))
            logits.append(self.heads[view_index](features))
        stacked = torch.stack(logits, dim=1)
        return (stacked * self.weights.view(1, -1, 1)).sum(dim=1)


class SnapshotMultiViewClassifier(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", dropout: float = 0.3, hidden_dim: int = 256, view_dropout: float = 0.0, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained)
        self.view_dropout = view_dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projection_fns = (center_slice, mean_projection, max_projection, std_projection)
        snapshots = []
        for view_index in range(images.shape[1]):
            view = images[:, view_index]
            for projection_fn in projection_fns:
                snapshots.append(crop_batch_2d(projection_fn(view)))
        stacked = torch.stack(snapshots, dim=1)
        if self.training and self.view_dropout > 0:
            mask = (torch.rand(stacked.shape[:2], device=images.device) > self.view_dropout).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            stacked = stacked * mask
        batch, snaps, channels, height, width = stacked.shape
        features = self.encoder(stacked.reshape(batch * snaps, channels, height, width))
        logits = self.classifier(features).reshape(batch, snaps, 2)
        return logits.mean(dim=1)


class Mil25DClassifier(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", hidden_dim: int = 256, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = Timm2DEncoder(encoder_name, in_chans=3, pretrained=pretrained)
        self.pool = AttentionPool(self.encoder.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        instances = []
        for view_index in range(images.shape[1]):
            triplets = stack_triplets(images[:, view_index])
            instances.append(triplets)
        bag = torch.cat(instances, dim=1)
        batch, instances_count, channels, height, width = bag.shape
        features = self.encoder(bag.reshape(batch * instances_count, channels, height, width))
        pooled, _ = self.pool(features.reshape(batch, instances_count, -1))
        return self.classifier(pooled)


class TriPlaneHybridClassifier(nn.Module):
    def __init__(self, encoder_name: str = "resnet50", hidden_dim: int = 256, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained)
        feature_dim = self.encoder.feature_dim
        handcrafted_dim = 18
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3 + handcrafted_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projections = roi_project_views(images, center_slice)
        batch, views, channels, height, width = projections.shape
        deep = self.encoder(projections.reshape(batch * views, channels, height, width)).reshape(batch, views, -1)
        feats = handcrafted_features(images)
        return self.classifier(torch.cat([deep.reshape(batch, -1), feats], dim=1))


class SequenceCnnLstmClassifier(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", hidden_dim: int = 256, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained)
        self.lstm = nn.LSTM(
            input_size=self.encoder.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for view_index in range(images.shape[1]):
            view = images[:, view_index]
            batch, slices, height, width = view.shape
            flat_features = self.encoder(view.reshape(batch * slices, 1, height, width)).reshape(batch, slices, -1)
            sequence, _ = self.lstm(flat_features)
            outputs.append(sequence[:, -1])
        return self.classifier(torch.cat(outputs, dim=1))


class Hybrid25D3DEnsembleClassifier(nn.Module):
    def __init__(
        self,
        encoder_25d: str = "vit_small_patch16_224",
        encoder_3d: str = "r3d18",
        alpha: float = 0.5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained_25d: bool = False,
        pool_25d: str = "attention",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.slice_encoder25 = SliceSetEncoder(
            Timm2DEncoder(encoder_25d, in_chans=1, pretrained=pretrained_25d),
            pool=pool_25d,
        )
        feature_dim25 = self.slice_encoder25.encoder.feature_dim
        self.head25 = nn.Sequential(
            nn.Linear(feature_dim25 * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.encoder3 = build_3d_encoder(encoder_3d)
        feature_dim3 = int(getattr(self.encoder3, "feature_dim"))
        self.head3 = nn.Sequential(
            nn.Linear(feature_dim3 * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def set_training_stage(self, mode: str, unfreeze_last_blocks: int = 2) -> None:
        self._set_trainable(self.slice_encoder25, False)
        self._set_trainable(self.encoder3, False)
        self._set_trainable(self.head25, True)
        self._set_trainable(self.head3, True)

        if mode == "head_only":
            return
        if mode == "partial_25d":
            encoder25 = self.slice_encoder25.encoder.model
            blocks = getattr(encoder25, "blocks", None)
            if blocks is not None and len(blocks) > 0:
                self._set_trainable(encoder25.norm, True)
                for block in blocks[-max(1, int(unfreeze_last_blocks)) :]:
                    self._set_trainable(block, True)
            else:
                self._set_trainable(self.slice_encoder25, True)
            return
        if mode == "full":
            self._set_trainable(self.slice_encoder25, True)
            self._set_trainable(self.encoder3, True)
            return
        raise ValueError(f"Unsupported D4 training stage: {mode}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat25 = self.slice_encoder25(images)
        logits25 = self.head25(feat25.reshape(feat25.shape[0], -1))

        features3d = []
        for view_index in range(images.shape[1]):
            volume = images[:, view_index].unsqueeze(1)
            features3d.append(self.encoder3(volume))
        logits3 = self.head3(torch.cat(features3d, dim=1))
        return self.alpha * logits25 + (1.0 - self.alpha) * logits3


class AnatomyAwarePrototypeClassifier(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", hidden_dim: int = 256, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = Timm2DEncoder(encoder_name, in_chans=1, pretrained=pretrained)
        self.pre_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prototype_head = PrototypeHead(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projections = roi_project_views(images, mean_projection)
        batch, views, channels, height, width = projections.shape
        features = self.encoder(projections.reshape(batch * views, channels, height, width)).reshape(batch, views, -1)
        embedding = self.pre_head(features.reshape(batch, -1))
        return self.prototype_head(embedding)


class DenseVoteUNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        aggregation: str = "topk",
        patch_size: Sequence[int] = (8, 64, 64),
        patch_stride: Sequence[int] = (4, 48, 48),
        max_patches_per_view: int = 6,
        roi_threshold: float = 0.55,
        roi_margin: Sequence[int] = (1, 12, 12),
        voxel_topk_fraction: float = 0.05,
        patch_topk_fraction: float = 0.25,
        majority_temperature: float = 12.0,
        base_channels: int = 8,
    ) -> None:
        super().__init__()
        self.unet = SimpleUNet3D(base=base_channels)
        self.aggregation = aggregation
        self.patch_size = tuple(int(item) for item in patch_size)
        self.patch_stride = tuple(int(item) for item in patch_stride)
        self.max_patches_per_view = int(max_patches_per_view)
        self.roi_threshold = float(roi_threshold)
        self.roi_margin = tuple(int(item) for item in roi_margin)
        self.voxel_topk_fraction = float(voxel_topk_fraction)
        self.patch_topk_fraction = float(patch_topk_fraction)
        self.majority_temperature = float(majority_temperature)

    def _voxel_score(self, positive_map: torch.Tensor) -> torch.Tensor:
        flat = positive_map.flatten(1)
        if self.aggregation == "topk":
            topk = max(1, int(math.ceil(flat.shape[1] * self.voxel_topk_fraction)))
            return flat.topk(topk, dim=1).values.mean(dim=1)
        if self.aggregation == "majority":
            return torch.sigmoid((positive_map - 0.5) * self.majority_temperature).mean(dim=(1, 2, 3))
        return flat.mean(dim=1)

    def _bag_score(self, patch_scores: torch.Tensor) -> torch.Tensor:
        if patch_scores.numel() == 0:
            return patch_scores.new_tensor(0.5)
        if self.aggregation == "topk":
            topk = max(1, int(math.ceil(patch_scores.numel() * self.patch_topk_fraction)))
            return patch_scores.topk(topk).values.mean()
        if self.aggregation == "majority":
            return torch.sigmoid((patch_scores - 0.5) * self.majority_temperature).mean()
        return patch_scores.mean()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_batch, patch_counts = extract_candidate_volume_patches(
            images,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            roi_threshold=self.roi_threshold,
            roi_margin=self.roi_margin,
            max_patches_per_view=self.max_patches_per_view,
        )
        dense_logits = self.unet(patch_batch.unsqueeze(1))
        positive_map = torch.softmax(dense_logits, dim=1)[:, 1]
        patch_scores = self._voxel_score(positive_map)

        sample_scores = []
        cursor = 0
        for patch_count in patch_counts:
            sample_patch_scores = patch_scores[cursor : cursor + patch_count]
            sample_scores.append(self._bag_score(sample_patch_scores))
            cursor += patch_count
        stacked_scores = torch.stack(sample_scores, dim=0)
        return torch.stack([1.0 - stacked_scores, stacked_scores], dim=1)


def build_c3_expert_model(config: dict, view_index: int) -> nn.Module:
    model_cfg = config["model"]
    return DecisionFusionSingleExpertClassifier(
        view_index=view_index,
        encoder_name=str(model_cfg.get("encoder_name", "densenet121")),
        dropout=float(model_cfg.get("dropout", 0.3)),
        pretrained=bool(model_cfg.get("pretrained", False)),
    )


def build_paper_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    family = str(model_cfg["family"]).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    dropout = float(model_cfg.get("dropout", 0.3))
    encoder_name = str(model_cfg.get("encoder_name", "resnet18"))
    pretrained = bool(model_cfg.get("pretrained", False))

    if family == "s1_3d_efficient":
        return TriViewVolumeClassifier(
            encoder_name=str(model_cfg.get("encoder_3d", "efficient3d")),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if family == "s2_se_resnet50":
        return SeResNet50TripViewClassifier(
            encoder_name=encoder_name,
            pool=str(model_cfg.get("slice_pool", "attention")),
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "s3_anatomy_prototype":
        return AnatomyAwarePrototypeClassifier(
            encoder_name=encoder_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "s4_cnn_lstm":
        return SequenceCnnLstmClassifier(
            encoder_name=encoder_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "d1_mil_25d":
        return Mil25DClassifier(
            encoder_name=encoder_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "d2_triplane_hybrid":
        return TriPlaneHybridClassifier(
            encoder_name=encoder_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "d4_hybrid_25d_3d":
        return Hybrid25D3DEnsembleClassifier(
            encoder_25d=str(model_cfg.get("encoder_25d", "vit_small_patch16_224")),
            encoder_3d=str(model_cfg.get("encoder_3d", "r3d18")),
            alpha=float(model_cfg.get("alpha", 0.5)),
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained_25d=bool(model_cfg.get("pretrained_25d", False)),
            pool_25d=str(model_cfg.get("pool_25d", "attention")),
        )
    if family == "c1_xfmamba_lite":
        return XFMambaLiteClassifier(
            encoder_name=encoder_name,
            pool=str(model_cfg.get("slice_pool", "attention")),
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "c2_snapshot_multiview":
        return SnapshotMultiViewClassifier(
            encoder_name=encoder_name,
            dropout=dropout,
            hidden_dim=hidden_dim,
            view_dropout=float(model_cfg.get("view_dropout", 0.0)),
            pretrained=pretrained,
        )
    if family == "c3_decision_fusion":
        return DecisionFusionExpertsClassifier(
            encoder_name=encoder_name,
            weights=model_cfg.get("decision_weights"),
            dropout=dropout,
            pretrained=pretrained,
        )
    if family == "r1_fracnet_weak":
        return DenseVoteUNetClassifier(
            aggregation="topk",
            patch_size=model_cfg.get("patch_size", (8, 64, 64)),
            patch_stride=model_cfg.get("patch_stride", (4, 48, 48)),
            max_patches_per_view=int(model_cfg.get("max_patches_per_view", 6)),
            roi_threshold=float(model_cfg.get("roi_threshold", 0.55)),
            roi_margin=model_cfg.get("roi_margin", (1, 12, 12)),
            voxel_topk_fraction=float(model_cfg.get("voxel_topk_fraction", 0.05)),
            patch_topk_fraction=float(model_cfg.get("patch_topk_fraction", 0.25)),
            majority_temperature=float(model_cfg.get("majority_temperature", 12.0)),
            base_channels=int(model_cfg.get("base_channels", 8)),
        )
    if family == "r4_dense_vote":
        return DenseVoteUNetClassifier(
            aggregation="majority",
            patch_size=model_cfg.get("patch_size", (8, 64, 64)),
            patch_stride=model_cfg.get("patch_stride", (4, 48, 48)),
            max_patches_per_view=int(model_cfg.get("max_patches_per_view", 6)),
            roi_threshold=float(model_cfg.get("roi_threshold", 0.55)),
            roi_margin=model_cfg.get("roi_margin", (1, 12, 12)),
            voxel_topk_fraction=float(model_cfg.get("voxel_topk_fraction", 0.05)),
            patch_topk_fraction=float(model_cfg.get("patch_topk_fraction", 0.25)),
            majority_temperature=float(model_cfg.get("majority_temperature", 12.0)),
            base_channels=int(model_cfg.get("base_channels", 8)),
        )
    raise ValueError(f"Unsupported paper model family: {family}")
