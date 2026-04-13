"""Cross-View Attention module.

Allows the three anatomical views (axial, coronal, sagittal) to exchange
information via multi-head self-attention before being fused into a
patient-level representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossViewAttention(nn.Module):
    """Multi-head self-attention across views.

    Input : ``(B, num_views, D)``
    Output: ``(B, num_views, D)`` – view features enriched with cross-view context.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of each view feature (e.g. 512 for ResNet18).
    num_heads : int
        Number of attention heads.  Must divide ``feature_dim``.
    num_layers : int
        Number of stacked Transformer encoder layers.
    dropout : float
        Dropout rate inside attention and feed-forward.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        view_features : Tensor
            Shape ``(B, num_views, D)``.

        Returns
        -------
        Tensor
            Shape ``(B, num_views, D)`` – cross-view-attended features.
        """
        # Transformer encoder with residual connection
        enhanced = self.encoder(view_features)
        enhanced = self.norm(enhanced + view_features)
        return enhanced


class CrossViewMeanContextMixer(nn.Module):
    """Lightweight residual mixer using the mean of the other views.

    Compared with transformer-style cross-view attention, this module injects
    only a small amount of context after slice pooling:
    1. For each view token, compute the mean feature of the other views.
    2. Concatenate ``[self_feature, other_view_mean]``.
    3. Predict a residual update through a bottleneck MLP.
    4. Add the residual back and normalize.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """Apply low-capacity cross-view context mixing.

        Parameters
        ----------
        view_features : Tensor
            Shape ``(B, num_views, D)``.

        Returns
        -------
        Tensor
            Shape ``(B, num_views, D)``.
        """
        num_views = view_features.shape[1]
        if num_views < 2:
            return view_features

        summed = view_features.sum(dim=1, keepdim=True)
        other_view_mean = (summed - view_features) / float(num_views - 1)
        residual = self.mlp(torch.cat([view_features, other_view_mean], dim=-1))
        return self.norm(view_features + residual)
