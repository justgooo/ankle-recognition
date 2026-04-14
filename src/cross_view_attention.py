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
        attention_dim: int | None = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        attention_dim = int(attention_dim or feature_dim)
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads.")

        self.input_proj: nn.Module
        self.output_proj: nn.Module
        if attention_dim == feature_dim:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(feature_dim, attention_dim)
            self.output_proj = nn.Linear(attention_dim, feature_dim)

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
        self.norm = nn.LayerNorm(feature_dim)
        self.residual_scale = float(residual_scale)

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
        # Scale the cross-view residual branch when a lighter mixer is desired.
        mixed_features = self.input_proj(view_features)
        enhanced = self.encoder(mixed_features)
        enhanced = self.output_proj(enhanced)
        enhanced = self.norm(view_features + self.residual_scale * enhanced)
        return enhanced
