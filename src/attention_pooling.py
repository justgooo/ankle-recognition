"""Attention Pooling module.

Replaces naive mean-pooling over slice features with a learnable
attention-weighted aggregation so the model can focus on the most
informative slices within each view.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learnable attention pooling over a sequence dimension.

    Given input of shape ``(B, S, D)`` (batch, sequence, feature-dim),
    it produces ``(B, D)`` by computing attention weights over the
    sequence and returning the weighted sum.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of each token / slice feature.
    hidden_dim : int, optional
        Hidden size inside the attention scorer.  Defaults to ``feature_dim // 2``.
    """

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 2
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, S, D)``.

        Returns
        -------
        Tensor
            Shape ``(B, D)`` – attention-pooled feature.
        """
        # (B, S, 1)
        scores = self.attention(x)
        # (B, S, 1)
        weights = F.softmax(scores, dim=1)
        # (B, D)
        pooled = (x * weights).sum(dim=1)
        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled
