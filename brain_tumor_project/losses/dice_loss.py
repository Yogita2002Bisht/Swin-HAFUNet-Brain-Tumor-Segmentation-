"""Soft Dice Loss for binary segmentation.

Works on raw logits — applies sigmoid internally so it can be
combined directly with ``BCEWithLogitsLoss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Differentiable Dice loss.

    Args:
        smooth: Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss.

        Args:
            logits: Raw model output ``(B, 1, H, W)``.
            targets: Ground-truth masks ``(B, 1, H, W)`` with values in {0, 1}.

        Returns:
            Scalar Dice loss (1 − Dice coefficient).
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dims
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
