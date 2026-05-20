"""Combined loss: weighted sum of Dice Loss and BCE-with-Logits Loss.

    L = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from losses.dice_loss import DiceLoss


class CombinedLoss(nn.Module):
    """Weighted Dice + Binary Cross-Entropy compound loss.

    Args:
        dice_weight: Weight for the Dice component.
        bce_weight: Weight for the BCE component.
        smooth: Dice smoothing constant.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            logits: Raw model output ``(B, 1, H, W)``.
            targets: Ground-truth masks ``(B, 1, H, W)`` with values in {0, 1}.

        Returns:
            Scalar combined loss value.
        """
        d_loss = self.dice_loss(logits, targets)
        b_loss = self.bce_loss(logits, targets)
        return self.dice_weight * d_loss + self.bce_weight * b_loss
