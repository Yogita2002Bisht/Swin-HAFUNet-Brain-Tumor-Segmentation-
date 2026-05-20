"""Segmentation evaluation metrics.

All metrics operate on **binary** predictions and targets.
Logits are converted to binary predictions using a configurable threshold.

Supported metrics:
  - **Dice Score** (F1 / Sørensen–Dice coefficient)
  - **IoU** (Intersection over Union / Jaccard index)
  - **Precision** (Positive Predictive Value)
  - **Recall** (Sensitivity / True Positive Rate)
"""

from __future__ import annotations

from typing import Dict

import torch


class SegmentationMetrics:
    """Accumulates per-batch statistics and computes epoch-level metrics.

    Args:
        threshold: Probability threshold for binarising sigmoid outputs.
        smooth: Small constant to avoid division by zero.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6) -> None:
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset accumulated counters."""
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0
        self._tn = 0.0
        self._dice_sum = 0.0
        self._iou_sum = 0.0
        self._count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions.

        Args:
            logits: Raw model output ``(B, 1, H, W)``.
            targets: Ground-truth binary masks ``(B, 1, H, W)``.
        """
        preds = (torch.sigmoid(logits) >= self.threshold).float()
        targets = targets.float()

        # Flatten each sample in batch
        preds_flat = preds.view(preds.size(0), -1)
        tgt_flat = targets.view(targets.size(0), -1)

        tp = (preds_flat * tgt_flat).sum(dim=1)
        fp = (preds_flat * (1 - tgt_flat)).sum(dim=1)
        fn = ((1 - preds_flat) * tgt_flat).sum(dim=1)
        tn = ((1 - preds_flat) * (1 - tgt_flat)).sum(dim=1)

        # Per-sample dice & IoU
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        self._tp += tp.sum().item()
        self._fp += fp.sum().item()
        self._fn += fn.sum().item()
        self._tn += tn.sum().item()
        self._dice_sum += dice.sum().item()
        self._iou_sum += iou.sum().item()
        self._count += preds.size(0)

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics over all accumulated batches.

        Returns:
            Dictionary with keys: ``dice``, ``iou``, ``precision``, ``recall``.
        """
        eps = self.smooth
        precision = self._tp / (self._tp + self._fp + eps)
        recall = self._tp / (self._tp + self._fn + eps)

        return {
            "dice": self._dice_sum / max(self._count, 1),
            "iou": self._iou_sum / max(self._count, 1),
            "precision": precision,
            "recall": recall,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """Quick single-batch Dice score (no accumulation)."""
        preds = (torch.sigmoid(logits) >= threshold).float()
        preds_flat = preds.view(-1)
        tgt_flat = targets.view(-1)
        inter = (preds_flat * tgt_flat).sum()
        return (2.0 * inter / (preds_flat.sum() + tgt_flat.sum() + 1e-6)).item()
