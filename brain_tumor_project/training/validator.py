"""Validation loop — runs one full pass over the validation set."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import SegmentationMetrics


class Validator:
    """Runs model evaluation on a DataLoader without gradient computation.

    Args:
        model: Segmentation model.
        criterion: Loss function.
        device: Torch device.
        threshold: Binarisation threshold for metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.metrics = SegmentationMetrics(threshold=threshold)

    @torch.no_grad()
    def run(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one full validation epoch.

        Args:
            dataloader: Validation ``DataLoader``.

        Returns:
            Dictionary with ``val_loss``, ``val_dice``, ``val_iou``,
            ``val_precision``, ``val_recall``.
        """
        self.model.eval()
        self.metrics.reset()
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            running_loss += loss.item()
            n_batches += 1
            self.metrics.update(logits, masks)

        m = self.metrics.compute()
        return {
            "val_loss": running_loss / max(n_batches, 1),
            "val_dice": m["dice"],
            "val_iou": m["iou"],
            "val_precision": m["precision"],
            "val_recall": m["recall"],
        }
