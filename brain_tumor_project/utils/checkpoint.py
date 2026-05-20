"""Checkpoint management — save / load / track best model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manages saving, loading, and tracking of model checkpoints.

    Args:
        ckpt_dir: Directory to store checkpoint files.
        monitor: Metric name to monitor (e.g. ``"val_dice"``).
        mode: ``"max"`` if higher is better, ``"min"`` if lower is better.
    """

    def __init__(
        self,
        ckpt_dir: str = "checkpoints",
        monitor: str = "val_dice",
        mode: str = "max",
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score: Optional[float] = None
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        metric_value: float,
        save_best: bool = True,
        save_last: bool = True,
    ) -> None:
        """Save checkpoint, optionally updating the best-model file.

        Args:
            state: Dictionary containing ``model_state_dict``, ``optimizer_state_dict``,
                   ``epoch``, ``metrics``, etc.
            epoch: Current epoch number.
            metric_value: Value of the monitored metric.
            save_best: Whether to overwrite ``best_model.pth`` if improved.
            save_last: Whether to always save ``last_model.pth``.
        """
        if save_last:
            path = os.path.join(self.ckpt_dir, "last_model.pth")
            torch.save(state, path)

        if save_best and self._is_improvement(metric_value):
            self.best_score = metric_value
            path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(state, path)

    def load(self, path: str, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        """Load a checkpoint from disk.

        Args:
            path: Path to the ``.pth`` file.
            device: Device to map tensors onto.

        Returns:
            Dictionary with saved state.
        """
        return torch.load(path, map_location=device, weights_only=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return value > self.best_score
        return value < self.best_score
