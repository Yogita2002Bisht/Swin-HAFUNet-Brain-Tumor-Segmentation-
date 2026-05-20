"""Training loop for Swin-HAFUNet with mixed-precision, gradient clipping,
early stopping, and checkpoint management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import SegmentationMetrics
from training.validator import Validator
from utils.checkpoint import CheckpointManager
from utils.logger import get_tensorboard_writer
from utils.results_tracker import ResultsTracker


class EarlyStopping:
    """Monitors a metric and signals when to stop training.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to qualify as progress.
        mode: ``"max"`` if higher is better, ``"min"`` otherwise.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        """Update with current metric value. Returns ``True`` if training should stop."""
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class Trainer:
    """Full training orchestrator.

    Args:
        model: The segmentation model.
        criterion: Loss module.
        optimizer: Optimizer instance.
        scheduler: LR scheduler (optional).
        train_loader: Training ``DataLoader``.
        val_loader: Validation ``DataLoader``.
        device: Torch device.
        cfg: Full configuration dictionary.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: Dict[str, Any],
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg

        # Training config
        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 150)
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.clip_norm = train_cfg.get("gradient_clip_max_norm", 1.0)
        self.accum_steps = train_cfg.get("accumulation_steps", 1)

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode="max",  # monitoring dice
        )
        self.monitor_key = es_cfg.get("monitor", "val_dice")

        # Validation
        eval_cfg = cfg.get("evaluation", {})
        self.validator = Validator(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            threshold=eval_cfg.get("threshold", 0.5),
        )

        # Checkpointing
        ckpt_cfg = cfg.get("checkpoint", {})
        self.ckpt_mgr = CheckpointManager(
            ckpt_dir=ckpt_cfg.get("dir", "checkpoints"),
            monitor=ckpt_cfg.get("monitor", "val_dice"),
            mode=ckpt_cfg.get("mode", "max"),
        )
        self.save_best = ckpt_cfg.get("save_best", True)
        self.save_last = ckpt_cfg.get("save_last", True)

        # Logging
        log_cfg = cfg.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "logs")
        self.log_every = log_cfg.get("log_every_n_steps", 10)
        self.logger = logging.getLogger("swin_hafunet")
        self.tb_writer = get_tensorboard_writer(self.log_dir) if log_cfg.get("tensorboard", True) else None

        # Metrics tracker for training
        self.train_metrics = SegmentationMetrics(threshold=eval_cfg.get("threshold", 0.5))

        # Results tracker — CSV tables, charts, timing
        results_dir = cfg.get("logging", {}).get("results_dir", "results")
        self.results_tracker = ResultsTracker(output_dir=results_dir)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the complete training loop."""
        self.logger.info(
            f"Starting training — {self.epochs} epochs, "
            f"AMP={'on' if self.use_amp else 'off'}, "
            f"device={self.device}"
        )

        self.results_tracker.start_training()

        for epoch in range(1, self.epochs + 1):
            self.results_tracker.start_epoch()

            # ── Train phase ──────────────────────────────────────────
            self.results_tracker.start_phase()
            train_metrics = self._train_one_epoch(epoch)
            train_secs = self.results_tracker.end_train_phase()

            # ── Validation phase ─────────────────────────────────────
            self.results_tracker.start_phase()
            val_metrics = self.validator.run(self.val_loader)
            val_secs = self.results_tracker.end_val_phase()

            epoch_secs = self.results_tracker.end_epoch()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"train_loss={train_metrics['train_loss']:.4f}  "
                f"train_dice={train_metrics['train_dice']:.4f}  |  "
                f"val_loss={val_metrics['val_loss']:.4f}  "
                f"val_dice={val_metrics['val_dice']:.4f}  "
                f"val_iou={val_metrics['val_iou']:.4f}  |  "
                f"lr={lr:.2e}  |  "
                f"time={epoch_secs:.1f}s (train={train_secs:.1f}s val={val_secs:.1f}s)"
            )

            if self.tb_writer is not None:
                self._log_tensorboard(epoch, train_metrics, val_metrics, lr)

            # Record to results tracker (CSV + accumulators)
            self.results_tracker.record_epoch(epoch, train_metrics, val_metrics, lr)

            # Checkpoint
            monitored = val_metrics.get(self.monitor_key, val_metrics["val_dice"])
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict(),
                "metrics": {**train_metrics, **val_metrics},
            }
            self.ckpt_mgr.save(state, epoch, monitored, self.save_best, self.save_last)

            # Early stopping
            if self.early_stopping.step(monitored):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.early_stopping.patience} epochs)."
                )
                break

        # ── End of training — save all results ───────────────────────
        self.results_tracker.end_training()
        out_dir = self.results_tracker.save_all(cfg=self.cfg)
        self.logger.info(f"Results saved to: {out_dir}/")

        if self.tb_writer is not None:
            self.tb_writer.close()

        self.logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch and return metrics.

        Returns:
            Dictionary with ``train_loss`` and ``train_dice``.
        """
        self.model.train()
        self.train_metrics.reset()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, 1):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if step % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * self.accum_steps
            n_batches += 1

            self.train_metrics.update(logits.detach(), masks)

            if step % self.log_every == 0:
                pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        m = self.train_metrics.compute()
        return {
            "train_loss": running_loss / max(n_batches, 1),
            "train_dice": m["dice"],
        }

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------

    def _log_tensorboard(
        self,
        epoch: int,
        train_m: Dict[str, float],
        val_m: Dict[str, float],
        lr: float,
    ) -> None:
        w = self.tb_writer
        w.add_scalar("Loss/train", train_m["train_loss"], epoch)
        w.add_scalar("Loss/val", val_m["val_loss"], epoch)
        w.add_scalar("Dice/train", train_m["train_dice"], epoch)
        w.add_scalar("Dice/val", val_m["val_dice"], epoch)
        w.add_scalar("IoU/val", val_m["val_iou"], epoch)
        w.add_scalar("Precision/val", val_m["val_precision"], epoch)
        w.add_scalar("Recall/val", val_m["val_recall"], epoch)
        w.add_scalar("LR", lr, epoch)
