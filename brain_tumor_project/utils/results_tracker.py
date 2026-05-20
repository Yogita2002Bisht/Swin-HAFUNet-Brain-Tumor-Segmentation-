"""Results tracker — saves CSV tables, matplotlib charts, and timing logs.

Generates the following outputs inside ``<output_dir>/``:
  - ``results.csv``           — structured table of all epoch metrics
  - ``timing.csv``            — per-epoch wall-clock times (train, val, total)
  - ``training_summary.txt``  — human-readable final summary
  - ``loss_curve.png``        — train/val loss plot
  - ``dice_curve.png``        — train/val Dice plot
  - ``iou_curve.png``         — val IoU plot
  - ``precision_recall.png``  — val precision & recall plot
  - ``lr_schedule.png``       — learning-rate schedule plot
  - ``timing_chart.png``      — per-epoch timing breakdown
  - ``all_metrics.png``       — combined 2×3 dashboard
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResultsTracker:
    """Accumulates per-epoch results and writes CSV / charts / summary.

    Args:
        output_dir: Root directory for all outputs.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Accumulators
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_dices: List[float] = []
        self.val_dices: List[float] = []
        self.val_ious: List[float] = []
        self.val_precisions: List[float] = []
        self.val_recalls: List[float] = []
        self.learning_rates: List[float] = []

        # Timing
        self.train_times: List[float] = []   # seconds
        self.val_times: List[float] = []
        self.epoch_times: List[float] = []

        # Global timers
        self._training_start: Optional[float] = None
        self._training_end: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self._phase_start: Optional[float] = None

        # CSV files — write headers once
        self._results_csv = os.path.join(output_dir, "results.csv")
        self._timing_csv = os.path.join(output_dir, "timing.csv")
        self._write_csv_headers()

    # ------------------------------------------------------------------
    # Timer helpers
    # ------------------------------------------------------------------

    def start_training(self) -> None:
        """Call once before the training loop begins."""
        self._training_start = time.time()

    def end_training(self) -> None:
        """Call once after the training loop ends."""
        self._training_end = time.time()

    def start_epoch(self) -> None:
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()

    def start_phase(self) -> None:
        """Call at the start of a phase (train or val)."""
        self._phase_start = time.time()

    def end_train_phase(self) -> float:
        """Call after the training phase ends. Returns seconds elapsed."""
        elapsed = time.time() - self._phase_start
        self.train_times.append(elapsed)
        return elapsed

    def end_val_phase(self) -> float:
        """Call after the validation phase ends. Returns seconds elapsed."""
        elapsed = time.time() - self._phase_start
        self.val_times.append(elapsed)
        return elapsed

    def end_epoch(self) -> float:
        """Call at the end of each epoch. Returns total epoch seconds."""
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        return elapsed

    # ------------------------------------------------------------------
    # Record metrics
    # ------------------------------------------------------------------

    def record_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        """Store one epoch's metrics and append to CSVs."""
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics["train_loss"])
        self.train_dices.append(train_metrics["train_dice"])
        self.val_losses.append(val_metrics["val_loss"])
        self.val_dices.append(val_metrics["val_dice"])
        self.val_ious.append(val_metrics["val_iou"])
        self.val_precisions.append(val_metrics["val_precision"])
        self.val_recalls.append(val_metrics["val_recall"])
        self.learning_rates.append(lr)

        # Append row to results CSV
        with open(self._results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['train_loss']:.6f}",
                f"{train_metrics['train_dice']:.6f}",
                f"{val_metrics['val_loss']:.6f}",
                f"{val_metrics['val_dice']:.6f}",
                f"{val_metrics['val_iou']:.6f}",
                f"{val_metrics['val_precision']:.6f}",
                f"{val_metrics['val_recall']:.6f}",
                f"{lr:.8f}",
            ])

        # Append timing row
        train_t = self.train_times[-1] if self.train_times else 0.0
        val_t = self.val_times[-1] if self.val_times else 0.0
        epoch_t = self.epoch_times[-1] if self.epoch_times else 0.0
        with open(self._timing_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_t:.2f}",
                f"{val_t:.2f}",
                f"{epoch_t:.2f}",
                f"{train_t / 60:.2f}",
                f"{val_t / 60:.2f}",
                f"{epoch_t / 60:.2f}",
            ])

    # ------------------------------------------------------------------
    # Generate all outputs at the end
    # ------------------------------------------------------------------

    def save_all(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        """Generate charts + summary. Call after training completes.

        Returns:
            Path to the output directory.
        """
        self._plot_loss_curve()
        self._plot_dice_curve()
        self._plot_iou_curve()
        self._plot_precision_recall()
        self._plot_lr_schedule()
        self._plot_timing()
        self._plot_dashboard()
        self._write_summary(cfg)
        return self.output_dir

    # ------------------------------------------------------------------
    # CSV headers
    # ------------------------------------------------------------------

    def _write_csv_headers(self) -> None:
        with open(self._results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_dice",
                "val_loss", "val_dice", "val_iou",
                "val_precision", "val_recall", "learning_rate",
            ])
        with open(self._timing_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_sec", "val_sec", "total_sec",
                "train_min", "val_min", "total_min",
            ])

    # ------------------------------------------------------------------
    # Matplotlib charts
    # ------------------------------------------------------------------

    def _plot_loss_curve(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.train_losses, "b-o", markersize=3, label="Train Loss")
        ax.plot(self.epochs, self.val_losses, "r-o", markersize=3, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_dice_curve(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.train_dices, "b-o", markersize=3, label="Train Dice")
        ax.plot(self.epochs, self.val_dices, "r-o", markersize=3, label="Val Dice")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice Score")
        ax.set_title("Training & Validation Dice Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "dice_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_iou_curve(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.val_ious, "g-o", markersize=3, label="Val IoU")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_title("Validation IoU (Jaccard Index)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "iou_curve.png"), dpi=150)
        plt.close(fig)

    def _plot_precision_recall(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.val_precisions, "m-o", markersize=3, label="Val Precision")
        ax.plot(self.epochs, self.val_recalls, "c-o", markersize=3, label="Val Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Precision & Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "precision_recall.png"), dpi=150)
        plt.close(fig)

    def _plot_lr_schedule(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.learning_rates, "k-", linewidth=2, label="Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "lr_schedule.png"), dpi=150)
        plt.close(fig)

    def _plot_timing(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        x = list(range(len(self.epochs)))

        train_mins = [t / 60 for t in self.train_times]
        val_mins = [t / 60 for t in self.val_times]

        ax.bar([i - bar_width / 2 for i in x], train_mins, bar_width, label="Train (min)", color="steelblue")
        ax.bar([i + bar_width / 2 for i in x], val_mins, bar_width, label="Val (min)", color="coral")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (minutes)")
        ax.set_title("Per-Epoch Timing Breakdown")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Only show a subset of x-tick labels if many epochs
        if len(self.epochs) > 20:
            step = max(1, len(self.epochs) // 15)
            ax.set_xticks([x[i] for i in range(0, len(x), step)])
            ax.set_xticklabels([self.epochs[i] for i in range(0, len(self.epochs), step)])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(self.epochs)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "timing_chart.png"), dpi=150)
        plt.close(fig)

    def _plot_dashboard(self) -> None:
        """Combined 2×3 subplot dashboard with all key charts."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # (0,0) Loss
        axes[0, 0].plot(self.epochs, self.train_losses, "b-", label="Train")
        axes[0, 0].plot(self.epochs, self.val_losses, "r-", label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # (0,1) Dice
        axes[0, 1].plot(self.epochs, self.train_dices, "b-", label="Train")
        axes[0, 1].plot(self.epochs, self.val_dices, "r-", label="Val")
        axes[0, 1].set_title("Dice Score")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # (0,2) IoU
        axes[0, 2].plot(self.epochs, self.val_ious, "g-", label="Val IoU")
        axes[0, 2].set_title("IoU")
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # (1,0) Precision & Recall
        axes[1, 0].plot(self.epochs, self.val_precisions, "m-", label="Precision")
        axes[1, 0].plot(self.epochs, self.val_recalls, "c-", label="Recall")
        axes[1, 0].set_title("Precision & Recall")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # (1,1) LR
        axes[1, 1].plot(self.epochs, self.learning_rates, "k-")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axes[1, 1].grid(True, alpha=0.3)

        # (1,2) Timing
        if self.epoch_times:
            epoch_mins = [t / 60 for t in self.epoch_times]
            axes[1, 2].bar(self.epochs, epoch_mins, color="steelblue", alpha=0.8)
            axes[1, 2].set_title("Epoch Time (min)")
            axes[1, 2].grid(True, alpha=0.3, axis="y")

        for ax in axes.flat:
            ax.set_xlabel("Epoch")

        fig.suptitle("Swin-HAFUNet Training Dashboard", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(self.output_dir, "all_metrics.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def _write_summary(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        """Write a human-readable training summary text file."""
        import torch

        total_time = (self._training_end or 0) - (self._training_start or 0)

        best_val_dice = max(self.val_dices) if self.val_dices else 0.0
        best_epoch = self.epochs[self.val_dices.index(best_val_dice)] if self.val_dices else 0
        best_idx = self.val_dices.index(best_val_dice) if self.val_dices else 0

        lines = [
            "=" * 70,
            "  SWIN-HAFUNET BRAIN TUMOR SEGMENTATION — TRAINING SUMMARY",
            "=" * 70,
            "",
            "─── Configuration ───────────────────────────────────────────",
        ]

        if cfg:
            lines += [
                f"  Project         : {cfg.get('project', {}).get('name', 'N/A')}",
                f"  Dataset         : {cfg.get('data', {}).get('dataset', 'N/A')}",
                f"  Image size      : {cfg.get('transforms', {}).get('image_size', 'N/A')}",
                f"  Batch size      : {cfg.get('training', {}).get('batch_size', 'N/A')}",
                f"  Epochs (config) : {cfg.get('training', {}).get('epochs', 'N/A')}",
                f"  Epochs (actual) : {len(self.epochs)}",
                f"  Optimizer       : AdamW (lr={cfg.get('optimizer', {}).get('lr', 'N/A')})",
                f"  Scheduler       : CosineAnnealing",
                f"  Loss            : {cfg.get('loss', {}).get('name', 'N/A')}",
                f"  Mixed Precision : {cfg.get('training', {}).get('mixed_precision', False)}",
                f"  Seed            : {cfg.get('project', {}).get('seed', 'N/A')}",
            ]

        lines += [
            "",
            "─── Device ──────────────────────────────────────────────────",
            f"  Device          : {'cuda' if torch.cuda.is_available() else 'cpu'}",
        ]
        if torch.cuda.is_available():
            lines += [
                f"  GPU             : {torch.cuda.get_device_name(0)}",
                f"  VRAM            : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB",
            ]

        lines += [
            "",
            "─── Best Results ────────────────────────────────────────────",
            f"  Best Val Dice   : {best_val_dice:.6f}  (epoch {best_epoch})",
            f"  Val IoU         : {self.val_ious[best_idx]:.6f}",
            f"  Val Precision   : {self.val_precisions[best_idx]:.6f}",
            f"  Val Recall      : {self.val_recalls[best_idx]:.6f}",
            f"  Val Loss        : {self.val_losses[best_idx]:.6f}",
            "",
            "─── Final Epoch Results ─────────────────────────────────────",
        ]

        if self.epochs:
            lines += [
                f"  Train Loss      : {self.train_losses[-1]:.6f}",
                f"  Train Dice      : {self.train_dices[-1]:.6f}",
                f"  Val Loss        : {self.val_losses[-1]:.6f}",
                f"  Val Dice        : {self.val_dices[-1]:.6f}",
                f"  Val IoU         : {self.val_ious[-1]:.6f}",
                f"  Val Precision   : {self.val_precisions[-1]:.6f}",
                f"  Val Recall      : {self.val_recalls[-1]:.6f}",
            ]

        lines += [
            "",
            "─── Timing ──────────────────────────────────────────────────",
            f"  Total training  : {total_time:.1f} sec ({total_time / 60:.2f} min / {total_time / 3600:.2f} hr)",
        ]

        if self.epoch_times:
            avg_epoch = sum(self.epoch_times) / len(self.epoch_times)
            avg_train = sum(self.train_times) / len(self.train_times) if self.train_times else 0
            avg_val = sum(self.val_times) / len(self.val_times) if self.val_times else 0
            fastest = min(self.epoch_times)
            slowest = max(self.epoch_times)
            lines += [
                f"  Avg epoch time  : {avg_epoch:.1f} sec ({avg_epoch / 60:.2f} min)",
                f"  Avg train phase : {avg_train:.1f} sec ({avg_train / 60:.2f} min)",
                f"  Avg val phase   : {avg_val:.1f} sec ({avg_val / 60:.2f} min)",
                f"  Fastest epoch   : {fastest:.1f} sec",
                f"  Slowest epoch   : {slowest:.1f} sec",
                f"  Train/Val ratio : {avg_train / max(avg_val, 0.01):.2f}x",
            ]

        lines += [
            "",
            "─── Output Files ────────────────────────────────────────────",
            f"  Results CSV     : {self._results_csv}",
            f"  Timing CSV      : {self._timing_csv}",
            f"  Charts          : {self.output_dir}/",
            f"  Checkpoints     : checkpoints/",
            f"  TensorBoard     : logs/tensorboard/",
            f"  Training log    : logs/train.log",
            "",
            "=" * 70,
        ]

        summary_path = os.path.join(self.output_dir, "training_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
