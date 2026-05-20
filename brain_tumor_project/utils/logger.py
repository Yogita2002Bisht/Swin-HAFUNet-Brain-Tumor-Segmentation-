"""Logging utilities — file + console + optional TensorBoard."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def setup_logger(
    name: str = "swin_hafunet",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to console and a log file.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_tensorboard_writer(log_dir: str = "logs") -> "SummaryWriter":
    """Return a TensorBoard ``SummaryWriter``.

    Args:
        log_dir: Directory for TensorBoard event files.

    Returns:
        A ``SummaryWriter`` instance.
    """
    from torch.utils.tensorboard import SummaryWriter

    tb_dir = os.path.join(log_dir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=tb_dir)
