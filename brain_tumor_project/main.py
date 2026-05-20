#!/usr/bin/env python3
"""Swin-HAFUNet Brain Tumor Segmentation — Main Entry Point.

Usage::

    python main.py --config configs/base_config.yaml

Orchestrates the full pipeline:
  1. Load YAML configuration
  2. Seed everything for reproducibility
  3. Build datasets & data loaders
  4. Instantiate model, loss, optimizer, scheduler
  5. Launch training with logging, checkpointing, and early stopping
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

# ── Local imports ────────────────────────────────────────────────────────────
from utils.seed import set_seed
from utils.logger import setup_logger
from data.brisc_dataset import BRISCDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.swin_hafunet import SwinHAFUNet
from losses.combined_loss import CombinedLoss
from losses.dice_loss import DiceLoss
from training.trainer import Trainer


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Swin-HAFUNet Brain Tumor Segmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training.",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    The training set is split into train/val based on ``data.val_split``.
    The split is stratified by file name to ensure both partitions see all
    tumour types.
    """
    data_cfg = cfg["data"]
    brisc_cfg = data_cfg["brisc"]
    t_cfg = cfg["transforms"]

    root = brisc_cfg["root_dir"]
    img_dir = os.path.join(root, brisc_cfg["train_images"])
    mask_dir = os.path.join(root, brisc_cfg["train_masks"])
    image_ext = brisc_cfg.get("image_ext", ".jpg")
    mask_ext = brisc_cfg.get("mask_ext", ".png")

    # ── Transforms ───────────────────────────────────────────────────────
    train_tf = get_train_transforms(
        image_size=t_cfg["image_size"],
        mean=t_cfg["mean"],
        std=t_cfg["std"],
        aug_cfg=t_cfg.get("augmentation", {}),
    )
    val_tf = get_val_transforms(
        image_size=t_cfg["image_size"],
        mean=t_cfg["mean"],
        std=t_cfg["std"],
    )

    # ── Full dataset (just to get file list for splitting) ───────────────
    full_ds = BRISCDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        transform=None,
        image_ext=image_ext,
        mask_ext=mask_ext,
    )

    # ── Stratified split by tumour type ──────────────────────────────────
    filenames = full_ds.image_files
    # Extract tumour-type label for stratification (4th token)
    labels = []
    for fn in filenames:
        parts = Path(fn).stem.split("_")
        labels.append(parts[3] if len(parts) >= 4 else "unknown")

    val_frac = data_cfg.get("val_split", 0.15)
    train_files, val_files = train_test_split(
        filenames, test_size=val_frac, random_state=cfg["project"]["seed"],
        stratify=labels,
    )

    # ── Build separate datasets with appropriate transforms ──────────────
    train_ds = BRISCDataset(img_dir, mask_dir, transform=train_tf, image_ext=image_ext, mask_ext=mask_ext)
    val_ds = BRISCDataset(img_dir, mask_dir, transform=val_tf, image_ext=image_ext, mask_ext=mask_ext)

    # Filter file lists to match the split
    train_ds.image_files = train_files
    val_ds.image_files = val_files

    train_cfg_t = cfg.get("training", {})
    batch_size = train_cfg_t.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════════════
# Model / Loss / Optimizer / Scheduler
# ═════════════════════════════════════════════════════════════════════════════

def build_model(cfg: Dict[str, Any]) -> SwinHAFUNet:
    """Instantiate the Swin-HAFUNet model from config."""
    m = cfg["model"]
    enc = m["encoder"]
    dec = m["decoder"]
    return SwinHAFUNet(
        backbone=enc["backbone"],
        pretrained=enc["pretrained"],
        in_channels=enc.get("in_channels", 3),
        encoder_channels=enc.get("feature_channels", [96, 192, 384, 768]),
        decoder_channels=dec.get("channels", [256, 128, 64, 32]),
        num_classes=m.get("num_classes", 1),
        dropout=dec.get("dropout", 0.1),
    )


def build_criterion(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg["loss"]
    name = loss_cfg.get("name", "combined")
    if name == "combined":
        return CombinedLoss(
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            smooth=loss_cfg.get("smooth", 1.0),
        )
    elif name == "dice":
        return DiceLoss(smooth=loss_cfg.get("smooth", 1.0))
    elif name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> AdamW:
    """Build AdamW optimizer from config."""
    opt_cfg = cfg["optimizer"]
    return AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(optimizer: AdamW, cfg: Dict[str, Any]) -> CosineAnnealingLR:
    """Build cosine-annealing LR scheduler from config."""
    sch_cfg = cfg["scheduler"]
    return CosineAnnealingLR(
        optimizer,
        T_max=sch_cfg.get("T_max", 150),
        eta_min=sch_cfg.get("eta_min", 1e-6),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ── Reproducibility ──────────────────────────────────────────────────
    seed = cfg["project"]["seed"]
    set_seed(seed)

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    logger = setup_logger(name="swin_hafunet", log_dir=log_dir)
    logger.info(f"Project: {cfg['project']['name']}  |  Seed: {seed}")

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info("Building data loaders …")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        f"Train samples: {len(train_loader.dataset)}  |  "
        f"Val samples: {len(val_loader.dataset)}  |  "
        f"Batch size: {cfg['training']['batch_size']}"
    )

    # ── Model ────────────────────────────────────────────────────────────
    logger.info("Building Swin-HAFUNet model …")
    model = build_model(cfg)
    params = model.count_parameters()
    logger.info(
        f"Parameters — Encoder: {params['encoder']:,}  "
        f"Decoder: {params['decoder']:,}  "
        f"Total: {params['total']:,}"
    )

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger.info(f"Loss: {cfg['loss']['name']}  |  Optimizer: AdamW  |  Scheduler: CosineAnnealing")

    # ── Resume from checkpoint (optional) ────────────────────────────────
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        from utils.checkpoint import CheckpointManager
        ckpt_mgr = CheckpointManager()
        state = ckpt_mgr.load(args.resume, device=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info(f"Resumed at epoch {state['epoch']}")

    # ── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
