"""Swin Transformer Encoder — extracts multi-scale features using timm.

The encoder wraps a Swin Transformer backbone (e.g. ``swin_tiny_patch4_window7_224``)
and returns hierarchical feature maps at four stages with channel dimensions
typically [96, 192, 384, 768] for Swin-Tiny.

Usage::

    encoder = SwinEncoder(backbone="swin_tiny_patch4_window7_224", pretrained=True)
    features = encoder(x)          # list of 4 tensors at increasing depth
"""

from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn


class SwinEncoder(nn.Module):
    """Hierarchical Swin Transformer encoder based on ``timm``.

    Args:
        backbone: ``timm`` model name (must be a Swin variant).
        pretrained: Whether to load ImageNet-pretrained weights.
        in_channels: Number of input channels (3 for RGB).
        feature_channels: Expected channel sizes at each stage (for validation only).
    """

    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        in_channels: int = 3,
        feature_channels: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels or [96, 192, 384, 768]

        # Create Swin backbone with multi-scale feature output
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract hierarchical features.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            List of four feature tensors in NCHW format at resolutions
            ``[H/4, H/8, H/16, H/32]``.
        """
        features = self.backbone(x)
        # timm Swin outputs are (B, H, W, C) — convert to (B, C, H, W)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        return features

    def get_channels(self) -> List[int]:
        """Return the channel dimension at each feature stage."""
        return self.feature_channels
