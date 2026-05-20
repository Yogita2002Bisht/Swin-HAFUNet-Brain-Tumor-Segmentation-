"""Swin-HAFUNet — Swin Transformer Encoder + Hierarchical Attention Fusion Decoder.

This is the top-level model that wires the ``SwinEncoder`` and ``HAFDecoder``
into a single :class:`torch.nn.Module`.  It accepts an RGB image and produces
a pixel-level segmentation logit map at the same spatial resolution.

Usage::

    from models import SwinHAFUNet

    model = SwinHAFUNet(
        backbone="swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=1,
    )
    logits = model(torch.randn(1, 3, 224, 224))
    # logits.shape → (1, 1, 224, 224)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin_encoder import SwinEncoder
from models.haf_decoder import HAFDecoder


class SwinHAFUNet(nn.Module):
    """Swin-HAFUNet for binary brain-tumor segmentation.

    Args:
        backbone: ``timm`` backbone name for the Swin encoder.
        pretrained: Load pretrained weights for the encoder.
        in_channels: Number of image channels.
        encoder_channels: Channel counts at each encoder stage.
        decoder_channels: Channel counts at each decoder stage.
        num_classes: Number of segmentation classes (1 for binary).
        dropout: Dropout ratio in the decoder.
    """

    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        in_channels: int = 3,
        encoder_channels: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_channels = encoder_channels or [96, 192, 384, 768]
        decoder_channels = decoder_channels or [256, 128, 64, 32]

        self.encoder = SwinEncoder(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=in_channels,
            feature_channels=encoder_channels,
        )

        self.decoder = HAFDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor ``(B, C, H, W)``.

        Returns:
            Segmentation logit tensor ``(B, num_classes, H, W)``,
            same spatial size as the input.
        """
        h, w = x.shape[2], x.shape[3]

        # Encoder → multi-scale features (shallow → deep)
        features = self.encoder(x)

        # Decoder → logits
        logits = self.decoder(features)

        # Ensure output matches input resolution
        if logits.shape[2] != h or logits.shape[3] != w:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts for encoder, decoder, and total."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}
