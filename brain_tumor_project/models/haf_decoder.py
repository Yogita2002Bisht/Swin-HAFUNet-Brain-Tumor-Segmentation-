"""Hierarchical Attention Fusion (HAF) Decoder.

Implements the decoder side of Swin-HAFUNet with:
  1. **Channel Attention** — squeeze-and-excitation style recalibration.
  2. **Spatial Attention** — channel-wise pooling → conv → sigmoid gate.
  3. **Multi-Scale Fusion Blocks** — combine encoder skip features with
     up-sampled decoder features through attention-guided fusion.

The decoder progressively up-samples from the bottleneck resolution back to
1/4 of the input size, followed by a final up-sample + segmentation head to
produce a full-resolution prediction map.

Reference:
  [1] Swin-HAFUNet — Hierarchical Attention Fusion UNet (BRISC / BraTS).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Attention modules
# =========================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention via channel-max and channel-mean pooling.

    Args:
        kernel_size: Convolution kernel for the spatial gate.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class HAFAttentionBlock(nn.Module):
    """Combined Channel + Spatial attention block used in each fusion stage."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# =========================================================================
# Decoder building blocks
# =========================================================================

class ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FusionBlock(nn.Module):
    """Single decoder stage: up-sample → concat skip → double conv → attention.

    Args:
        in_ch: Channels from the lower decoder level (before up-sampling).
        skip_ch: Channels from the encoder skip connection.
        out_ch: Output channels after fusion.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBnReLU(in_ch // 2 + skip_ch, out_ch)
        self.conv2 = ConvBnReLU(out_ch, out_ch)
        self.attention = HAFAttentionBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle spatial size mismatches
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = self.dropout(x)
        return x


# =========================================================================
# Full decoder
# =========================================================================

class HAFDecoder(nn.Module):
    """Hierarchical Attention Fusion decoder for Swin-HAFUNet.

    Expects four encoder feature maps (from deepest to shallowest) and
    progressively fuses them through attention-guided up-sampling blocks.

    Args:
        encoder_channels: Channel counts at each encoder stage,
            ordered **shallowest → deepest** (e.g. ``[96, 192, 384, 768]``).
        decoder_channels: Output channel counts for each decoder stage
            (e.g. ``[256, 128, 64, 32]``).
        num_classes: Number of output segmentation classes.
        dropout: Dropout probability in each fusion block.
    """

    def __init__(
        self,
        encoder_channels: List[int] | None = None,
        decoder_channels: List[int] | None = None,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        enc = encoder_channels or [96, 192, 384, 768]
        dec = decoder_channels or [256, 128, 64, 32]

        # Bottleneck bridge (deepest encoder features)
        self.bottleneck = ConvBnReLU(enc[3], enc[3])

        # Decoder fusion stages (deep → shallow)
        self.stage4 = FusionBlock(enc[3], enc[2], dec[0], dropout)
        self.stage3 = FusionBlock(dec[0], enc[1], dec[1], dropout)
        self.stage2 = FusionBlock(dec[1], enc[0], dec[2], dropout)

        # Final up-sample to restore spatial resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dec[2], dec[3], kernel_size=2, stride=2),
            ConvBnReLU(dec[3], dec[3]),
            ConvBnReLU(dec[3], dec[3]),
        )

        # Segmentation head — one more 2× up-sample to match input
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(dec[3], dec[3], kernel_size=2, stride=2),
            nn.Conv2d(dec[3], num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale features into a segmentation map.

        Args:
            features: List of four encoder feature tensors ordered
                **shallowest → deepest** (``[f1, f2, f3, f4]``).

        Returns:
            Logit tensor of shape ``(B, num_classes, H, W)`` where
            ``(H, W)`` matches the original input spatial size.
        """
        f1, f2, f3, f4 = features  # shallow → deep

        x = self.bottleneck(f4)
        x = self.stage4(x, f3)
        x = self.stage3(x, f2)
        x = self.stage2(x, f1)
        x = self.final_up(x)
        x = self.seg_head(x)
        return x
