"""Image / mask transform pipelines using torchvision + custom dual transforms.

Provides joint image-mask augmentation so that spatial transforms (flip,
rotation, resize) are applied identically to both the image and the mask
while colour-space augmentations are applied to the image only.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# =========================================================================
# Joint (image + mask) transform helpers
# =========================================================================

class JointCompose:
    """Chain multiple joint transforms that operate on (image, mask) pairs."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class JointResize:
    def __init__(self, size: int) -> None:
        self.size = (size, size)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class JointHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class JointVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class JointRandomRotation:
    def __init__(self, degrees: int = 15, p: float = 0.5) -> None:
        self.degrees = degrees
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
        return image, mask


class ImageColorJitter:
    """Apply colour jitter to the IMAGE only (not the mask)."""

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=0.5):
        self.p = p
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast,
                                    saturation=saturation, hue=hue)

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            image = self.jitter(image)
        return image, mask


class JointToTensor:
    """Convert (PIL Image, PIL Mask) → (Tensor image, Tensor mask)."""

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.to_tensor(image)              # (C, H, W) float [0, 1]
        image = self.normalize(image)

        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)    # binarise
        mask = torch.from_numpy(mask).unsqueeze(0) # (1, H, W)
        return image, mask


# =========================================================================
# Public API
# =========================================================================

def get_train_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    aug_cfg: Optional[Dict] = None,
) -> JointCompose:
    """Build the training augmentation pipeline.

    Args:
        image_size: Target spatial size (square).
        mean: Per-channel mean for normalisation.
        std: Per-channel std for normalisation.
        aug_cfg: Augmentation parameters from config.

    Returns:
        A ``JointCompose`` pipeline that takes ``(image, mask)`` PIL images
        and returns ``(image_tensor, mask_tensor)``.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    aug_cfg = aug_cfg or {}

    transforms_list: list = [JointResize(image_size)]

    if aug_cfg.get("horizontal_flip", False):
        transforms_list.append(JointHorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", False):
        transforms_list.append(JointVerticalFlip(p=0.5))

    rotation = aug_cfg.get("random_rotation", 0)
    if rotation > 0:
        transforms_list.append(JointRandomRotation(degrees=rotation, p=0.5))

    cj = aug_cfg.get("color_jitter", {})
    if cj:
        transforms_list.append(
            ImageColorJitter(
                brightness=cj.get("brightness", 0.0),
                contrast=cj.get("contrast", 0.0),
                saturation=cj.get("saturation", 0.0),
                hue=cj.get("hue", 0.0),
                p=0.5,
            )
        )

    transforms_list.append(JointToTensor(mean=mean, std=std))
    return JointCompose(transforms_list)


def get_val_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> JointCompose:
    """Build the validation / test transform pipeline (no augmentation).

    Args:
        image_size: Target spatial size (square).
        mean: Per-channel mean for normalisation.
        std: Per-channel std for normalisation.

    Returns:
        A ``JointCompose`` pipeline.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    return JointCompose([
        JointResize(image_size),
        JointToTensor(mean=mean, std=std),
    ])
