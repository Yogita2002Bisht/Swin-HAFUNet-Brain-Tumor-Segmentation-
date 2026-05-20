"""BRISC 2025 Segmentation Dataset.

File naming convention (discovered from data exploration):
  Images: ``brisc2025_{split}_{id}_{type}_{view}_t1.jpg``
  Masks:  ``brisc2025_{split}_{id}_{type}_{view}_t1.png``

  type ∈ {gl (glioma), me (meningioma), pi (pituitary)}
  view ∈ {ax (axial), co (coronal), sa (sagittal)}

The dataset pairs each ``.jpg`` image with the ``.png`` mask sharing the
same stem name.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BRISCDataset(Dataset):
    """PyTorch Dataset for BRISC 2025 brain tumor segmentation task.

    Args:
        image_dir: Path to the folder containing ``.jpg`` images.
        mask_dir: Path to the folder containing ``.png`` masks.
        transform: A ``JointCompose`` transform from ``data.transforms``
            that accepts ``(PIL Image, PIL Mask)`` and returns
            ``(image_tensor, mask_tensor)``.
        image_ext: Extension for image files (default ``".jpg"``).
        mask_ext: Extension for mask files (default ``".png"``).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.mask_ext = mask_ext

        # Collect and sort image file names
        self.image_files: List[str] = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(self.image_ext)]
        )

        # Validate that every image has a corresponding mask
        self._validate_pairs()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a dictionary with keys ``image`` and ``mask``.

        ``image`` — ``(C, H, W)`` float tensor  
        ``mask``  — ``(1, H, W)`` float tensor with values in {0, 1}
        """
        img_name = self.image_files[idx]
        mask_name = img_name.replace(self.image_ext, self.mask_ext)

        image = Image.open(str(self.image_dir / img_name)).convert("RGB")
        mask = Image.open(str(self.mask_dir / mask_name)).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Fallback: basic tensor conversion without transform
            import torchvision.transforms.functional as TF
            image = TF.to_tensor(image)
            mask_np = np.array(mask, dtype=np.float32)
            mask_np = (mask_np > 127).astype(np.float32)
            mask = torch.from_numpy(mask_np).unsqueeze(0)

        return {"image": image, "mask": mask, "filename": img_name}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_pairs(self) -> None:
        """Check that every image has a matching mask file."""
        missing: List[str] = []
        for img_name in self.image_files:
            mask_name = img_name.replace(self.image_ext, self.mask_ext)
            if not (self.mask_dir / mask_name).exists():
                missing.append(mask_name)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} masks not found. First 5: {missing[:5]}"
            )

    def get_class_distribution(self) -> Dict[str, int]:
        """Return per-tumor-type counts based on file naming convention.

        Tumor type is extracted from the 4th underscore-separated token.
        E.g. ``brisc2025_train_00001_gl_ax_t1.jpg`` → ``gl``.
        """
        counts: Dict[str, int] = {}
        for fname in self.image_files:
            parts = Path(fname).stem.split("_")
            if len(parts) >= 4:
                tumor_type = parts[3]
                counts[tumor_type] = counts.get(tumor_type, 0) + 1
        return counts
