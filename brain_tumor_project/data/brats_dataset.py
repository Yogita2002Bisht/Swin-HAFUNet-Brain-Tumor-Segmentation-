"""BraTS 2021 Dataset — placeholder for future 3-D NIfTI support.

This module provides the scaffolding for loading BraTS 2021 volumetric
MRI data (``*.nii`` files) once the pipeline is extended to 3-D
segmentation.  Currently it is **not** used by the training loop;
the active dataset is :class:`data.brisc_dataset.BRISCDataset`.

Each BraTS case folder contains five volumes:
  - ``{case}_t1.nii``
  - ``{case}_t1ce.nii``
  - ``{case}_t2.nii``
  - ``{case}_flair.nii``
  - ``{case}_seg.nii``   (ground-truth segmentation)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS 2021 (future use).

    Args:
        root_dir: Path to ``BraTS2021_Training_Data/``.
        modalities: List of modality suffixes to load (e.g. ``["t1", "t1ce", "t2", "flair"]``).
        transform: Optional callable applied to the stacked volume + mask.
        slice_axis: Axis along which to extract 2-D slices (0=sagittal, 1=coronal, 2=axial).
            If ``None``, the full 3-D volume is returned.
    """

    def __init__(
        self,
        root_dir: str,
        modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        slice_axis: Optional[int] = 2,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.modalities = modalities or ["t1", "t1ce", "t2", "flair"]
        self.transform = transform
        self.slice_axis = slice_axis

        # Discover case folders
        self.cases: List[str] = sorted(
            [d for d in os.listdir(self.root_dir) if (self.root_dir / d).is_dir()]
        )

        # Build a flat index of (case_idx, slice_idx) if slicing is enabled
        self._index: List[tuple] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a dict with ``image`` (C, H, W) and ``mask`` (1, H, W)."""
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel is required for BraTS data — pip install nibabel") from exc

        case_idx, slice_idx = self._index[idx]
        case_name = self.cases[case_idx]
        case_dir = self.root_dir / case_name

        # Load modalities → stack into (C, H, W, D)
        channels = []
        for mod in self.modalities:
            nii_path = case_dir / f"{case_name}_{mod}.nii"
            vol = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
            channels.append(vol)
        volume = np.stack(channels, axis=0)  # (C, H, W, D)

        # Load segmentation
        seg_path = case_dir / f"{case_name}_seg.nii"
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)

        # Extract slice
        if self.slice_axis is not None:
            volume = np.take(volume, slice_idx, axis=self.slice_axis + 1)  # +1 for channel dim
            seg = np.take(seg, slice_idx, axis=self.slice_axis)

        # Binarize segmentation (whole tumor = any label > 0)
        mask = (seg > 0).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # (1, H, W)

        sample = {"image": volume, "mask": mask, "case": case_name, "slice_idx": slice_idx}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build a flat list of (case_idx, slice_idx) tuples."""
        if self.slice_axis is None:
            # Return full 3-D volumes
            self._index = [(i, 0) for i in range(len(self.cases))]
            return

        # For slice-based indexing we need the volume shape.
        # Peek at the first case to determine the slice count along the
        # chosen axis.  BraTS volumes are typically 240×240×155.
        axis_sizes = {0: 240, 1: 240, 2: 155}  # defaults
        try:
            import nibabel as nib

            first = self.cases[0]
            vol = nib.load(
                str(self.root_dir / first / f"{first}_t1.nii")
            ).header.get_data_shape()
            axis_sizes = {0: vol[0], 1: vol[1], 2: vol[2]}
        except Exception:
            pass  # Use defaults

        n_slices = axis_sizes.get(self.slice_axis, 155)
        for ci in range(len(self.cases)):
            for si in range(n_slices):
                self._index.append((ci, si))
