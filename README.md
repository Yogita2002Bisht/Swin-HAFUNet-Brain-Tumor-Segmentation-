# 🧠 Swin-HAFUNet: Brain Tumor Segmentation on BraTS 2020

> Hybrid Attention Fusion U-Net for multi-class glioma subregion segmentation from multi-modal MRI — trained on a **6 GB consumer GPU**.


---

## 📖 Overview

This repository presents **Swin-HAFUNet**, a hybrid Swin Transformer + U-Net architecture for automatic brain tumor segmentation on the BraTS 2020 benchmark. The model segments three clinically critical glioma subregions — necrotic core (NCR/NET), peritumoral edema (ED), and enhancing tumor (ET) — from four co-registered MRI modalities.

### Key Training Results

| Metric | Final Value |
|---|---|
| Train Dice | ~0.92 |
| Val Dice | ~0.87 |
| Val IoU (Jaccard) | ~0.80 |
| Val Precision | ~0.90 |
| Val Recall | ~0.88 |
| Train Loss | ~0.045 |
| Val Loss | ~0.08 |
| Avg Epoch Time | ~1.37 min |
| Total Epochs | 63 |

---

## 📊 Training Curves

### Full Training Dashboard

The dashboard below shows all six training metrics across 63 epochs: Loss, Dice Score, IoU, Precision & Recall, Learning Rate schedule, and per-epoch wall-clock time (~1.37 min/epoch consistently).

![All Metrics](results/all_metrics.png)

---

### Loss

Both training and validation loss drop sharply in the first 5 epochs and converge smoothly. Final train loss ~0.045, val loss ~0.08 with no signs of overfitting.

![Loss Curve](results/loss_curve.png)

---

### Dice Score

Training Dice climbs steadily to ~0.92. Validation Dice plateaus at ~0.87 from epoch 20 onward, indicating stable generalization.

![Dice Curve](results/dice_curve.png)

---

### Validation IoU (Jaccard Index)

Validation IoU rises from 0.64 at epoch 1 to a stable ~0.80, consistent with the Dice trajectory.

![IoU Curve](results/iou_curve.png)

---

### Validation Precision & Recall

Precision (magenta) stabilizes at ~0.90 and recall (cyan) at ~0.88 after the initial warm-up, showing balanced detection with a slight precision advantage.

![Precision & Recall](results/precision_recall.png)

---

### Learning Rate Schedule

Cosine annealing from 1.0 × 10⁻⁴ decaying smoothly to ~0.64 × 10⁻⁴ over 63 epochs.

![LR Schedule](results/lr_schedule.png)

---

## 🏗️ Architecture

**Swin-HAFUNet** combines a Swin Transformer encoder with a U-Net decoder featuring hybrid attention fusion:

```
Input: [B, 4, 96, 96, 96]  ← FLAIR, T1, T1CE, T2
  │
  ├─ Swin Transformer Encoder  ← shifted-window self-attention
  │   hierarchical feature extraction at multiple scales
  │
  ├─ Hybrid Attention Fusion   ← fuses local CNN + global transformer features
  │
  ├─ U-Net Decoder             ← transposed conv upsampling + skip connections
  │
  └─ Output: [B, 4, 96, 96, 96]  ← softmax → {Background, NCR/NET, ED, ET}
```

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/swin-hafunet.git
cd swin-hafunet

pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install monai==1.4.0 nibabel numpy tqdm
```

> **Requirements:** Python 3.8+, CUDA GPU with ≥ 6 GB VRAM

---

## 📁 Dataset

Download **BraTS 2020** from Kaggle:

```bash
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
```

Expected structure:

```
data/
└── BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   ├── BraTS20_Training_001_t1.nii.gz
    │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   ├── BraTS20_Training_001_t2.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    └── ...
```

| Property | Value |
|---|---|
| Total cases | 369 |
| Train / Val split | 295 / 74 (80/20) |
| Volume size | 240 × 240 × 155 |
| Resolution | 1 × 1 × 1 mm³ |
| Classes | Background, NCR/NET, ED, ET |

---

## 🚀 Usage

### Training

```bash
python train.py --data_dir ./data/BraTS2020_TrainingData \
                --output_dir ./outputs \
                --epochs 63
```

### Inference

```bash
python inference.py --checkpoint ./outputs/best_model.pth \
                    --input_dir ./data/BraTS2020_ValidationData \
                    --output_dir ./predictions
```

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Initial learning rate | 1 × 10⁻⁴ |
| LR schedule | Cosine Annealing |
| Total epochs | 63 |
| Patch size | 96 × 96 × 96 |
| Foreground sampling prob. | 0.75 |
| AMP | FP16/FP32 mixed |
| Loss | DiceCE (Dice + CrossEntropy) |
| Avg epoch time | ~1.37 min |
| Hardware | NVIDIA RTX 3060 (6 GB VRAM) |
| Framework | PyTorch 2.6.0 + MONAI 1.4.0 |

---

## 🔭 Future Work

- [ ] k-fold cross-validation for statistically robust evaluation
- [ ] Test-time augmentation (TTA)
- [ ] Multi-scale attention gates in decoder skip connections
- [ ] Extend to BraTS 2021/2023 datasets

---


```

---

## 🙏 Acknowledgements

- [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) challenge organizers
- [MONAI](https://monai.io/) — Medical Open Network for AI
- Dataset on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

---
