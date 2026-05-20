"""Quick smoke test — verifies the entire pipeline end-to-end."""

import yaml
import torch
from data.transforms import get_train_transforms, get_val_transforms
from data.brisc_dataset import BRISCDataset
from models.swin_hafunet import SwinHAFUNet
from losses.combined_loss import CombinedLoss
from evaluation.metrics import SegmentationMetrics

cfg = yaml.safe_load(open("configs/base_config.yaml"))

# 1. Dataset
print("=" * 60)
print("1. DATASET TEST")
t_cfg = cfg["transforms"]
tf = get_train_transforms(t_cfg["image_size"], t_cfg["mean"], t_cfg["std"], t_cfg.get("augmentation", {}))
root = cfg["data"]["brisc"]["root_dir"]
ds = BRISCDataset(f"{root}/train/images", f"{root}/train/masks", transform=tf)
print(f"   Dataset size: {len(ds)}")
sample = ds[0]
print(f"   Image shape: {sample['image'].shape}")
print(f"   Mask shape:  {sample['mask'].shape}")
print(f"   Mask unique: {sample['mask'].unique().tolist()}")
print(f"   Class dist:  {ds.get_class_distribution()}")
print("   PASS")

# 2. Model forward pass
print("=" * 60)
print("2. MODEL FORWARD PASS")
model = SwinHAFUNet(pretrained=False)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"   Input:  {x.shape}")
print(f"   Output: {out.shape}")
params = model.count_parameters()
print(f"   Params: encoder={params['encoder']:,}  decoder={params['decoder']:,}  total={params['total']:,}")
assert out.shape == (2, 1, 224, 224), f"Expected (2,1,224,224), got {out.shape}"
print("   PASS")

# 3. Loss
print("=" * 60)
print("3. LOSS FUNCTION")
criterion = CombinedLoss()
target = torch.zeros(2, 1, 224, 224)
loss = criterion(out, target)
print(f"   Loss value: {loss.item():.4f}")
assert loss.requires_grad, "Loss should have grad"
print("   PASS")

# 4. Metrics
print("=" * 60)
print("4. METRICS")
metrics = SegmentationMetrics()
metrics.update(out, target)
m = metrics.compute()
print(f"   Dice: {m['dice']:.4f}  IoU: {m['iou']:.4f}  Prec: {m['precision']:.4f}  Recall: {m['recall']:.4f}")
print("   PASS")

# 5. DataLoader
print("=" * 60)
print("5. DATALOADER")
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"   Batch image: {batch['image'].shape}")
print(f"   Batch mask:  {batch['mask'].shape}")
print("   PASS")

print("=" * 60)
print("ALL TESTS PASSED!")
