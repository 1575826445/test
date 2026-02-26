#!/usr/bin/env python3
"""
Edge Expert Pretraining Script
==============================

Pretrains the Edge Expert network (3-conv architecture) to detect tumor/organ
boundaries from multi-modal MRI inputs. The pretrained weights are then loaded
into the full SG-CMN model.

Architecture: matches EdgeExpertExtractor.edge_net in models/sgcmn/experts.py
  Conv3d(3->32) -> InstanceNorm -> LeakyReLU ->
  Conv3d(32->64) -> InstanceNorm -> LeakyReLU ->
  Conv3d(64->1)  -> Sigmoid

Input:  3-channel (T1 + T2 + missing_modality/zeros)
Output: edge probability map tau in [0, 1]
Label:  morphological gradient of the segmentation mask

Usage:
  python3 pretrain_edge_expert.py --data_dir /path/to/data \
                                  --splits_file /path/to/splits.json \
                                  --output_path pretrained_models/edge_expert.pth \
                                  --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom, binary_dilation, binary_erosion
from tqdm import tqdm


# ==================== Edge GT Generation ====================

def get_edge_gt(label_batch, kernel_size=3):
    """
    Generate edge ground truth from binary segmentation labels
    using morphological gradient (dilation - erosion).

    Args:
        label_batch: [B, 1, D, H, W] tensor or numpy array.
        kernel_size: structuring element size.

    Returns:
        edges: [B, 1, D, H, W] float tensor of edge labels.
    """
    label_np = label_batch.cpu().numpy() if isinstance(label_batch, torch.Tensor) else label_batch
    edges = np.zeros_like(label_np)
    struct = np.ones((kernel_size,) * 3)

    for b in range(label_np.shape[0]):
        seg = label_np[b, 0] > 0.5
        if seg.sum() == 0:
            continue
        dilated = binary_dilation(seg, structure=struct, iterations=1)
        eroded = binary_erosion(seg, structure=struct, iterations=1)
        edges[b, 0] = (dilated.astype(np.float32) - eroded.astype(np.float32)).clip(0, 1)

    return torch.from_numpy(edges).float()


# ==================== Simple Edge Network ====================

class SimpleEdgeNet(nn.Module):
    """
    Edge detection network identical to EdgeExpertExtractor.edge_net.
    """
    def __init__(self):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Conv3d(3, 32, 3, 1, 1),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv3d(32, 64, 3, 1, 1),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv3d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.edge_net(x)


# ==================== Dataset ====================

class EdgePretrainDataset(Dataset):
    """
    Generic dataset for edge expert pretraining.
    Expects NIfTI files organized per the BraTS or Parotid format.
    """
    def __init__(self, case_ids, data_dir, target_shape, dataset_type='brats',
                 label_dir=None, mensa_dropout=0.3):
        self.case_ids = case_ids
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.dataset_type = dataset_type
        self.label_dir = label_dir
        self.mensa_dropout = mensa_dropout

    def __len__(self):
        return len(self.case_ids)

    def _zscore(self, arr):
        mask = arr > 0
        if mask.sum() > 0:
            arr -= arr[mask].mean()
            arr /= (arr[mask].std() + 1e-8)
        return arr

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        ts = self.target_shape

        if self.dataset_type == 'brats':
            case_dir = os.path.join(self.data_dir, case_id)
            t1 = nib.load(f"{case_dir}/{case_id}-t2w.nii").get_fdata().astype(np.float32)
            t2 = nib.load(f"{case_dir}/{case_id}-t2f.nii").get_fdata().astype(np.float32)
            mensa = nib.load(f"{case_dir}/{case_id}-t1c.nii").get_fdata().astype(np.float32)
            label = nib.load(f"{case_dir}/{case_id}-seg.nii").get_fdata().astype(np.float32)
            label[label > 0] = 1.0
        else:
            t1 = nib.load(f"{self.data_dir}/{case_id}_0001.nii.gz").get_fdata().astype(np.float32)
            t2 = nib.load(f"{self.data_dir}/{case_id}_0002.nii.gz").get_fdata().astype(np.float32)
            mensa = nib.load(f"{self.data_dir}/{case_id}_0000.nii.gz").get_fdata().astype(np.float32)
            label = nib.load(f"{self.label_dir}/{case_id}.nii.gz").get_fdata().astype(np.float32)

        # Resize
        t1 = zoom(t1, [ts[i] / t1.shape[i] for i in range(3)], order=1)
        t2 = zoom(t2, [ts[i] / t2.shape[i] for i in range(3)], order=1)
        mensa = zoom(mensa, [ts[i] / mensa.shape[i] for i in range(3)], order=1)
        label = zoom(label, [ts[i] / label.shape[i] for i in range(3)], order=0)

        # Normalize
        self._zscore(t1)
        self._zscore(t2)
        self._zscore(mensa)

        # Mensa dropout
        if np.random.random() < self.mensa_dropout:
            mensa = np.zeros_like(mensa)

        # Stack as 3-channel input
        inp = np.stack([t1, t2, mensa], axis=0)  # [3, D, H, W]
        label = label[np.newaxis]                  # [1, D, H, W]

        return torch.from_numpy(inp).float(), torch.from_numpy(label).float()


# ==================== Loss ====================

class EdgeLoss(nn.Module):
    """Focal + Dice loss for edge detection."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Focal loss
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        loss_focal = focal.mean()

        # Dice loss
        smooth = 1e-5
        pred_f = pred.view(-1)
        target_f = target.view(-1)
        intersection = (pred_f * target_f).sum()
        loss_dice = 1 - (2 * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth)

        return loss_focal + loss_dice


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Edge Expert Pretraining')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Label directory (required for parotid)')
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='brats', choices=['brats', 'parotid'])
    parser.add_argument('--output_path', type=str, default='pretrained_models/edge_expert.pth')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--mensa_dropout', type=float, default=0.3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target_shape', type=int, nargs=3, default=None)

    args = parser.parse_args()

    if args.target_shape is None:
        args.target_shape = (96, 96, 96) if args.dataset == 'brats' else (64, 128, 128)
    else:
        args.target_shape = tuple(args.target_shape)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load splits
    with open(args.splits_file) as f:
        splits = json.load(f)

    if isinstance(splits, list):
        # Cross-validation format: use fold 0
        train_ids = splits[0]['train']
        val_ids = splits[0]['val']
    else:
        train_ids = splits['train']
        val_ids = splits['val']

    print("=" * 60)
    print("Edge Expert Pretraining")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Train:      {len(train_ids)} cases")
    print(f"  Val:        {len(val_ids)} cases")
    print(f"  Shape:      {args.target_shape}")
    print(f"  Device:     {device}")
    print("=" * 60)

    train_ds = EdgePretrainDataset(train_ids, args.data_dir, args.target_shape,
                                   dataset_type=args.dataset, label_dir=args.label_dir,
                                   mensa_dropout=args.mensa_dropout)
    val_ds = EdgePretrainDataset(val_ids, args.data_dir, args.target_shape,
                                 dataset_type=args.dataset, label_dir=args.label_dir,
                                 mensa_dropout=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = SimpleEdgeNet().to(device)
    criterion = EdgeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for inp, label in train_loader:
            inp, label = inp.to(device), label.to(device)
            edge_gt = get_edge_gt(label).to(device)

            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, edge_gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_dices = []
        with torch.no_grad():
            for inp, label in val_loader:
                inp, label = inp.to(device), label.to(device)
                edge_gt = get_edge_gt(label).to(device)
                pred = model(inp)
                pred_bin = (pred > 0.5).float()
                # Edge Dice
                smooth = 1e-5
                inter = (pred_bin * edge_gt).sum()
                dice = (2 * inter + smooth) / (pred_bin.sum() + edge_gt.sum() + smooth)
                val_dices.append(dice.item())

        val_dice = np.mean(val_dices)
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_state = model.edge_net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        marker = " *BEST*" if is_best else ""
        print(f"Epoch {epoch+1:02d}/{args.epochs}: Loss={train_loss:.4f}, "
              f"EdgeDice={val_dice:.4f}, Best={best_val_dice:.4f}{marker}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save
    torch.save(best_state, args.output_path)
    print(f"\nEdge Expert saved: {args.output_path} (Best Edge Dice: {best_val_dice:.4f})")


if __name__ == '__main__':
    main()
