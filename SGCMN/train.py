#!/usr/bin/env python3
"""
SG-CMN Training Script
======================

Supports two datasets:
  - BraTS 2024 (brain tumor, fixed hold-out split)
  - Parotid gland (5-fold cross-validation)

Usage:
  # BraTS 2024 training
  python3 train.py --dataset brats --data_dir /path/to/BraTS2024/train \
                   --splits_file /path/to/splits_brats.json \
                   --edge_expert_ckpt pretrained_models/edge_expert.pth \
                   --output_dir results/brats --gpu 0

  # Parotid 5-fold training
  python3 train.py --dataset parotid --data_dir /path/to/imagesTr \
                   --label_dir /path/to/labelsTr \
                   --splits_file /path/to/splits_5fold.json \
                   --edge_expert_ckpt pretrained_models/edge_expert.pth \
                   --output_dir results/parotid_5fold --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom, label as scipy_label
from tqdm import tqdm

from models.sgcmn.network import SGCMN
from models.sgcmn.losses import CombinedLoss
from utils import compute_dice, largest_connected_component, tta_predict_3input, zscore_normalize


# ==================== Datasets ====================

class BraTSDataset(Dataset):
    """BraTS 2024 dataset. Modality mapping: T2w->T1, FLAIR->T2, T1ce->Mensa(missing)."""
    def __init__(self, case_ids, data_dir, target_shape, mensa_dropout=0.0):
        self.case_ids = case_ids
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.mensa_dropout = mensa_dropout

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        case_dir = os.path.join(self.data_dir, case_id)

        t2w = nib.load(f"{case_dir}/{case_id}-t2w.nii").get_fdata().astype(np.float32)
        t2f = nib.load(f"{case_dir}/{case_id}-t2f.nii").get_fdata().astype(np.float32)
        t1c = nib.load(f"{case_dir}/{case_id}-t1c.nii").get_fdata().astype(np.float32)
        label = nib.load(f"{case_dir}/{case_id}-seg.nii").get_fdata().astype(np.float32)

        # Binary label (Whole Tumor)
        label[label > 0] = 1.0

        # Resize
        ts = self.target_shape
        t2w = zoom(t2w, [ts[i] / t2w.shape[i] for i in range(3)], order=1)
        t2f = zoom(t2f, [ts[i] / t2f.shape[i] for i in range(3)], order=1)
        t1c = zoom(t1c, [ts[i] / t1c.shape[i] for i in range(3)], order=1)
        label = zoom(label, [ts[i] / label.shape[i] for i in range(3)], order=0)

        # Z-score normalization
        for arr in [t2w, t2f, t1c]:
            zscore_normalize(arr)

        # Mensa dropout
        if self.mensa_dropout > 0 and np.random.random() < self.mensa_dropout:
            t1c = np.zeros_like(t1c)

        return (torch.from_numpy(t2w).unsqueeze(0),
                torch.from_numpy(t2f).unsqueeze(0),
                torch.from_numpy(t1c).unsqueeze(0),
                torch.from_numpy(label).unsqueeze(0),
                case_id)


class ParotidDataset(Dataset):
    """Parotid gland dataset with 3 modalities."""
    def __init__(self, case_ids, data_dir, label_dir, target_shape, mensa_dropout=0.0):
        self.case_ids = case_ids
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.target_shape = target_shape
        self.mensa_dropout = mensa_dropout

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        t1 = nib.load(f"{self.data_dir}/{case_id}_0001.nii.gz").get_fdata().astype(np.float32)
        t2 = nib.load(f"{self.data_dir}/{case_id}_0002.nii.gz").get_fdata().astype(np.float32)
        mensa = nib.load(f"{self.data_dir}/{case_id}_0000.nii.gz").get_fdata().astype(np.float32)
        label = nib.load(f"{self.label_dir}/{case_id}.nii.gz").get_fdata().astype(np.float32)

        ts = self.target_shape
        t1 = zoom(t1, [ts[i] / t1.shape[i] for i in range(3)], order=1)
        t2 = zoom(t2, [ts[i] / t2.shape[i] for i in range(3)], order=1)
        mensa = zoom(mensa, [ts[i] / mensa.shape[i] for i in range(3)], order=1)
        label = zoom(label, [ts[i] / label.shape[i] for i in range(3)], order=0)

        for arr in [t1, t2, mensa]:
            zscore_normalize(arr)

        if self.mensa_dropout > 0 and np.random.random() < self.mensa_dropout:
            mensa = np.zeros_like(mensa)

        return (torch.from_numpy(t1).unsqueeze(0),
                torch.from_numpy(t2).unsqueeze(0),
                torch.from_numpy(mensa).unsqueeze(0),
                torch.from_numpy(label).unsqueeze(0),
                case_id)


# ==================== Training ====================

def dice_score_tensor(pred, target, smooth=1e-5):
    """Dice score for tensors."""
    pred_f = pred.view(-1)
    target_f = target.view(-1)
    intersection = (pred_f * target_f).sum()
    return (2 * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth)


def train_one_fold(train_ids, val_ids, args, device, fold_idx=0):
    """Train a single fold/split."""

    # Dataset
    if args.dataset == 'brats':
        train_ds = BraTSDataset(train_ids, args.data_dir, args.target_shape,
                                mensa_dropout=args.mensa_dropout)
        val_ds = BraTSDataset(val_ids, args.data_dir, args.target_shape, mensa_dropout=0.0)
    else:
        train_ds = ParotidDataset(train_ids, args.data_dir, args.label_dir,
                                  args.target_shape, mensa_dropout=args.mensa_dropout)
        val_ds = ParotidDataset(val_ids, args.data_dir, args.label_dir,
                                args.target_shape, mensa_dropout=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = SGCMN(edge_expert_ckpt=args.edge_expert_ckpt, base_channels=args.base_channels)
    if args.freeze_edge_expert:
        model.freeze_edge_expert()
    else:
        model.unfreeze_edge_expert()
    model = model.to(device)

    # Loss
    criterion = CombinedLoss(
        lambda_causal=args.lambda_causal,
        lambda_style=args.lambda_style,
        lambda_sentinel=args.lambda_sentinel
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        for t1, t2, mensa, label, _ in train_loader:
            t1, t2, mensa, label = t1.to(device), t2.to(device), mensa.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(t1, t2, mensa)

            # Compute real Dice for Sentinel training
            with torch.no_grad():
                pred_bin = (output['y1'].detach() > 0.5).float()
                true_dice = torch.stack([
                    dice_score_tensor(pred_bin[i], label[i]) for i in range(pred_bin.size(0))
                ]).unsqueeze(1).to(device)

            loss, loss_dict = criterion(output, label, true_dice)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        scheduler.step()

        # Validate
        model.eval()
        val_dices = []
        with torch.no_grad():
            for t1, t2, mensa, label, _ in val_loader:
                t1, t2, mensa = t1.to(device), t2.to(device), mensa.to(device)
                label_np = label.numpy()[0, 0]

                # Validate with missing modality (mensa=0)
                mensa_zero = torch.zeros_like(mensa)
                pred = tta_predict_3input(model, t1, t2, mensa_zero, output_key='y1')
                pred_np = (pred[0, 0].cpu().numpy() > 0.5).astype(np.float32)
                pred_np = largest_connected_component(pred_np)
                val_dices.append(compute_dice(pred_np, label_np))

        val_dice = np.mean(val_dices)
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        avg_loss = np.mean([l['total'] for l in train_losses])
        marker = " *BEST*" if is_best else ""
        print(f"[Fold {fold_idx}] Epoch {epoch+1:02d}/{args.epochs}: "
              f"Loss={avg_loss:.4f}, ValDice={val_dice:.4f}, Best={best_val_dice:.4f}{marker}")

        if patience_counter >= args.patience:
            print(f"[Fold {fold_idx}] Early stopping at epoch {epoch+1}")
            break

    return best_val_dice, best_model_state


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='SG-CMN Training')
    # Dataset
    parser.add_argument('--dataset', type=str, required=True, choices=['brats', 'parotid'],
                        help='Dataset name: brats or parotid')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to image data directory')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Path to label directory (required for parotid)')
    parser.add_argument('--splits_file', type=str, required=True,
                        help='Path to data splits JSON file')
    parser.add_argument('--output_dir', type=str, default='results/train',
                        help='Output directory for checkpoints and logs')
    # Model
    parser.add_argument('--edge_expert_ckpt', type=str, default=None,
                        help='Path to pretrained Edge Expert checkpoint')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of feature channels')
    parser.add_argument('--freeze_edge_expert', action='store_true',
                        help='Freeze Edge Expert parameters during training')
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--mensa_dropout', type=float, default=0.3,
                        help='Probability of dropping the third modality during training')
    parser.add_argument('--lambda_causal', type=float, default=0.5,
                        help='Weight for causal prediction loss')
    parser.add_argument('--lambda_style', type=float, default=0.1,
                        help='Weight for style loss')
    parser.add_argument('--lambda_sentinel', type=float, default=0.5,
                        help='Weight for quality sentinel loss')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    # Hardware
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    # Data shape
    parser.add_argument('--target_shape', type=int, nargs=3, default=None,
                        help='Target volume shape (D H W). Default: 96 96 96 for brats, 64 128 128 for parotid')
    # 5-fold
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds (1 for single split, 5 for cross-validation)')

    args = parser.parse_args()

    # Set default target shapes
    if args.target_shape is None:
        if args.dataset == 'brats':
            args.target_shape = (96, 96, 96)
        else:
            args.target_shape = (64, 128, 128)
    else:
        args.target_shape = tuple(args.target_shape)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SG-CMN Training")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Target shape: {args.target_shape}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Mensa dropout:{args.mensa_dropout}")
    print(f"  Lambda causal:{args.lambda_causal}")
    print(f"  Lambda style: {args.lambda_style}")
    print(f"  Lambda sent.: {args.lambda_sentinel}")
    print(f"  Num folds:    {args.num_folds}")
    print("=" * 60)

    # Load splits
    with open(args.splits_file) as f:
        splits = json.load(f)

    if args.num_folds == 1:
        # Single split mode
        train_ids = splits['train']
        val_ids = splits['val']
        test_ids = splits.get('test', [])

        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

        val_dice, model_state = train_one_fold(train_ids, val_ids, args, device, fold_idx=0)

        # Save model
        save_path = os.path.join(args.output_dir, 'best_model.pth')
        torch.save(model_state, save_path)
        print(f"\nBest model saved: {save_path} (Val Dice: {val_dice:.4f})")

    else:
        # Cross-validation mode (splits is a list of folds)
        fold_results = []
        for fold_idx in range(args.num_folds):
            fold_data = splits[fold_idx]
            train_ids = fold_data['train']
            val_ids = fold_data['val']

            print(f"\n{'=' * 60}")
            print(f"Fold {fold_idx + 1}/{args.num_folds}: Train={len(train_ids)}, Val={len(val_ids)}")
            print(f"{'=' * 60}")

            val_dice, model_state = train_one_fold(train_ids, val_ids, args, device, fold_idx=fold_idx)

            # Save fold model
            save_path = os.path.join(args.output_dir, f'fold{fold_idx}_best.pth')
            torch.save({'model_state_dict': model_state, 'best_val_dice': val_dice, 'fold': fold_idx},
                       save_path)

            fold_results.append({'fold': fold_idx, 'val_dice': float(val_dice)})
            print(f"Fold {fold_idx + 1} saved: {save_path} (Val Dice: {val_dice:.4f})")

            del model_state
            torch.cuda.empty_cache()

        # Summary
        print(f"\n{'=' * 60}")
        print("Cross-Validation Summary")
        print(f"{'=' * 60}")
        for r in fold_results:
            print(f"  Fold {r['fold'] + 1}: Val Dice = {r['val_dice']:.4f}")
        avg_val = np.mean([r['val_dice'] for r in fold_results])
        print(f"  Average: {avg_val:.4f}")

        # Save summary
        with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
            json.dump({'fold_results': fold_results, 'avg_val_dice': float(avg_val)}, f, indent=2)

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
