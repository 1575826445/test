#!/usr/bin/env python3
"""
SG-CMN Evaluation Script
========================

Comprehensive evaluation with 5 metrics: Dice, HD95, ASSD, Sensitivity, Precision.
Supports TTA (Test-Time Augmentation) and LCC (Largest Connected Component) post-processing.

Usage:
  # BraTS evaluation
  python3 evaluate.py --dataset brats --data_dir /path/to/BraTS2024/train \
                      --splits_file /path/to/splits_brats.json \
                      --checkpoint results/brats/best_model.pth \
                      --edge_expert_ckpt pretrained_models/edge_expert.pth \
                      --output_dir results/brats_eval --gpu 0

  # Parotid 5-fold ensemble evaluation
  python3 evaluate.py --dataset parotid --data_dir /path/to/imagesTr \
                      --label_dir /path/to/labelsTr \
                      --splits_file /path/to/splits_5fold.json \
                      --checkpoint_dir results/parotid_5fold \
                      --edge_expert_ckpt pretrained_models/edge_expert.pth \
                      --output_dir results/parotid_eval --num_folds 5 --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

from models.sgcmn.network import SGCMN
from utils import (
    compute_dice, compute_hd95, compute_assd,
    compute_sensitivity, compute_precision,
    largest_connected_component, tta_predict_3input, tta_predict_8flip,
    zscore_normalize,
)


def load_brats_case(case_id, data_dir, target_shape):
    """Load and preprocess a BraTS case for evaluation."""
    case_dir = os.path.join(data_dir, case_id)
    t2w = nib.load(f"{case_dir}/{case_id}-t2w.nii").get_fdata().astype(np.float32)
    t2f = nib.load(f"{case_dir}/{case_id}-t2f.nii").get_fdata().astype(np.float32)
    label = nib.load(f"{case_dir}/{case_id}-seg.nii").get_fdata().astype(np.float32)
    label[label > 0] = 1.0

    ts = target_shape
    t2w = zoom(t2w, [ts[i] / t2w.shape[i] for i in range(3)], order=1)
    t2f = zoom(t2f, [ts[i] / t2f.shape[i] for i in range(3)], order=1)
    label = zoom(label, [ts[i] / label.shape[i] for i in range(3)], order=0)

    zscore_normalize(t2w)
    zscore_normalize(t2f)

    return t2w, t2f, label


def load_parotid_case(case_id, data_dir, label_dir, target_shape):
    """Load and preprocess a Parotid case for evaluation."""
    t1 = nib.load(f"{data_dir}/{case_id}_0001.nii.gz").get_fdata().astype(np.float32)
    t2 = nib.load(f"{data_dir}/{case_id}_0002.nii.gz").get_fdata().astype(np.float32)
    label = nib.load(f"{label_dir}/{case_id}.nii.gz").get_fdata().astype(np.float32)

    ts = target_shape
    t1 = zoom(t1, [ts[i] / t1.shape[i] for i in range(3)], order=1)
    t2 = zoom(t2, [ts[i] / t2.shape[i] for i in range(3)], order=1)
    label = zoom(label, [ts[i] / label.shape[i] for i in range(3)], order=0)

    zscore_normalize(t1)
    zscore_normalize(t2)

    return t1, t2, label


def evaluate_single_model(model, test_ids, args, device):
    """Evaluate a single model on all test cases."""
    model.eval()
    results = {}

    for case_id in tqdm(test_ids, desc="Evaluating"):
        if args.dataset == 'brats':
            t1_np, t2_np, label_np = load_brats_case(case_id, args.data_dir, args.target_shape)
        else:
            t1_np, t2_np, label_np = load_parotid_case(case_id, args.data_dir,
                                                         args.label_dir, args.target_shape)

        t1_t = torch.from_numpy(t1_np).float().unsqueeze(0).unsqueeze(0).to(device)
        t2_t = torch.from_numpy(t2_np).float().unsqueeze(0).unsqueeze(0).to(device)
        mensa_zero = torch.zeros_like(t1_t)

        with torch.no_grad():
            if args.tta == '8flip':
                pred = tta_predict_8flip(model, t1_t, t2_t, mensa_zero, output_key='y1')
            else:
                pred = tta_predict_3input(model, t1_t, t2_t, mensa_zero, output_key='y1')

        pred_np = pred.squeeze().cpu().numpy()
        pred_bin = largest_connected_component((pred_np > 0.5).astype(np.float32))
        label_bin = (label_np > 0.5).astype(np.float32)

        dice = compute_dice(pred_bin, label_bin)
        hd95 = compute_hd95(pred_bin, label_bin)
        assd = compute_assd(pred_bin, label_bin)
        sens = compute_sensitivity(pred_bin, label_bin)
        prec = compute_precision(pred_bin, label_bin)

        results[case_id] = {
            'dice': dice, 'hd95': hd95, 'assd': assd,
            'sensitivity': sens, 'precision': prec
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='SG-CMN Evaluation')
    parser.add_argument('--dataset', type=str, required=True, choices=['brats', 'parotid'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to single model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing fold checkpoints (for cross-validation)')
    parser.add_argument('--edge_expert_ckpt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/eval')
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tta', type=str, default='4flip', choices=['4flip', '8flip'],
                        help='TTA mode: 4flip (3 axes + original) or 8flip (all combinations)')
    parser.add_argument('--num_folds', type=int, default=1)
    parser.add_argument('--target_shape', type=int, nargs=3, default=None)

    args = parser.parse_args()

    if args.target_shape is None:
        args.target_shape = (96, 96, 96) if args.dataset == 'brats' else (64, 128, 128)
    else:
        args.target_shape = tuple(args.target_shape)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load splits and determine test cases
    with open(args.splits_file) as f:
        splits = json.load(f)

    if args.num_folds == 1:
        test_ids = splits.get('test', [])
    else:
        # For cross-validation, test cases are separate
        if 'test' in splits[0]:
            test_ids = splits[0]['test']
        else:
            # Assume test_ids provided separately or all cases not in train/val
            test_ids = splits.get('test', [])

    print("=" * 60)
    print("SG-CMN Evaluation")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Test cases: {len(test_ids)}")
    print(f"  TTA mode:   {args.tta}")
    print(f"  Device:     {device}")
    print("=" * 60)

    if args.num_folds == 1:
        # Single model evaluation
        model = SGCMN(edge_expert_ckpt=args.edge_expert_ckpt, base_channels=args.base_channels)
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

        results = evaluate_single_model(model, test_ids, args, device)
    else:
        # Multi-fold ensemble: average metrics across folds
        all_fold_results = []
        for fold_idx in range(args.num_folds):
            ckpt_path = os.path.join(args.checkpoint_dir, f'fold{fold_idx}_best.pth')
            print(f"\nFold {fold_idx + 1}/{args.num_folds}: {ckpt_path}")

            model = SGCMN(edge_expert_ckpt=args.edge_expert_ckpt, base_channels=args.base_channels)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)

            fold_results = evaluate_single_model(model, test_ids, args, device)
            all_fold_results.append(fold_results)

            del model
            torch.cuda.empty_cache()

        # Average across folds
        results = {}
        for case_id in test_ids:
            metrics = {}
            for metric in ['dice', 'hd95', 'assd', 'sensitivity', 'precision']:
                vals = [fr[case_id][metric] for fr in all_fold_results if case_id in fr]
                metrics[metric] = float(np.mean(vals))
            results[case_id] = metrics

    # Print results
    print(f"\n{'=' * 70}")
    print(f"{'Case':<25} {'Dice':>8} {'HD95':>8} {'ASSD':>8} {'Sens':>8} {'Prec':>8}")
    print(f"{'-' * 70}")
    for case_id in sorted(results.keys()):
        r = results[case_id]
        print(f"{case_id:<25} {r['dice']:>8.4f} {r['hd95']:>8.2f} {r['assd']:>8.3f} "
              f"{r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    # Summary
    metrics_summary = {}
    for metric in ['dice', 'hd95', 'assd', 'sensitivity', 'precision']:
        vals = [results[c][metric] for c in results]
        metrics_summary[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    print(f"{'-' * 70}")
    print(f"{'Mean':<25} {metrics_summary['dice']['mean']:>8.4f} "
          f"{metrics_summary['hd95']['mean']:>8.2f} {metrics_summary['assd']['mean']:>8.3f} "
          f"{metrics_summary['sensitivity']['mean']:>8.4f} {metrics_summary['precision']['mean']:>8.4f}")
    print(f"{'Std':<25} {metrics_summary['dice']['std']:>8.4f} "
          f"{metrics_summary['hd95']['std']:>8.2f} {metrics_summary['assd']['std']:>8.3f} "
          f"{metrics_summary['sensitivity']['std']:>8.4f} {metrics_summary['precision']['std']:>8.4f}")

    # Save results
    output = {
        'per_case': results,
        'summary': metrics_summary,
        'config': {
            'dataset': args.dataset,
            'tta': args.tta,
            'num_folds': args.num_folds,
            'target_shape': list(args.target_shape),
        }
    }
    out_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
