"""
Shared utility functions: metrics, post-processing, TTA.
"""

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from scipy.spatial import cKDTree


# ==================== Metrics ====================

def compute_dice(pred, target, smooth=1e-5):
    """Compute Dice score between two binary masks."""
    pred_f = pred.flatten()
    target_f = target.flatten()
    intersection = (pred_f * target_f).sum()
    union = pred_f.sum() + target_f.sum()
    if union == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (union + smooth))


def compute_hd95(pred, target, voxel_spacing=(1.0, 1.0, 1.0), max_penalty=50.0):
    """Compute 95th percentile Hausdorff Distance."""
    pred_pts = np.argwhere(pred > 0.5)
    target_pts = np.argwhere(target > 0.5)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return max_penalty

    pred_pts_mm = pred_pts * np.array(voxel_spacing)
    target_pts_mm = target_pts * np.array(voxel_spacing)

    tree_pred = cKDTree(pred_pts_mm)
    tree_target = cKDTree(target_pts_mm)

    d_pred2target, _ = tree_pred.query(target_pts_mm)
    d_target2pred, _ = tree_target.query(pred_pts_mm)

    hd95 = max(np.percentile(d_pred2target, 95), np.percentile(d_target2pred, 95))
    return min(float(hd95), max_penalty)


def compute_assd(pred, target, voxel_spacing=(1.0, 1.0, 1.0), max_penalty=50.0):
    """Compute Average Symmetric Surface Distance."""
    pred_pts = np.argwhere(pred > 0.5)
    target_pts = np.argwhere(target > 0.5)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return max_penalty

    pred_pts_mm = pred_pts * np.array(voxel_spacing)
    target_pts_mm = target_pts * np.array(voxel_spacing)

    tree_pred = cKDTree(pred_pts_mm)
    tree_target = cKDTree(target_pts_mm)

    d_pred2target, _ = tree_pred.query(target_pts_mm)
    d_target2pred, _ = tree_target.query(pred_pts_mm)

    assd = (d_pred2target.mean() + d_target2pred.mean()) / 2.0
    return min(float(assd), max_penalty)


def compute_sensitivity(pred, target):
    """Compute sensitivity (recall)."""
    tp = ((pred > 0.5) & (target > 0.5)).sum()
    fn = ((pred <= 0.5) & (target > 0.5)).sum()
    if tp + fn == 0:
        return 1.0
    return float(tp) / float(tp + fn)


def compute_precision(pred, target):
    """Compute precision."""
    tp = ((pred > 0.5) & (target > 0.5)).sum()
    fp = ((pred > 0.5) & (target <= 0.5)).sum()
    if tp + fp == 0:
        return 1.0
    return float(tp) / float(tp + fp)


# ==================== Post-processing ====================

def largest_connected_component(binary_mask):
    """Keep only the largest connected component in a binary mask."""
    if binary_mask.sum() == 0:
        return binary_mask
    labeled_mask, num_features = scipy_label(binary_mask)
    if num_features == 0:
        return binary_mask
    sizes = np.bincount(labeled_mask.ravel())[1:]
    largest_label = np.argmax(sizes) + 1
    return (labeled_mask == largest_label).astype(np.float32)


# ==================== Test-Time Augmentation ====================

def tta_predict_3input(model, t1, t2, mensa, output_key='y1'):
    """
    Test-time augmentation with 4 flip orientations (3 axes + original).

    Args:
        model: SG-CMN model (already on device, eval mode).
        t1, t2, mensa: [1, 1, D, H, W] tensors on device.
        output_key: key in model output dict to aggregate.

    Returns:
        pred: [1, 1, D, H, W] averaged prediction.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        # Original
        out = model(t1, t2, mensa)
        preds.append(out[output_key])
        # Flip along each spatial axis
        for dim in [2, 3, 4]:
            t1_f = torch.flip(t1, [dim])
            t2_f = torch.flip(t2, [dim])
            mensa_f = torch.flip(mensa, [dim])
            out_f = model(t1_f, t2_f, mensa_f)
            preds.append(torch.flip(out_f[output_key], [dim]))
    return torch.stack(preds).mean(0)


def tta_predict_8flip(model, t1, t2, mensa, output_key='y1'):
    """
    Test-time augmentation with 8 flip combinations.

    Args:
        model: SG-CMN model (already on device, eval mode).
        t1, t2, mensa: [1, 1, D, H, W] tensors on device.
        output_key: key in model output dict to aggregate.

    Returns:
        pred: [1, 1, D, H, W] averaged prediction.
    """
    flip_combos = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
    model.eval()
    preds = []
    with torch.no_grad():
        for flip_dims in flip_combos:
            t1_a = torch.flip(t1, flip_dims) if flip_dims else t1
            t2_a = torch.flip(t2, flip_dims) if flip_dims else t2
            m_a = torch.flip(mensa, flip_dims) if flip_dims else mensa
            out = model(t1_a, t2_a, m_a)
            pred = out[output_key]
            if flip_dims:
                pred = torch.flip(pred, flip_dims)
            preds.append(pred)
    return torch.stack(preds).mean(0)


# ==================== Preprocessing ====================

def zscore_normalize(volume):
    """Z-score normalization on non-zero voxels (in-place)."""
    mask = volume > 0
    if mask.sum() > 0:
        volume -= volume[mask].mean()
        volume /= (volume[mask].std() + 1e-8)
    return volume
