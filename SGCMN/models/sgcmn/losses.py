"""
Loss functions: StyleLoss, QualitySentinel, and CombinedLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramMatrix(nn.Module):
    """Compute Gram matrix for capturing feature style/texture information."""
    def forward(self, x):
        b, c, d, h, w = x.size()
        features = x.view(b, c, d * h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * d * h * w)


class StyleLoss(nn.Module):
    """
    Physics-Aware Style Loss.

    Matches the texture distribution between predicted and real missing-modality
    features via Gram matrix comparison, mitigating the blurriness caused by MSE.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.gram = GramMatrix()
        self.weight = weight
        self.mse = nn.MSELoss()

    def forward(self, pred_feat, real_feat):
        """
        Args:
            pred_feat: [B, C, D, H, W] predicted missing-modality features.
            real_feat: [B, C, D, H, W] real missing-modality features.

        Returns:
            style_loss: Gram matrix matching loss.
        """
        pred_gram = self.gram(pred_feat)
        real_gram = self.gram(real_feat.detach())
        return self.weight * self.mse(pred_gram, real_gram)


class QualitySentinel(nn.Module):
    """
    Holistic Quality Sentinel.

    Predicts segmentation quality (Dice score) from fused bottleneck features.
    Used for failure detection: flags cases below a confidence threshold
    for manual review.

    Input:  bottleneck features [B, C, D/8, H/8, W/8]
    Output: predicted Dice score [B, 1] in range (0, 1)
    """
    def __init__(self, feature_dim=512):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, bottleneck_feat):
        """
        Args:
            bottleneck_feat: [B, C, D/8, H/8, W/8]

        Returns:
            pred_dice: [B, 1] predicted Dice score.
        """
        x = self.global_pool(bottleneck_feat)  # [B, C, 1, 1, 1]
        x = x.view(x.size(0), -1)              # [B, C]
        pred_dice = self.mlp(x)                 # [B, 1]
        return pred_dice


class CombinedLoss(nn.Module):
    """
    Combined training loss.

    L_total = L_dice + L_bce
              + lambda_causal * L_causal
              + lambda_style  * L_style
              + lambda_sentinel * L_sentinel
    """
    def __init__(self, lambda_causal=0.1, lambda_style=0.1, lambda_sentinel=0.5):
        super().__init__()
        self.lambda_causal = lambda_causal
        self.lambda_style = lambda_style
        self.lambda_sentinel = lambda_sentinel

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.style_loss = StyleLoss(weight=1.0)

    def dice_loss(self, pred, target, smooth=1e-5):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def forward(self, output, target, real_dice=None):
        """
        Args:
            output: model output dict with keys 'y1', 'mensa_pred_feat',
                    'mensa_real_feat', 'pred_dice'.
            target: [B, 1, D, H, W] ground-truth label.
            real_dice: [B, 1] real Dice score (for Sentinel training).

        Returns:
            total_loss: scalar loss.
            loss_dict: dict of individual loss components (float values).
        """
        y1 = output['y1']

        # Main task losses (cast to float32 for AMP compatibility)
        y1_f32 = y1.float()
        target_f32 = target.float()
        loss_dice = self.dice_loss(y1_f32, target_f32)
        loss_bce = self.bce(y1_f32, target_f32)

        total_loss = loss_dice + loss_bce
        loss_dict = {'dice': loss_dice.item(), 'bce': loss_bce.item()}

        # Causal prediction loss
        if 'mensa_pred_feat' in output and 'mensa_real_feat' in output:
            loss_causal = self.mse(output['mensa_pred_feat'], output['mensa_real_feat'].detach())
            total_loss = total_loss + self.lambda_causal * loss_causal
            loss_dict['causal'] = loss_causal.item()

        # Style loss
        if 'mensa_pred_feat' in output and 'mensa_real_feat' in output and self.lambda_style > 0:
            loss_style = self.style_loss(output['mensa_pred_feat'], output['mensa_real_feat'])
            total_loss = total_loss + self.lambda_style * loss_style
            loss_dict['style'] = loss_style.item()

        # Sentinel loss
        if 'pred_dice' in output and real_dice is not None and self.lambda_sentinel > 0:
            loss_sentinel = self.mse(output['pred_dice'], real_dice)
            total_loss = total_loss + self.lambda_sentinel * loss_sentinel
            loss_dict['sentinel'] = loss_sentinel.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
