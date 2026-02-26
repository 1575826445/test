"""
SG-CMN: Structural-Guided Causal Modulation Network
====================================================

Main network for multi-modal medical image segmentation with missing modality handling.

Core innovations:
1. Causal Prediction: predicts missing modality features from available modalities.
2. Edge Guidance: Edge Expert extracts boundary information to guide segmentation.
3. Multi-scale Fusion: three independent encoders + SE attention fusion.
4. Quality Sentinel: predicts segmentation quality for failure detection.
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import EnhancedDecoder
from .experts import EdgeExpertExtractor, CausalPredictor
from .modules import SEBlock
from .losses import QualitySentinel


class SGCMN(nn.Module):
    """
    SG-CMN: Structural-Guided Causal Modulation Network.

    Args:
        edge_expert_ckpt: path to Edge Expert pretrained weights (or None).
        base_channels: base number of feature channels (default: 64).

    Input:
        t1:    [B, 1, D, H, W] first available modality.
        t2:    [B, 1, D, H, W] second available modality.
        mensa: [B, 1, D, H, W] or None, potentially missing modality.

    Output:
        dict with keys:
            - 'y1':              segmentation prediction [B, 1, D, H, W]
            - 'y0':              placeholder (kept for compatibility)
            - 'tau':             edge weights [B, 1, D, H, W]
            - 'mensa_real_feat': real missing-modality bottleneck features
            - 'mensa_pred_feat': predicted missing-modality bottleneck features
            - 'pred_dice':       predicted segmentation quality [B, 1]
            - 'fused_bn':        fused bottleneck (for visualization)
    """
    def __init__(self, edge_expert_ckpt=None, base_channels=64):
        super().__init__()

        # Three independent encoders
        self.encoder_t1 = Encoder(in_channels=1, base_channels=base_channels)
        self.encoder_t2 = Encoder(in_channels=1, base_channels=base_channels)
        self.encoder_mensa = Encoder(in_channels=1, base_channels=base_channels)

        # Edge Expert
        self.expert_extractor = EdgeExpertExtractor(edge_expert_ckpt)

        # Causal Predictor (T1 + T2 -> missing modality features)
        self.causal_predictor = CausalPredictor(feature_dim=base_channels * 8)

        # Three-modality fusion + SE attention
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(base_channels * 8 * 3, base_channels * 8, 1, bias=False),
            nn.InstanceNorm3d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.se_block = SEBlock(base_channels * 8, reduction=16)

        # Decoder
        self.decoder = EnhancedDecoder(base_channels=base_channels)

        # Quality Sentinel
        self.quality_sentinel = QualitySentinel(feature_dim=base_channels * 8)

    def forward(self, t1, t2, mensa=None):
        """
        Forward pass.

        Returns:
            dict containing segmentation prediction and intermediate features.
        """
        # 1. Handle missing modality input
        if mensa is None:
            mensa = torch.zeros_like(t1)
            has_mensa = False
        else:
            has_mensa = (mensa.abs().max() > 1e-6)

        # 2. Edge Expert extracts boundary information
        y0_placeholder, tau = self.expert_extractor(
            t1, t2, mensa if has_mensa else torch.zeros_like(t1)
        )

        # 3. Three encoders extract features
        t1_feats, t1_bn = self.encoder_t1(t1)
        t2_feats, t2_bn = self.encoder_t2(t2)
        mensa_feats, mensa_real_bn = self.encoder_mensa(mensa)

        # 4. Causal prediction: T1 + T2 -> missing modality features
        mensa_pred_bn = self.causal_predictor(t1_bn, t2_bn)

        # 5. Select features (real or predicted)
        mensa_used_bn = mensa_real_bn if has_mensa else mensa_pred_bn

        # 6. Three-modality fusion + SE attention
        bottleneck_combined = torch.cat([t1_bn, t2_bn, mensa_used_bn], dim=1)
        fused_bn = self.fusion_conv(bottleneck_combined)
        fused_bn = self.se_block(fused_bn)

        # 7. Decode (edge-guided)
        y1 = self.decoder(fused_bn, t1_feats, t2_feats, y0_placeholder, tau)

        # 8. Quality Sentinel
        pred_dice = self.quality_sentinel(fused_bn)

        return {
            'y1': y1,
            'y0': y0_placeholder,
            'tau': tau,
            'mensa_real_feat': mensa_real_bn,
            'mensa_pred_feat': mensa_pred_bn,
            'pred_dice': pred_dice,
            'fused_bn': fused_bn,
        }

    def freeze_edge_expert(self):
        """Freeze Edge Expert parameters."""
        for param in self.expert_extractor.parameters():
            param.requires_grad = False
        print("[INFO] Edge Expert frozen.")

    def unfreeze_edge_expert(self):
        """Unfreeze Edge Expert parameters."""
        for param in self.expert_extractor.parameters():
            param.requires_grad = True
        print("[INFO] Edge Expert unfrozen.")

    def get_confidence(self, t1, t2, mensa=None, threshold=0.7):
        """
        Inference with confidence estimation.

        Returns:
            pred: segmentation prediction.
            confidence: predicted quality score (0~1).
            need_review: whether the case needs manual review.
        """
        with torch.no_grad():
            output = self.forward(t1, t2, mensa)
            pred = output['y1']
            confidence = output['pred_dice'].item()
            need_review = confidence < threshold

        return pred, confidence, need_review
