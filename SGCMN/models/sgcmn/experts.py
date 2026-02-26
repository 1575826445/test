"""
Expert modules: Edge Expert and Causal Predictor.
"""

import torch
import torch.nn as nn
from .modules import ResidualBlock


class EdgeExpertExtractor(nn.Module):
    """
    Edge Expert Extractor.

    Extracts edge information (tau) from T1 + T2 + missing modality
    to guide the decoder for boundary-aware segmentation.
    """
    def __init__(self, edge_expert_ckpt=None):
        super().__init__()

        # Edge detection network
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

        # Load pretrained weights if provided
        self.use_pretrained = False
        if edge_expert_ckpt is not None:
            state_dict = torch.load(edge_expert_ckpt, map_location='cpu')

            # Handle different state_dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            self.edge_net.load_state_dict(state_dict, strict=False)

            # Freeze Edge Expert parameters
            for param in self.edge_net.parameters():
                param.requires_grad = False

            print("[INFO] Edge Expert loaded and frozen.")
            self.use_pretrained = True
        else:
            print("[WARN] Edge Expert initialized randomly (not frozen).")

    def forward(self, t1, t2, mensa):
        """
        Args:
            t1, t2, mensa: [B, 1, D, H, W]

        Returns:
            y0_placeholder: zero tensor (body expert removed in this version).
            tau: [B, 1, D, H, W] edge attention weights.
        """
        combined = torch.cat([t1, t2, mensa], dim=1)
        tau = self.edge_net(combined)
        y0_placeholder = torch.zeros_like(tau)
        return y0_placeholder, tau


class CausalPredictor(nn.Module):
    """
    Causal Predictor (Structural Causal Model).

    Predicts the missing modality features from T1 + T2 bottleneck features.
    When the third modality is absent, the predicted features replace it.
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_dim * 2, feature_dim, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(feature_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            ResidualBlock(feature_dim),
            ResidualBlock(feature_dim)
        )

    def forward(self, t1_features, t2_features):
        """
        Args:
            t1_features: [B, C, D/8, H/8, W/8] T1 bottleneck features.
            t2_features: [B, C, D/8, H/8, W/8] T2 bottleneck features.

        Returns:
            mensa_pred: [B, C, D/8, H/8, W/8] predicted missing-modality features.
        """
        combined = torch.cat([t1_features, t2_features], dim=1)
        return self.fusion(combined)
