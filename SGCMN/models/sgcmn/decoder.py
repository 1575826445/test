"""
Edge-guided Decoder: fuses multi-scale features with edge attention enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BasicConv3d, ResidualBlock


class EnhancedDecoder(nn.Module):
    """
    Edge-guided Enhanced Decoder.

    Features:
    - Fuses T1 and T2 multi-scale skip connections.
    - Uses Edge Expert output (tau) for boundary enhancement.
    - Progressive upsampling to restore full resolution.
    """
    def __init__(self, base_channels=64):
        super().__init__()

        # Level 4: 1/8 -> 1/4
        self.dec4 = nn.Sequential(
            BasicConv3d(base_channels * 8 + base_channels * 8 * 2, base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        self.up4 = nn.ConvTranspose3d(base_channels * 4, base_channels * 4, 2, 2)

        # Level 3: 1/4 -> 1/2
        self.dec3 = nn.Sequential(
            BasicConv3d(base_channels * 4 + base_channels * 4 * 2, base_channels * 2),
            ResidualBlock(base_channels * 2)
        )
        self.up3 = nn.ConvTranspose3d(base_channels * 2, base_channels * 2, 2, 2)

        # Level 2: 1/2 -> 1x
        self.dec2 = nn.Sequential(
            BasicConv3d(base_channels * 2 + base_channels * 2 * 2, base_channels),
            ResidualBlock(base_channels)
        )
        self.up2 = nn.ConvTranspose3d(base_channels, base_channels, 2, 2)

        # Level 1: 1x (final)
        self.dec1 = nn.Sequential(
            BasicConv3d(base_channels + base_channels * 2, base_channels),
            ResidualBlock(base_channels)
        )

        # Output head
        self.final_conv = nn.Conv3d(base_channels, 1, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, bottleneck, t1_feats, t2_feats, y0_placeholder, tau):
        """
        Args:
            bottleneck: [B, 8C, D/8, H/8, W/8] fused bottleneck features.
            t1_feats: [s1, s2, s3, s4] multi-scale features from T1 encoder.
            t2_feats: [s1, s2, s3, s4] multi-scale features from T2 encoder.
            y0_placeholder: placeholder tensor (kept for interface compatibility).
            tau: [B, 1, D, H, W] edge attention weights from Edge Expert.

        Returns:
            y1: [B, 1, D, H, W] segmentation prediction.
        """
        s1_t1, s2_t1, s3_t1, s4_t1 = t1_feats
        s1_t2, s2_t2, s3_t2, s4_t2 = t2_feats

        # Level 4: bottleneck + s4 -> up4
        s4_fused = torch.cat([s4_t1, s4_t2], dim=1)
        x = torch.cat([bottleneck, s4_fused], dim=1)
        x = self.dec4(x)
        x = self.up4(x)

        # Level 3: x + s3 -> up3
        s3_fused = torch.cat([s3_t1, s3_t2], dim=1)
        x = torch.cat([x, s3_fused], dim=1)
        x = self.dec3(x)
        x = self.up3(x)

        # Level 2: x + s2 -> up2
        s2_fused = torch.cat([s2_t1, s2_t2], dim=1)
        x = torch.cat([x, s2_fused], dim=1)
        x = self.dec2(x)
        x = self.up2(x)

        # Level 1: x + s1 -> final
        s1_fused = torch.cat([s1_t1, s1_t2], dim=1)
        x = torch.cat([x, s1_fused], dim=1)
        x = self.dec1(x)

        # Prediction
        x = self.final_conv(x)

        # Edge enhancement: tau acts as an attention weight
        tau_resized = F.interpolate(tau, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = x * (1 + tau_resized)

        x = self.final_act(x)

        return x
