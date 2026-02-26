"""
Multi-scale 3D Encoder: extracts multi-resolution features from each modality.
"""

import torch
import torch.nn as nn
from .modules import BasicConv3d, ResidualBlock


class Encoder(nn.Module):
    """
    Multi-scale 3D Encoder.

    Produces 4 levels of feature maps:
    - s1: 1x   (original resolution)
    - s2: 1/2
    - s3: 1/4
    - s4: 1/8
    - bottleneck: 1/8 (deepest features)
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # Stage 1: original resolution
        self.conv1 = BasicConv3d(in_channels, base_channels, 3, 1, 1)
        self.res1 = ResidualBlock(base_channels)

        # Stage 2: 1/2 resolution
        self.conv2 = BasicConv3d(base_channels, base_channels * 2, 3, 2, 1)
        self.res2 = ResidualBlock(base_channels * 2)

        # Stage 3: 1/4 resolution
        self.conv3 = BasicConv3d(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.res3 = ResidualBlock(base_channels * 4)

        # Stage 4: 1/8 resolution
        self.conv4 = BasicConv3d(base_channels * 4, base_channels * 8, 3, 2, 1)
        self.res4 = ResidualBlock(base_channels * 8)

        # Bottleneck
        self.bottleneck = BasicConv3d(base_channels * 8, base_channels * 8, 3, 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] input volume.

        Returns:
            features: list [s1, s2, s3, s4] of multi-scale feature maps.
            bottleneck: deepest feature tensor.
        """
        s1 = self.res1(self.conv1(x))   # [B, C,   D,   H,   W]
        s2 = self.res2(self.conv2(s1))  # [B, 2C,  D/2, H/2, W/2]
        s3 = self.res3(self.conv3(s2))  # [B, 4C,  D/4, H/4, W/4]
        s4 = self.res4(self.conv4(s3))  # [B, 8C,  D/8, H/8, W/8]

        bottleneck = self.bottleneck(s4)  # [B, 8C, D/8, H/8, W/8]

        return [s1, s2, s3, s4], bottleneck
