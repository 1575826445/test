"""
Basic building blocks: convolution blocks, residual blocks, attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class BasicConv3d(nn.Module):
    """Basic 3D convolution block: Conv3d + InstanceNorm3d + LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two 3x3x3 convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = BasicConv3d(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.01, inplace=True)
