"""
SG-CMN: Structural-Guided Causal Modulation Network
====================================================

A causal modulation network for multi-modal medical image segmentation
with missing modality handling.

Main components:
- SGCMN: Main network model
- Encoder: Multi-scale 3D encoder
- EnhancedDecoder: Edge-guided decoder
- EdgeExpertExtractor: Edge expert module
- CausalPredictor: Causal prediction module for missing modality
- CombinedLoss: Combined training loss

Usage:
    from models.sgcmn import SGCMN

    model = SGCMN(edge_expert_ckpt='path/to/edge_expert.pth')
    output = model(t1, t2, mensa)  # mensa is optional
    pred = output['y1']
"""

from .network import SGCMN
from .encoder import Encoder
from .decoder import EnhancedDecoder
from .modules import SEBlock, BasicConv3d, ResidualBlock
from .experts import EdgeExpertExtractor, CausalPredictor
from .losses import StyleLoss, QualitySentinel, CombinedLoss, GramMatrix

__all__ = [
    'SGCMN',
    'Encoder',
    'EnhancedDecoder',
    'EdgeExpertExtractor',
    'CausalPredictor',
    'SEBlock',
    'BasicConv3d',
    'ResidualBlock',
    'StyleLoss',
    'QualitySentinel',
    'CombinedLoss',
    'GramMatrix',
]

__version__ = '1.0.0'
