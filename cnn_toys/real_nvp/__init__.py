"""
An implementation of RealNVP:
https://arxiv.org/abs/1605.08803
"""

from .layer import NVPLayer, FactorHalf, Squeeze, MaskedConv, checkerboard_mask, depth_mask
