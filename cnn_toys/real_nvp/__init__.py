"""
An implementation of RealNVP:
https://arxiv.org/abs/1605.08803
"""

from .layer import (FactorHalf, MaskedConv, NVPLayer, Network, PaddedLogit, Squeeze,
                    checkerboard_mask, depth_mask)
