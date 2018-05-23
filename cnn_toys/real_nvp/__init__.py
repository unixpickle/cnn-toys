"""
An implementation of RealNVP:
https://arxiv.org/abs/1605.08803
"""

from .interp import interpolate, interpolate_linear
from .layer import (FactorHalf, MaskedConv, MaskedFC, NVPLayer, Network, PaddedLogit, Squeeze,
                    checkerboard_mask, depth_mask, one_cold_mask)
from .models import simple_network
from .objective import bits_per_pixel, log_likelihood
