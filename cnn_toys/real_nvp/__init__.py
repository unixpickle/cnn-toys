"""
An implementation of RealNVP:
https://arxiv.org/abs/1605.08803
"""

from .interp import interpolate, interpolate_linear
from .layer import (FactorHalf, MaskedConv, MaskedFC, NVPLayer, Network, PaddedLogit, Squeeze,
                    checkerboard_mask, depth_mask, one_cold_mask)
from .models import simple_network
from .objective import (bits_per_pixel, bits_per_pixel_and_grad, log_likelihood,
                        output_log_likelihood)
