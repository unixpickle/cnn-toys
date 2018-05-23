"""
Pre-built networks.
"""

from functools import partial

from .layer import (FactorHalf, MaskedConv, Network, PaddedLogit, Squeeze,
                    checkerboard_mask, depth_mask)


def simple_network():
    """
    A Network that is good for experimenting.
    """
    main_layers = [
        MaskedConv(partial(checkerboard_mask, True), 2),
        MaskedConv(partial(checkerboard_mask, False), 2),
        MaskedConv(partial(checkerboard_mask, True), 2),
        Squeeze(),
        MaskedConv(partial(depth_mask, True), 2),
        MaskedConv(partial(depth_mask, False), 2),
        MaskedConv(partial(depth_mask, True), 2),
        FactorHalf()
    ]
    return Network([PaddedLogit()] + (main_layers * 3))
