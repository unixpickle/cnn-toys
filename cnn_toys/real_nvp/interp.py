"""
Interpolation in latent space.
"""

import tensorflow as tf


def interpolate(latents, fracs):
    """
    Interpolate between two images' latent variables.

    Args:
      latents: latents from running a batch of two images
        through an NVPLayer.
      fracs: a sequence of interpolation fractions, where
        0 means the first image and 1 means the second.

    Returns:
      A new set of latents of batch size `len(fracs)`.
    """
    new_latents = []
    for latent in latents:
        img_1 = latent[0]
        img_2 = latent[1]
        spread = [img_1 * f + img_2 * (1 - f) for f in fracs]
        new_latents.append(tf.stack(spread))
    return new_latents


def interpolate_linear(latents, num_stops):
    """
    Linearly interpolate between two images' latent
    variables.

    Args:
      latents: latents from running a batch of two images
        through an NVPLayer.
      num_stops: the number of samples to produce.

    Returns:
      A new set of latents of batch size num_stops.
    """
    return interpolate(latents, [i / (num_stops - 1) for i in range(num_stops)])
