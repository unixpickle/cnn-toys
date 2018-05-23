"""
Likelihood objectives.
"""

import numpy as np
import tensorflow as tf

from .layer import sum_batch


def bits_per_pixel(layer, inputs, noise=1.0 / 255.0):
    """
    Compute the bits per pixel for each input image.
    """
    # Compute a monte carlo integral with one sample.
    sampled_noise = tf.random_uniform(tf.shape(inputs), maxval=noise)
    log_probs = log_likelihood(layer, inputs + sampled_noise)
    num_pixels = int(np.prod([x.value for x in inputs.get_shape()[1:]]))
    return -(log_probs / num_pixels + tf.log(float(noise))) / tf.log(2.0)


def log_likelihood(layer, inputs):
    """
    Compute the log likelihood for each input in a batch,
    assuming a Gaussian latent distribution.
    """
    outputs, latents, log_probs = layer.forward(inputs)
    assert outputs is None, 'extraneous non-latent outputs'
    for latent in latents:
        log_probs = log_probs + gaussian_log_prob(latent)
    return log_probs


def gaussian_log_prob(tensor):
    """
    For each sub-tensor in a batch, compute the Gaussian
    log-density.
    """
    dist = tf.distributions.Normal(0.0, 1.0)
    return sum_batch(dist.log_prob(tensor))
