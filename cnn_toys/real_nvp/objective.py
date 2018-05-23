"""
Likelihood objectives.
"""

import tensorflow as tf

from .layer import sum_batch


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
