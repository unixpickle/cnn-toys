"""
Likelihood objectives.
"""

import numpy as np
import tensorflow as tf

from .layer import sum_batch


def bits_per_pixel(layer, inputs, noise=1.0 / 255.0):
    """
    Compute the bits per pixel for each input image.

    Args:
      layer: the network to apply.
      inputs: the input images.
      noise: the amount of noise to add for a Monte Carlo
        integral. This is used to turn discrete inputs
        into continuous inputs.
    """
    # Compute a Monte Carlo integral with one sample.
    sampled_noise = tf.random_uniform(tf.shape(inputs), maxval=noise)
    log_probs = log_likelihood(layer, inputs + sampled_noise)
    num_pixels = int(np.prod([x.value for x in inputs.get_shape()[1:]]))
    return -(log_probs / num_pixels + tf.log(float(noise))) / tf.log(2.0)


def bits_per_pixel_and_grad(layer, inputs, noise=1.0 / 255.0, var_list=None):
    """
    Like bits_per_pixel(), but also computes the gradients
    for the mean bits per pixel.

    Returns:
      A pair (bits, grads):
        bits: a 1-D Tensor of bits-per-pixel values.
        grads: a list of (gradient, variable) pairs.
    """
    sampled_noise = tf.random_uniform(tf.shape(inputs), maxval=noise)
    outputs, latents, log_dets = layer.forward(inputs + sampled_noise)
    assert outputs is None
    log_probs = output_log_likelihood(latents, log_dets)
    num_pixels = int(np.prod([x.value for x in inputs.get_shape()[1:]]))
    bits = -(log_probs / num_pixels + tf.log(float(noise))) / tf.log(2.0)
    loss = tf.reduce_mean(bits)
    grads = layer.gradients(outputs, latents, log_dets, loss, var_list=var_list)
    return bits, grads


def log_likelihood(layer, inputs):
    """
    Compute the log likelihood for each input in a batch,
    assuming a Gaussian latent distribution.
    """
    outputs, latents, log_dets = layer.forward(inputs)
    assert outputs is None, 'extraneous non-latent outputs'
    return output_log_likelihood(latents, log_dets)


def output_log_likelihood(latents, log_dets):
    """
    Like log_likelihood(), but with a pre-computed output
    from an NVPLayer.
    """
    log_probs = log_dets
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
