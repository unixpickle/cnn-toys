"""
Tests for real NVP layers.
"""

from functools import partial

import numpy as np
import pytest
import tensorflow as tf

from .layer import (FactorHalf, MaskedConv, Network, PaddedLogit, Squeeze,
                    checkerboard_mask, depth_mask)

def test_squeeze_forward():
    """
    Test the forward pass of the Squeeze layer.
    """
    inputs = np.array([
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24]]
        ]
    ], dtype='float32')
    with tf.Session() as sess:
        actual = sess.run(Squeeze().forward(tf.constant(inputs))[0])
        expected = np.array([
            [
                [[1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]],
                [[13, 16, 19, 22, 14, 17, 20, 23, 15, 18, 21, 24]]
            ]
        ], dtype='float32')
        assert not np.isnan(actual).any()
        assert np.allclose(actual, expected)

def test_padded_logit_inverse():
    """
    A specialized test for PaddedLogit inverses.
    """
    inputs = np.random.random(size=(1, 3, 3, 7)).astype('float32') * 0.95
    _inverse_test(PaddedLogit(), inputs)

@pytest.mark.parametrize("layer,shape",
                         [(FactorHalf(), (3, 27, 15, 8)),
                          (MaskedConv(partial(checkerboard_mask, True), 1), (3, 28, 14, 8)),
                          (MaskedConv(partial(depth_mask, False), 1), (3, 28, 14, 8)),
                          (Network([FactorHalf(), Squeeze()]), (3, 28, 14, 4)),
                          (Squeeze(), (4, 8, 18, 4))])
def test_inverses(layer, shape):
    """
    Tests for inverses on unbounded inputs.
    """
    inputs = np.random.normal(size=shape).astype('float32')
    _inverse_test(layer, inputs)

def _inverse_test(layer, inputs):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            in_constant = tf.constant(inputs)
            with tf.variable_scope('model'):
                out, latent, _ = layer.forward(in_constant)
            with tf.variable_scope('model', reuse=True):
                inverse = layer.inverse(out, latent)
            sess.run(tf.global_variables_initializer())
            actual = sess.run(inverse)
            assert not np.isnan(actual).any()
            assert np.allclose(actual, inputs, atol=1e-4, rtol=1e-4)
