"""
Tests for real NVP layers.
"""

import numpy as np
import tensorflow as tf

from .layer import Squeeze

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

def test_squeeze_inverse():
    """
    Test the inverse of the Squeeze layer.
    """
    inputs = np.random.normal(size=(3, 28, 14, 7)).astype('float32')
    with tf.Session() as sess:
        actual = sess.run(Squeeze().inverse(*Squeeze().forward(tf.constant(inputs))[:2]))
        assert not np.isnan(actual).any()
        assert np.allclose(actual, inputs)
