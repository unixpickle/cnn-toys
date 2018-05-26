"""
Tests for real NVP layers.
"""

from functools import partial

import numpy as np
import pytest
import tensorflow as tf

from .layer import (FactorHalf, MaskedConv, MaskedFC, Network, PaddedLogit, Squeeze,
                    checkerboard_mask, depth_mask, one_cold_mask)


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
                          (MaskedFC(partial(one_cold_mask, 3)), (3, 6)),
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
            _randomized_init(sess)
            actual = sess.run(inverse)
            assert not np.isnan(actual).any()
            assert np.allclose(actual, inputs, atol=1e-4, rtol=1e-4)


def test_padded_logit_log_det():
    """
    A specialized test for PaddedLogit determinants.
    """
    inputs = np.random.random(size=(1, 3, 3, 2)).astype('float32') * 0.95
    _log_det_test(PaddedLogit(), inputs)


@pytest.mark.parametrize("layer,shape",
                         [(MaskedConv(partial(checkerboard_mask, True), 1), (1, 4, 6, 2)),
                          (MaskedConv(partial(depth_mask, False), 1), (1, 4, 4, 8)),
                          (MaskedFC(partial(one_cold_mask, 3)), (1, 6))])
def test_log_det(layer, shape):
    """
    Tests log determinants.
    """
    inputs = np.random.normal(size=shape).astype('float32')
    _log_det_test(layer, inputs)


def _log_det_test(layer, inputs):
    assert len(inputs) == 1, 'only works with batch-size 1'
    with tf.Graph().as_default():
        with tf.Session() as sess:
            in_vecs = tf.constant(np.reshape(inputs, [inputs.shape[0], -1]))
            in_tensor = tf.reshape(in_vecs, inputs.shape)
            with tf.variable_scope('model'):
                out, _, log_det = layer.forward(in_tensor)
                out_vecs = tf.reshape(out, in_vecs.get_shape())
            jacobian = _compute_jacobian(in_vecs, out_vecs)
            jacobians = tf.stack([jacobian], axis=0)
            real_log_det = tf.linalg.slogdet(jacobians)[1][0]
            _randomized_init(sess)
            real_log_det, log_det = sess.run([real_log_det, log_det])
            assert log_det.shape == (1,)
            assert not np.isnan(log_det).any()
            assert not np.isnan(real_log_det).any()
            assert np.allclose(real_log_det, log_det[0], atol=1e-4, rtol=1e-4)


def _compute_jacobian(in_vecs, out_vecs):
    num_dims = in_vecs.get_shape()[-1].value
    res = []
    for comp in range(num_dims):
        res.append(tf.gradients(out_vecs[:, comp], in_vecs)[0][0])
    return tf.stack(res, axis=0)


def _randomized_init(sess):
    """
    Initialize all the TF variables in a way that prevents
    default identity behavior.

    Without a random init, some layers are essentially
    just identity transforms.
    """
    sess.run(tf.global_variables_initializer())
    for variable in tf.trainable_variables():
        shape = [x.value for x in variable.get_shape()]
        val = tf.glorot_uniform_initializer()(shape)
        sess.run(tf.assign(variable, val))
