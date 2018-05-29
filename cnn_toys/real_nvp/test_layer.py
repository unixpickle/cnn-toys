"""
Tests for real NVP layers.
"""

from functools import partial
from random import random

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


def test_gradients():
    """
    Test that manual gradient computation works properly.
    """
    with tf.Graph().as_default():  # pylint: disable=E1129
        layers = [
            PaddedLogit(),
            MaskedConv(partial(checkerboard_mask, True), 2),
            MaskedConv(partial(checkerboard_mask, False), 2),
            Squeeze(),
            MaskedConv(partial(depth_mask, True), 2),
            FactorHalf(),
        ]
        network = Network(layers)
        inputs = tf.random_uniform([3, 8, 8, 4])
        outputs, latents, log_det = network.forward(inputs)
        loss = (tf.reduce_sum(tf.stack([(random() + 1) * tf.reduce_sum(x) for x in latents])) +
                (random() + 1) * tf.reduce_sum(log_det))

        manual_grads = network.gradients(outputs, latents, log_det, loss)
        manual_grads = {var: grad for grad, var in manual_grads}
        manual_grads = [manual_grads[v] for v in tf.trainable_variables()]

        true_grads = tf.gradients(loss, tf.trainable_variables())

        diffs = [tf.reduce_max(x - y) for x, y in zip(manual_grads, true_grads)]
        max_diff = tf.reduce_max(tf.stack(diffs))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _randomized_init(sess)
            assert sess.run(max_diff) < 1e-4


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
    with tf.Graph().as_default():  # pylint: disable=E1129
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
                         [(MaskedConv(partial(checkerboard_mask, True), 1), (3, 4, 6, 2)),
                          (MaskedConv(partial(depth_mask, False), 1), (3, 4, 4, 8)),
                          (MaskedFC(partial(one_cold_mask, 3)), (3, 6))])
def test_log_det(layer, shape):
    """
    Tests log determinants.
    """
    inputs = np.random.normal(size=shape).astype('float32')
    _log_det_test(layer, inputs)


def _log_det_test(layer, inputs):
    with tf.Graph().as_default():  # pylint: disable=E1129
        with tf.Session() as sess:
            in_vecs = tf.constant(np.reshape(inputs, [inputs.shape[0], -1]))
            in_tensor = tf.reshape(in_vecs, inputs.shape)
            with tf.variable_scope('model'):
                out, _, log_dets = layer.forward(in_tensor)
                out_vecs = tf.reshape(out, in_vecs.get_shape())
            jacobians = _compute_jacobians(in_vecs, out_vecs)
            real_log_dets = tf.linalg.slogdet(jacobians)[1]
            _randomized_init(sess)
            real_log_dets, log_dets = sess.run([real_log_dets, log_dets])
            assert log_dets.shape == (inputs.shape[0],)
            assert not np.isnan(log_dets).any()
            assert not np.isnan(real_log_dets).any()
            assert np.allclose(real_log_dets, log_dets, atol=1e-4, rtol=1e-4)


def _compute_jacobians(in_vecs, out_vecs):
    num_dims = in_vecs.get_shape()[-1].value
    res = []
    for i in range(in_vecs.get_shape()[0].value):
        rows = []
        for comp in range(num_dims):
            rows.append(tf.gradients(out_vecs[i, comp], in_vecs)[0][i])
        res.append(tf.stack(rows, axis=0))
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
