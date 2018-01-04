"""Tests for image histories."""

# pylint: disable=E1129

import tensorflow as tf

from .history import history_image

def test_history_image_append():
    """Test underfull image histories"""
    with tf.Graph().as_default():
        in_image = tf.random_normal((5, 5))
        hist = history_image(in_image, buffer_size=5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(5):
                in_arr, out_arr = sess.run((in_image, hist))
                assert (in_arr == out_arr).all()

def test_history_image_sample():
    """Test sampling from image histories"""
    with tf.Graph().as_default():
        in_image = tf.random_normal((5, 5))
        hist = history_image(in_image, buffer_size=5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            history = []
            for _ in range(5):
                history.append(sess.run(hist))
            sample_count = 0
            while sample_count < 5:
                in_arr, out_arr = sess.run((in_image, hist))
                if (in_arr == out_arr).all():
                    continue
                found = False
                for i, hist_entry in enumerate(history):
                    if (hist_entry == out_arr).all():
                        found = True
                        history[i] = in_arr
                assert found
                sample_count += 1

def test_history_image_single():
    """Test a buffer with one sample"""
    with tf.Graph().as_default():
        in_image = tf.random_normal((5, 5))
        hist = history_image(in_image, buffer_size=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            history = sess.run(hist)
            sample_count = 0
            while sample_count < 5:
                in_arr, out_arr = sess.run((in_image, hist))
                if (in_arr == out_arr).all():
                    continue
                assert (history == out_arr).all()
                history = in_arr
                sample_count += 1
