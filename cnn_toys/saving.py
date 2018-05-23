"""
Tools for saving/restoring models.
"""

import os
import pickle

import tensorflow as tf


def save_state(sess, path):
    """Export all TensorFlow variables."""
    with open(path, 'wb+') as outfile:
        pickle.dump(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)), outfile)


def restore_state(sess, path):
    """Import all TensorFlow variables."""
    if not os.path.exists(path):
        return
    with open(path, 'rb') as infile:
        state = pickle.load(infile)
    for var, val in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), state):
        sess.run(tf.assign(var, val))
