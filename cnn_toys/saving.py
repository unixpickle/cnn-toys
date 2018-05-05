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
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    placeholders = [tf.placeholder(dtype=v.dtype.base_dtype, shape=v.get_shape()) for v in all_vars]
    assigns = [tf.assign(var, ph) for var, ph in zip(all_vars, placeholders)]
    sess.run(tf.group(*assigns), feed_dict=dict(zip(placeholders, state)))
