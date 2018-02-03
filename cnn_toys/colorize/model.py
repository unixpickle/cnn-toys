"""
A model for image colorization.
"""

import tensorflow as tf

def sample_loss(input_ph):
    """
    Generate a loss for converting a color image to
    grayscale and then back again.
    """
    colorized = colorize(tf.reduce_mean(input_ph, axis=-1, keep_dims=True))
    return tf.reduce_mean(tf.abs(colorized - input_ph))

def colorize(input_ph):
    """
    Apply a neural network to produce a colorized version
    of the input images.
    """
    activation = lambda x: tf.nn.relu(tf.contrib.layers.layer_norm(x))
    output = input_ph
    output = tf.pad(output, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    output = tf.layers.conv2d(output, 32, 7, activation=tf.nn.relu)
    for features in [64, 128]:
        output = tf.layers.conv2d(output, features, 3, activation=activation)
    for _ in range(6):
        old_output = output
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        output = tf.layers.conv2d(output, 128, 3, activation=activation)
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        output = tf.layers.conv2d(output, 128, 3, activation=tf.contrib.layers.layer_norm)
        output = tf.nn.relu(old_output + output)
    output = tf.layers.conv2d_transpose(output, 64, 3, strides=2, activation=activation)
    output = tf.layers.conv2d_transpose(output, 32, 3, strides=2, activation=activation)
    output = tf.pad(output, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    return tf.layers.conv2d(output, 3, 7, activation=tf.sigmoid)
