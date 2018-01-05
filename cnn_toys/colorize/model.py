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
    output = input_ph
    for features in [32, 64, 128, 256, 128, 64, 32, 3]:
        output = tf.pad(output, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
        activation = None
        if features != 3:
            activation = tf.nn.relu
        output = tf.layers.conv2d(output, features, 5, padding='valid', activation=activation)
    return tf.sigmoid(output)
