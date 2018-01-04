"""
Architectures for GANs.
"""

import tensorflow as tf

from .history import history_image    

def discriminator(images):
    """Get a batch of discriminator outputs, one per image."""
    activation = tf.nn.leaky_relu
    outputs = tf.layers.conv2d(images, 64, 4, strides=2, activation=activation)
    for num_filters in [128, 256, 512]:
        outputs = tf.layers.conv2d(outputs, num_filters, 4, strides=2, use_bias=False)
        outputs = activation(instance_norm(outputs))
    flat = tf.reshape(outputs, [outputs.get_shape()[0].value, -1])
    return tf.layers.dense(flat, 1, activation=None)

def generator(images):
    """Generate images in Y using the batch of images in X."""
    activation = lambda x: tf.nn.relu(instance_norm(x))
    output = reflection_pad(images, 7)
    output = tf.layers.conv2d(output, 32, 7, activation=activation)
    for num_filters in [64, 128]:
        output = reflection_pad(output, 3)
        output = tf.layers.conv2d(output, num_filters, 3, strides=2, activation=activation)
    for _ in range(6):
        new_out = output
        for _ in range(2):
            new_out = reflection_pad(new_out, 3)
            new_out = tf.layers.conv2d(tf.layers.conv2d(output, 128, 3), 128, 3)
        output = output + new_out
    for num_filters in [64, 32]:
        output = tf.layers.conv2d_transpose(output, num_filters, 3, strides=2, padding='same',
                                            activation=activation)
    output = reflection_pad(output, 7)
    return tf.sigmoid(tf.layers.conv2d(output, 3, 7))

def instance_norm(images, epsilon=1e-5, name='instance_norm'):
    """Apply instance normalization to the batch."""
    means = tf.reduce_mean(images, axis=[1, 2], keep_dims=True)
    stddevs = tf.sqrt(tf.reduce_mean(tf.square(images - means), axis=[1, 2], keep_dims=True))
    results = (images - means) / (stddevs + epsilon)
    with tf.variable_scope(None, default_name=name):
        biases = tf.get_variable('biases', shape=images.get_shape()[-1].value, dtype=images.dtype,
                                 initializer=tf.zeros_initializer())
        scales = tf.get_variable('scales', shape=images.get_shape()[-1].value, dtype=images.dtype,
                                 initializer=tf.ones_initializer())
        return results*scales + biases

def reflection_pad(images, filter_size):
    """Perform reflection padding for a convolution."""
    num = filter_size // 2
    return tf.pad(images, [[0, 0], [num, num], [num, num], [0, 0]], mode='REFLECT')
