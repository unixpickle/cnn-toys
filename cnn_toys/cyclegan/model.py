"""
Architectures for GANs.
"""

import tensorflow as tf

from .history import history_image

# pylint: disable=R0902,R0903
class CycleGAN:
    """
    A class representing all the components of a CycleGAN
    in a single package.
    """
    def __init__(self, real_x, real_y, buffer_size=50, cycle_weight=10):
        self.real_x = _add_image_noise(real_x)
        self.real_y = _add_image_noise(real_y)
        self._setup_generators()
        self._setup_discriminators(buffer_size)
        self._setup_cycles(cycle_weight)
        self._setup_gradients()

    def optimize(self, learning_rate=0.0002, global_step=None):
        """Create an Op that takes a training step."""
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return opt.apply_gradients(self.gradients, global_step=global_step)

    def _setup_generators(self):
        start_vars = tf.trainable_variables()
        with tf.variable_scope('gen_x'):
            self.gen_x = generator(self.real_y)
        with tf.variable_scope('gen_y'):
            self.gen_y = generator(self.real_x)
        self.gen_vars = [v for v in tf.trainable_variables() if v not in start_vars]

    def _setup_discriminators(self, buffer_size):
        start_vars = tf.trainable_variables()
        with tf.variable_scope('disc_x'):
            disc_x_loss, gen_x_loss = gan_loss(self.real_x, self.gen_x, buffer_size)
        with tf.variable_scope('disc_y'):
            disc_y_loss, gen_y_loss = gan_loss(self.real_y, self.gen_y, buffer_size)
        self.disc_vars = [v for v in tf.trainable_variables() if v not in start_vars]
        self.disc_loss = disc_x_loss + disc_y_loss
        self.gen_loss = gen_x_loss + gen_y_loss

    def _setup_cycles(self, weight):
        with tf.variable_scope('gen_y', reuse=True):
            self.cycle_y = generator(self.gen_x)
        with tf.variable_scope('gen_x', reuse=True):
            self.cycle_x = generator(self.gen_y)
        self.cycle_loss = weight * (tf.reduce_mean(tf.abs(self.real_x - self.cycle_x)) +
                                    tf.reduce_mean(tf.abs(self.real_y - self.cycle_y)))

    def _setup_gradients(self):
        gen_grad = _grad_dict(self.gen_loss + self.cycle_loss, self.gen_vars)
        disc_grad = _grad_dict(self.disc_loss, self.disc_vars)
        total_grad = _add_grad_dicts(gen_grad, disc_grad)
        self.gradients = [(g, v) for v, g in total_grad.items()]

def discriminator(images):
    """Get a batch of discriminator outputs, one per image."""
    activation = tf.nn.leaky_relu
    outputs = tf.layers.conv2d(images, 64, 4, strides=2, activation=activation)
    for num_filters in [128, 256, 512]:
        outputs = tf.layers.conv2d(outputs, num_filters, 4, strides=2, use_bias=False)
        outputs = activation(instance_norm(outputs))
    return tf.layers.conv2d(outputs, 1, 4)

def gan_loss(real_image, gen_image, buffer_size):
    """Apply a discriminator and get (disc_loss, gen_loss)."""
    buf_image = history_image(gen_image, buffer_size)
    batch = tf.stack([real_image, gen_image, buf_image])
    disc = discriminator(batch)
    real_outs, gen_outs, buf_outs = disc[0], disc[1], disc[2]
    disc_loss = tf.reduce_mean(tf.square(real_outs - 1)) + tf.reduce_mean(tf.square(buf_outs))
    gen_loss = tf.reduce_mean(tf.square(gen_outs - 1))
    return disc_loss, gen_loss

def generator(image):
    """Generate an image in Y using an image in X."""
    activation = lambda x: tf.nn.relu(instance_norm(x))
    output = reflection_pad(tf.expand_dims(image, 0), 7)
    output = tf.layers.conv2d(output, 32, 7, padding='valid', activation=activation)
    for num_filters in [64, 128]:
        output = reflection_pad(output, 3)
        output = tf.layers.conv2d(output, num_filters, 3, strides=2, padding='valid',
                                  activation=activation)
    for _ in range(6):
        new_out = output
        for i in range(2):
            activation = instance_norm if i == 1 else lambda x: tf.nn.relu(instance_norm(x))
            new_out = reflection_pad(new_out, 3)
            new_out = tf.layers.conv2d(new_out, 128, 3, padding='valid', activation=activation)
        output = output + new_out
    for num_filters in [64, 32]:
        output = tf.layers.conv2d_transpose(output, num_filters, 3, strides=2, padding='same',
                                            activation=activation)
    output = reflection_pad(output, 7)
    return tf.sigmoid(tf.layers.conv2d(output, 3, 7, padding='valid'))[0]

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

def _grad_dict(term, variables):
    grads = tf.gradients(term, variables)
    res = {}
    for var, grad in zip(variables, grads):
        if grad is not None:
            res[var] = grad
    return res

def _add_grad_dicts(dict1, dict2):
    res = dict1.copy()
    for var, grad in dict2.items():
        if var in res:
            res[var] += grad
        else:
            res[var] = grad
    return res

def _add_image_noise(image):
    return tf.clip_by_value(image + tf.random_normal(tf.shape(image), stddev=0.001), 0, 1)
