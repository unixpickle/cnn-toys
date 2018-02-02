"""
Architectures for CycleGANs.
"""

import tensorflow as tf

from .history import history_image

# pylint: disable=R0902,R0903
class CycleGAN:
    """
    A CycleGAN model.

    The generator() and discriminator() methods can be
    overridden to change the model architecture.

    Unless otherwise stated, all pixels are assumed to
    range from [0, 1].
    """
    def __init__(self, real_x, real_y, buffer_size=50, cycle_weight=10):
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        self.buffer_size = buffer_size
        with tf.variable_scope('cyclegan', initializer=initializer):
            self.real_x = _add_image_noise(real_x)
            self.real_y = _add_image_noise(real_y)
            self._setup_generators()
            self._setup_discriminators()
            self._setup_cycles(cycle_weight)
            self._setup_gradients()

    def optimize(self, learning_rate=0.0002, beta1=0.5, global_step=None):
        """Create an Op that takes a training step."""
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        return opt.apply_gradients(self.gradients, global_step=global_step)

    def generator(self, image):
        """Apply the generator to an image."""
        return standard_generator(image, self._num_residual())

    def discriminator(self, images): # pylint: disable=R0201
        """Apply the descriminator to a batch of images."""
        return standard_discriminator(images)

    def _setup_generators(self):
        start_vars = tf.trainable_variables()
        with tf.variable_scope('gen_x'):
            self.gen_x = self.generator(self.real_y)
        with tf.variable_scope('gen_y'):
            self.gen_y = self.generator(self.real_x)
        self.gen_vars = [v for v in tf.trainable_variables() if v not in start_vars]

    def _setup_discriminators(self):
        start_vars = tf.trainable_variables()
        with tf.variable_scope('disc_x'):
            disc_x_loss, gen_x_loss = self._discriminate(self.real_x, self.gen_x)
        with tf.variable_scope('disc_y'):
            disc_y_loss, gen_y_loss = self._discriminate(self.real_y, self.gen_y)
        self.disc_vars = [v for v in tf.trainable_variables() if v not in start_vars]
        self.disc_loss = (disc_x_loss + disc_y_loss) / 2
        self.gen_loss = gen_x_loss + gen_y_loss

    def _setup_cycles(self, weight):
        with tf.variable_scope('gen_y', reuse=True):
            self.cycle_y = self.generator(self.gen_x)
        with tf.variable_scope('gen_x', reuse=True):
            self.cycle_x = self.generator(self.gen_y)
        self.cycle_loss = weight * (tf.reduce_mean(tf.abs(self.real_x - self.cycle_x)) +
                                    tf.reduce_mean(tf.abs(self.real_y - self.cycle_y)))

    def _setup_gradients(self):
        gen_grad = _grad_dict(self.gen_loss + self.cycle_loss, self.gen_vars)
        disc_grad = _grad_dict(self.disc_loss, self.disc_vars)
        total_grad = _add_grad_dicts(gen_grad, disc_grad)
        self.gradients = [(g, v) for v, g in total_grad.items()]

    def _discriminate(self, real_image, gen_image):
        """
        Run samples through a discriminator to get the GAN
        losses.

        Returns:
          A tuple (discriminator_loss, generator_loss).
        """
        buf_image = history_image(gen_image, self.buffer_size)
        batch = tf.stack([real_image, gen_image, buf_image])
        disc = self.discriminator(batch)
        real_outs, gen_outs, buf_outs = disc[0], disc[1], disc[2]
        disc_loss = tf.reduce_mean(tf.square(real_outs - 1)) + tf.reduce_mean(tf.square(buf_outs))
        gen_loss = tf.reduce_mean(tf.square(gen_outs - 1))
        return disc_loss, gen_loss

    def _num_residual(self):
        if self.real_x.get_shape()[1].value > 128:
            return 9
        return 6

def standard_discriminator(images):
    """
    Apply the standard CycleGAN discriminator to a batch
    of images.

    The output Tensor may have any rank, but the outer
    dimension must match the batch size.
    """
    activation = tf.nn.leaky_relu
    outputs = 2 * images - 1
    outputs = tf.layers.conv2d(outputs, 64, 4, strides=2, activation=activation)
    for num_filters in [128, 256, 512]:
        strides = 2
        if num_filters == 512:
            strides = 1
        outputs = tf.layers.conv2d(outputs, num_filters, 4, strides=strides, use_bias=False)
        outputs = activation(instance_norm(outputs))
    return tf.layers.conv2d(outputs, 1, 1)

def standard_generator(image, num_residual):
    """
    Apply the standard CycleGAN generator to an image.

    Args:
      image: an image Tensor with pixel values in the
        range [0, 1].
      num_residual: the number of residual layers. In the
        original paper, this varied by image size.
    """
    activation = lambda x: tf.nn.relu(instance_norm(x))
    output = 2 * image - 1
    output = reflection_pad(tf.expand_dims(output, 0), 7)
    output = tf.layers.conv2d(output, 32, 7, padding='valid', activation=activation, use_bias=False)
    for num_filters in [64, 128]:
        output = reflection_pad(output, 3)
        output = tf.layers.conv2d(output, num_filters, 3, strides=2, padding='valid',
                                  activation=activation, use_bias=False)
    for _ in range(num_residual):
        new_out = output
        for i in range(2):
            activation = instance_norm if i == 1 else lambda x: tf.nn.relu(instance_norm(x))
            new_out = reflection_pad(new_out, 3)
            new_out = tf.layers.conv2d(new_out, 128, 3, padding='valid', activation=activation,
                                       use_bias=False)
        output = output + new_out
    for num_filters in [64, 32]:
        output = tf.layers.conv2d_transpose(output, num_filters, 3, strides=2, padding='same',
                                            activation=activation, use_bias=False)
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
