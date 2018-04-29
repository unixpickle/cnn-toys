"""
Real-valued non-volume preserving transformations.
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

class NVPLayer(ABC):
    """
    A layer in a real NVP model.
    """
    @property
    def num_latents(self):
        """
        Get the size of the latent tuple returned by
        forward().
        """
        return 0

    @abstractmethod
    def forward(self, inputs):
        """
        Apply the layer to a batch of inputs.

        Args:
          inputs: an input batch for the layer.

        Returns:
          A tuple (outputs, latents, log_det):
            outputs: the values to be passed to the next
              layer of the network. May be None for the
              last layer of the network.
            latents: A tuple of factored out Tensors.
              This may be an empty tuple.
            log_det: the log of the determinant of the Jacobian
        """
        pass

    @abstractmethod
    def inverse(self, outputs, latents):
        """
        Apply the inverse of the model.

        Args:
          outputs: the outputs from the layer.
          latents: the latent outputs from the layer.

        Returns:
          The recovered input batch for the layer.
        """
        pass

class Network(NVPLayer):
    """
    A feed-forward composition of NVP layers.
    """
    def __init__(self, layers):
        self.layers = layers

    @property
    def num_latents(self):
        return 1 + sum(l.num_latents for l in self.layers)

    def forward(self, inputs):
        latents = []
        outputs = inputs
        log_det = tf.zeros(shape=(), dtype=inputs.dtype)
        for layer in self.layers:
            outputs, sub_latents, sub_log_det = layer.forward(outputs)
            latents.extend(sub_latents)
            log_det = log_det + sub_log_det
        latents.append(outputs)
        return None, tuple(latents), log_det

    def inverse(self, outputs, latents):
        assert outputs is None
        assert len(latents) == self.num_latents
        latents = latents[:-1]
        inputs = latents[0]
        for layer in self.layers:
            sub_latents = latents[-layer.num_latents:]
            latents = latents[:-layer.num_latents]
            inputs = layer.inverse(inputs, sub_latents)
        return inputs

class FactorHalf(NVPLayer):
    """
    A layer that factors out half of the inputs.
    """
    @property
    def num_latents(self):
        return 1

    def forward(self, inputs):
        half_depth = inputs.get_shape()[-1].value // 2
        return inputs[:half_depth], (inputs[half_depth:],), tf.constant(0, dtype=inputs.dtype)

    def inverse(self, outputs, latents):
        return tf.concat([outputs, latents], axis=-1)

class Squeeze(NVPLayer):
    """
    A layer that squeezes 2x2x1 blocks into 1x1x4 blocks.
    """
    def forward(self, inputs):
        assert all([x.value % 2 == 0 for x in inputs.get_shape()[1:3]]), 'even shape required'
        conv_filter = self._permutation_filter(inputs.get_shape()[-1].value, inputs.dtype)
        return (tf.nn.conv2d(inputs, conv_filter, [1, 2, 2, 1], 'VALID'),
                (), tf.constant(0, dtype=inputs.dtype))

    def inverse(self, outputs, latents):
        assert latents == ()
        in_depth = outputs.get_shape()[-1].value // 4
        conv_filter = self._permutation_filter(in_depth, outputs.dtype)
        out_shape = ([tf.shape(outputs)[0]] + [x.value * 2 for x in outputs.get_shape()[1:3]] +
                     [in_depth])
        return tf.nn.conv2d_transpose(outputs, conv_filter, out_shape, [1, 2, 2, 1], 'VALID')

    @staticmethod
    def _permutation_filter(depth, dtype):
        """
        Generate a convolutional filter that performs the
        squeeze operation.
        """
        res = np.zeros((2, 2, depth, depth * 4))
        for i in range(depth):
            for row in range(2):
                for col in range(2):
                    res[row, col, i, 4 * i + row * 2 + col] = 1
        return tf.constant(res, dtype=dtype)

class MaskedConv(NVPLayer):
    """
    A masked convolution NVP transformation.
    """
    def __init__(self, mask_fn, kernel_size, **conv_kwargs):
        """
        Create a masked convolution layer.

        Args:
          mask_fn: a function which takes a Tensor and
            produces a boolean mask Tensor.
          kernel_size: the convolutional kernel size.
          conv_kwargs: other arguments for conv2d().
        """
        self.mask_fn = mask_fn
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        self._tanh_scale = None

    def forward(self, inputs):
        biases, log_scales = self._apply_masked(inputs)
        log_det = tf.reduce_sum(log_scales)
        return inputs * tf.exp(log_scales) + biases, (), log_det

    def inverse(self, outputs, latents):
        assert latents == ()
        biases, log_scales = self._apply_masked(outputs)
        return (outputs - biases) * tf.exp(-log_scales)

    def _apply_masked(self, inputs):
        """
        Get (biases, log_scales) for the inputs.
        """
        mask = self.mask_fn(inputs)
        depth = inputs.get_shape()[3].value
        masked = tf.cast(mask, inputs.dtype) * inputs
        output = tf.layers.conv2d(masked, depth * 2, self.kernel_size, **self.conv_kwargs)
        biases = output[..., :depth]
        scale_params = output[..., depth:]
        log_scales = tf.where(mask,
                              tf.zeros_like(inputs),
                              tf.tanh(scale_params) * self._get_tanh_scale(inputs))
        return biases, log_scales

    def _get_tanh_scale(self, in_out):
        if self._tanh_scale is None:
            self._tanh_scale = tf.get_variable('tanh_scale',
                                               shape=[x.value for x in in_out.get_shape()[1:]],
                                               dtype=in_out.dtype,
                                               initializer=tf.ones_initializer())
        return self._tanh_scale

def checkerboard_mask(is_even, tensor):
    """
    Create a checkerboard mask in the shape of a Tensor.

    Args:
      is_even: determines which of two masks to use.
      tensor: the Tensor whose shape to match.
    """
    result = np.zeros([tensor.get_shape()[1].value, tensor.get_shape()[2].value, 1], dtype='bool')
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            result[row, col] = (((row + col) % 2 == 0) == is_even)
    # Broadcast into the shape of the tensor.
    return tf.zeros(shape=tf.shape(tensor), dtype=tf.bool) + tf.constant(result)

def depth_mask(is_even, tensor):
    """
    Create a depth mask in the shape of a Tensor.

    Args:
      is_even: determines which of two masks to use.
      tensor: the Tensor whose shape to match.
    """
    if is_even:
        mask = [True, True, False, False]
    else:
        mask = [False, False, True, True]
    return tf.zeros(shape=tf.shape(tensor), dtype=tf.bool) + tf.constant(mask)
