"""
Real-valued non-volume preserving transformations.
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class NVPLayer(ABC):
    """
    A layer in a real NVP model.

    Subclasses must override _forward() and _inverse().
    Subclasses may also override test_feed_dict() and
    num_latents() if appropriate.
    """
    @property
    def num_latents(self):
        """
        Get the size of the latent tuple returned by
        forward().
        """
        return 0

    def test_feed_dict(self):
        """
        Get a feed_dict to pass to TensorFlow when testing
        the model. Typically, this will tell BatchNorm to
        use pre-computed statistics.
        """
        return {}

    @abstractmethod
    def _forward(self, inputs):
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
            log_det: a batch of log of the determinants.
        """
        pass

    @abstractmethod
    def _inverse(self, outputs, latents):
        """
        Apply the inverse of the layer.

        Args:
          outputs: the outputs from the layer.
          latents: the latent outputs from the layer.

        Returns:
          The recovered input batch for the layer.
        """
        pass

    def forward(self, inputs, name='layer', reuse=False):
        """
        Apply the layer to a batch of inputs.

        Args:
          inputs: an input batch for the layer.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.

        Returns:
          A tuple (outputs, latents, log_det):
            outputs: the values to be passed to the next
              layer of the network. May be None for the
              last layer of the network.
            latents: A tuple of factored out Tensors.
              This may be an empty tuple.
            log_det: a batch of log of the determinants.
        """
        with tf.variable_scope(name, reuse=reuse):
            return self._forward(inputs)

    def inverse(self, outputs, latents, name='layer', reuse=False):
        """
        Apply the inverse of the layer.

        Args:
          outputs: the outputs from the layer.
          latents: the latent outputs from the layer.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.

        Returns:
          The recovered input batch for the layer.
        """
        with tf.variable_scope(name, reuse=reuse):
            return self._inverse(outputs, latents)

    def backward(self, outputs, outputs_grad, latents, latents_grad, log_det_grad,
                 var_list=None, name='layer', reuse=False):
        """
        Compute a gradient through the layer.

        This is optimized for memory consumption.
        Currently, it does not support 2nd derivatives.

        Args:
          outputs: the outputs of the layer. May be None.
          outputs_grad: the gradient of the objective with
            respect to the outputs. May be None.
          latents: the latent outputs from the layer.
          latents_grad: the gradient of the objective with
            respect to the latents.
          log_det_grad: the gradient of the objective with
            respect to the log determinant.
          var_list: the list of variables to differentiate
            with respect to. If None, use all trainable
            variables.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.

        Returns:
          A tuple (upstream, grads):
            inputs: the recovered inputs to the layer.
            upstream: a Tensor representing the gradient
              of the objective with respect to the inputs
              to the layer.
            grads: a list of (gradient, variable) pairs
              for the parameters of the layer.
        """
        inputs = tf.stop_gradient(self.inverse(outputs, latents, name=name, reuse=reuse))
        new_outputs, new_latents, new_log_dets = self.forward(inputs,
                                                              name=name,
                                                              reuse=True)
        objective = tf.reduce_sum(new_log_dets * tf.stop_gradient(log_det_grad))
        if new_outputs is not None:
            objective += tf.reduce_sum(new_outputs * tf.stop_gradient(outputs_grad))
        for latent, latent_grad in zip(new_latents, latents_grad):
            objective += tf.reduce_sum(latent * tf.stop_gradient(latent_grad))
        input_grad = tf.gradients(objective, inputs)[0]
        if input_grad is None:
            input_grad = tf.Print(tf.zeros_like(input_grad), [],
                                  message='WARNING: gradient does not flow to inputs',
                                  first_n=1)
        variables = var_list if var_list is not None else tf.trainable_variables()
        var_grads = [x for x in zip(tf.gradients(objective, variables), variables)
                     if x[0] is not None]
        return inputs, input_grad, var_grads

    def gradients(self, outputs, latents, log_det, loss, var_list=None, name='layer', reuse=True):
        """
        Perform backpropagation through the layer using
        the backward() method.

        This computes gradients without needing to store
        intermediate Tensors from the forward pass.
        Currently, it does not support 2nd derivatives.

        Args:
          outputs: the layer outputs.
          latents: the layer's latent outputs.
          log_det: the output log determinants.
          loss: the loss value resulting from the latents
            and log determinants.
          var_list: the variables to find gradients for,
            or None to use all trainable variables.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.

        Returns:
          A list of (gradient, variable) pairs.
        """
        assert len(latents) == self.num_latents
        if outputs is not None:
            outputs_grad = tf.gradients(loss, outputs)[0]
            if outputs_grad is None:
                outputs_grad = tf.zeros_like(outputs)
        else:
            outputs_grad = None
        latents_grad = [grad if grad is not None else tf.zeros_like(latent)
                        for grad, latent in zip(tf.gradients(loss, latents), latents)]
        log_det_grad = tf.gradients(loss, log_det)[0]
        if log_det_grad is None:
            log_det_grad = tf.zeros_like(log_det)
        return self.backward(outputs, outputs_grad, latents, latents_grad, log_det_grad,
                             var_list=var_list, name=name, reuse=reuse)[2]


class Network(NVPLayer):
    """
    A feed-forward composition of NVP layers.
    """

    def __init__(self, layers):
        self.layers = layers

    @property
    def num_latents(self):
        return 1 + sum(l.num_latents for l in self.layers)

    def test_feed_dict(self):
        res = {}
        for layer in self.layers:
            res.update(layer.test_feed_dict())

    def _forward(self, inputs):
        latents = []
        outputs = inputs
        log_det = tf.zeros(shape=[tf.shape(inputs)[0]], dtype=inputs.dtype)
        for i, layer in enumerate(self.layers):
            outputs, sub_latents, sub_log_det = layer.forward(outputs, name='layer_%d' % i)
            latents.extend(sub_latents)
            log_det = log_det + sub_log_det
        latents.append(outputs)
        return None, tuple(latents), log_det

    def _inverse(self, outputs, latents):
        assert outputs is None
        assert len(latents) == self.num_latents
        inputs = latents[-1]
        latents = latents[:-1]
        for i, layer in list(enumerate(self.layers))[::-1]:
            if layer.num_latents > 0:
                sub_latents = latents[-layer.num_latents:]
                latents = latents[:-layer.num_latents]
            else:
                sub_latents = ()
            inputs = layer.inverse(inputs, sub_latents, name='layer_%d' % i)
        return inputs

    def backward(self, outputs, outputs_grad, latents, latents_grad, log_det_grad,
                 var_list=None, name='layer', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            assert outputs is None
            assert outputs_grad is None
            outputs = latents[-1]
            outputs_grad = latents_grad[-1]
            latents = latents[:-1]
            latents_grad = latents_grad[:-1]
            total_grads = {}
            prev_grads = []
            for i, layer in list(enumerate(self.layers))[::-1]:
                if layer.num_latents > 0:
                    sub_latents = latents[-layer.num_latents:]
                    sub_latents_grad = latents_grad[-layer.num_latents:]
                    latents = latents[:-layer.num_latents]
                    latents_grad = latents_grad[:-layer.num_latents]
                else:
                    sub_latents = ()
                    sub_latents_grad = ()
                with tf.control_dependencies(prev_grads):
                    outputs, outputs_grad, vars_grad = layer.backward(outputs,
                                                                      outputs_grad,
                                                                      sub_latents,
                                                                      sub_latents_grad,
                                                                      log_det_grad,
                                                                      var_list=var_list,
                                                                      name='layer_%d' % i)
                for grad, var in vars_grad:
                    if var in total_grads:
                        total_grads[var] += grad
                    else:
                        total_grads[var] = grad
                prev_grads = [g for g, _ in vars_grad]
            return outputs, outputs_grad, [(grad, var) for var, grad in total_grads.items()]


class PaddedLogit(NVPLayer):
    """
    An NVP layer that applies `logit(a + (1-2a)x)`.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def _forward(self, inputs):
        padded = self.alpha + (1 - 2 * self.alpha) * inputs
        logits = tf.log(padded / (1 - padded))
        log_dets = tf.log(1 / padded + 1 / (1 - padded)) + tf.log((1 - 2 * self.alpha))
        return logits, (), sum_batch(log_dets)

    def _inverse(self, outputs, latents):
        assert latents == ()
        sigmoids = tf.nn.sigmoid(outputs)
        return (sigmoids - self.alpha) / (1 - 2 * self.alpha)


class FactorHalf(NVPLayer):
    """
    A layer that factors out half of the inputs.
    """
    @property
    def num_latents(self):
        return 1

    def _forward(self, inputs):
        return (inputs[..., ::2], (inputs[..., 1::2],),
                tf.constant(0, dtype=inputs.dtype))

    def _inverse(self, outputs, latents):
        assert len(latents) == 1
        # Trick to undo the alternating split.
        expanded_1 = tf.expand_dims(outputs, axis=-1)
        expanded_2 = tf.expand_dims(latents[0], axis=-1)
        concatenated = tf.concat([expanded_1, expanded_2], axis=-1)
        new_shape = [tf.shape(outputs)[0]] + [x.value for x in outputs.get_shape()[1:]]
        new_shape[-1] *= 2
        return tf.reshape(concatenated, new_shape)


class Squeeze(NVPLayer):
    """
    A layer that squeezes 2x2x1 blocks into 1x1x4 blocks.
    """

    def _forward(self, inputs):
        assert all([x.value % 2 == 0 for x in inputs.get_shape()[1:3]]), 'even shape required'
        conv_filter = self._permutation_filter(inputs.get_shape()[-1].value, inputs.dtype)
        return (tf.nn.conv2d(inputs, conv_filter, [1, 2, 2, 1], 'VALID'),
                (), tf.constant(0, dtype=inputs.dtype))

    def _inverse(self, outputs, latents):
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


class MaskedLayer(NVPLayer):
    """
    An abstract NVP transformation that uses a masked
    neural network.
    """

    def _forward(self, inputs):
        biases, log_scales = self._apply_masked(inputs)
        log_det = sum_batch(log_scales)
        return inputs * tf.exp(log_scales) + biases, (), log_det

    def _inverse(self, outputs, latents):
        assert latents == ()
        biases, log_scales = self._apply_masked(outputs)
        return (outputs - biases) * tf.exp(-log_scales)

    @abstractmethod
    def _apply_masked(self, inputs):
        """
        Get (biases, log_scales) for the inputs.
        """
        pass


class MaskedConv(MaskedLayer):
    """
    A masked convolution NVP transformation.
    """

    def __init__(self, mask_fn, num_residual, num_features=32, kernel_size=3, **conv_kwargs):
        """
        Create a masked convolution layer.

        Args:
          mask_fn: a function which takes a Tensor and
            produces a boolean mask Tensor.
          num_residual: the number of residual blocks.
          num_features: the number of latent features.
          kernel_size: the convolutional kernel size.
          conv_kwargs: other arguments for conv2d().
        """
        self.mask_fn = mask_fn
        self.num_residual = num_residual
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        self._training = tf.constant(True)

    def test_feed_dict(self):
        return {self._training: False}

    def _apply_masked(self, inputs):
        """
        Get (biases, log_scales) for the inputs.
        """
        mask = self.mask_fn(inputs)
        depth = inputs.get_shape()[3].value
        masked = tf.where(mask, inputs, tf.zeros_like(inputs))
        latent = tf.layers.conv2d(masked, self.num_features, 1)
        for _ in range(self.num_residual):
            latent = self._residual_block(tf.nn.relu(latent))
        with tf.variable_scope(None, default_name='mask_biases'):
            bias_params = tf.layers.conv2d(latent, depth, 1)
            biases = tf.where(mask, tf.zeros_like(inputs), bias_params)
        with tf.variable_scope(None, default_name='mask_scales'):
            scale_params = tf.layers.conv2d(latent, depth, 1)
            log_scales = tf.where(mask,
                                  tf.zeros_like(inputs),
                                  tf.tanh(scale_params) * self._get_tanh_scale(inputs))
        return biases, log_scales

    def _residual_block(self, inputs):
        with tf.variable_scope(None, default_name='residual'):
            output = tf.layers.conv2d(inputs, self.num_features, self.kernel_size, padding='same',
                                      **self.conv_kwargs)
            output = tf.nn.relu(self._batch_norm(output))
            output = tf.layers.conv2d(output, self.num_features, self.kernel_size, padding='same',
                                      **self.conv_kwargs)
            output = self._batch_norm(output)
            return output + inputs

    def _batch_norm(self, values):
        return tf.layers.batch_normalization(values, training=self._training)

    @staticmethod
    def _get_tanh_scale(in_out):
        with tf.variable_scope(None, default_name='mask_params'):
            return tf.get_variable('tanh_scale',
                                   shape=[x.value for x in in_out.get_shape()[1:]],
                                   dtype=in_out.dtype,
                                   initializer=tf.zeros_initializer())


class MaskedFC(MaskedLayer):
    """
    A fully-connected layer that scales certain dimensions
    using information from other dimensions.
    """

    def __init__(self, mask_fn, num_features=64, num_layers=2):
        """
        Create a masked layer.

        Args:
          mask_fn: a function which takes a Tensor and
            produces a boolean mask Tensor.
          num_features: the number of hidden units.
          num_layers: the number of hidden layers.
        """
        self.mask_fn = mask_fn
        self.num_features = num_features
        self.num_layers = num_layers

    def _apply_masked(self, inputs):
        """
        Get (biases, log_scales) for the inputs.
        """
        depth = inputs.get_shape()[-1]
        mask = self.mask_fn(inputs)

        masked_in = tf.where(mask, inputs, tf.zeros_like(inputs))
        out = tf.layers.dense(masked_in, self.num_features, activation=tf.nn.relu)
        for _ in range(self.num_layers - 1):
            out = tf.layers.dense(out, self.num_features, activation=tf.nn.relu)
        log_scales = tf.layers.dense(out, depth, kernel_initializer=tf.zeros_initializer())
        log_scales = tf.where(mask, tf.zeros_like(inputs), log_scales)
        biases = tf.layers.dense(out, depth, kernel_initializer=tf.zeros_initializer())
        biases = tf.where(mask, tf.zeros_like(inputs), biases)

        return biases, log_scales


def checkerboard_mask(is_even, tensor):
    """
    Create a checkerboard mask in the shape of a Tensor.

    Args:
      is_even: determines which of two masks to use.
      tensor: the Tensor whose shape to match.
    """
    result = np.zeros([x.value for x in tensor.get_shape()[1:]], dtype='bool')
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            result[row, col, :] = (((row + col) % 2 == 0) == is_even)
    return tf.tile(tf.expand_dims(result, axis=0), [tf.shape(tensor)[0], 1, 1, 1])


def one_cold_mask(idx, tensor):
    """
    Create a mask that only masks out one channel in a 2-D
    input Tensor.
    """
    result = np.ones([tensor.get_shape()[-1].value], dtype='bool')
    result[idx] = False
    return tf.tile(tf.expand_dims(result, axis=0), [tf.shape(tensor)[0], 1])


def depth_mask(is_even, tensor):
    """
    Create a depth mask in the shape of a Tensor.

    Args:
      is_even: determines which of two masks to use.
      tensor: the Tensor whose shape to match.
    """
    assert tensor.get_shape()[-1] % 4 == 0, 'depth must be divisible by 4'
    if is_even:
        mask = [True, True, False, False]
    else:
        mask = [False, False, True, True]
    one_dim = tf.tile(tf.constant(mask, dtype=tf.bool), [tensor.get_shape()[-1].value // 4])
    # Broadcast, since + doesn't work for booleans.
    return tf.logical_or(one_dim, tf.zeros(shape=tf.shape(tensor), dtype=tf.bool))


def sum_batch(tensor):
    """
    Compute a 1-D batch of sums.
    """
    return tf.reduce_sum(tf.reshape(tensor, [tf.shape(tensor)[0], -1]), axis=1)
