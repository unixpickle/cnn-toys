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
    @abstractmethod
    def forward(self, inputs):
        """
        Apply the layer to a batch of inputs.

        Args:
          inputs: an input batch for the layer.

        Returns:
          A tuple (outputs, latents, log_det):
            outputs: the values to be passed to the next
              layer of the network.
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

class FactorHalf(NVPLayer):
    """
    An NVPLayer that factors out half of the inputs.
    """
    def forward(self, inputs):
        half_depth = inputs.get_shape()[-1].value // 2
        return inputs[:half_depth], (inputs[half_depth:]), tf.constant(0, dtype=inputs.dtype)

    def inverse(self, outputs, latents):
        return tf.concat([outputs, latents], axis=-1)
