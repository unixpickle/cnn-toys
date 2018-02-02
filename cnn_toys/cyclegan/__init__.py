"""The CycleGAN architecture."""

from .history import history_image
from .model import (CycleGAN, standard_discriminator, standard_generator, instance_norm,
                    reflection_pad)
