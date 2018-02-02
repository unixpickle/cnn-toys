"""Learning rate schedules."""

import tensorflow as tf

def half_annealed_lr(initial, iters, global_step):
    """
    Create a learning rate that stays at an initial value
    for the first half of training, then is linearly
    annealed for the second half of training.

    Args:
      initial: the initial LR.
      iters: the total number of iterations.
      global_step: the step counter Tensor.
    """
    frac_done = 1 - tf.cast(iters - global_step, tf.float32) / float(iters)
    return tf.cond(frac_done < 0.5, lambda: initial, lambda: (1 - frac_done) * 2 * initial)
