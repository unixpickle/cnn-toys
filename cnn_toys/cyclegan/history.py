"""
An in-graph image history buffer.
"""

import tensorflow as tf

def history_image(image, buffer_size, name='image_buffer'):
    """
    Get an image from a history buffer and submit the
    image to the same buffer.

    Args:
      image: an image Tensor. The shape must be known in
        advance.
      buffer_size: the number of images to store in the
        history buffer.
      name: the default name of the scope in which the
        buffer is stored. Uniquified as needed.
    """
    with tf.variable_scope(None, default_name=name):
        buf = tf.get_variable('images', shape=[buffer_size] + [x.value for x in image.get_shape()],
                              dtype=image.dtype, trainable=False)
        size = tf.get_variable('size', dtype=tf.int32, initializer=0, trainable=False)
        def _insert_new():
            insert_idx = tf.assign_add(size, 1) - 1
            return _assign_buf_entry(buf, insert_idx, image)
        def _sample_old():
            idx = tf.random_uniform((), maxval=buffer_size, dtype=tf.int32)
            # `+ 0` hack to deal with buffer_size == 1.
            # See https://github.com/tensorflow/tensorflow/issues/4663.
            old = buf[idx] + 0
            with tf.control_dependencies([old]):
                assign = _assign_buf_entry(buf, idx, image)
                with tf.control_dependencies([assign]):
                    return tf.identity(old)
        return tf.cond(size < buffer_size,
                       _insert_new,
                       lambda: tf.cond(tf.random_uniform(()) < 0.5,
                                       _sample_old,
                                       lambda: image))

def _assign_buf_entry(buf, idx, image):
    pieces = [buf[:idx], tf.expand_dims(image, 0), buf[idx+1:]]
    return tf.assign(buf, tf.concat(pieces, 0))[idx]
