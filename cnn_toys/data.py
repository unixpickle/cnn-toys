"""
Tools for dealing with datasets of images.
"""

import glob
from hashlib import md5
import os

import tensorflow as tf


def dir_train_val(image_dir, size, bigger_size=None):
    """
    Create a training and validation Dataset by reading
    and splitting a directory of images.

    Returns (train, validation).
    """
    paths = _find_paths(image_dir)
    train_paths = [p for p in paths if not _use_for_val(p)]
    val_paths = [p for p in paths if _use_for_val(p)]
    if not train_paths or not val_paths:
        raise RuntimeError('not enough data')
    return (images_dataset(train_paths, size, bigger_size=bigger_size),
            images_dataset(val_paths, size, bigger_size=bigger_size))


def dir_dataset(image_dir, size, bigger_size=None):
    """Create a Dataset of images from a directory."""
    return images_dataset(_find_paths(image_dir), size, bigger_size=bigger_size)


def images_dataset(paths, size, bigger_size=None):
    """
    Create a Dataset of images from image file paths.

    Args:
      paths: a sequence of image paths.
      size: the size of the resulting images.
      bigger_size: if not None, the images are scaled to
        this size before being randomly cropped to size.
    """
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)

    def _read_image(path_tensor):
        data_tensor = tf.read_file(path_tensor)
        image_tensor = tf.image.decode_image(data_tensor, channels=3)
        image_tensor.set_shape((None, None, 3))
        if bigger_size is None:
            return tf.cast(tf.image.resize_images(image_tensor, [size, size]), tf.float32) / 0xff
        big = tf.image.resize_images(image_tensor, [bigger_size, bigger_size])
        small = tf.random_crop(big, [size, size, 3])
        return tf.cast(small, tf.float32) / 0xff
    return paths_ds.shuffle(buffer_size=len(paths)).map(_read_image)


def _use_for_val(path):
    return md5(bytes(path, 'utf-8')).digest()[0] < 0x80


def _find_paths(image_dir):
    if not os.path.isdir(image_dir):
        if '*' in image_dir:
            return glob.glob(image_dir)
        else:
            raise RuntimeError('image directory not found: ' + image_dir)
    paths = []
    for name in os.listdir(image_dir):
        if not name.startswith('.'):
            paths.append(os.path.join(image_dir, name))
    return paths
