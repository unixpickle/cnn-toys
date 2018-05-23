"""Sample from a colorization model."""

import argparse

import numpy as np
import tensorflow as tf

from cnn_toys.colorize.model import colorize
from cnn_toys.data import dir_train_val
from cnn_toys.graphics import save_image_grid
from cnn_toys.saving import restore_state


def main(args):
    """Sample a batch of colorized images."""
    _, val_set = dir_train_val(args.data_dir, args.size)
    images = val_set.batch(args.batch).repeat().make_one_shot_iterator().get_next()
    grayscale = tf.reduce_mean(images, axis=-1, keep_dims=True)
    with tf.variable_scope('colorize'):
        colorized = colorize(grayscale)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_state(sess, args.state_file)
        rows = sess.run([images, tf.tile(grayscale, [1, 1, 1, 3]), colorized])
        save_image_grid(np.array(rows), 'images.png')


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', help='data directory', default='data')
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--batch', help='number of samples', type=int, default=16)
    parser.add_argument('--state-file', help='state input file', default='state.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    main(_parse_args())
