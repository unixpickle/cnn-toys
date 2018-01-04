"""Sample from a colorization model."""

import argparse

from PIL import Image
import numpy as np
import tensorflow as tf

from data import dir_datasets
from model import colorize, restore_state

def main(args):
    """Sample a batch of colorized images."""
    _, val_set = dir_datasets(args.data_dir, args.size)
    images = val_set.batch(args.batch).repeat().make_one_shot_iterator().get_next()
    grayscale = tf.reduce_mean(images, axis=-1, keep_dims=True)
    with tf.variable_scope('colorize'):
        colorized = colorize(grayscale)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_state(sess, args.state_file)
        images, grayscale, colorized = sess.run((images, grayscale, colorized))
        res = np.zeros((args.size * args.batch, args.size * 3, 3), dtype='float32')
        for i in range(args.batch):
            res[i*args.size:(i+1)*args.size, 0:args.size, :] = images[i]
            res[i*args.size:(i+1)*args.size, args.size:2*args.size, :] = grayscale[i]
            res[i*args.size:(i+1)*args.size, 2*args.size:3*args.size, :] = colorized[i]
        img = Image.fromarray((res * 0xff).astype('uint8'), 'RGB')
        img.save("images.png")

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', help='data directory', default='data')
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--batch', help='number of samples', type=int, default=16)
    parser.add_argument('--state-file', help='state input file', default='state.pkl')
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
