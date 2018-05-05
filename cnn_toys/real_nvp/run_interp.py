"""Interpolate between samples with a real NVP model."""

import argparse

import tensorflow as tf

from cnn_toys.data import images_dataset
from cnn_toys.graphics import save_image_grid
from cnn_toys.real_nvp import interpolate_linear, simple_network
from cnn_toys.saving import restore_state

def main(args):
    """Interpolation entry-point."""
    print('loading images...')
    dataset = images_dataset([args.image_1, args.image_2], args.size)
    images = dataset.batch(2).make_one_shot_iterator().get_next()
    print('setting up model...')
    network = simple_network()
    with tf.variable_scope('model'):
        _, latents, _ = network.forward(images)
    latents = interpolate_linear(latents, args.rows * args.cols)
    with tf.variable_scope('model', reuse=True):
        images = network.inverse(None, latents)
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('generating images...')
        samples = sess.run(tf.reshape(images, [args.rows, args.cols, args.size, args.size, 3]),
                           feed_dict=network.test_feed_dict())
        save_image_grid(samples, args.out_file)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--rows', help='rows in output', type=int, default=4)
    parser.add_argument('--cols', help='columns in output', type=int, default=4)
    parser.add_argument('--state-file', help='state output file', default='state.pkl')
    parser.add_argument('--out-file', help='image output file', default='interp.png')
    parser.add_argument('image_1', help='first input image')
    parser.add_argument('image_2', help='first input image')
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
