"""Sample a real NVP model."""

import argparse

import tensorflow as tf

from cnn_toys.real_nvp import simple_network
from cnn_toys.graphics import save_image_grid
from cnn_toys.saving import restore_state

def main(args):
    """Sampling entry-point."""
    if args.seed:
        tf.set_random_seed(args.seed)
    print('setting up model...')
    network = simple_network()
    with tf.variable_scope('model'):
        fake_batch = tf.zeros((args.rows * args.cols, args.size, args.size, 3), dtype=tf.float32)
        _, latents, _ = network.forward(fake_batch)
    with tf.variable_scope('model', reuse=True):
        gauss_latents = [tf.random_normal(latent.shape, seed=args.seed) for latent in latents]
        images = network.inverse(None, gauss_latents)
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('generating samples...')
        samples = sess.run(tf.reshape(images, [args.rows, args.cols, args.size, args.size, 3]),
                           feed_dict=network.test_feed_dict())
        save_image_grid(samples, args.out_file)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--rows', help='rows in output', type=int, default=4)
    parser.add_argument('--cols', help='columns in output', type=int, default=4)
    parser.add_argument('--state-file', help='state output file', default='state.pkl')
    parser.add_argument('--out-file', help='image output file', default='samples.png')
    parser.add_argument('--seed', help='seed for outputs', type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
