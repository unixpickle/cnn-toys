"""Train a real NVP model."""

import argparse
from itertools import count

import tensorflow as tf

from cnn_toys.real_nvp import log_likelihood, simple_network
from cnn_toys.data import dir_dataset
from cnn_toys.saving import save_state, restore_state


def main(args):
    """The main training loop."""
    print('loading dataset...')
    dataset = dir_dataset(args.data_dir, args.size)
    images = dataset.repeat().batch(args.batch).make_one_shot_iterator().get_next()
    images = images + tf.random_uniform(tf.shape(images), maxval=0.01)
    print('setting up model...')
    network = simple_network()
    with tf.variable_scope('model'):
        loss = -tf.reduce_mean(log_likelihood(network, images))
    optimize = tf.train.AdamOptimizer(learning_rate=args.step_size).minimize(loss)
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('training...')
        for i in count():
            cur_loss, _ = sess.run((loss, optimize))
            print('step %d: loss=%f' % (i, cur_loss))
            if i % args.save_interval == 0:
                save_state(sess, args.state_file)


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', help='data directory', default='data')
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--batch', help='batch size', type=int, default=32)
    parser.add_argument('--step-size', help='training step size', type=float, default=2e-4)
    parser.add_argument('--state-file', help='state output file', default='state.pkl')
    parser.add_argument('--save-interval', help='iterations per save', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    main(_parse_args())
