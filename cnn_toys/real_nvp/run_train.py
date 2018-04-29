"""Train a real NVP model."""

import argparse
from functools import partial
from itertools import count

import tensorflow as tf

import cnn_toys.real_nvp as nvp
from cnn_toys.data import dir_dataset
from cnn_toys.saving import save_state, restore_state

def main(args):
    """The main training loop."""
    print('loading dataset...')
    dataset = dir_dataset(args.data_dir, args.size)
    images = dataset.repeat().batch(args.batch).make_one_shot_iterator().get_next()
    print('setting up model...')
    main_layers = [
        nvp.MaskedConv(partial(nvp.checkerboard_mask, True), 3),
        nvp.MaskedConv(partial(nvp.checkerboard_mask, False), 3),
        nvp.MaskedConv(partial(nvp.checkerboard_mask, True), 3),
        nvp.Squeeze(),
        nvp.MaskedConv(partial(nvp.depth_mask, True), 3),
        nvp.MaskedConv(partial(nvp.depth_mask, False), 3),
        nvp.MaskedConv(partial(nvp.depth_mask, True), 3),
        nvp.FactorHalf()
    ]
    network = nvp.Network([nvp.PaddedLogit()] + (main_layers * 3))
    loss = tf.reduce_mean(nvp.log_likelihood(network, images))
    optimize = tf.train.AdamOptimizer(learning_rate=args.step_size).minimize(loss)
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('training...')
        for i in count():
            loss, _ = sess.run((loss, optimize))
            print('step %d: loss=%f' % (i, loss))
            if i % args.save_interval == 0:
                save_state(sess, args.state_file)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', help='data directory', default='data')
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--step-size', help='training step size', type=float, default=2e-4)
    parser.add_argument('--state-file', help='state output file', default='state.pkl')
    parser.add_argument('--save-interval', help='iterations per save', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
