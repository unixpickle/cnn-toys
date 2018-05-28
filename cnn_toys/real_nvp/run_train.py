"""Train a real NVP model."""

import argparse
from itertools import count

import tensorflow as tf

from cnn_toys.data import dir_train_val
from cnn_toys.real_nvp import bits_per_pixel, bits_per_pixel_and_grad, simple_network
from cnn_toys.saving import save_state, restore_state


def main(args):
    """The main training loop."""
    print('loading dataset...')
    train_data, val_data = dir_train_val(args.data_dir, args.size)
    train_images = train_data.repeat().batch(args.batch).make_one_shot_iterator().get_next()
    val_images = val_data.repeat().batch(args.batch).make_one_shot_iterator().get_next()
    print('setting up model...')
    network = simple_network()
    with tf.variable_scope('model'):
        if args.low_mem:
            bpp, train_gradients = bits_per_pixel_and_grad(network, train_images)
            train_loss = tf.reduce_mean(bpp)
        else:
            train_loss = tf.reduce_mean(bits_per_pixel(network, train_images))
    with tf.variable_scope('model', reuse=True):
        val_loss = tf.reduce_mean(bits_per_pixel(network, val_images))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.step_size)
        if args.low_mem:
            optimize = optimizer.apply_gradients(train_gradients)
        else:
            optimize = optimizer.minimize(train_loss)
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('training...')
        for i in count():
            cur_loss, _ = sess.run((train_loss, optimize))
            if i % args.val_interval == 0:
                cur_val_loss = sess.run(val_loss, feed_dict=network.test_feed_dict())
                print('step %d: loss=%f val=%f' % (i, cur_loss, cur_val_loss))
            else:
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
    parser.add_argument('--val-interval', help='iterations per validation', type=int, default=10)
    parser.add_argument('--low-mem', help='use memory-efficient backprop', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(_parse_args())
