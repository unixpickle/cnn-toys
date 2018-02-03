"""Train a colorization model."""

import argparse
import itertools

import tensorflow as tf

from cnn_toys.data import dir_train_val
from cnn_toys.saving import save_state, restore_state
from cnn_toys.colorize.model import sample_loss

def main(args):
    """Training outer loop."""
    train, val = [d.batch(args.batch).repeat().make_one_shot_iterator().get_next()
                  for d in dir_train_val(args.data_dir, args.size)]
    with tf.variable_scope('colorize'):
        train_loss = sample_loss(train)
    with tf.variable_scope('colorize', reuse=True):
        val_loss = sample_loss(val)
    with tf.control_dependencies([train_loss, val_loss]):
        optimize = tf.train.AdamOptimizer(learning_rate=args.step_size).minimize(train_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_state(sess, args.state_file)
        for i in itertools.count():
            losses, _ = sess.run([(train_loss, val_loss), optimize])
            print('step %d: train=%f val=%f' % ((i,) + losses))
            if i % args.save_interval == 0:
                save_state(sess, args.state_file)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', help='data directory', default='data')
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--batch', help='batch size', type=int, default=16)
    parser.add_argument('--step-size', help='training step size', type=float, default=1e-3)
    parser.add_argument('--state-file', help='state output file', default='state.pkl')
    parser.add_argument('--save-interval', help='steps per save', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
