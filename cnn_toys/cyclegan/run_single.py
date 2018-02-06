"""
Run a trained CycleGAN on a single image.

Runs both generators on the image and produces a grid
containing both outputs.
"""

import argparse

import numpy as np
import tensorflow as tf

from cnn_toys.cyclegan.model import CycleGAN
from cnn_toys.data import images_dataset
from cnn_toys.graphics import save_image_grid
from cnn_toys.saving import restore_state

def main(args):
    """Load and use a model."""
    print('loading input image...')
    dataset = images_dataset([args.in_file], args.size, bigger_size=args.bigger_size)
    image = dataset.repeat().make_one_shot_iterator().get_next()
    print('setting up model...')
    model = CycleGAN(image, image)
    tf.get_variable('global_step', dtype=tf.int64, shape=(), initializer=tf.zeros_initializer())
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('running model...')
        row = sess.run([model.gen_x, model.gen_y])
        save_image_grid(np.array([row]), args.out_file)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', help='image size', type=int, default=256)
    parser.add_argument('--bigger-size', help='size to crop from', type=int, default=286)
    parser.add_argument('--state-file', help='state input file', default='state.pkl')
    parser.add_argument('--iters', help='number of training steps', type=int, default=100000)
    parser.add_argument('in_file', help='path to input file')
    parser.add_argument('out_file', help='path to output file')
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
