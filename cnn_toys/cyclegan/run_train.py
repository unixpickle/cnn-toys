"""Train a CycleGAN model."""

import argparse
import os

from PIL import Image
import numpy as np
import tensorflow as tf

from cnn_toys.data import dir_dataset
from cnn_toys.saving import save_state, restore_state
from cnn_toys.cyclegan.model import CycleGAN

def main(args):
    """The main training loop."""
    print('loading datasets...')
    real_x = _load_dataset(args.data_dir_1, args.size)
    real_y = _load_dataset(args.data_dir_2, args.size)
    print('setting up model...')
    model = CycleGAN(real_x, real_y)
    global_step = tf.get_variable('global_step', dtype=tf.int64, shape=(),
                                  initializer=tf.zeros_initializer())
    optimize = model.optimize(
        learning_rate=_annealed_learning_rate(args.step_size, args.iters, global_step),
        global_step=global_step)
    with tf.Session() as sess:
        print('initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('attempting to restore model...')
        restore_state(sess, args.state_file)
        print('training...')
        while sess.run(global_step) < args.iters:
            terms = sess.run((optimize, model.disc_loss, model.gen_loss, model.cycle_loss))
            step = sess.run(global_step)
            print('step %d: disc=%f gen=%f cycle=%f' % ((step,) + terms[1:]))
            if step % args.sample_interval == 0:
                save_state(sess, args.state_file)
                print('saving samples...')
                _generate_samples(sess, args, model, step)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir-1', help='first data directory', default='data_1')
    parser.add_argument('--data-dir-2', help='second data directory', default='data_2')
    parser.add_argument('--size', help='image size', type=int, default=128)
    parser.add_argument('--step-size', help='training step size', type=float, default=2e-4)
    parser.add_argument('--state-file', help='state output file', default='state.pkl')
    parser.add_argument('--iters', help='number of training steps', type=int, default=100000)
    parser.add_argument('--sample-interval', help='iters per sample', type=int, default=1000)
    parser.add_argument('--sample-dir', help='directory to dump samples', default='samples')
    parser.add_argument('--sample-count', help='number of samples to draw', type=int, default=16)
    return parser.parse_args()

def _load_dataset(dir_path, size):
    return dir_dataset(dir_path, size).repeat().make_one_shot_iterator().get_next()

def _annealed_learning_rate(initial, iters, global_step):
    frac_done = tf.cast(iters - global_step, tf.float32) / float(iters)
    return tf.cond(frac_done < 0.5, lambda: initial, lambda: (1 - frac_done) * 2 * initial)

def _generate_samples(sess, args, model, step):
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)
    grid_img = np.zeros((args.size * args.sample_count, args.size * 4, 3), dtype='float32')
    for i in range(args.sample_count):
        images = sess.run((model.real_x, model.gen_y, model.real_y, model.gen_x))
        for j, image in enumerate(images):
            grid_img[i*args.size:(i+1)*args.size, j*args.size:(j+1)*args.size, :] = image
    img = Image.fromarray((grid_img * 0xff).astype('uint8'), 'RGB')
    img.save(os.path.join(args.sample_dir, 'samples_%d.png' % step))

if __name__ == '__main__':
    main(_parse_args())
