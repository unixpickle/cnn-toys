"""Train a CycleGAN model."""

import argparse

def main(args):
    """The main training loop."""
    # TODO: this.

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir-1', help='first data directory', default='data_1')
    parser.add_argument('--data-dir-2', help='second data directory', default='data_2')
    parser.add_argument('--size', help='image size', type=int, default=128)
    parser.add_argument('--step-size', help='training step size', type=float, default=2e-4)
    parser.add_argument('--state-dir', help='state output directory', default='state')
    return parser.parse_args()

if __name__ == '__main__':
    main(_parse_args())
