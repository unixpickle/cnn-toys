"""
Tools for creating graphics from CNN models.
"""

from PIL import Image
import numpy as np


def save_image_grid(grid, out_file, padding=10):
    """
    Save a grid of RGB images to a file.

    Args:
      grid: a 5-dimensional np.array of images. The shape
        is [rows x cols x img_height x img_width x 3].
        Pixel values should range from 0 to 1.
      out_file: the path where the file should be saved.
      padding: pixels of space to put around each image.
    """
    grid = np.clip(grid, 0, 1)
    num_rows = grid.shape[0]
    num_cols = grid.shape[1]
    img_height = grid.shape[2]
    img_width = grid.shape[3]
    grid_img = np.zeros((num_rows * img_height + (num_rows + 1) * padding,
                         num_cols * img_width + (num_cols + 1) * padding, 3), dtype='float32')
    # White background for the border.
    grid_img += 1
    for row, row_imgs in enumerate(grid):
        for col, img in enumerate(row_imgs):
            row_start = row * img_height + (row + 1) * padding
            col_start = col * img_width + (col + 1) * padding
            grid_img[row_start: row_start + img_height,
                     col_start: col_start + img_width] = img
    img = Image.fromarray((grid_img * 0xff).astype('uint8'), 'RGB')
    img.save(out_file)
