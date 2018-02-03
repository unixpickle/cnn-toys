"""
Tools for creating graphics from CNN models.
"""

from PIL import Image
import numpy as np

def save_image_grid(grid, out_file):
    """
    Save a grid of RGB images to a file.

    Args:
      grid: a 5-dimensional np.array of images. The shape
        is [rows x cols x img_height x img_width x 3].
        Pixel values should range from 0 to 1.
      out_file: the path where the file should be saved.
    """
    num_rows = grid.shape[0]
    num_cols = grid.shape[1]
    img_height = grid.shape[2]
    img_width = grid.shape[3]
    grid_img = np.zeros((num_rows * img_height, num_cols * img_width, 3), dtype='float32')
    for row, row_imgs in enumerate(grid):
        for col, img in enumerate(row_imgs):
            grid_img[row * img_height : (row + 1) * img_height,
                     col * img_width : (col + 1) * img_width] = img
    img = Image.fromarray((grid_img * 0xff).astype('uint8'), 'RGB')
    img.save(out_file)
