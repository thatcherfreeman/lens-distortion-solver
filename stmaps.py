
import numpy as np
from tqdm import tqdm

def bilinear_sample(image: np.ndarray, x: float, y: float) -> np.ndarray:
    # Sample where (0, 0) is the top left corner of the image.
    height, width, channels = image.shape
    f_x = x * (width - 1)
    f_y = y * (height - 1)

    x_low = int(f_x)
    x_high = int(f_x + 1)
    y_low = int(f_y)
    y_high = int(f_y + 1)
    x_high = min(max(x_high, 0), width - 1)
    y_high = min(max(y_high, 0), height - 1)
    x_low = min(max(x_low, 0), width - 1)
    y_low = min(max(y_low, 0), height - 1)

    c_ll = image[y_low, x_low, :]
    c_lh = image[y_high, x_low, :]
    c_hl = image[y_low, x_high, :]
    c_hh = image[y_high, x_high, :]

    mix_x = f_x - x_low
    mix_y = f_y - y_low
    c_l = c_ll + (c_hl - c_ll) * mix_x
    c_h = c_lh + (c_hh - c_lh) * mix_x
    c = c_l + (c_h - c_l) * mix_y
    return c

def apply_stmap(image: np.ndarray, stmap: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    channels = image.shape[2]
    output = np.zeros((output_height, output_width, channels))
    for row in tqdm(range(output_height)):
        for col in range(output_width):
            x = col / (output_width - 1)
            y = row / (output_height - 1)

            s, t = bilinear_sample(stmap, x, y)
            color = bilinear_sample(image, s, t)
            output[row, col] = color
    return output


def default_stmap(height: int, width: int) -> np.ndarray:
    return np.stack(np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    ), axis=2)


def make_distort_stmap_from_model(fn, height: int, width: int) -> np.ndarray:
    initial_stmap = default_stmap(height, width)
    output_stmap = np.zeros_like(initial_stmap)
    rows, columns, chans = initial_stmap.shape
    for r in tqdm(range(rows)):
        for c in range(columns):
            xc, yc = fn(initial_stmap[r, c, 0] - 0.5, initial_stmap[r, c, 1] - 0.5)
            output_stmap[r, c, :] = [xc + 0.5, yc + 0.5]
    return output_stmap