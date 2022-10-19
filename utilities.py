import numpy as np
import cv2
import os
from tqdm import tqdm
from typing import Tuple, List


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pre-reverse first axis
    img = img[::-1, :, :]
    return img

def write_image(image_fn: str, img: np.ndarray):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    h,w,c = img.shape
    if c < 3:
        needed_channels = 3 - c
        img = np.concatenate([img, np.zeros((h, w, needed_channels))], axis=2)
    # Reverse first axis.
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_fn, img)


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


def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    m, n, c = kernel.shape
    mi, ni, ci = image.shape
    assert c == ci, "mismatched channels."
    assert m == n, "kernel not square."
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    new_image = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image


def cubic_solver(a=0, b=0, c=0, d=0):
    # Finds roots of polynomial ax^3 + 0x^2 + cx + d = 0
    assert b == 0, "cubic solver only works with b=0"
    if a == 0:
        return -d / c
    a, b, c, d = float(a), float(b), float(c), float(d)

    # Cardano's formula
    c /= a
    d /= a
    Q = c / 3
    R = -d / 2
    delta = Q**3 + R**2
    if delta > 0:
        # One root.
        C = (R + (delta**0.5))**(1/3)
        return C - (Q / C)
    else:
        S: complex = (R + (delta**0.5))**(1/3)
        T: complex = (R - (delta**0.5))**(1/3)
        # out1: complex = S + T
        # out2: complex = -(S + T) / 2 + (S - T) * 1j * (3**0.5) / 2
        out3: complex = -(S + T) / 2 - (S - T) * 1j * (3**0.5) / 2
        return out3.real

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


def fit_parabola_horizontal_line(points: List[Tuple[float]]) -> Tuple[float, float, float]:
    # Returns A, B, C for which:
    # y = Ax**2 + Bx + C
    x1, x2, x3 = points[0][0], points[1][0], points[2][0]
    y1, y2, y3 = points[0][1], points[1][1], points[2][1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    return (A, B, C)

def fit_parabola_vertical_line(points: List[Tuple[float]]) -> Tuple[float, float, float]:
    # Returns A, B, C for which:
    # x = Ay**2 + By + C
    inverted_points = [(y, x) for x,y in points]
    return fit_parabola_horizontal_line(inverted_points)

def convert_uv_to_xy(u, v, aspect) -> Tuple[float, float]:
    # Assumes u,v have (0,0) in the top left corner. Returns x,y where (0,0) is the center.
    x,y = u - 0.5, v-0.5
    x *= aspect
    return x,y
