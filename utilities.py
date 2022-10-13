import numpy as np
import cv2
import os


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(img.dtype)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_image(image_fn: str, img: np.ndarray):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    h,w,c = img.shape
    if c < 3:
        needed_channels = 3 - c
        img = np.concatenate([img, np.zeros((h, w, needed_channels))], axis=2).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_fn, img)


def default_stmap(height: int, width: int) -> np.ndarray:
    return np.stack(np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(1, 0, height),
    ), axis=2)


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
    assert b == 0

    if a == 0:
        return -d / c
    # Cardano's formula
    c /= a
    d /= a

    Q = c / 3
    R = -d / 2
    delta = Q**3 + R**2

    if delta > 0:
        # One root.
        C = (R + (delta**0.5))**(1/3)
        return C + (Q / C)
    else:
        S = ((R**2 - delta)**0.5)**(1/3)
        T = (np.arctan(((-delta)**0.5) / R))/3
        return -S * np.cos(T) + S * (3**0.5) * np.sin(T)



