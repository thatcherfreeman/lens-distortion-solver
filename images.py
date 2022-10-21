import os
import cv2
import numpy as np


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pre-reverse first axis
    img = img[::-1, :, :]
    return img

def show_image(image: np.ndarray):
    m, n, c = image.shape
    image = image.astype(np.float32)
    if c == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image[::-1]
    cv2.imshow("image", image)
    cv2.waitKey(0)

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

def overlay(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    assert image1.shape == image2.shape, "expected same shape for overlay operation."
    new_image = image1.copy()
    new_image = new_image + image2
    return new_image
