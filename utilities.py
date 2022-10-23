import numpy as np
from typing import Tuple, List, Union, Literal
from scipy.ndimage import convolve, minimum_filter, gaussian_laplace, gaussian_gradient_magnitude
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import images

def zero_out_edges(image: np.ndarray, border=0.05) -> np.ndarray:
    num_rows, num_cols = int(border * image.shape[0]), int(border * image.shape[1])
    image[:num_rows] *= 0
    image[-num_rows:] *= 0
    image[:, :num_cols] *= 0
    image[:, -num_cols:] *= 0
    return image

def get_coords_from_edges(edges: np.ndarray, t: float) -> List[Tuple[int]]:
    # Takes in an image where edges are detected (mostly black, some white lines)
    # then outputs (r, c) coordinates where the pixel in the image exceeds the
    # threshold.
    h, w, c = edges.shape
    assert c == 1, "Expected one channel."
    row_idx, col_idx = (edges.reshape((h, w)) > t).nonzero()
    return [(r, c) for r, c in zip(row_idx, col_idx)]


def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    m, n, c = kernel.shape
    mi, ni, ci = image.shape
    assert c == ci, "mismatched channels."
    assert m == n, "kernel not square."
    assert m % 2 == 1, "Kernel needs to have odd shape"
    origin = int(-(m // 2))
    new_image = convolve(
        image,
        kernel,
        mode='nearest',
        origin=(origin, origin, 0),
    )
    print(new_image.shape)
    return new_image

def get_edges(image: np.ndarray, direction: Union[Literal['vertical'], Literal['horizontal']]) -> np.ndarray:
    h, w, c = image.shape
    window_size = int((h * 0.02) // 2 * 2 + 1)
    if direction == 'vertical':
        sigma = (window_size, 0, 0)
        shape = (window_size, 1, 1)
    elif direction == 'horizontal':
        sigma = (0, window_size, 0)
        shape = (1, window_size, 1)
    else:
        raise NotImplementedError(f"Unexpected edge direction {direction}")
    new_image = 1 - gaussian_laplace(image, sigma=sigma, mode='nearest')
    # new_image = minimum_filter_2d(1 - image, shape)
    return new_image

def minimum_filter_2d(image: np.ndarray, shape: Tuple) -> np.ndarray:
    assert all([x % 2 == 1 for x in shape])
    origin = (int(-(x // 2)) for x in shape)
    new_image = minimum_filter(
        image,
        shape,
        mode='constant',
        cval=1.0,
        origin=origin,
    )
    return new_image


def get_threshold(image: np.ndarray, draw_images=False) -> float:
    # Identify a threshold which divides the bright and dark pixels in the image.
    h, w, c = image.shape
    assert c == 1
    image = image.reshape((h*w,))
    candidates = np.linspace(0.0, 1.0, 20)
    quantities = np.array([np.mean(image < c) for c in candidates])
    if draw_images:
        plt.plot(candidates, quantities)
        plt.show()
    gain = [0] + list(quantities[1:] - quantities[:-1])
    for i, (c, q, g) in enumerate(zip(candidates, quantities, gain)):
        if q > 0.95 and g < 0.01:
            return c


    return 0.5

def extract_key_points(edges: List[Tuple[int, int]], height: int, width: int, draw_images=False) -> List[List[Tuple[float, float]]]:
    # Extracts 3 points from each line for use with parabola fitting.
    # Identify which points are along the same line.
    if len(edges) == 0:
        return []
    clustering_model = DBSCAN(eps=4/1000 * height, min_samples=5, n_jobs=-1)
    npedges = np.array([list(x) for x in edges], dtype=int)
    labels = clustering_model.fit_predict(npedges)

    # Group dbscan output into clusters
    clusters = {label: [] for label in range(-1, np.max(labels) + 1)}
    for point, label in zip(npedges, labels):
        clusters[label].append(point)

    # Preprocess by removing unclustered pixels, small clusters
    cluster_sizes = np.array([len(cluster) for cluster in clusters.values()])
    t = 0.02 * np.sum(cluster_sizes)
    clusters = {k: v for k,v in clusters.items() if len(v) > t}
    if -1 in clusters:
        del clusters[-1]

    if draw_images:
        colors = {
            l: np.random.rand(3) for l in range(np.max(labels) + 1)
        }
        colors[-1] = np.ones(3)
        output_image = np.zeros((height, width, 3))
        for l, cluster in clusters.items():
            for r,c in cluster:
                output_image[r,c] = colors[l]
        images.show_image(output_image)

    output_points = []
    for cluster in clusters.values():
        # Make bounding box around points..
        coords = np.array(cluster)
        max_row = np.max(coords[:, 0])
        min_row = np.min(coords[:, 0])
        max_col = np.max(coords[:, 1])
        min_col = np.min(coords[:, 1])

        # Figure out if the box is tall or wide
        if abs(max_row - min_row) > abs(max_col - min_col):
            # Tall
            top = np.mean(coords[coords[:, 0] == min_row], axis=0)
            bottom = np.mean(coords[coords[:, 0] == max_row], axis=0)
            mid_row = int((min_row + max_row) / 2)
            mid = np.mean(coords[coords[:, 0] == mid_row], axis=0)
            if len(cluster) > 100:
                output_points.append([tuple(top), tuple(bottom), tuple(mid)])
        else:
            left = np.mean(coords[coords[:, 1] == min_col], axis=0)
            right = np.mean(coords[coords[:, 1] == max_col], axis=0)
            mid_col = int((min_col + max_col) / 2)
            mid = np.mean(coords[coords[:, 1] == mid_col], axis=0)
            if len(cluster) > 100:
                output_points.append([tuple(left), tuple(right), tuple(mid)])

    if draw_images:
        output_image = np.zeros((height, width, 3))
        for three_points in output_points:
            color = np.random.rand(3)
            for r,c in three_points:
                output_image[int(r)-2:int(r)+2, int(c)-2:int(c)+2] = color
        images.show_image(output_image)

    return output_points

def convert_uv_to_xy(u, v, aspect) -> Tuple[float, float]:
    # Assumes u,v have (0,0) in the top left corner. Returns x,y where (0,0) is the center.
    x,y = u - 0.5, v-0.5
    x *= aspect
    return x,y

def convert_xy_to_rc(x, y, height, width) -> Tuple[int, int]:
    aspect = width / height
    return int((y + 0.5) * (height - 1)), int((x/aspect + 0.5) * (width - 1))
