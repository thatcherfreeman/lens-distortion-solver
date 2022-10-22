from argparse import ArgumentParser
import sys
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from models import (
    first_order_spherical,
    parabola,
)
import images
import solvers
import stmaps
import utilities


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        help="Path to image of a correction chart.",
    )
    parser.add_argument(
        "-k",
        default=None,
        type=float,
        help="distortion parameter",
    )
    parser.add_argument(
        "--stmap-width",
        default=None,
        type=int,
        help="Width of the generated ST Maps"
    )
    parser.add_argument(
        "--stmap-height",
        default=None,
        type=int,
        help="Height of the generated ST Maps"
    )
    parser.add_argument(
        "--draw-images",
        action='store_true',
        help="Draw images for debugging."
    )
    args = parser.parse_args()

    img = images.open_image(args.filename)
    img_height, img_width, channels = img.shape
    if args.stmap_width is not None and args.stmap_height is not None:
        stmap_height, stmap_width = args.stmap_height, args.stmap_width
    else:
        stmap_height, stmap_width = img_height, img_width
    aspect = img_width / img_height
    print(f"Generating STMaps of resolution {stmap_height} by {stmap_width}")

    print("Running horizontal and vertical edge detection...")
    img_gray = np.mean(img, axis=2, keepdims=True)
    max_val, min_val = np.max(img_gray), np.min(img_gray)
    img_gray = (img_gray - min_val) / (max_val - min_val)
    window_size = int((img_height * 0.01) // 2 * 2 + 1)
    print(f"Running minimum filter with length {window_size}")
    vertical_edges = utilities.zero_out_edges(utilities.minimum_filter_2d((1 - img_gray), (window_size, 1, 1)))
    horizontal_edges = utilities.zero_out_edges(utilities.minimum_filter_2d((1 - img_gray), (1, window_size, 1)))
    if args.draw_images:
        images.show_image(img_gray)
        images.show_image(vertical_edges)
        images.show_image(horizontal_edges)

    print("Identifying appropriate threshold for edge detection...")
    t = utilities.get_threshold(vertical_edges)
    print(f"Selected edge detection threshold of {t}")

    vertical_edge_coords = utilities.get_coords_from_edges(vertical_edges, t)
    horizontal_edge_coords = utilities.get_coords_from_edges(horizontal_edges, t)
    print(f"Found {len(vertical_edge_coords) + len(horizontal_edge_coords)} edge pixels.")

    print("Clustering edges...")
    dataset = {}
    dataset['vertical'] = utilities.extract_key_points(vertical_edge_coords, img_height, img_width, args.draw_images)
    dataset['horizontal'] = utilities.extract_key_points(horizontal_edge_coords, img_height, img_width, args.draw_images)

    dataset = {k: [list(map(lambda point: (point[1]/img_width, point[0]/img_height), line)) for line in v] for k, v in dataset.items()}

    # convert UV coordinates to X,Y coordinates, centered in the middle of the image.
    dataset["horizontal"] = [list(map(lambda tup: utilities.convert_uv_to_xy(tup[0], tup[1], aspect), points)) for points in dataset["horizontal"]]
    dataset["vertical"] = [list(map(lambda tup: utilities.convert_uv_to_xy(tup[0], tup[1], aspect), points)) for points in dataset["vertical"]]

    # Fit parabolas
    parameters = {}
    parameters['horizontal']: List[Tuple[float, float, float]] = list(map(solvers.fit_parabola_horizontal_line, dataset['horizontal']))
    parameters['vertical']: List[Tuple[float, float, float]] = list(map(solvers.fit_parabola_vertical_line, dataset['vertical']))
    parabolas = [parabola(A,B,C) for A,B,C in parameters['horizontal']] + [parabola(A,B,C) for A,B,C in parameters['vertical']]

    if args.draw_images:
        parabola_drawing = img.copy()
        # parabola_drawing = np.zeros_like(img)
        for i, p in enumerate(parabolas):
            color = np.random.rand(3)
            if i < len(parameters['horizontal']):
                for x in np.linspace(-0.5 * aspect, 0.5 * aspect, 1920):
                    y = p.forward(x)
                    r, c = utilities.convert_xy_to_rc(x, y, img_height, img_width)
                    if 0 <= r and r < img_height:
                        parabola_drawing[r, c] = color
            else:
                for y in np.linspace(-0.5, 0.5, 1080):
                    x = p.forward(y)
                    r, c = utilities.convert_xy_to_rc(x, y, img_height, img_width)
                    if 0 <= c and c < img_width:
                        parabola_drawing[r, c] = color
        images.show_image(parabola_drawing)

    # Estimate k
    k_estimates = [p.estimate_k() for p in parabolas]
    if args.draw_images:
        plt.scatter([abs(p.c) for p in parabolas], k_estimates)
        plt.show()
    # Remove outliers from k_estimates
    weights = np.array([abs(p.c) for p in parabolas])

    k = np.average(
        k_estimates,
        weights=weights,
    )
    if args.k is not None:
        k = args.k
    print(f"Using distortion paramter k = {k}")

    model = first_order_spherical(k, aspect)
    output_stmap = stmaps.make_distort_stmap_from_model(model.reverse, stmap_height, stmap_width)
    images.write_image("sample_images/output_stmap.exr", output_stmap)
    # img_undistorted = stmaps.apply_stmap(img, output_stmap, stmap_height, stmap_width)
    # images.write_image("sample_images/undistorted.exr", img_undistorted)
    redistort_stmap = stmaps.make_distort_stmap_from_model(model.forward, stmap_height, stmap_width)
    images.write_image("sample_images/output_reverse_stmap.exr", redistort_stmap)
    # img_redistorted = stmaps.apply_stmap(img_undistorted, redistort_stmap, stmap_height, stmap_width)
    # images.write_image("sample_images/redistorted.exr", img_redistorted)

