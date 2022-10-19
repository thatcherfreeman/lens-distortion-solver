from argparse import ArgumentParser
import numpy as np
import cv2
from typing import List, Tuple, Dict

from models import (
    first_order_spherical,
    parabola,
)
from utilities import (
    open_image,
    write_image,
    make_distort_stmap_from_model,
    apply_stmap,
    fit_parabola_horizontal_line,
    fit_parabola_vertical_line,
    convert_uv_to_xy,
)
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        help="Path to image of a correction chart.",
    )
    parser.add_argument(
        "-k",
        default=0.0,
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
    args = parser.parse_args()

    img = open_image(args.filename)
    img_height, img_width, channels = img.shape
    if args.stmap_width is not None and args.stmap_height is not None:
        stmap_height, stmap_width = args.stmap_height, args.stmap_width
    else:
        stmap_height, stmap_width = img_height, img_width
    print(f"Generating STMaps of resolution {stmap_height} by {stmap_width}")

    aspect = img_width / img_height

    # model = first_order_spherical(k=args.k, aspect_ratio=img_width/img_height)

    # print("Making undistortion stmap...")
    # output_stmap = make_distort_stmap_from_model(model.forward, 1080, 1920)
    # write_image("sample_images/output_stmap.exr", output_stmap)

    # print("Making undistorted image...")
    # img_undistorted = apply_stmap(img, output_stmap, 1080, 1920)
    # write_image("sample_images/undistorted.exr", img_undistorted)

    # print("Making distortion stmap...")
    # output_stmap = make_distort_stmap_from_model(model.reverse, 1080, 1920)
    # write_image("sample_images/output_reverse_stmap.exr", output_stmap)

    # print("Making redistorted image...")
    # img_redistorted = apply_stmap(img_undistorted, output_stmap, 1080, 1920)
    # write_image("sample_images/redistorted.exr", img_redistorted)

    dataset: Dict[str, List[List[Tuple[float]]]] = {
        "horizontal": [
            [(996, 725), (197, 691), (1694, 694)],
            [(208, 841), (910, 902), (1414, 878)],
        ],
        "vertical": [
            [(1272, 1012), (1299, 566), (1279, 122)],
            [(1448, 990), (1481, 540), (1454, 135)],
        ],
    }
    dataset = {k: [map(lambda point: (point[0]/img_width, point[1]/img_height), line) for line in v] for k, v in dataset.items()}

    # convert UV coordinates to X,Y coordinates, centered in the middle of the image.
    dataset["horizontal"] = [list(map(lambda tup: convert_uv_to_xy(tup[0], tup[1], aspect), points)) for points in dataset["horizontal"]]
    dataset["vertical"] = [list(map(lambda tup: convert_uv_to_xy(tup[0], tup[1], aspect), points)) for points in dataset["vertical"]]

    # Fit parabolas
    parameters = {}
    parameters['horizontal']: List[Tuple[float, float, float]] = list(map(fit_parabola_horizontal_line, dataset['horizontal']))
    parameters['vertical']: List[Tuple[float, float, float]] = list(map(fit_parabola_vertical_line, dataset['vertical']))
    parabolas = [parabola(A,B,C) for A,B,C in parameters['horizontal']] + [parabola(A,B,C) for A,B,C in parameters['vertical']]

    # Estimate k
    k = np.average(
        [p.estimate_k() for p in parabolas],
        weights=[p.c for p in parabolas],
    )
    print(f"Using distortion paramter k = {k}")

    model = first_order_spherical(k, aspect)
    output_stmap = make_distort_stmap_from_model(model.reverse, stmap_height, stmap_width)
    write_image("sample_images/output_stmap.exr", output_stmap)
    img_undistorted = apply_stmap(img, output_stmap, stmap_height, stmap_width)
    write_image("sample_images/undistorted.exr", img_undistorted)
    redistort_stmap = make_distort_stmap_from_model(model.forward, stmap_height, stmap_width)
    write_image("sample_images/output_reverse_stmap.exr", redistort_stmap)


