from argparse import ArgumentParser
import numpy as np
import cv2

from models import (
    first_order_spherical,
)
from utilities import (
    open_image,
    write_image,
    make_distort_stmap_from_model,
    apply_stmap,
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
    args = parser.parse_args()

    img = open_image(args.filename)

    model = first_order_spherical(args.k)

    output_stmap = make_distort_stmap_from_model(model.forward, 1080, 1920)
    write_image("sample_images/output_stmap.exr", output_stmap)

    img_undistorted = apply_stmap(img, output_stmap, 1080, 1920)
    write_image("sample_images/undistorted.exr", img_undistorted)

    output_stmap = make_distort_stmap_from_model(model.reverse, 1080, 1920)
    write_image("sample_images/output_reverse_stmap.exr", output_stmap)

    img_redistorted = apply_stmap(img_undistorted, output_stmap, 1080, 1920)
    write_image("sample_images/redistorted.exr", img_redistorted)
