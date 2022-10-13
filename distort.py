from argparse import ArgumentParser
import numpy as np
import cv2

from models import (
    first_order_spherical,
)
from utilities import (
    write_image,
    make_distort_stmap_from_model,
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

    # img = open_image(args.filename)

    model = first_order_spherical(args.k)
    output_stmap = make_distort_stmap_from_model(model, 1080, 1920)
    write_image("sample_images/output_stmap.exr", output_stmap)
