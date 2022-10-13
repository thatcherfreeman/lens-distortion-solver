from argparse import ArgumentParser
import numpy as np
import cv2

from models import (
    first_order_spherical,
)
from utilities import (
    open_image,
    default_stmap,
    write_image,
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

    initial_stmap = default_stmap(1080, 1920)
    output_stmap = np.zeros_like(initial_stmap)
    rows, columns, chans = initial_stmap.shape
    for x in tqdm(range(rows)):
        for y in range(columns):
            xc, yc = model.forward(initial_stmap[x,y,0] - 0.5, initial_stmap[x,y,1] - 0.5)
            output_stmap[x, y, :] = [xc + 0.5, yc + 0.5]

    write_image("sample_images/output_stmap.exr", output_stmap)
