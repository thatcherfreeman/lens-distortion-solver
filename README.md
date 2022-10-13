# Lens Distortion Solver

## Objective
Consume an image of a grid taken with a camera (with square pixels) and a spherical lens. Extract the edges and points from the image and for some reasonable lens distortion model, best fit some parameters to undistort the image. Generate the necessary ST Maps (at a good resolution) that can be used in other software to distort or undistort the image.

## Methodology
1. Ingest image captured of some sort of grid or checkerboard
2. Run Edge Detection
3. Separate edge detected pixels into Vertical and Horizontal edge groups (convolve with 3x3 matrices for this probably)
4. Cluster pixels within the vertical and horizontal groups using dbscan, consider these clusters to be their own lines
5. Choose some parameters for the lens distortion model, correct the pixel coordinates from part (4)
6. Measure curvature of clusters in the corrected image by selecting pixels within clusters and compute a radius of a circle that contains the three pixels (via sine law) and then take the reciprocal of this radius as a straightness heuristic (minimized at zero)
7. Continue to finess the lens distortion model parameters until the overall line error is decreased to zero. Probably should use some variation of binary search/hill climbing.
8. Generate STMap as EXR using the resulting lens distortion model.


## Files
`stmap.dctl`: Applies an ST Map to an image. Pipe the ST map into channels 2 and 3 and the luminance of your target image into channel 1, and it'll output the luminance channel with the ST map applied. Just for testing my code.