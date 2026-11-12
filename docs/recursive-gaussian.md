# Recursive Gaussian (issue #176)

The CPU `effectcraft-raster::gaussian_blur` path uses the shared `craft-blur`
Young–van Vliet recurrence from 2.5 px and the shared calibrated sampled kernel
below that. The old three-box cascade quantized
the requested width and allowed relative variance error up to `1 / (3 sigma)`;
the shared recurrence calibrates its poles to match the requested second
moment.

`repeat = true` maps to clamp-edge continuation, while `repeat = false` maps to
zero extension without border renormalization, matching the old box code. The
four RGBA channels are filtered independently, as before. Tests in
`craft-blur` compare the impulse response with a normalized sampled Gaussian:
tested discrete variance error is below 0.5%; maximum absolute shape error is
0.023721 at sigma 2.5 and 0.011705 at sigma 4, and below 0.01 from sigma 8
onward. These impulse-shape bounds do not guarantee a bound for arbitrary
image content. The shared crate also enforces generated-sigma limits of 0.03
shape error on [2.5, 8), 0.01 on [8, 64], and variance error below 0.5% on
[2.5, 64]. See its README for the full accuracy and edge contracts.
