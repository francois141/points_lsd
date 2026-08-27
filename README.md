# points-lsd

Point-seeded LSD (Line Segment Detector) bindings used by
[UPAL](https://github.com/francois141/upal) (*Unified and Efficient Point-Line Local Features*, ECCV 2026).

This is a fork of [pytlsd](https://github.com/iago-suarez/pytlsd) by Iago Suárez, itself a
pybind11 wrapper around the original C implementation of LSD by Rafael Grompone von Gioi.
On top of the upstream bindings it adds `lsd_from_points`, which grows LSD regions only from a
set of seed pixels (e.g. learned keypoints) instead of scanning the full image, and accepts
externally computed gradients so that a learned line field can steer the detector.

![](https://raw.githubusercontent.com/francois141/points_lsd/main/resources/example.jpg)

## Install

Prebuilt wheels are published for Linux (x86_64, aarch64), macOS (arm64, x86_64) and
Windows (AMD64), CPython 3.10–3.14:

```bash
pip install points-lsd
```

Building from source needs a C++17 compiler and CMake ≥ 3.15 (no other native
dependencies):

```bash
git clone --recursive https://github.com/francois141/points_lsd.git
pip install ./points_lsd
```

## Usage

`lsd_from_points` needs the image, integer `(x, y)` seed pixels, and gradient norm/angle maps.
The maps are float64 arrays of the image's shape with undefined pixels set to `-1024.0`; the
snippet below computes them the way LSD does (2x2 finite differences), but any gradient — e.g.
one derived from a learned line distance field — can be supplied.

```python
import numpy as np
import points_lsd

gray = ...  # H x W float64 (or uint8) image
seeds = np.array([[x0, y0], [x1, y1]], dtype=np.int32)  # (x, y) pixel seeds inside the image


def lsd_gradients(img, not_defined=-1024.0, threshold=5.2262518595055063):
    norm = np.full(img.shape, not_defined)
    angle = np.full(img.shape, not_defined)
    a, b, c, d = img[:-1, :-1], img[:-1, 1:], img[1:, :-1], img[1:, 1:]
    gx, gy = b + d - a - c, c + d - a - b
    norm[:-1, :-1] = 0.5 * np.hypot(gx, gy)
    angle[:-1, :-1] = np.arctan2(gx, -gy)
    angle[norm <= threshold] = not_defined
    return norm, angle


gradnorm, gradangle = lsd_gradients(gray.astype(np.float64))

# N x 5 float32 array: [x1, y1, x2, y2, p] per segment, where p is LSD's angle precision.
segments = points_lsd.lsd_from_points(gray, seeds, 1.0, 0.6, 0.0, gradnorm, gradangle)

# The upstream full-image detector is still available (gradients optional there).
segments = points_lsd.lsd(gray)
```

Seeds outside the image, seed arrays that are not `N x 2`, or missing gradient maps raise
`ValueError` / `TypeError`. Region growing starts only from seeds whose gradient angle is
defined, so seeds should lie on (or one pixel before) an intensity edge.

## Development

```bash
pip install -e ".[test]"
pytest tests            # test_smoke.py is numpy-only; test_lsd.py needs the [test] extra
```

Wheels are built with [cibuildwheel](https://cibuildwheel.pypa.io) in
`.github/workflows/wheels.yml` and published to PyPI on `v*` tags. To build them locally:

```bash
pipx run cibuildwheel --platform macos   # or linux (needs Docker)
```

## License

The binding code and the `lsd_from_points` extension retain the MIT license of upstream
`pytlsd` (see [LICENSE](LICENSE)), while pybind11 is BSD-3-Clause. The bundled core detector
`src/lsd.cpp` is © Rafael Grompone von Gioi and licensed under the **GNU Affero General
Public License v3 or later** (see
[LICENSES/AGPL-3.0-or-later.txt](LICENSES/AGPL-3.0-or-later.txt)); it is linked statically,
so the distributed wheels as a whole are subject to the AGPL terms.
