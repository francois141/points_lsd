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
snippet below computes them the way LSD does (2x2 finite differences), but any gradient, such
as one derived from a learned line distance field, can be supplied.

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

## Wheels and publishing

### Supported wheels

Built with [cibuildwheel](https://cibuildwheel.pypa.io) (config in `pyproject.toml`
under `[tool.cibuildwheel]`) for CPython 3.10, 3.11, 3.12, 3.13 and 3.14:

| Platform | Architectures | Wheel tag |
|---|---|---|
| Linux (glibc ≥ 2.24) | x86_64, aarch64 | `manylinux_2_24_*.manylinux_2_28_*` |
| macOS | arm64 (≥ 11.0), x86_64 (≥ 10.9–10.15, per Python build) | `macosx_*` |
| Windows | AMD64 | `win_amd64` |

That is 25 wheels plus an sdist per release. Not built: musllinux (Alpine), PyPy,
free-threaded CPython and 32-bit platforms. On a compatible platform, pip can build from
the sdist with a C++17 compiler and CMake ≥ 3.15. The extension has no required external
native libraries. OpenMP support and the OpenCV-dependent C++ tests are opt-in and disabled
for wheels.

### How a release works

`.github/workflows/wheels.yml` runs on every push to `main`, every pull request, every
`v*` tag, and on manual dispatch. Every run builds all wheels and the sdist, smoke-tests
the wheels in clean environments where the runner can execute them (macOS x86_64 is
skipped on the arm64 runner), and runs the full test suite on Linux, macOS and Windows.
What happens with the artifacts depends on the trigger:

| Trigger | Publishes to |
|---|---|
| push to `main` / pull request | nothing (build + test only) |
| manual run (*Actions → Wheels → Run workflow*) with **Upload to TestPyPI** ticked | [TestPyPI](https://test.pypi.org/project/points-lsd/) |
| push of a tag `vX.Y.Z` | [PyPI](https://pypi.org/project/points-lsd/) |

Uploads use [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
so no API tokens are stored. Before the first upload to an index, configure that index to
trust this repository's `wheels.yml` workflow and its corresponding `pypi` or `testpypi`
GitHub environment.

**The version comes from `pyproject.toml`, not from git.** The tag only *triggers* the
upload, and the `check_version` job refuses a tag that does not match
`project.version` (tag `v0.1.0` ↔ `version = "0.1.0"`). A version can be uploaded to an
index more than once only with new distribution filenames. Existing files cannot be
overwritten, so replacing published artifacts requires a new version number.

Release checklist:

```bash
# 1. update project.version in pyproject.toml and land it on main
git add pyproject.toml
git commit -m "release: 0.2.0"
git push origin main                                 # CI must be green

# 2. optional dry run: Actions > Wheels > Run workflow > "Upload to TestPyPI", then
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "points-lsd==0.2.0"

# 3. publish
git tag v0.2.0
git push origin v0.2.0                               # publish_pypi job uploads to PyPI
```

### Building wheels locally

```bash
pipx run cibuildwheel --platform linux    # needs Docker; native arch, or set CIBW_ARCHS_LINUX
pipx run cibuildwheel --platform macos    # needs the python.org CPython installers
CIBW_BUILD="cp313-*" pipx run cibuildwheel --platform linux   # one interpreter only
```

cibuildwheel's Windows build must run on Windows (or in CI). A plain `pip wheel .` builds
a single wheel for the current interpreter without cibuildwheel's portability repair and
compatibility checks.

## License

The binding code and the `lsd_from_points` extension retain the MIT license of upstream
`pytlsd` (see [LICENSE](LICENSE)), while pybind11 is BSD-3-Clause. The bundled core detector
`src/lsd.cpp` is © Rafael Grompone von Gioi and licensed under the **GNU Affero General
Public License v3 or later** (see
[LICENSES/AGPL-3.0-or-later.txt](LICENSES/AGPL-3.0-or-later.txt)); it is linked statically,
so the distributed wheels as a whole are subject to the AGPL terms.
