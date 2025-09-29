# Image Utils

[![Run tests](https://github.com/alexanderswerdlow/image_utils/actions/workflows/ci.yml/badge.svg)](https://github.com/alexanderswerdlow/image_utils/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://image-utils.readthedocs.io/)
![Supported python versions](https://raw.githubusercontent.com/alexanderswerdlow/image_utils/master/docs/python_badge.svg)


Are you tired of having to constantly switch between NumPy arrays, PyTorch Tensors, and PIL images? Simply wrap your NumPy array, PyTorch Tensor, or PIL image with `Im()` and let it handle conversions between shapes, formats, etc.

We can replace this:
```
numpy_img = (torch.rand(10, 3, 256, 256).permute(0, 2, 3, 1).detach().float().cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(numpy_img[0]).save('output.png')
```

With this:
```
Im(torch.rand(10, 3, 256, 256)).save()
```

The powerful part is not that this works for a specific input shape/dtype/range, but [almost] any combination.

## Features

- Supports NumPy arrays, PyTorch Tensors, and PIL Images
- _All_ operations support batching over arbitrary shapes `[..., H, W, C] or [..., C, H, W]` which are preserved through (most) transformations.
- Handles all common data types `[Float, Integer, Boolean]`, and Ranges `[0, 1], [0, 255]` with automatic detection (e.g., no `.permute()` is required).
- Vertical/Horizontal concatenation of images of varying shapes with automatic padding, device conversion, etc.
- Many common transformations unified across formats (resize / crop / square / scale / normalize / denormalize).
- Video support (encodes a batched sequence to `[mp4, gif]`).
- Minimal dependencies (primarily just NumPy & PIL).
- Writing text on images, and much more!

## Installation

_Warning_: The library is currently in alpha and the API is subject to change. If you use this library as part of another application, consider pinning to a specific commit. Moreover, you can copy the `src/image_utils/im.py` file as it works standalone!

To install from PyPI:

```
pip install image_utilities
```

## Usage

Below is an example of using the primary `Im` class:

```
from image_utils import Im

img = np.random.randint(0, 256, (2, 10, 256, 256, 3), np.uint8)
img = Im(img)
img = img.write_text("Hello World!")
img = img.scale(2) 
img = img.crop(200, 300, 0, 100)

img = Im.concat_horizontal(*img, spacing=15)
img.save()
img.save_video(fps=2) # We now have a 10 frame video!
```

This does the following:

- Initializes 20 (2 x 10) random images
- Writes "Hello World!" on all images
- Scales all images, preserving aspect ratio. (Use resize(), scale_to_width(), or scale_to_width() for more control.)
- Crops all images
- Takes [2, 10, ...] and horizontally concats along the first dim to get an output of [10, ...] (where each image has ~2x the height). This is an example of the unpacking operator (on the Im object!), but it also supports regular array slicing notation (e.g., Im()[:1])
- Saves the batch of [10, ...] to an image (viewed as a grid by default). Uses a timestamp for the name and PNG format by default.
- Saves the batch of [10, ...] as a video. Uses a timestamp for the name and MP4 format by default.

For additional usage examples, see the [test suite](tests/test_im_utils.py).

## When you should use image_utils

- If you need to quickly visualize and work with images/videos.
- If your code frequently switches between NumPy, PyTorch, and PIL.

## When you shouldn't use image_utils

- You shouldn't use `image_utils` in a pre-processing pipeline. There is no guarantee that a given operation [e.g., resize] will have exact consistency between versions.

- If you are concerned with performance, you also shouldn't use `image_utils` as it focuses on flexibility over performance. We focus on unified operations which sometimes requires internal conversions which could otherwise be avoided.

## Extra Goodies

Another handy feature is provided by `library_ops`. This overrides the `__repr__` for NumPy arrays and PyTorch Tensors. For example:

```console
>>> import torch
>>> torch.randn(5, 5)
tensor([[-0.5524,  1.2306,  1.3209,  0.0336, -0.2458],
        [ 0.0448, -0.5564,  1.7019,  1.3689, -2.7115],
        [ 0.3842, -0.9593, -1.3799,  0.8625, -0.4071],
        [ 1.1263,  0.8479, -0.0585,  0.2687, -1.1983],
        [-0.5371, -0.5553, -0.7780, -0.8373,  0.2803]])
>>> from image_utils import library_ops
>>> torch.randn(5, 5)
[5,5] torch.float32 cpu finite
elems: 25, avg: 0.306, min: -1.399, max: 2.467
tensor([[ 0.782,  1.755,  0.975,  2.467, -0.646],
        [ 0.899,  2.344, -1.178, -0.291, -1.399],
        [ 0.676,  1.095,  0.289,  0.104, -0.294],
        [-0.152,  1.120, -0.844,  0.698,  0.647],
        [ 0.158, -0.048,  0.338, -0.838, -1.008]])
[5,5] torch.float32 cpu finite
```

Instead of only seeing the array contents, we can now view the shape, dtype, device, and more. `finite` or `infinite` signifies whether the array contains any `NaN` of `Inf` values.

If you want a dedicated library for this, check out [lovely-tensors](https://github.com/xl0/lovely-tensors)!

## Tests

*Note*: If you want to know more about how to use specific methods or which formats we test on, check out `tests/test_im_utils.py`.

- To run all tests, simply run: `pytest`
- To break with pdb on error, use: `pytest --pdb -s`
- To run a specific test use: `pytest -k 'test_concat' --pdb -s`

## Development
### Local Installation

To install locally in a self-contained environment with [UV](https://docs.astral.sh/uv/):

```
git clone https://github.com/alexanderswerdlow/image_utils.git
uv sync --extra video --extra dev --extra cpu
```

### Build Docs

```
uv pip install myst_parser sphinx sphinx-rtd-theme
cd docs
make html
```