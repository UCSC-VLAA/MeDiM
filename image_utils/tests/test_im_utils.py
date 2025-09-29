from typing import Iterable, Union
from image_utils import Im, strip_unsafe
from PIL import Image
import torch
import numpy as np
import pytest
from pathlib import Path
from einops import rearrange, repeat

img_path = Path("tests/high_res.png")
save_path = Path(__file__).parent / "output"


def get_img(
    img_type: Union[np.ndarray, Image.Image, torch.Tensor],
    hwc_order=True,
    dtype=None,
    normalize=False,
    device=None,
    bw_img=False,
    batch_shape=None,
    contiguous: bool = True,
):
    if bw_img:
        if dtype is None:
            img = Image.fromarray(np.random.rand(128, 128) > 0.5)
        else:
            img = Image.fromarray(np.random.randint(256, size=(128, 128)).astype(dtype))
    else:
        img = Image.open(img_path)

    if img_type == Image.Image:
        return img

    img = np.array(img)
    if img_type == torch.Tensor:
        img = torch.from_numpy(img)

    if not contiguous:
        if img_type == torch.Tensor:
            img = torch.randn((np.prod(img.shape),)).view(*img.shape)
        else:
            img = np.random.randn(np.prod(img.shape)).reshape(*img.shape)

    if not hwc_order:
        img = rearrange(img, "h w c -> c h w")

    if dtype is not None:
        img = img / 255.0
        if img_type == torch.Tensor:
            img = img.to(dtype=dtype)
        elif img_type == np.ndarray:
            img = img.astype(dtype)

    if normalize:
        pass

    if device is not None and img_type == torch.Tensor:
        img = img.to(device=device)

    if batch_shape is not None:
        if len(img.shape) == 2:
            img = img[None]
        img = repeat(img, f'... -> {" ".join(sorted(list(batch_shape)))} ...', **batch_shape)

    return img


@pytest.mark.parametrize("dim_size", [4, 10, 100])
def test_single_arg_even(dim_size):
    dims = (dim_size, dim_size)
    rand_float_tensor = torch.FloatTensor(*dims).uniform_()
    rand_bool_tensor = torch.FloatTensor(*dims).uniform_() > 0.5
    rant_int_tensor = torch.randint(0, 100, dims)

    rand_float_array = np.random.rand(*dims)
    rand_bool_array = np.random.rand(*dims) > 0.5
    rand_int_array = np.random.randint(100, size=dims)

    print(rand_float_tensor)
    print(rand_bool_tensor)
    print(rant_int_tensor)

    print(rand_float_array)
    print(rand_bool_array)
    print(rand_int_array)


valid_configs = [
    {"img_type": Image.Image},
    {"img_type": np.ndarray},
    {"img_type": np.ndarray, "contiguous": False},
    {
        "img_type": np.ndarray,
        "hwc_order": False,
    },
    {
        "img_type": np.ndarray,
        "dtype": np.float16,
    },
    {"img_type": np.ndarray, "hwc_order": False, "dtype": np.float16},
    {"img_type": np.ndarray, "hwc_order": False, "dtype": np.float32, "normalize": True},
    {"img_type": torch.Tensor},
    {"img_type": np.ndarray, "contiguous": False},
    {
        "img_type": torch.Tensor,
        "hwc_order": False,
    },
    {
        "img_type": torch.Tensor,
        "dtype": torch.float32,
    },
    {"img_type": np.ndarray, "hwc_order": False, "batch_shape": {"a": 2, "b": 3, "c": 4}},
    {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.float16},
    {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.bfloat16},
    {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.float, "normalize": True},
    {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.float16, "normalize": True},
    {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.bfloat16, "normalize": True},
    {"img_type": np.ndarray, "bw_img": True},
    {"img_type": np.ndarray, "bw_img": True, "dtype": np.uint8},
    {"img_type": np.ndarray, "batch_shape": {"a": 2}},
    {"img_type": np.ndarray, "batch_shape": {"a": 2, "b": 3, "c": 4}},
    {"img_type": np.ndarray, "batch_shape": {"a": 2, "b": 3}},
]


def get_file_path(img_params: dict, name: str):
    file_path = save_path / strip_unsafe("__".join([f"{k}_{v}" for k, v in img_params.items()]))
    return file_path.parent / f"{file_path.name}_{name}"


@pytest.mark.parametrize("img_params", valid_configs)
def test_save(img_params):
    img = Im(get_img(**img_params))
    img.copy.save(get_file_path(img_params, "save"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_grid(img_params):
    img = Im(get_img(**img_params))
    img.grid().save(get_file_path(img_params, "save"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_write_text(img_params):
    img = Im(get_img(**img_params))
    img.copy.write_text("test").save(get_file_path(img_params, "text"))
    img.copy.write_text("This is a typing test.", color=(8, 128, 82), size=1.0, thickness=2.0).save(get_file_path(img_params, "text_scaled"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_add_border(img_params):
    img = Im(get_img(**img_params))
    img.copy.add_border(border=50, color=(128, 128, 128)).save(get_file_path(img_params, "border"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_resize(img_params):
    img = Im(get_img(**img_params))
    img.copy.resize(128, 128).save(get_file_path(img_params, "resize"))
    img.copy.scale(0.25).save(get_file_path(img_params, "downscale"))
    img.copy.scale_to_width(128).save(get_file_path(img_params, "scale_width"))
    img.copy.scale_to_height(128).save(get_file_path(img_params, "scale_height"))
    img.copy.scale(0.5).scale_to_width(128).resize(512, 1024).scale_to_width(512).save(get_file_path(img_params, "multiple_resize"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_normalization(img_params):
    img = Im(get_img(**img_params))
    if img_params.get("bw_img", False):
        return
    img.normalize().denormalize().save(get_file_path(img_params, "normalize0"))
    img.denormalize().normalize().save(get_file_path(img_params, "normalize1"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_format(img_params):
    img = Im(get_img(**img_params))
    pil_img = img.pil
    torch_img = img.torch
    np_img = img.np
    cv_img = img.opencv
    torch_img_ = Im(img).torch
    torch.allclose(torch_img, torch_img_)


@pytest.mark.parametrize("img_params", valid_configs)
def test_pickle(img_params):
    import pickle

    img = Im(get_img(**img_params))
    pil_img = img.pil
    pil_img = pil_img[0] if isinstance(pil_img, Iterable) else pil_img
    pil_img = Im(pil_img)
    torch_img = Im(img.torch)
    np_img = Im(img.np)
    cv_img = Im(img.opencv)
    for i in [img, pil_img, torch_img, np_img, cv_img]:
        with open(get_file_path(img_params, "pickle"), "wb") as f:
            pickle.dump(i, f)


@pytest.mark.parametrize("img_params", valid_configs)
def test_concat(img_params):
    img = Im(get_img(**img_params))

    input_data = [img, img, img]
    # Test the standard way
    Im.concat_horizontal(*input_data, spacing=15).save(get_file_path(img_params, "concat_horizontal_spacing"))
    Im.concat_vertical(*input_data, spacing=0)

    # Test inputting a list directly
    Im.concat_horizontal(input_data, spacing=50, fill=(255, 255, 0)).save(get_file_path(img_params, "concat_horizontal_spacing_fill_color"))
    Im.concat_vertical(input_data, spacing=0)

    # Test different underlying types
    Im.concat_vertical(*[img, Im(img.np), Im(img.torch)], spacing=0)

    # Test inputting raw arrays
    Im.concat_horizontal(*[x.arr for x in input_data], spacing=5)
    Im.concat_vertical(*[x.arr for x in input_data], spacing=5)

    # Test unequal sizes in both the direction of concat and not
    h_ = img.np.shape[-3]
    Im.concat_vertical(*[img.np, img.np[..., : h_ // 2, :, :], img.np[..., h_ // 2 :, :, :]], spacing=5)
    Im.concat_vertical(*[img.np, img.np[..., :, : h_ // 2], img.np[..., :, h_ // 2 :, :]], spacing=5)
    Im.concat_vertical(*[img.np, img.np[..., : h_ // 2, : h_ // 2]], spacing=5)
    Im.concat_vertical(*[img.np, img.np[..., : h_ // 2]], spacing=5)

    Im.concat_horizontal(*[img.np, img.np[..., : h_ // 2, :, :], img.np[..., h_ // 2 :, :, :]], spacing=5)
    Im.concat_horizontal(*[img.np, img.np[..., : h_ // 2, : h_ // 2, :], img.np[..., :, h_ // 2 :, :]], spacing=5)
    Im.concat_horizontal(*[img.np, img.np[..., :, : h_ // 2, :]], spacing=5)


@pytest.mark.parametrize("hw", [(16, 16), (64, 64)])
def test_concat_broadcast(hw):
    good_cases = [
        (torch.randn(*hw, 3), torch.randn(*hw, 3)),
        (torch.randn(*hw, 3), torch.randn(1, hw[0], hw[1] * 2, 3)),
        (torch.randn(1, *hw, 3), torch.randn(hw[0], hw[1] * 2, 3)),
        (torch.randn(*hw, 3), torch.randn(hw[0], hw[1] * 2, 3)),
        (torch.randn(1, *hw, 3), torch.randn(1, *hw, 3)),
        (torch.randn(*hw, 3), torch.randn(1, *hw, 3)),
        (torch.randn(1, *hw, 3), torch.randn(*hw, 3)),
        (torch.randn(5, *hw, 3), torch.randn(*hw, 3)),
        (torch.randn(5, *hw, 3), torch.randn(1, *hw, 3)),
        (torch.randn(5, 3, 2, *hw, 3), torch.randn(*hw, 3)),
        (torch.randn(5, 3, 2, *hw, 3), torch.randn(2, *hw, 3)),
        (torch.randn(5, 3, 2, *hw, 3), torch.randn(3, 2, *hw, 3)),
        (torch.randn(2, *hw, 3), torch.randn(3, 2, *hw, 3)),
    ]

    for j in range(2):
        for i, (im1, im2) in enumerate(good_cases):
            if j == 1:
                im1, im2 = im1.numpy(), im2.numpy()
            Im.concat_horizontal(im1, im2)

    error_cases = [
        (torch.randn(4, *hw, 3), torch.randn(3, *hw, 3)),
        (torch.randn(2, *hw, 3), torch.randn(4, *hw, 3)),
    ]

    for j in range(2):
        for i, (im1, im2) in enumerate(error_cases, start=len(good_cases) + 1):
            try:
                if j == 1:
                    im1, im2 = im1.numpy(), im2.numpy()
                Im.concat_horizontal(im1, im2)
            except ValueError as e:
                pass


@pytest.mark.parametrize("img_params", valid_configs[:4])
@pytest.mark.parametrize("format", ["mp4", "gif", "webm"])
@pytest.mark.parametrize("frames", [1, 2, 4, 16])
@pytest.mark.parametrize("fps", [2, 16])
def test_encode_video(img_params, format, frames, fps):
    img_params["batch_shape"] = {"a": frames}
    if img_params["img_type"] == Image.Image:
        return
    img = Im(get_img(**img_params))
    img.encode_video(fps=fps, format=format)
    img.save_video(get_file_path(img_params, "video"), fps=fps, format=format)


@pytest.mark.parametrize("img_params", valid_configs)
def test_complicated(img_params):
    img = get_img(**img_params)
    orig_shape = None
    if isinstance(img, (torch.Tensor, np.ndarray)):
        orig_shape = img.shape
    img = Im(img)
    img = img.scale(0.5).resize(128, 128).crop(10, 120, 10, 120).write_text("Hello world!")
    img = img.add_border(border=50, color=(128, 128, 128)).normalize(mean=(0.5, 0.75, 0.5), std=(0.1, 0.01, 0.01))
    img = img.concat_horizontal(img, img, spacing=15).concat_vertical(img, img, spacing=15)
    img = img.torch
    img = Im(img).denormalize(mean=(0.5, 0.75, 0.5), std=(0.1, 0.01, 0.01))
    img = img.colorize()
    img.save(get_file_path(img_params, "complicated"))
    if orig_shape is not None:
        assert img.torch.shape[:-3] == orig_shape[:-3]


@pytest.mark.parametrize(
    "img_params",
    [
        {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.float16, "batch_shape": {"a": 2}},
        {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.bfloat16, "batch_shape": {"a": 2, "b": 3, "c": 4}},
        {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.float, "normalize": True, "batch_shape": {"a": 2, "b": 3}},
        {"img_type": np.ndarray, "batch_shape": {"a": 2}},
        {"img_type": np.ndarray, "batch_shape": {"a": 2, "b": 3, "c": 4}},
        {"img_type": np.ndarray, "batch_shape": {"a": 2, "b": 3}},
    ],
)
def test_slicing(img_params):
    img = get_img(**img_params)
    img = Im(img)
    h_ = img.np.shape[-3]
    if img_params.get("hwc_order", True):
        assert np.allclose(img.np[..., : h_ // 2, :10, :], img[..., : h_ // 2, :10, :].np)
        assert torch.allclose(img.torch[..., :, : h_ // 2, :10], img[..., : h_ // 2, :10, :].torch)
    else:
        assert np.allclose(img.np[..., :, : h_ // 2, :10, :], img[..., :, : h_ // 2, :10].np)
        assert torch.allclose(img.torch[..., :, : h_ // 2, :10], img[..., :, : h_ // 2, :10].torch)


@pytest.mark.parametrize("img_params", valid_configs)
def test_crop(img_params):
    img = Im(get_img(**img_params))
    img = img.crop(0, 128, 0, 128)
    img.save(get_file_path(img_params, "complicated"))


@pytest.mark.parametrize("hw", [(16, 16), (64, 64)])
def test_single_channel(hw):
    Im(torch.randn(hw)).save(get_file_path({"img_type": np.ndarray}, "single_channel"))
    Im(torch.rand(hw)).save(get_file_path({"img_type": np.ndarray}, "single_channel"))
    Im(torch.randn(hw) * 1000).save(get_file_path({"img_type": np.ndarray}, "single_channel"))


@pytest.mark.parametrize(
    "img_params",
    [
        {"img_type": Image.Image},
    ],
)
def test_pil_list(img_params):
    img = Im([get_img(**img_params) for _ in range(5)])
    assert img.batch_size == 5
    img.copy.save(get_file_path(img_params, "save"))


@pytest.mark.parametrize("img_params", valid_configs)
def test_square(img_params):
    """Test the square method for both NumPy and PyTorch images."""
    img = Im(get_img(**img_params))
    channels = img.channels
    target_size = 64

    squared = img.square(target_size)
    assert squared.width == target_size, f"Expected width {target_size}, got {squared.width}"
    assert squared.height == target_size, f"Expected height {target_size}, got {squared.height}"
    if not (img_params["img_type"] == np.ndarray and img.channels == 1):  # TODO: Fix this
        assert squared.channels == channels, f"Expected {channels} channels, got {squared.channels}"

        np_img = squared.get_np()
        expected_shape = (target_size, target_size, channels)
        assert np_img.shape[-3:] == expected_shape, f"Expected shape {expected_shape}, got {np_img.shape}"

        torch_img = squared.get_torch()
        expected_shape = (channels, target_size, target_size)
        assert torch_img.shape[-3:] == expected_shape, f"Expected tensor shape {expected_shape}, got {torch_img.shape}"


# @pytest.mark.parametrize("hw", [(16, 16), (64, 64)])
# def test_complicated_concat(hw):
#     Im.concat_horizontal(
#         torch.randn(hw),
#         torch.randn(1, *hw),
#         torch.randn(1, *hw, 3),
#         torch.randn(hw) > 0.5,
#         torch.randn(1, *hw) > 0.5,
#         np.random.randn(*hw),
#         np.random.randn(1, *hw),
#         np.random.rand(1, *hw, 3),
#         np.random.randint(256, size=(*hw, 3)).astype(np.uint8),
#         np.random.randint(256, size=(1, *hw, 3)).astype(np.uint8),
#         np.random.rand(*hw) > 0.5,
#         np.random.rand(1, *hw) > 0.5,
#         np.random.rand(1, *hw, 1) > 0.5,
#     ).save(get_file_path({"img_type": np.ndarray}, "complicated_concat"))
