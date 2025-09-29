from __future__ import annotations

import copy
import importlib.util
import os
import string
import tempfile
import warnings
from enum import Enum
from functools import partial
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple, Type, Union

import numpy as np
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Integer
from numpy import ndarray
from PIL import Image

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch import Tensor, device  # type: ignore
else:

    class device:
        def __init__(self, type: str):
            self.type = type

    class Tensor:  # type: ignore
        def __init__(self):
            pass


if importlib.util.find_spec("image_utils") is not None:
    from image_utils.file_utils import get_date_time_str, load_cached_from_url
    from image_utils.standalone_image_utils import pca

if importlib.util.find_spec("imageio") is not None:
    from imageio import v3 as iio

if TYPE_CHECKING:
    from torch import Tensor

    ImArr = Union[ndarray, Tensor]  # The actual array itself
    ImArrType = Type[Union[ndarray, Tensor]]  # The object itself is just a type
    ImDtype = Union[torch.dtype, np.dtype]

colorize_weights = {}
enable_warnings = os.getenv("IMAGE_UTILS_DISABLE_WARNINGS") is None


class callable_staticmethod(staticmethod):
    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)


class staticproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


def warning_guard(message: str):
    if enable_warnings:
        warnings.warn(message, stacklevel=2)


def is_dtype(arr: ImArr, dtype: Union[Float, Integer, Bool]):
    return isinstance(arr, dtype[ndarray, "..."]) or isinstance(arr, dtype[Tensor, "..."])


def is_tensor(obj: ImArr):
    return isinstance(obj, Tensor)


def is_ndarray(obj: ImArr):
    return isinstance(obj, ndarray)


def is_pil(obj: ImArr):
    return isinstance(obj, Image.Image)


def is_arr(obj: ImArr):
    return is_ndarray(obj) | is_tensor(obj)


def dispatch_op(obj: ImArr, np_op, torch_op, *args):
    if is_ndarray(obj):
        return np_op(obj, *args)
    elif is_tensor(obj):
        return torch_op(obj, *args)
    else:
        raise ValueError(f"obj must be numpy array or torch tensor, not {type(obj)}")


class ChannelOrder(Enum):
    HWC = "HWC"
    CHW = "CHW"


class ChannelRange(Enum):
    UINT8 = "UINT8"
    FLOAT = "FLOAT"
    BOOL = "BOOL"


def identity(x):
    return x


class Im:
    """
    This class represents an image [or collection of batched images] and allows for simple conversion between formats
    (PIL/NumPy ndarray/PyTorch Tensor) and support for common image operations, regardless of input dtype, batching, normalization, etc.

    Note: Be careful when using this class directly as part of a training pipeline. Many operations will cause the underlying data to convert between formats (e.g., Tensor -> Pillow) and move the data back to system memory and/or incur loss of precision (e.g., float -> uint8). In addition, we do not guarantee bit-consistency over different versions as we may the internal representation or backend computation of a function. Some operations are in place operations even if they return an Im object.

    Specifically, we make the following design choices:

    - All inputs are internally represented by either a ndarray or tensor with shape (B, H, W, C) or (B, C, H, W)
    - Convinience is prioritized over efficiency. We make copies or perform in-place operation with little consistency. We may re-evaluate this design decision in the future.
    """

    # Common ImageNet normalization values
    default_normalize_mean = [0.4265, 0.4489, 0.4769]
    default_normalize_std = [0.2053, 0.2206, 0.2578]

    def __init__(
        self,
        arr: Union["Im", Tensor, Image.Image, list[Image.Image], tuple[Image.Image], ndarray, str, Path],
        channel_range: Optional[ChannelRange] = None,
        **kwargs,
    ):
        # TODO: Add real URL checking here
        if isinstance(arr, (str, Path)) and Path(arr).exists():
            arr = Im.open(arr)
        elif isinstance(arr, str):
            arr = Image.open(load_cached_from_url(arr))

        if isinstance(arr, Im):
            # We allow Im(im) and make it a no-op
            for attr in dir(arr):
                if not attr.startswith("__") and not isinstance(getattr(type(arr), attr, None), property):
                    setattr(self, attr, getattr(arr, attr))
            return

        self.device: device
        self.arr_type: ImArrType

        # To handle things in a unified manner, we choose to always convert PIL Images -> NumPy internally
        if isinstance(arr, Image.Image):
            arr = np.array(arr)
        elif (isinstance(arr, list) or isinstance(arr, tuple)) and all(isinstance(a, Image.Image) for a in arr):
            arr = np.stack([np.array(a) for a in arr])

        assert isinstance(arr, (ndarray, Tensor)), f"arr must be numpy array, pillow image, or torch tensor, not {type(arr)}"
        self.arr: ImArr = arr
        if isinstance(self.arr, ndarray):
            self.arr_type = ndarray
            self.device = device("cpu")
        elif isinstance(self.arr, Tensor):
            self.device = self.arr.device
            self.arr_type = Tensor
            if self.arr.requires_grad:
                warning_guard("Input tensor has requires_grad=True. We are detaching the tensor.")
                self.arr = self.arr.detach()
        else:
            raise ValueError("Must be numpy array, pillow image, or torch tensor")

        # TODO: Consider normalizing to HWC order for all arrays, similar to how arr_transform works
        # These views should be very efficient and make things more unified
        self.channel_order: ChannelOrder = ChannelOrder.HWC if self.arr.shape[-1] < min(self.arr.shape[-3:-1]) else ChannelOrder.CHW
        self.dtype: ImDtype = self.arr.dtype

        # We normalize all arrays to (B, H, W, C) and record their original shape so
        # we can re-transform then when we need to output them
        self.arr_transform: Callable[[ImArr], ImArr]
        if len(self.arr.shape) == 2:
            self.channel_order = ChannelOrder.HWC
            self.arr_transform = partial(rearrange, pattern="() h w () -> h w")
        elif len(self.arr.shape) == 3:
            self.arr_transform = partial(rearrange, pattern="() h w c -> h w c")
        elif len(self.arr.shape) == 4:
            self.arr_transform = partial(identity)
        elif len(self.arr.shape) >= 5:
            extra_dims = self.arr.shape[:-3]
            mapping = {k: v for k, v in zip(string.ascii_uppercase, extra_dims)}
            transform_str = f'({" ".join(sorted(list(mapping.keys())))}) a b c -> {" ".join(sorted(list(mapping.keys())))} a b c'
            self.arr_transform = partial(rearrange, pattern=transform_str, **mapping)  # type: ignore # lambda x: rearrange(x, transform_str, g)
        else:
            raise ValueError("Must be between 3-5 dims")

        if len(self.arr.shape) == 2:
            self.arr = rearrange(self.arr, "h w -> h w ()")

        self.arr = rearrange(self.arr, "... h w c -> (...) h w c")

        # We use some very simple hueristics to guess what kind of image we have
        if channel_range is not None:  # Optionally specify the type
            self.channel_range = channel_range
        elif is_dtype(arr, Integer):
            assert self.arr.max() >= 0, "Integer array must be non-negative"
            if self.channels > 1 or self.arr.max() > 1:
                self.channel_range = ChannelRange.UINT8
            else:  # We assume an integer array with 0s and 1s is a BW image
                self.channel_range = ChannelRange.BOOL
        elif is_dtype(arr, Float):
            self.channel_range = ChannelRange.FLOAT
        elif is_dtype(arr, Bool):
            self.channel_range = ChannelRange.BOOL
        else:
            raise ValueError("Invalid Type")

    def __getitem__(self, sl):
        # TODO: Decide on a consistent way to handle slicing. We should either always slice according to the original shape,
        # or always slice according to [..., C, H, W].
        return Im(self.arr_transform(self.arr)[sl])

    def __repr__(self):
        if self.arr_type == ndarray:
            arr_name = "ndarray"
        elif self.arr_type == Tensor:
            arr_name = "tensor"
        else:
            raise ValueError("Must be numpy array or torch tensor")

        if is_pil(self.arr):
            shape_str = repr(self.arr)
        else:
            shape_str = f"type: {arr_name}, shape: {self.arr_transform(self.arr).shape}"

        return f"Im of {shape_str}, device: {self.device}"

    def _convert(
        self, desired_datatype: ImArrType, desired_order: ChannelOrder = ChannelOrder.HWC, desired_range: ChannelRange = ChannelRange.UINT8
    ) -> Im:
        if self.arr_type != desired_datatype or self.channel_order != desired_order or self.channel_range != desired_range:
            # We preserve the original dtype, shape, and device
            orig_transform, orig_device, orig_dtype = self.arr_transform, self.device, self.arr.dtype

            if desired_datatype == ndarray:
                self = Im(self.get_np(order=desired_order, range=desired_range))
            elif desired_datatype == Tensor:
                self = Im(self.get_torch(order=desired_order, range=desired_range))

            self.device = orig_device
            self.arr_transform = orig_transform
            self.dtype = orig_dtype

        return self

    @callable_staticmethod
    def _convert_to_datatype(desired_datatype: ImArrType, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8):
        def custom_decorator(func):
            def wrapper(self: Im, *args, **kwargs):
                self = self._convert(desired_datatype, desired_order, desired_range)
                return func(self, *args, **kwargs)

            return wrapper

        return custom_decorator

    def _handle_order_transform(self, im: ImArr, desired_order: ChannelOrder, desired_range: ChannelRange, select_batch=None) -> ImArr:
        if select_batch is not None:
            im = im[select_batch]
        else:
            im = self.arr_transform(im)

        if self.channels == 1 and len(im.shape) == 2:
            im = rearrange(im, "h w -> h w ()")
            assert self.channel_order == ChannelOrder.HWC

        if desired_order == ChannelOrder.CHW and self.channel_order == ChannelOrder.HWC:
            im = rearrange(im, "... h w c -> ... c h w")
        elif desired_order == ChannelOrder.HWC and self.channel_order == ChannelOrder.CHW:
            im = rearrange(im, "... c h w -> ... h w c")

        start_cur_order = "h w ()" if desired_order == ChannelOrder.HWC else "() h w"
        end_cur_order = start_cur_order.replace("()", "c")

        if self.channel_range != desired_range:
            assert is_ndarray(im) or is_tensor(im)
            if self.channel_range == ChannelRange.FLOAT and desired_range == ChannelRange.UINT8:
                if self.channels == 1 and im.max() > im.min():
                    im = (im - im.min()) / (im.max() - im.min())
                im = im * 255
                if self.channels == 1:
                    im = repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3)
            elif self.channel_range == ChannelRange.UINT8 and desired_range == ChannelRange.FLOAT:
                im = im / 255.0
            elif self.channel_range == ChannelRange.BOOL:
                assert self.channels == 1
                if "pattern" in self.arr_transform.keywords:
                    self.arr_transform.keywords["pattern"] = "() ... -> ..."
                im = repeat(im, f"... {start_cur_order} -> ... {end_cur_order}", c=3)
                if desired_range == ChannelRange.UINT8:
                    im = im * 255
            else:
                raise ValueError("Not supported")

            if desired_range == ChannelRange.UINT8:
                im = im.astype(np.uint8) if isinstance(im, ndarray) else im.to(torch.uint8)
            elif desired_range == ChannelRange.FLOAT:
                im = im.astype(np.float32) if isinstance(im, ndarray) else im.to(torch.float32)

        return im

    def get_np(self, order=ChannelOrder.HWC, range=ChannelRange.UINT8) -> ndarray:
        """Converts the image to a NumPy Array with specified channel order and range."""
        arr = self.arr
        if is_tensor(arr):
            arr = torch_to_numpy(arr)  # type: ignore

        arr = self._handle_order_transform(arr, order, range)
        assert isinstance(arr, ndarray)
        return arr

    def get_torch(self, order=ChannelOrder.CHW, range=ChannelRange.FLOAT) -> Tensor:
        """Converts the image to a PyTorch Tensor with specified channel order and range."""
        arr = self.arr
        if is_ndarray(arr):
            arr = torch.from_numpy(arr)

        arr = self._handle_order_transform(arr, order, range)
        assert isinstance(arr, Tensor)
        if self.device is not None:
            arr = arr.to(self.device)
        return arr

    def get_pil(self) -> Union[Image.Image, list[Image.Image]]:
        """Converts the image to a PIL Image. Returns a list for batched images."""
        if self.batch_size == 1:
            img = self.get_np()
            if img.shape[-1] == 1:
                img = rearrange(img, "... () -> ...")

            img = rearrange(img, "... h w c -> (...) h w c").squeeze(0)
            return Image.fromarray(img)
        else:
            img = rearrange(self.get_np(), "... h w c -> (...) h w c")
            if img.shape[0] == 1:
                return Image.fromarray(img[0].squeeze(-1) if (self.channels == 1 and img[0].shape[-1] == 1) else img[0])
            else:
                return [Image.fromarray(img[i]) for i in range(img.shape[0])]

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def get_opencv(self):
        return self.arr

    @property
    def copy(self):
        """Returns a deep copy of the image."""
        return copy.deepcopy(self)

    @property
    def height(self):
        """Returns the height of the image."""
        return self.image_shape[0]

    @property
    def width(self):
        """Returns the width of the image."""
        return self.image_shape[1]

    @property
    def batch_size(self):
        return self.arr.shape[0]  # TODO: This is a bit hacky

    @property
    def channels(self):
        """Returns the number of channels in the image (e.g., 3 for RGB or 1 for BW)"""
        if self.channel_order == ChannelOrder.HWC:
            return self.arr.shape[-1]
        else:
            return self.arr.shape[-3]

    @property
    def range_max(self):
        """Returns the maximum value of the image range (e.g., 255 for UINT8 or 1.0 for FLOAT)"""
        assert self.channel_range in (ChannelRange.UINT8, ChannelRange.FLOAT)
        return 255 if self.channel_range == ChannelRange.UINT8 else 1.0

    @property
    def image_shape(self):
        """Returns the height and width of the image as a tuple (H, W)"""
        if self.channel_order == ChannelOrder.HWC:
            return (self.arr.shape[-3], self.arr.shape[-2])
        else:
            return (self.arr.shape[-2], self.arr.shape[-1])

    @callable_staticmethod
    def open(filepath: Path, use_imageio=False) -> Im:
        """Opens an image from disk and returns an Im object"""
        if use_imageio:
            img = iio.imread(filepath)
        else:
            img = Image.open(filepath)
        return Im(img)

    @callable_staticmethod
    def new(h: int = 256, w: int = 256, color=(255, 255, 255)):
        """Creates a new image with the specified height and width and color"""
        return Im(Image.new("RGB", (w, h), color))

    @callable_staticmethod
    def random(h: int = 256, w: int = 256, cache: bool = False) -> Im:
        """Creates a random image from unsplash or picsum"""
        try:
            return Im(Image.open(load_cached_from_url(f"https://unsplash.it/{w}/{h}?random", cache=cache)))
        except:
            return Im(Image.open(load_cached_from_url(f"https://picsum.photos/{w}/{h}?random", cache=cache)))

    @staticproperty
    def ex(self):
        """Returns an example image."""
        return self.new()

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def resize(self, height: int, width: int, resampling_mode: str = "bilinear"):
        """Resizes image to a new height/width using the specified resampling mode (default: bilinear)."""
        from torchvision.transforms.functional import InterpolationMode, resize

        assert isinstance(self.arr, torch.Tensor)
        arr = resize(self.arr, [height, width], InterpolationMode(resampling_mode), antialias=True)
        arr = self.arr_transform(arr)
        return Im(arr)

    def scale(self, scale: float, **kwargs) -> Im:
        """Scales the image by a factor, preserving the aspect ratio."""
        width, height = self.width, self.height
        return self.resize(int(height * scale), int(width * scale), **kwargs)

    def scale_to_width(self, new_width: int, **kwargs) -> Im:
        """Scales the image to desired width, preserving the aspect ratio."""
        width, height = self.width, self.height
        wpercent = new_width / float(width)
        hsize = int((float(height) * float(wpercent)))
        return self.resize(hsize, new_width, **kwargs)

    def scale_to_height(self, new_height: int, **kwargs) -> Im:
        """Scales the image to desired height, preserving the aspect ratio."""
        width, height = self.width, self.height
        hpercent = new_height / float(height)
        wsize = int((float(width) * float(hpercent)))
        return self.resize(new_height, wsize, **kwargs)

    def square(self, size: int) -> Im:
        """Returns a square image, resizing and padding while preserving aspect ratio"""
        if self.width == self.height:
            return self.resize(size, size)
        else:

            def add_padding(arr: ImArr, n: int, direction: str = "vertical") -> ImArr:
                assert direction in ("vertical", "horizontal")
                if isinstance(arr, np.ndarray):
                    pad_width = [(0, 0)] * arr.ndim
                    pad_width[-3 if direction == "vertical" else -2] = (n, n)
                    return np.pad(arr, pad_width, mode="constant")
                elif isinstance(arr, torch.Tensor):
                    if direction == "vertical":
                        pad_arr = torch.zeros(*arr.shape[:-3], n, arr.shape[-2], arr.shape[-1], dtype=arr.dtype, device=arr.device)
                        return torch.cat([pad_arr, arr, pad_arr], dim=-3)
                    elif direction == "horizontal":
                        pad_arr = torch.zeros(*arr.shape[:-3], arr.shape[-3], n, arr.shape[-1], dtype=arr.dtype, device=arr.device)
                        return torch.cat([pad_arr, arr, pad_arr], dim=-2)

            if is_ndarray(self.arr):
                self = Im(self.get_np(ChannelOrder.HWC, ChannelRange.FLOAT))
            elif is_tensor(self.arr):
                self = Im(self.get_torch(ChannelOrder.HWC, ChannelRange.FLOAT))

            if self.width > self.height:
                new_height = (self.width - self.height) // 2
                self = Im(add_padding(self.arr, new_height, "vertical"))
            else:
                new_width = (self.height - self.width) // 2
                self = Im(add_padding(self.arr, new_width, "horizontal"))

            return self.resize(size, size)

    @callable_staticmethod
    def _save_data(filepath: Path = Path(get_date_time_str()), filetype: str = "png") -> Path:
        filepath = Path(filepath)
        if filepath.suffix == "":
            filepath = filepath.with_suffix(f".{filetype}")

        if len(filepath.parents) == 1:
            save_directory = Path(os.environ.get("IMAGE_UTILS_OUTPUT_DIR", "output"))
            filepath = save_directory / filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)

        return filepath

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def grid(self, **kwargs) -> Im:
        """Converts a batched image to a single grid image"""
        from torchvision import utils

        img = utils.make_grid(self.arr, **kwargs)  # type: ignore
        return Im(img)

    def save(self, filepath: Optional[Path] = None, filetype: str = "png", optimize: bool = False, quality: Optional[float] = None, **kwargs) -> Path:
        """
        Saves the image to a file, optionally optimizing and compressing the image.
        By default, the image is saved to $CWD/outputs with a timestamp as the filename, and a PNG filetype.
        If the image is batched, the images will be saved as a grid.
        """
        if filepath is None:
            filepath = Path(get_date_time_str())

        filepath: Path = Im._save_data(filepath, filetype)

        if self.batch_size > 1:
            img = self.get_torch()
            self = self.grid(**kwargs)

        img = self.get_pil()
        assert isinstance(img, Image.Image)

        flags = {"optimize": True, "quality": quality if quality else 0.95} if optimize or quality else {}

        img.save(filepath, **flags)

        return filepath.resolve()

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def write_text(
        self,
        text: Union[str, list[str]],
        color: tuple[int, int, int] = (255, 0, 0),
        position: tuple[float, float] = (0.9725, 0.01),  # yx position in relative coordinates [0, 1]. Defaults to bottom left.
        size: float = 1.0,
        thickness: float = 1.0,
        font: int = 0,  # cv2.FONT_HERSHEY_SIMPLEX
        relative_font_scale: Optional[float] = None,
    ) -> Im:
        """Writes text to the image. If the image is batched, the user can specify a list of text strings, otherwise, all images will have the same text. Requires OpenCV."""
        try:
            import cv2
        except:
            raise ImportError("OpenCV is required to write text to an image.")

        FONT_SCALE = 3e-3 * size
        THICKNESS_SCALE = 2e-3 * thickness
        new_im = self.copy

        if relative_font_scale is not None:
            warning_guard("relative_font_scale is deprecated. Use font_scale instead.")
            FONT_SCALE = relative_font_scale

        for i in range(new_im.arr.shape[0]):
            text_to_write = text[i] if isinstance(text, list) else text
            assert isinstance(new_im.arr[i], ndarray)
            # We could convert to BGR and back but since we specify colors in RGB, we don't need to
            im = cv2.putText(
                img=np.ascontiguousarray(new_im.arr[i]),  #
                text=text_to_write,
                org=(int(position[1] * new_im.width), int(position[0] * new_im.height)),
                fontFace=font,
                fontScale=FONT_SCALE * min(new_im.height, new_im.width),
                color=color,
                thickness=ceil(min(new_im.height, new_im.width) * THICKNESS_SCALE),
                lineType=cv2.LINE_AA,
            )  # type: ignore
            new_im.arr[i] = im  # type: ignore

        return new_im

    def add_border(self, border: int, color: Tuple[int, int, int]):
        """Adds solid color border to all sides of an image"""
        imgs = self.pil
        if not isinstance(imgs, Iterable):
            imgs = [imgs]

        from PIL import ImageOps

        arr = np.stack([Im(ImageOps.expand(img, border=border, fill=color)).np for img in imgs], axis=0)
        arr = self.arr_transform(arr)
        return Im(arr)

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def crop(self, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0):
        """Crops the image. The image is sliced as [..., top:bottom, left:right]"""
        arr = self.arr[..., top:bottom, left:right]
        arr = self.arr_transform(arr)
        return Im(arr)

    def normalize_setup(self, mean=default_normalize_mean, std=default_normalize_std):
        def convert_instance_np(arr_1, arr_2):
            assert isinstance(self.dtype, np.dtype)
            return np.array(arr_2).astype(self.dtype)

        def convert_instance_torch(arr_1, arr_2):
            assert isinstance(self.dtype, torch.dtype)
            if self.dtype in (torch.float16, torch.bfloat16, torch.half):
                return Tensor(arr_2).to(dtype=torch.float, device=self.device)
            else:
                return Tensor(arr_2).to(dtype=self.dtype, device=self.device)

        if is_ndarray(self.arr):
            self = Im(self.get_np(ChannelOrder.HWC, ChannelRange.FLOAT))
        elif is_tensor(self.arr):
            self = Im(self.get_torch(ChannelOrder.HWC, ChannelRange.FLOAT))

        mean = dispatch_op(self.arr, convert_instance_np, convert_instance_torch, mean)
        std = dispatch_op(self.arr, convert_instance_np, convert_instance_torch, std)

        return self, mean, std

    def normalize(self, normalize_min_max: bool = False, **kwargs) -> Im:
        """Normalizes image using either the current min-max or given a mean & std."""
        if normalize_min_max:
            # TODO: Make this more general
            self = Im(self.get_np(ChannelOrder.HWC, ChannelRange.FLOAT))
            self.arr = self.arr / self.arr.max() - self.arr.min()
        else:
            self, mean, std = self.normalize_setup(**kwargs)
            self.arr = (self.arr - mean) / std
        return self

    def denormalize(self, clamp: tuple[float, float] = (0, 1.0), **kwargs) -> Im:
        """De-normalizes image, optionally clamping values to specified range."""
        self, mean, std = self.normalize_setup(**kwargs)
        self.arr = (self.arr * std) + mean
        if isinstance(self.arr, ndarray):
            self.arr = self.arr.clip(*clamp) if clamp else self.arr
        elif isinstance(self.arr, Tensor):
            self.arr = self.arr.clamp(*clamp) if clamp else self.arr
        return self

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def convert_opencv_color(self, color: int):
        """E.g.,cv2.COLOR_RGB2BGR"""
        assert isinstance(self.arr, ndarray)
        from cv2 import cvtColor

        self.arr = cvtColor(self.arr, color)

    @callable_staticmethod
    def concat_vertical(*args, **kwargs) -> Im:
        """Concatenates images vertically (i.e. stacked on top of each other)"""
        return concat_variable(concat_vertical_, *args, **kwargs)

    @callable_staticmethod
    def concat_horizontal(*args, **kwargs) -> Im:
        """Concatenates images horizontally (i.e. left to right)"""
        return concat_variable(concat_horizontal_, *args, **kwargs)

    def save_video(self, filepath: Optional[Path] = None, fps: int = 4, format="mp4", use_pyav: bool = False):
        """Saves a video to disk. If filepath is not specified, the video will be saved to $CWD/outputs with a timestamp as the filename."""
        if filepath is None:
            filepath = Path(get_date_time_str())

        filepath: Path = Im._save_data(filepath, format)

        if use_pyav:
            from image_utils.video_utils import write_video

            self = self._convert(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
            assert isinstance(self.arr, ndarray)
            write_video(self.arr, filepath, fps=fps)
        else:
            byte_stream = self.encode_video(fps, format)
            with open(filepath, "wb") as f:
                f.write(byte_stream.getvalue())

    @_convert_to_datatype(desired_datatype=ndarray, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def encode_video(self, fps: int, format="mp4") -> BytesIO:
        """Encodes a batched image to a video. Requires ImageIO."""
        assert len(self.arr.shape) == 4, "Video data must be 4D (time, height, width, channels)"
        byte_stream = BytesIO()

        # TODO: We shouldn't need to write -> read. An imageio/ffmpeg issue is causing this.
        with tempfile.NamedTemporaryFile(suffix=f".{format}") as ntp:
            try:
                import imageio
            except ImportError:
                raise ImportError("ImageIO is required to encode videos. Please install it with `pip install imageio[ffmpeg]`.")

            if format == "webm":
                writer = imageio.get_writer(ntp.name, format="webm", codec="libvpx-vp9", pixelformat="yuv420p", output_params=["-lossless", "1"], fps=fps)  # type: ignore
            elif format == "gif":
                writer = imageio.get_writer(ntp.name, format="GIF", mode="I", duration=(1000 * 1 / fps))  # type: ignore
            elif format == "mp4":
                writer = imageio.get_writer(ntp.name, quality=10, pixelformat="yuv420p", codec="libx264", fps=fps)
            else:
                raise NotImplementedError(f"Format {format} not implemented.")

            for frame in self.arr:
                writer.append_data(frame)

            writer.close()
            with open(ntp.name, "rb") as f:
                byte_stream.write(f.read())

        byte_stream.seek(0)
        return byte_stream

    def to(self, device: torch.device):
        """Move tensor to device. In-place operation."""
        if isinstance(self.arr, Tensor):
            self.arr = self.arr.to(device)
        else:
            assert device == torch.device("cpu")

        self.device = device
        return self

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.CHW, desired_range=ChannelRange.FLOAT)
    def colorize(self) -> Im:
        """Creates a colorized RGB image from e.g., a feature map with channels > 3."""
        if self.channels not in colorize_weights:
            colorize_weights[self.channels] = torch.randn(3, self.channels, 1, 1)

        assert isinstance(self.arr, Tensor)
        import torch.nn.functional as F

        arr = F.conv2d(self.arr, weight=colorize_weights[self.channels])
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = self.arr_transform(arr)
        return Im(arr)

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.FLOAT)
    def pca(self, **kwargs) -> Im:
        """
        Computes principal components over all "pixels" in the batched image array.
        You may optionally specify principal components.
        """
        assert isinstance(self.arr, Tensor)
        b, h, w, _ = self.arr.shape
        pca_arr: Tensor = rearrange(self.arr, "... c -> (...) c")
        output = pca(pca_arr, **kwargs)
        output: Tensor = rearrange(output, "(b h w) c -> b h w c", b=b, h=h, w=w)
        return Im(output)

    def show(self):
        """Displays the image in the default image viewer (e.g., in the terminal or in ipython)."""
        import subprocess

        method = None
        if subprocess.run(["which", "imgcat"], capture_output=True).returncode == 0:
            method = "iterm2-imgcat"
        elif subprocess.run(["which", "xdg-open"], capture_output=True).returncode == 0:
            method = "xdg-open"

        if method is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = self.save(Path(temp_dir))
                if method == "iterm2-imgcat":
                    print("\n" * 4)
                    print("\033[4F")
                    subprocess.check_call(["imgcat", filename])
                    print("\033[4B")
                else:
                    subprocess.check_call(["xdg-open", filename])
        else:
            filename = self.save()
            print(f"Failed to view image.Image saved to {filename}")

    @_convert_to_datatype(desired_datatype=Tensor, desired_order=ChannelOrder.HWC, desired_range=ChannelRange.UINT8)
    def bool_to_rgb(self) -> Im:
        """Converts a boolean array to a RGB image (B / W)."""
        return self

    pil = property(get_pil)
    np = property(get_np)
    torch = property(get_torch)
    opencv = property(get_opencv)


def concat_variable(concat_func: Callable[..., Im], *args: Im, **kwargs) -> Im:
    """Helper function to concatenate variable number of images using a specified concatenation type."""
    if len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], (Im, ndarray, Tensor)):
        args = args[0]  # We allow passing in a single list without prior unpacking

    output_img = None
    for img in args:
        if not isinstance(img, Im):
            img = Im(img)

        if img.channel_range == ChannelRange.BOOL:
            warning_guard("Concatenating boolean images. We are converting to NumPy.")
            img = Im(img.np)

        if output_img is None:
            output_img = img
        else:
            if output_img.arr_type != img.arr_type or output_img.channel_order != img.channel_order or output_img.channel_range != img.channel_range:
                img = img._convert(output_img.arr_type, output_img.channel_order, output_img.channel_range)

            if output_img.device != img.device:
                warning_guard("Concatenating images on different devices. We are moving both to CPU.")
                img = img.to(torch.device("cpu"))
                output_img = output_img.to(torch.device("cpu"))

            output_img = concat_func(output_img, img, **kwargs)

    assert isinstance(output_img, Im)
    return output_img


def torch_to_numpy(arr: Tensor):
    if arr.dtype == torch.bfloat16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()


def get_arr_hwc(im: Im):
    if im.channels == 1:
        im = im.bool_to_rgb()
    return im._handle_order_transform(im.arr, desired_order=ChannelOrder.HWC, desired_range=im.channel_range)


def new_like(arr, shape, fill: Optional[tuple[int]] = None) -> ImArr:
    if is_ndarray(arr):
        new_arr = np.zeros_like(arr, shape=shape)
    elif is_tensor(arr):
        new_arr = arr.new_zeros(shape)
    else:
        raise ValueError("Must be numpy array or torch tensor")

    if fill is not None:
        assert len(fill) == 3 and new_arr.shape[-1] == 3
        fill_ = fill if is_dtype(arr, Integer) else tuple(f / 255 for f in fill)  # type: ignore
        new_arr[..., 0] = fill_[0]
        new_arr[..., 1] = fill_[1]
        new_arr[..., 2] = fill_[2]

    return new_arr


def concat_along_dim(arr_1: ImArr, arr_2: ImArr, dim: int):
    if is_ndarray(arr_1) and is_ndarray(arr_2):
        assert isinstance(arr_1, np.ndarray)
        assert isinstance(arr_2, np.ndarray)
        return np.concatenate((arr_1, arr_2), axis=dim)
    elif is_tensor(arr_1) and is_tensor(arr_2):
        return torch.cat([arr_1, arr_2], dim=dim)  # type: ignore
    else:
        raise ValueError("Must be numpy array or torch tensor")


def broadcast_arrays(im1_arr, im2_arr) -> Tuple[ImArr, ImArr]:
    """
    Broadcasts two image arrays to compatible shapes for concatenation operations.
    Specifically, takes [..., H, W, C] and [..., H, W, C] and broadcasts them to the same shape.

    TODO: Support broadcasting with different H/W/C. E.g., currently:
    [1, H, W, C] and [H // 2, W, C] fail to broadcast
    """
    if isinstance(im1_arr, torch.Tensor):
        expand_func = lambda x, shape: x.expand(shape)
    elif isinstance(im1_arr, np.ndarray):
        expand_func = np.broadcast_to
    else:
        raise ValueError("Unsupported array type.")

    im1_shape, im2_shape = im1_arr.shape, im2_arr.shape

    if len(im1_shape) != len(im2_shape):  # Check if the number of dimensions are different
        warning_guard("Attempting to concat images with different numbers of leading dimensions. Broadcasting...")
        if len(im1_shape) > len(im2_shape):
            if len(im1_shape) == 4 and len(im2_shape) == 3 and im1_shape[-3:] != im2_shape[-3:]:
                im2_arr = im2_arr[None]
            else:
                im2_arr = expand_func(im2_arr, im1_shape)  # Broadcast im2 to match im1
        else:
            if len(im1_shape) == 3 and len(im2_shape) == 4 and im1_shape[-3:] != im2_shape[-3:]:
                im1_arr = im1_arr[None]
            else:
                im1_arr = expand_func(im1_arr, im2_shape)  # Broadcast im1 to match im2
    else:  # Same number of dimensions
        if im1_shape != im2_shape and len(im1_shape) > 3:
            if im1_shape[0] != im2_shape[0] and im1_shape[0] != 1 and im2_shape[0] != 1:
                raise ValueError("Error: Cannot broadcast arrays with incompatible leading dimensions.")
            warning_guard("Attempting to concat images with batch sizes. Broadcasting...")
            if im1_shape[0] < im2_shape[0]:
                im1_arr = expand_func(im1_arr, im2_shape)  # Broadcast im1 to match im2
            elif im1_shape[0] > im2_shape[0]:
                im2_arr = expand_func(im2_arr, im1_shape)  # Broadcast im2 to match im1

    return im1_arr, im2_arr


def concat_horizontal_(im1: Im, im2: Im, spacing: int = 0, **kwargs) -> Im:
    """Concatenates two images horizontally with optional spacing between them."""

    # We convert to HWC but allow for tensor/ndarray with different shapes/dtypes
    im1_arr = get_arr_hwc(im1)
    im2_arr = get_arr_hwc(im2)
    im1_arr, im2_arr = broadcast_arrays(im1_arr, im2_arr)

    if im1.height != im2.height:
        warning_guard(f"Images have different heights: {im1.height} and {im2.height}. Padding to match height.")
        if im1.height > im2.height:
            new_im2_arr = new_like(im1_arr, (*im1_arr.shape[:-2], im2_arr.shape[-2], *im2_arr.shape[-1:]), **kwargs)
            new_im2_arr[..., : im2.height, :, :] = im2_arr
            im2_arr = new_im2_arr
        else:
            new_im1_arr = new_like(im2_arr, (*im2_arr.shape[:-2], im1_arr.shape[-2], *im1_arr.shape[-1:]), **kwargs)
            new_im1_arr[..., : im1.height, :, :] = im1_arr
            im1_arr = new_im1_arr

    if spacing > 0:
        new_im2_arr = new_like(im2_arr, (*im2_arr.shape[:-2], im2_arr.shape[-2] + spacing, *im2_arr.shape[-1:]), **kwargs)
        new_im2_arr[..., :, spacing:, :] = im2_arr
        im2_arr = new_im2_arr

    return Im(concat_along_dim(im1_arr, im2_arr, dim=-2))


def concat_vertical_(im1: Im, im2: Im, spacing: int = 0, **kwargs) -> Im:
    """Concatenates two images vertically with optional spacing between them."""

    # We convert to HWC but allow for tensor/ndarray with different shapes/dtypes
    im1_arr = get_arr_hwc(im1)
    im2_arr = get_arr_hwc(im2)
    im1_arr, im2_arr = broadcast_arrays(im1_arr, im2_arr)
    if im1.width != im2.width:
        warning_guard(f"Images have different widths: {im1.width} and {im2.width}. Padding to match width.")
        if im1.width > im2.width:
            new_im2_arr = new_like(im1_arr, (*im2_arr.shape[:-2], im1_arr.shape[-2], *im2_arr.shape[-1:]), **kwargs)
            new_im2_arr[..., :, : im2.width, :] = im2_arr
            im2_arr = new_im2_arr
        else:
            new_im1_arr = new_like(im2_arr, (*im1_arr.shape[:-2], im2_arr.shape[-2], *im1_arr.shape[-1:]), **kwargs)
            new_im1_arr[..., :, : im1.width, :] = im1_arr
            im1_arr = new_im1_arr

    if spacing > 0:
        new_im2_arr = new_like(im2_arr, (*im2_arr.shape[:-3], im2_arr.shape[-3] + spacing, *im2_arr.shape[-2:]), **kwargs)
        new_im2_arr[..., spacing:, :, :] = im2_arr
        im2_arr = new_im2_arr

    return Im(concat_along_dim(im1_arr, im2_arr, dim=-3))
