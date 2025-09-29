from __future__ import annotations

import colorsys
import io
from typing import Optional, Union

import numpy as np
from PIL import Image

import importlib

if importlib.util.find_spec("torch") is not None:
    import torch
    from torch import Tensor

if importlib.util.find_spec("torchvision") is not None:
    import torchvision.transforms.functional as T


def torch_to_numpy(arr: Tensor):
    if arr.dtype == torch.bfloat16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()


def get_layered_image_from_binary_mask(masks, flip=False, override_colors=None, colormap="gist_rainbow"):
    from image_utils.im import torch_to_numpy

    if torch.is_tensor(masks):
        masks = torch_to_numpy(masks)
    if flip:
        masks = np.flipud(masks)

    masks = masks.astype(np.bool_)

    nonzero_channels = np.apply_over_axes(np.sum, masks, [0, 1]).squeeze(0).squeeze(0) > 0
    colors = np.zeros((masks.shape[2], 3), dtype=np.uint8)
    if nonzero_channels.sum() > 0:
        colors[nonzero_channels] = list(get_color(nonzero_channels.sum(), colormap=colormap))

    img = np.zeros((*masks.shape[:2], 3))
    for i in range(masks.shape[2]):
        img[masks[..., i]] = colors[i]
        if override_colors is not None and i in override_colors:
            img[masks[..., i]] = override_colors[i]

    return Image.fromarray(img.astype(np.uint8))


def get_color(max_value: int, colormap="gist_rainbow"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colormap = plt.get_cmap(colormap)  # Pink is 0, Yellow is 1
    colors = [mcolors.to_rgb(colormap(i / max_value)) for i in range(max_value)]  # Generate colors
    return (np.array(colors) * 255).astype(int).tolist()


def get_n_distinct_colors(n):
    def HSVToRGB(h, s, v):
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return (int(255 * r), int(255 * g), int(255 * b))

    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))


def integer_to_color(seg: Union[torch.IntTensor, np.ndarray], num_classes: Optional[int] = None, ignore_empty: bool = True, **kwargs):
    """
    H, W np/torch int array to PIL Image.
    """
    if isinstance(seg, np.ndarray):
        seg = torch.from_numpy(seg)

    if ignore_empty:
        seg -= seg.min()

    num_classes = num_classes or seg.max() + 1
    onehot = torch.nn.functional.one_hot(seg, num_classes)
    return onehot_to_color(onehot, ignore_empty=ignore_empty, **kwargs)


def onehot_to_color(
    masks: Union[torch.FloatTensor, np.ndarray],
    flip: bool = False,
    override_colors: Optional[dict[int, tuple[int, int, int]]] = None,
    colormap: str = "gist_rainbow",
    ignore_empty: bool = True,
):
    """
    H, W, C np/torch bool array to PIL Image.
    Note that in cases where a single pixel has multiple channels that are true, we take the color of the last channel.
    """
    if torch.is_tensor(masks):
        masks = torch_to_numpy(masks)

    if flip:
        masks = np.flipud(masks)

    assert masks.ndim == 3
    masks = masks.astype(np.bool_)
    colors = np.zeros((masks.shape[2], 3), dtype=np.uint8)

    if ignore_empty:
        nonzero_channels = np.apply_over_axes(np.sum, masks, [0, 1]).squeeze(0).squeeze(0) > 0
        if nonzero_channels.sum() > 0:
            colors[nonzero_channels] = list(get_color(nonzero_channels.sum(), colormap=colormap))
    else:
        colors = list(get_color(masks.shape[2], colormap=colormap))

    img = np.zeros((*masks.shape[:2], 3))
    for i in range(masks.shape[2]):
        img[masks[..., i]] = colors[i]
        if override_colors is not None and i in override_colors:
            img[masks[..., i]] = override_colors[i]

    return Image.fromarray(img.astype(np.uint8))


def get_img_from_binary_masks(masks, flip=False):
    """
    Expects [H, W, C] np/torch array
    """
    arr = encode_binary_labels(masks)
    if flip:
        arr = np.flipud(arr)

    colors = np.asarray(list(get_n_distinct_colors(2 ** masks.shape[2])))
    return Image.fromarray(colors[arr].astype(np.uint8))


def encode_binary_labels(masks):
    """
    Turns boolean mask -> integer segmentation. Considers each channel as a bit.
    This is useful for visualizing masks that are overlapping.
    Expects [H, W, C] np/torch boolean array
    """
    if torch.is_tensor(masks):
        masks = torch_to_numpy(masks)

    assert masks.ndim == 3
    masks = masks.astype(np.bool_)
    masks = masks.transpose(2, 0, 1)
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def square_pad(image, h, w):
    h_1, w_1 = image.shape[-2:]
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):
        # padding to preserve aspect ratio
        hp = int(w_1 / ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        if hp > 0 and wp < 0:
            hp = hp // 2
            image = T.pad(image, (0, hp, 0, hp), 0, "constant")
            return T.resize(image, [h, w], antialias=True)

        elif hp < 0 and wp > 0:
            wp = wp // 2
            image = T.pad(image, (wp, 0, wp, 0), 0, "constant")
            return T.resize(image, [h, w], antialias=True)

    else:
        return T.resize(image, [h, w], antialias=True)


def calculate_principal_components(embeddings, num_components=3):
    """Calculates the principal components given the embedding features.

    Args:
        embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
        num_components: An integer indicates the number of principal
        components to return.

    Returns:
        A 2-D float tensor of shape `[num_pixels, num_components]`.
    """
    embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
    _, _, v = torch.svd(embeddings)
    return v[:, :num_components]


def pca(embeddings: Tensor, num_components: int = 3, principal_components: Optional[Tensor] = None) -> Tensor:
    """Conducts principal component analysis on the embedding features.

    This function is used to reduce the dimensionality of the embedding.

    Args:
        embeddings: An N-D float tensor with shape with the
        last dimension as `embedding_dim`.
        num_components: The number of principal components.
        principal_components: A 2-D float tensor used to convert the
        embedding features to PCA'ed space, also known as the U matrix
        from SVD. If not given, this function will calculate the
        principal_components given inputs.

    Returns:
        A N-D float tensor with the last dimension as  `num_components`.
    """
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])

    if principal_components is None:
        principal_components = calculate_principal_components(embeddings, num_components)
    embeddings = torch.mm(embeddings, principal_components)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)

    return embeddings


def hist(arr, save: bool = True, use_im: bool = True, **kwargs):
    from lovely_numpy import lo as np_lo

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    fig = np_lo(arr, **kwargs).plt.fig

    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    img = img.copy()

    if use_im:
        from image_utils import Im

        img = Im(img[..., :3])
        if save:
            img.save()
    else:
        if save:
            Image.fromarray(img).save("hist.png")

    return img
