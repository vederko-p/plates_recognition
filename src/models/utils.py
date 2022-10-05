
from typing import Tuple

import numpy as np
import torch
from torch import nn


def get_avg_patches(image: torch.tensor,
                    size: Tuple[int, int]) -> torch.tensor:
    _, h, w = image.shape
    kernel_size = h // size[0], w // size[0]
    _avg_pool = nn.AvgPool2d(kernel_size=kernel_size)
    return _avg_pool(image)


def rgb01_2_hex(color: Tuple[float, float, float]):
    hex_values = map(lambda x: hex(int(x*255))[-2:], color)
    return '#' + ''.join(hex_values)


def get_crop_from_np(image: np.array, xmin, ymin, xmax, ymax) -> np.array:
    return image[int(ymin): int(ymax), int(xmin): int(xmax)]


def np2tensor(
    array: np.array, dims=(2, 0, 1),
    brightness_normalize_coeff=255
) -> torch.tensor:
    return torch.tensor(array).permute(*dims) / brightness_normalize_coeff


def tensor2np(tensor: torch.tensor, dims=(1, 2, 0)) -> np.array:
    return tensor.permute(dims).numpy()
