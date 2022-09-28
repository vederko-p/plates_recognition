
from typing import Tuple

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
