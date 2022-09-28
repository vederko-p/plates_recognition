

import numpy as np
import torch
from typing import Tuple

import utils


def predict_color(
        image: torch.tensor,
        bins: int = 20
) -> Tuple[float, float, float]:
    """Predicts color by max amount of same colors for every channel."""
    predict = []
    for channel in image:
        channel_values = []
        for i in channel:
            for j in i:
                channel_values.append(j.item())
        hist = np.histogram(channel_values, bins=bins)
        predict.append(hist[1][hist[0].argmax()])
    return tuple(predict)


class LazyOCR:
    def __init__(self, size: Tuple[int, int] = (10, 10)):
        self.size = size

    def __call__(self, frame: torch.tensor, **kwargs):
        patches = utils.get_avg_patches(frame, self.size)
        rgb01_color = predict_color(patches, **kwargs)
        hex_color = utils.rgb01_2_hex(rgb01_color)
        return hex_color
