"""Pieces shared by the networks."""

import torch.nn as nn
import torch.nn.functional as F


def zero_conv(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def pad_to(image, factor):
    """Pad the bottom and right so both sides divide by `factor`. Reflect when it fits."""
    height, width = image.shape[-2:]
    pad_h = (factor - height % factor) % factor
    pad_w = (factor - width % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), height, width
