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


def bilinear_plus(low, residual, in_frames):
    """2× bilinear of the center low frame, plus the network correction."""
    mid = in_frames // 2
    center = low[:, mid * 3:(mid + 1) * 3]
    base = F.interpolate(center, scale_factor=2, mode="bilinear", align_corners=False)
    return base + residual


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers and a skip connection.

    Args:
        num_filters (int): Number of filters (channels) in the convolutional layers.
    """
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)
        return x + residual
