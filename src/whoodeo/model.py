import torch.nn as nn
import torch.nn.functional as F

def zero_conv(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def bilinear_plus(low, residual, in_frames):
    """2× bilinear of the center low frame, plus the network correction."""
    mid = in_frames // 2
    center = low[:, mid * 3:(mid + 1) * 3]
    base = F.interpolate(center, scale_factor=2, mode="bilinear", align_corners=False)
    return base + residual


class ESPCN(nn.Module):
    def __init__(self, upscale_factor=2):
        super(ESPCN, self).__init__()
        C = 3

        self.conv1 = nn.Conv2d(C, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, C * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        zero_conv(self.conv3)

    def forward(self, x):
        residual = F.tanh(self.conv1(x))
        residual = F.tanh(self.conv2(residual))
        residual = self.pixel_shuffle(self.conv3(residual))
        return bilinear_plus(x, residual, 1)


class Whoodeo(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Whoodeo, self).__init__()

        # self.conv1 = nn.Conv2d(3,  96, (7, 7), (1, 1), (3, 3))
        self.conv2 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(64,  32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32,  1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        print(x.shape)
        # x = F.tanh(self.conv1(x))
        # x = F.tanh(self.conv2(x))
        # x = F.tanh(self.conv3(x))
        # x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ModifiedESPCN(nn.Module):
    """
    A modified ESPCN model incorporating residual blocks for enhanced performance.

    Args:
        scale_factor (int): Upscaling factor (e.g., 2 for 2x upscaling). Default is 2.
        num_filters (int): Number of filters in the convolutional layers. Default is 64.
        num_res_blocks (int): Number of residual blocks. Default is 5.
        in_frames (int): Number of stacked RGB frames in the input. Default is 1.
    """
    def __init__(self, scale_factor=2, num_filters=64, num_res_blocks=5, in_frames=1):
        super(ModifiedESPCN, self).__init__()
        self.in_frames = in_frames
        # Initial convolution to extract features from stacked RGB frames
        self.initial_conv = nn.Conv2d(3 * in_frames, num_filters, kernel_size=3, padding=1)
        # Stack of residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        # Predicts the 2× correction, packed for the pixel shuffle.
        self.final_conv = nn.Conv2d(num_filters, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        zero_conv(self.final_conv)

    def forward(self, x):
        residual = F.relu(self.initial_conv(x))
        residual = self.res_blocks(residual)
        residual = self.pixel_shuffle(self.final_conv(residual))
        return bilinear_plus(x, residual, self.in_frames)


def _load_deform():
    try:
        from mps_deform_conv import DeformConv2d
    except ImportError as exc:
        raise SystemExit("arch deform needs the mps-deform-conv package") from exc
    return DeformConv2d


class NeighborAlign(nn.Module):
    """Sample a neighbor feature map onto the center. Offsets start at zero."""

    def __init__(self, channels):
        super().__init__()
        DeformConv2d = _load_deform()
        self.conv = DeformConv2d(channels, channels, kernel_size=3, padding=1)
        self.offset = nn.Conv2d(channels * 2, 18, kernel_size=3, padding=1)
        self.mask = nn.Conv2d(channels * 2, 9, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.mask.weight)
        # sigmoid(4) is near 1, so the nine taps start as an ordinary 3x3.
        nn.init.constant_(self.mask.bias, 4.0)

    def forward(self, center, neighbor):
        pair = torch.cat([center, neighbor], dim=1)
        offset = self.offset(pair)
        mask = torch.sigmoid(self.mask(pair))
        return self.conv(neighbor, offset, mask)


class DeformESPCN(nn.Module):
    """ModifiedESPCN with neighbor frames aligned by a deformable 3x3."""

    def __init__(self, scale_factor=2, num_filters=64, num_res_blocks=32, in_frames=5):
        super().__init__()
        if in_frames < 1 or in_frames % 2 != 1:
            raise SystemExit("deform in_frames must be a positive odd number")
        self.in_frames = in_frames
        self.stem = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.center = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.align = NeighborAlign(num_filters)
        self.fuse = nn.Conv2d(num_filters * in_frames, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )
        self.final_conv = nn.Conv2d(num_filters, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        zero_conv(self.final_conv)

    def forward(self, x):
        frames = x.chunk(self.in_frames, dim=1)
        feats = [F.relu(self.stem(frame)) for frame in frames]
        center_index = self.in_frames // 2
        center = feats[center_index]
        aligned = []
        for index, feat in enumerate(feats):
            if index == center_index:
                aligned.append(self.center(center))
            else:
                aligned.append(self.align(center, feat))
        fused = F.relu(self.fuse(torch.cat(aligned, dim=1)))
        residual = self.res_blocks(fused)
        residual = self.pixel_shuffle(self.final_conv(residual))
        return bilinear_plus(x, residual, self.in_frames)
