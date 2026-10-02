import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import ResidualBlock, bilinear_plus, zero_conv


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
