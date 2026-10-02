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


def _shift_views(image, radius):
    """Each integer offset in ±radius. A positive offset reads from the right and below."""
    height, width = image.shape[-2:]
    if radius:
        image = F.pad(image, (radius, radius, radius, radius))
    views = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            views.append(image[..., y0:y0 + height, x0:x0 + width])
    return views


class ShiftAlign(nn.Module):
    """Move a neighbor feature map onto the center by a dense integer search.

    Cosine similarity scores every offset in ±radius. A softmax mixes the
    shifted neighbor. `sharpness` multiplies the cosines and is not learned.
    With `reject`, one extra logit can drop the neighbor instead of moving it.
    Query and key start as identity, and the discard logit starts at zero, so a
    cosine of zero ties with discarding the neighbor.
    """

    def __init__(self, channels, radius, sharpness, reject):
        super().__init__()
        if radius < 0:
            raise SystemExit("radius must be zero or positive")
        self.radius = radius
        self.sharpness = float(sharpness)
        self.use_reject = bool(reject)
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.dirac_(self.query.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.dirac_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        if self.use_reject:
            self.discard = nn.Conv2d(channels, 1, kernel_size=1)
            nn.init.zeros_(self.discard.weight)
            nn.init.zeros_(self.discard.bias)

    def forward(self, center, neighbor):
        query = F.normalize(self.query(center), dim=1, eps=1e-6)
        key = F.normalize(self.key(neighbor), dim=1, eps=1e-6)
        key_views = _shift_views(key, self.radius)
        scores = [(query * view).sum(dim=1, keepdim=True) for view in key_views]
        scores = torch.cat(scores, dim=1) * self.sharpness
        if self.use_reject:
            scores = torch.cat([scores, self.discard(center)], dim=1)
        weights = torch.softmax(scores, dim=1)
        aligned = torch.zeros_like(neighbor)
        for index, view in enumerate(_shift_views(neighbor, self.radius)):
            aligned = aligned + weights[:, index:index + 1] * view
        return aligned


class ShiftESPCN(nn.Module):
    """Shared per-frame stem, integer alignment, then the residual ESPCN trunk."""

    def __init__(self, num_filters, num_res_blocks, in_frames, radius, stem, sharpness, reject, scale_factor=2):
        super().__init__()
        if in_frames < 3 or in_frames % 2 != 1:
            raise SystemExit("shift in_frames must be an odd number of at least 3")
        if stem < 0:
            raise SystemExit("stem must be zero or positive")
        self.in_frames = in_frames
        self.stem_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.stem_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(stem)])
        self.align = ShiftAlign(num_filters, radius, sharpness, reject)
        self.fuse = nn.Conv2d(num_filters * in_frames, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.final_conv = nn.Conv2d(num_filters, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        zero_conv(self.final_conv)

    def _stem(self, frame):
        return self.stem_blocks(F.relu(self.stem_conv(frame)))

    def forward(self, x):
        frames = x.chunk(self.in_frames, dim=1)
        feats = [self._stem(frame) for frame in frames]
        center_index = self.in_frames // 2
        center = feats[center_index]
        aligned = []
        for index, feat in enumerate(feats):
            if index == center_index:
                aligned.append(center)
            else:
                aligned.append(self.align(center, feat))
        fused = F.relu(self.fuse(torch.cat(aligned, dim=1)))
        residual = self.res_blocks(fused)
        residual = self.pixel_shuffle(self.final_conv(residual))
        return bilinear_plus(x, residual, self.in_frames)
