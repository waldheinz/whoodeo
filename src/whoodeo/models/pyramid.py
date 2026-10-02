"""Coarse-to-fine flow alignment, then the residual ESPCN trunk."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import ResidualBlock, bilinear_plus, zero_conv


# Cosine of 1 starts the gate near open. Cosine of 0 starts it at a half.
GATE_SHARPNESS = 4.0
# Added to the training loss. Adam follows it while the neighbor mix is still closed.
ALIGN_LOSS_WEIGHT = 0.1


def warp(image, flow):
    """Resample `image` at `flow` pixels. Positive flow reads from the right and below."""
    _, _, height, width = flow.shape
    y = torch.arange(height, device=flow.device, dtype=flow.dtype)
    x = torch.arange(width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x + flow[:, 0]
    grid_y = grid_y + flow[:, 1]
    grid_x = 2 * grid_x / max(width - 1, 1) - 1
    grid_y = 2 * grid_y / max(height - 1, 1) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )


def _feature_pyramid(image, levels):
    scales = [image]
    for _ in range(levels - 1):
        height, width = scales[-1].shape[-2:]
        if height < 2 or width < 2:
            break
        scales.append(F.avg_pool2d(scales[-1], kernel_size=2))
    return scales


def _upsample_flow(flow, size):
    """Enlarge a flow. A coarse pixel of movement becomes several fine pixels."""
    height, width = size
    up = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    scale = up.new_tensor((width / flow.shape[-1], height / flow.shape[-2]))
    return up * scale.view(1, 2, 1, 1)


def _cosine_distance(left, right):
    left = F.normalize(left, dim=1, eps=1e-6)
    right = F.normalize(right, dim=1, eps=1e-6)
    return (1 - (left * right).sum(dim=1)).mean()


def _zero_neighbor_in_weights(conv, channels, in_frames):
    """The fuse reads the center frame only, until training moves the other weights."""
    mid = in_frames // 2
    start = mid * channels
    end = start + channels
    weight = conv.weight.data
    weight[:, :start].zero_()
    weight[:, end:].zero_()


def _cost_volume(center, neighbor, radius):
    """One cosine per integer offset in ±radius. A positive offset reads from the right and below."""
    center = F.normalize(center, dim=1, eps=1e-6)
    neighbor = F.normalize(neighbor, dim=1, eps=1e-6)
    height, width = center.shape[-2:]
    if radius:
        neighbor = F.pad(neighbor, (radius, radius, radius, radius))
    scores = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            view = neighbor[..., y0:y0 + height, x0:x0 + width]
            scores.append((center * view).sum(dim=1, keepdim=True))
    return torch.cat(scores, dim=1)


def _best_offset(cost, radius):
    """The integer offset with the highest cosine, as a flow. No gradient."""
    index = cost.detach().argmax(dim=1)
    bins = 2 * radius + 1
    dx = (index % bins) - radius
    dy = torch.div(index, bins, rounding_mode="floor") - radius
    return torch.stack((dx, dy), dim=1).to(dtype=cost.dtype)


class FlowBlock(nn.Module):
    """A correction of at most `limit` pixels. The last layer starts at zero."""

    def __init__(self, channels, limit):
        super().__init__()
        self.limit = float(limit)
        hidden = 32
        self.conv1 = nn.Conv2d(channels * 2, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2)
        self.out = nn.Conv2d(hidden, 2, kernel_size=5, padding=2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pair):
        hidden = F.relu(self.conv1(pair))
        hidden = F.relu(self.conv2(hidden))
        return torch.tanh(self.out(hidden)) * self.limit


class CostHead(nn.Module):
    """Turn a coarse correlation into a flow. The last layer starts at zero.

    A smooth shift of several pixels barely changes the picture until the
    guess is already close, so a network that only sees the two pictures
    walks the wrong way. The correlation has a peak at the matching offset
    on the first step, and the peak loss below teaches this head to follow it.
    """

    def __init__(self, radius):
        super().__init__()
        self.limit = float(radius)
        bins = (2 * radius + 1) ** 2
        self.conv1 = nn.Conv2d(bins, 32, kernel_size=3, padding=1)
        self.out = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, cost):
        hidden = F.relu(self.conv1(cost))
        return torch.tanh(self.out(hidden)) * self.limit


class RejectGate(nn.Module):
    """Keep a moved neighbor where it points the same way as the center.

    The learned convolution starts at zero. Until it moves, the cosine alone
    decides: a match stays open, the opposite direction stays closed, and an
    unrelated neighbor starts half open.
    """

    def __init__(self, channels):
        super().__init__()
        self.gain = nn.Conv2d(channels * 2 + 1, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

    def forward(self, center, warped):
        cosine = (
            F.normalize(center, dim=1, eps=1e-6)
            * F.normalize(warped, dim=1, eps=1e-6)
        ).sum(dim=1, keepdim=True)
        learned = self.gain(torch.cat((center, warped, cosine), dim=1))
        return torch.sigmoid(learned + GATE_SHARPNESS * cosine)


class PyramidAlign(nn.Module):
    """Line a neighbor feature map up with the center, coarse scale first.

    `levels` counts the full-resolution scale. Three levels also look at half
    and quarter size. The coarsest scale reads a correlation and can jump by
    `radius` pixels of that scale. Each finer scale adds its own correction of
    at most `radius` pixels, after the coarser flow has been enlarged and held
    fixed. Every guess starts at zero.
    """

    def __init__(self, channels, levels, radius):
        super().__init__()
        if levels < 1:
            raise SystemExit("levels must be positive")
        if radius < 0:
            raise SystemExit("radius must be zero or positive")
        self.levels = levels
        self.radius = radius
        # One scale has nothing coarser to correlate, so it only corrects.
        if levels == 1:
            self.blocks = nn.ModuleList([FlowBlock(channels, radius)])
            self.coarse = None
        else:
            self.blocks = nn.ModuleList(
                [FlowBlock(channels, radius) for _ in range(levels - 1)]
            )
            self.coarse = CostHead(radius)
        self.gate = RejectGate(channels)

    def _estimate(self, center_scales, neighbor_scales):
        last = len(center_scales) - 1
        # Detach so the match cannot train the stem to a flat field that is easy to match.
        center_det = [scale.detach() for scale in center_scales]
        neighbor_det = [scale.detach() for scale in neighbor_scales]
        coarse = center_det[last]
        flow = coarse.new_zeros(coarse.shape[0], 2, coarse.shape[-2], coarse.shape[-1])
        photos = []
        for level in reversed(range(last + 1)):
            if level != last:
                # A finer score must not drag the coarser guess off the offset it found.
                base = _upsample_flow(flow, center_det[level].shape[-2:]).detach()
            else:
                base = flow
            moved = warp(neighbor_det[level], base)
            if self.coarse is not None and level == last:
                cost = _cost_volume(center_det[level], moved, self.radius)
                delta = self.coarse(cost)
                if self.training:
                    photos.append((delta - _best_offset(cost, self.radius)).abs().mean())
            else:
                pair = torch.cat((center_det[level], moved), dim=1)
                delta = self.blocks[level](pair)
            flow = base + delta
            if self.training:
                photos.append(
                    _cosine_distance(center_det[level], warp(neighbor_det[level], flow))
                )
        return flow, photos

    def warp_neighbor(self, center, neighbor):
        center_scales = _feature_pyramid(center, self.levels)
        neighbor_scales = _feature_pyramid(neighbor, self.levels)
        flow, photos = self._estimate(center_scales, neighbor_scales)
        return warp(neighbor, flow), photos

    def forward(self, center, neighbor):
        warped, photos = self.warp_neighbor(center, neighbor)
        return warped * self.gate(center, warped), photos


class PyramidESPCN(nn.Module):
    """Shared per-frame stem, pyramid flow, then the residual ESPCN trunk."""

    def __init__(self, num_filters, num_res_blocks, in_frames, levels, stem, radius, scale_factor=2):
        super().__init__()
        if in_frames < 3 or in_frames % 2 != 1:
            raise SystemExit("pyramid in_frames must be an odd number of at least 3")
        if levels < 1:
            raise SystemExit("levels must be positive")
        if stem < 0:
            raise SystemExit("stem must be zero or positive")
        self.in_frames = in_frames
        self.align_loss = None
        self.stem_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.stem_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(stem)])
        self.align = PyramidAlign(num_filters, levels, radius)
        self.fuse = nn.Conv2d(num_filters * in_frames, num_filters, kernel_size=3, padding=1)
        _zero_neighbor_in_weights(self.fuse, num_filters, in_frames)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.final_conv = nn.Conv2d(num_filters, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        zero_conv(self.final_conv)

    def _stem(self, frame):
        return self.stem_blocks(F.relu(self.stem_conv(frame)))

    def forward(self, x):
        self.align_loss = None
        frames = x.chunk(self.in_frames, dim=1)
        feats = [self._stem(frame) for frame in frames]
        center_index = self.in_frames // 2
        center = feats[center_index]
        aligned = []
        photos = []
        for index, feat in enumerate(feats):
            if index == center_index:
                aligned.append(center)
            else:
                warped, extra = self.align(center, feat)
                aligned.append(warped)
                photos.extend(extra)
        if photos:
            self.align_loss = ALIGN_LOSS_WEIGHT * torch.stack(photos).mean()
        fused = F.relu(self.fuse(torch.cat(aligned, dim=1)))
        residual = self.res_blocks(fused)
        residual = self.pixel_shuffle(self.final_conv(residual))
        return bilinear_plus(x, residual, self.in_frames)
