import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import ResidualBlock, bilinear_plus, zero_conv


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
