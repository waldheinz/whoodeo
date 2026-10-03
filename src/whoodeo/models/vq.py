"""One-frame residual ESPCN with a codebook on the correction.

Each patch of low-resolution features is replaced by the code that
points the same way. `patch` is the side length of that block. The straight-through path trains the encoder, and a
moving average trains the book. Codes the batch stopped using are put back
onto features from that batch, so the book cannot settle on one or two
entries. With bilinear left on, the enlargement stays and the last layer starts
at zero. With it off, that layer keeps its normal initialization and the
picture is only the codebook output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import ResidualBlock, bilinear_plus, zero_conv


# One code covers this many low-resolution pixels on each side, 8 on the master.
PATCH = 4
CODES = 1024
# VQ-VAE commitment weight. The train loop adds align_loss on top of the recipe.
COMMITMENT = 0.25
DECAY = 0.99
# A code the current batch did not use, and whose average stay is below this, is
# replaced by a feature from the batch. Without that, the first lookup keeps
# every position on the one or two shortest codes.
DEAD = 0.1


def _pad_to(image, factor):
    """Pad the bottom and right so both sides divide by `factor`. Reflect when it fits."""
    height, width = image.shape[-2:]
    pad_h = (factor - height % factor) % factor
    pad_w = (factor - width % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), height, width


class Codebook(nn.Module):
    """Nearest code by direction. The book follows the features with a moving average."""

    def __init__(self, codes, dim):
        super().__init__()
        self.codes = codes
        self.dim = dim
        book = F.normalize(torch.randn(codes, dim), dim=1)
        self.register_buffer("book", book)
        self.register_buffer("usage", torch.full((codes,), DEAD))
        self.register_buffer("average", book.clone())

    def forward(self, features):
        batch, _, height, width = features.shape
        flat = _unit(features.permute(0, 2, 3, 1).reshape(-1, self.dim))
        book = _unit(self.book)
        index = torch.matmul(flat, book.t()).argmax(dim=1)
        picked = book.index_select(0, index)
        quantized = picked.view(batch, height, width, self.dim).permute(0, 3, 1, 2).contiguous()
        if self.training:
            self._update(flat.detach(), index)
        normed = _unit(features)
        loss = COMMITMENT * F.mse_loss(normed, quantized.detach())
        return normed + (quantized - normed).detach(), loss

    @torch.no_grad()
    def _update(self, flat, index):
        onehot = F.one_hot(index, self.codes).to(dtype=flat.dtype)
        count = onehot.sum(dim=0)
        self.usage.mul_(DECAY).add_(count, alpha=1.0 - DECAY)
        self.average.mul_(DECAY).add_(onehot.t() @ flat, alpha=1.0 - DECAY)
        total = self.usage.sum().clamp(min=1e-5)
        smoothed = (self.usage + 1e-5) / (total + self.codes * 1e-5) * total
        self.book.copy_(_unit(self.average / smoothed.unsqueeze(1)))
        dead = (self.usage < DEAD) & (count == 0)
        n = int(dead.sum().item())
        if n == 0:
            return
        choice = torch.randint(0, flat.shape[0], (n,), device=flat.device)
        fresh = flat.index_select(0, choice)
        self.book[dead] = fresh
        self.average[dead] = fresh
        # The threshold itself, so one unused step drops the entry back under it.
        self.usage[dead] = DEAD


def _unit(vector):
    return F.normalize(vector, dim=-1, eps=1e-6)


class VQESPCN(nn.Module):
    """Modified ESPCN with the codebook between the trunk and the 2× correction."""

    def __init__(self, num_filters, num_res_blocks, scale_factor=2, bilinear=True, codes=CODES, patch=PATCH):
        super().__init__()
        if codes < 1:
            raise SystemExit("codes must be positive")
        if patch < 1:
            raise SystemExit("patch must be positive")
        self.in_frames = 1
        self.bilinear = bilinear
        self.patch = patch
        self.align_loss = None
        self.initial_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )
        self.down = nn.Conv2d(num_filters, num_filters, kernel_size=patch, stride=patch)
        self.codebook = Codebook(codes, num_filters)
        self.up = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        # Codes arrive as unit vectors, so each channel has variance 1/dim.
        # Kaiming assumes variance 1. Scale the weights once so a random start is noise.
        self.up.weight.data.mul_(num_filters ** 0.5)
        self.final_conv = nn.Conv2d(num_filters, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        if bilinear:
            zero_conv(self.final_conv)

    def forward(self, x):
        self.align_loss = None
        features = F.relu(self.initial_conv(x))
        features = self.res_blocks(features)
        padded, height, width = _pad_to(features, self.patch)
        quantized, self.align_loss = self.codebook(self.down(padded))
        restored = F.interpolate(quantized, scale_factor=self.patch, mode="nearest")
        restored = self.up(restored)[:, :, :height, :width]
        residual = self.pixel_shuffle(self.final_conv(restored))
        if not self.bilinear:
            return residual
        return bilinear_plus(x, residual, self.in_frames)
