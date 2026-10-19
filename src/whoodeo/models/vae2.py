"""A tapered convolutional VAE on one RGB frame, shaped for the Neural Engine.

The layers are 3x3 convolutions, SiLU, additions, and nearest-neighbor
doubling. There is no normalization and no attention, so a Core ML decoder
runs on the Neural Engine at a fixed latent size. Width grows as the picture
shrinks: `channels` at full resolution, twice that at half, four times that
from a quarter down. Full resolution has one residual block, the levels below
it two, and the latent level `blocks` of them with dilations 1, 2, and 4 for
reach. Each residual branch ends in a convolution that starts at zero, so a
fresh network is a plain chain and nothing has to be normalized.

The encoder maps the frame to a diagonal Gaussian of `latent` channels at
1/`factor` of each side. Training draws a sample, evaluation uses the mean.
The decoder output is not clamped during training. `decode(..., clamp=True)`
is what an export runs. A side that `factor` does not divide is padded and the
reconstruction is cut back to the original size.

`latent_shift` and `latent_scale` are buffers set at each validation from the
held-out means: (μ - shift) * scale has mean 0 and spread near 1. A model that
works on latents, such as an upscaler, should see that scaled code and hand
z / scale + shift back to the decoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import pad_to, zero_conv

# Width of each level as a multiple of `channels`, full resolution first.
WIDTHS = (1, 2, 4, 4, 4)
# Residual blocks at each level above the latent, full resolution first.
LEVEL_BLOCKS = (1, 2, 2, 2)
# Dilations of the latent blocks, repeated in this order.
DILATIONS = (1, 2, 4)


class Block(nn.Module):
    """x + conv(SiLU(conv(SiLU(x)))). The second convolution starts at zero."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        zero_conv(self.conv2)

    def forward(self, x):
        return x + self.conv2(F.silu(self.conv1(F.silu(x))))


def _blocks(channels, count, dilations=(1,)):
    return nn.Sequential(*(
        Block(channels, dilations[index % len(dilations)]) for index in range(count)
    ))


class _Down(nn.Module):
    def __init__(self, width, wider, blocks):
        super().__init__()
        self.blocks = _blocks(width, blocks)
        self.down = nn.Conv2d(width, wider, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(self.blocks(x))


class _Up(nn.Module):
    def __init__(self, wider, width, blocks):
        super().__init__()
        self.mix = nn.Conv2d(wider, width, kernel_size=3, padding=1)
        self.blocks = _blocks(width, blocks)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.blocks(self.mix(x))


class Encoder(nn.Module):
    def __init__(self, widths, blocks, latent):
        super().__init__()
        stages = len(widths) - 1
        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList(
            _Down(widths[level], widths[level + 1], LEVEL_BLOCKS[level])
            for level in range(stages)
        )
        self.mid = _blocks(widths[-1], blocks, DILATIONS)
        self.to_stats = nn.Conv2d(widths[-1], latent * 2, kernel_size=3, padding=1)

    def forward(self, image):
        hidden = self.stem(image)
        for stage in self.downs:
            hidden = stage(hidden)
        hidden = self.mid(hidden)
        return self.to_stats(F.silu(hidden))


class Decoder(nn.Module):
    def __init__(self, widths, blocks, latent):
        super().__init__()
        stages = len(widths) - 1
        self.from_latent = nn.Conv2d(latent, widths[-1], kernel_size=3, padding=1)
        self.mid = _blocks(widths[-1], blocks, DILATIONS)
        self.ups = nn.ModuleList(
            _Up(widths[level + 1], widths[level], LEVEL_BLOCKS[level])
            for level in reversed(range(stages))
        )
        self.to_rgb = nn.Conv2d(widths[0], 3, kernel_size=3, padding=1)

    def forward(self, latent):
        hidden = self.mid(self.from_latent(latent))
        for stage in self.ups:
            hidden = stage(hidden)
        return self.to_rgb(F.silu(hidden))


class VAE2(nn.Module):
    """Tapered widths, `blocks` dilated blocks at the latent, a Gaussian of `latent` channels."""

    def __init__(self, channels, blocks, latent, factor, beta):
        super().__init__()
        if channels < 1 or blocks < 1 or latent < 1:
            raise SystemExit("channels, blocks, and latent must be positive")
        if factor < 2 or factor > 16 or factor & (factor - 1):
            raise SystemExit("factor must be a power of two from 2 to 16")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) <= 0:
            raise SystemExit("beta must be a positive number")
        self.in_frames = 1
        self.channels = channels
        self.latent = latent
        self.factor = factor
        self.beta = float(beta)
        self.kl = None
        self.kl_loss = None
        stages = int(math.log2(factor))
        widths = tuple(channels * WIDTHS[level] for level in range(stages + 1))
        self.widths = widths
        self.encoder = Encoder(widths, blocks, latent)
        self.decoder = Decoder(widths, blocks, latent)
        self.register_buffer("latent_shift", torch.zeros(()))
        self.register_buffer("latent_scale", torch.ones(()))

    def encode(self, image):
        mu, logvar = self.encoder(image).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def decode(self, latent, clamp=False):
        image = self.decoder(latent)
        return image.clamp(0, 1) if clamp else image

    def last_layer(self):
        """The weight the adaptive GAN weight compares gradients on."""
        return self.decoder.to_rgb.weight

    def set_latent_stats(self, mean, spread):
        """Shift and scale so (μ - shift) * scale is centered with spread near 1."""
        self.latent_shift.fill_(float(mean))
        self.latent_scale.fill_(1.0 / max(float(spread), 1e-6))

    def forward(self, image):
        self.kl = None
        self.kl_loss = None
        padded, height, width = pad_to(image, self.factor)
        mu, logvar = self.encode(padded)
        if self.training:
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
        else:
            latent = mu
        recon = self.decode(latent)[:, :, :height, :width]
        mean = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
        self.kl = mean.detach()
        self.kl_loss = self.beta * mean
        self.mu = mu.detach()
        self.logvar = logvar.detach()
        return recon


class DecoderExport(nn.Module):
    """The decoder alone with the output clamped to [0, 1], for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent):
        return self.model.decode(latent, clamp=True)
