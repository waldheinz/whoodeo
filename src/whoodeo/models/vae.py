"""A convolutional VAE on one RGB frame.

The encoder maps the frame to a diagonal Gaussian. Training draws a sample.
Evaluation uses the mean, so a validation image stays put. The decoder paints
the frame. There is no skip from the encoder to the decoder, so the picture
has to pass through the latent. A side that `factor` does not divide is padded
and the reconstruction is cut back to the original size.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.models.common import ResidualBlock, pad_to


class _Down(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(self.blocks(x))


class _Up(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.mix = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.blocks(self.mix(x))


class VAE(nn.Module):
    """Residual blocks at every scale, then a spatial Gaussian of `latent` channels."""

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
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList(_Down(channels, blocks) for _ in range(stages))
        self.bottom = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.to_stats = nn.Conv2d(channels, latent * 2, kernel_size=3, padding=1)
        self.from_latent = nn.Conv2d(latent, channels, kernel_size=3, padding=1)
        self.restore = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.ups = nn.ModuleList(_Up(channels, blocks) for _ in range(stages))
        self.to_rgb = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def encode(self, image):
        hidden = F.relu(self.stem(image))
        for stage in self.downs:
            hidden = stage(hidden)
        hidden = self.bottom(hidden)
        mu, logvar = self.to_stats(hidden).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def decode(self, latent):
        hidden = F.relu(self.from_latent(latent))
        hidden = self.restore(hidden)
        for stage in self.ups:
            hidden = stage(hidden)
        return self.to_rgb(hidden)

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


class LatentTally:
    """Sums of μ and σ, so a split can be compared with the unit Gaussian.

    `spread` is the square root of var(μ) + mean(σ²). It sits near 1 when the
    code matches that Gaussian. A mu_std near 0 means the latent carries
    nothing. A sigma_mean near 0 means a deterministic code.
    """

    def __init__(self):
        self.count = 0
        self.sum_mu = 0.0
        self.sumsq_mu = 0.0
        self.sum_sigma = 0.0
        self.sumsq_sigma = 0.0
        self.channels = None
        self._hist = []

    def add(self, mu, logvar, sample=False):
        mu = mu.detach().float()
        sigma = torch.exp(0.5 * logvar.detach().float())
        self.count += mu.numel()
        self.sum_mu += float(mu.sum())
        self.sumsq_mu += float(mu.square().sum())
        self.sum_sigma += float(sigma.sum())
        self.sumsq_sigma += float(sigma.square().sum())
        summed = torch.stack((
            mu.sum(dim=(0, 2, 3)),
            mu.square().sum(dim=(0, 2, 3)),
            sigma.sum(dim=(0, 2, 3)),
        )).cpu()
        self.channels = summed if self.channels is None else self.channels + summed
        if sample:
            flat = mu.reshape(-1).cpu()
            take = min(4096, flat.numel())
            if flat.numel() > take:
                flat = flat[torch.randint(flat.numel(), (take,))]
            self._hist.append(flat)

    def report(self):
        if self.count == 0:
            return None
        mean = self.sum_mu / self.count
        var = max(0.0, self.sumsq_mu / self.count - mean * mean)
        mu_std = var ** 0.5
        sigma_mean = self.sum_sigma / self.count
        sigma_sq = self.sumsq_sigma / self.count
        width = self.channels.shape[1]
        each = self.count // width
        per_channel = []
        for index in range(width):
            total, squares, sig = (float(self.channels[row, index]) for row in range(3))
            center = total / each
            channel_var = max(0.0, squares / each - center * center)
            per_channel.append((channel_var ** 0.5, sig / each))
        hist = None
        if self._hist:
            hist = torch.cat(self._hist)
            if hist.numel() > 65536:
                hist = hist[torch.randint(hist.numel(), (65536,))]
        return {
            "mu_mean": mean,
            "mu_std": mu_std,
            "sigma_mean": sigma_mean,
            "spread": (var + sigma_sq) ** 0.5,
            "channels": per_channel,
            "hist": hist,
        }


def latent_scalars(report, prefix):
    """TensorBoard tags for one tally report. `prefix` is empty or `val/`."""
    tags = {
        f"{prefix}latent/mu_mean": report["mu_mean"],
        f"{prefix}latent/mu_std": report["mu_std"],
        f"{prefix}latent/sigma_mean": report["sigma_mean"],
        f"{prefix}latent/spread": report["spread"],
    }
    for index, (mu_std, sigma_mean) in enumerate(report["channels"]):
        tags[f"{prefix}latent/channel/{index}/mu_std"] = mu_std
        tags[f"{prefix}latent/channel/{index}/sigma_mean"] = sigma_mean
    return tags
