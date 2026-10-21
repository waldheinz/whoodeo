"""A tapered convolutional VAE on one RGB frame, shaped for the Neural Engine.

The layers are 3x3 convolutions, SiLU, additions, and nearest-neighbor
doubling. There is no normalization and no attention, so a Core ML decoder
runs on the Neural Engine at a fixed latent size. Width grows as the picture
shrinks: `channels` at full resolution, twice that at half, four times that
from a quarter down. Full resolution has one residual block, the levels below
it two, and the latent level `blocks` of them with dilations 1, 2, and 4 for
reach. Each residual branch ends in a convolution that starts at zero, so a
fresh network is a plain chain and nothing has to be normalized.

The levels can be set apart, full resolution first. `encoder_blocks` and
`decoder_blocks` hold one count per level above the latent, log2(factor) of
them. `widths` holds a multiple of `channels` per level, the latent level
last, one more entry. `decoder_widths` gives the decoder its own, so it can
be wider near the picture. Left out, each is the layout above.

The encoder maps the frame to a diagonal Gaussian of `latent` channels at
1/`factor` of each side. Training draws a sample, evaluation uses the mean.
`posterior: fixed` gives only the mean: no log-variance, no sample, no KL.

`noise_tau` above 0 adds latent noise in training, after any sample: each
frame draws s from |N(0, tau^2)| and gets s * std_c * N(0, 1), where std_c is
a running std of μ per channel (`latent_ema_std`, EMA 0.99, set from the
first batch). The decoder learns to read a code that is a little off.
Evaluation decodes μ.
The decoder output is not clamped during training. `decode(..., clamp=True)`
is what an export runs. A side that `factor` does not divide is padded and the
reconstruction is cut back to the original size.

`latent_shift` and `latent_scale` are per-channel buffers set at each
validation from the held-out μ: (μ - shift) * scale has mean 0 and std 1 in
each channel. A model that works on latents, such as an upscaler, should see
that scaled code and hand z / scale + shift back to the decoder. A checkpoint
with one shift and one scale for all channels loads with them repeated.
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
POSTERIORS = ("gaussian", "fixed")
# Running std of μ per channel, the scale of the training noise.
EMA_DECAY = 0.99


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


def layout(factor, encoder_blocks=None, decoder_blocks=None, widths=None, decoder_widths=None, where="vae2"):
    """Blocks and width multiples per level, full resolution first, defaults filled in.

    Returns (encoder_blocks, decoder_blocks, widths, decoder_widths) as tuples.
    """
    stages = int(math.log2(factor))
    default_blocks = LEVEL_BLOCKS[:stages]
    default_widths = WIDTHS[:stages + 1]

    def pick(name, value, default, count, least):
        if value is None:
            return tuple(default)
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise SystemExit(
                f"{where}: {name} needs {count} entries for factor {factor}, "
                f"full resolution first, not {value!r}"
            )
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < least:
                kind = "positive" if least > 0 else "zero or positive"
                raise SystemExit(f"{where}: {name} entries must be {kind} integers, not {value!r}")
        return tuple(value)

    encoder = pick("encoder_blocks", encoder_blocks, default_blocks, stages, 0)
    decoder = pick("decoder_blocks", decoder_blocks, default_blocks, stages, 0)
    enc_widths = pick("widths", widths, default_widths, stages + 1, 1)
    dec_widths = pick("decoder_widths", decoder_widths, enc_widths, stages + 1, 1)
    return encoder, decoder, enc_widths, dec_widths


def layout_text(factor, encoder_blocks=None, decoder_blocks=None, widths=None, decoder_widths=None):
    """The levels that differ from the default layout, for a log line. Empty for the default."""
    chosen = layout(factor, encoder_blocks, decoder_blocks, widths, decoder_widths)
    names = ("encoder_blocks", "decoder_blocks", "widths", "decoder_widths")
    parts = [
        f" {name} {'/'.join(str(item) for item in value)}"
        for name, value, default in zip(names, chosen, layout(factor))
        if value != default
    ]
    return "".join(parts)


class Encoder(nn.Module):
    def __init__(self, widths, blocks, latent, level_blocks, stats=2):
        super().__init__()
        stages = len(widths) - 1
        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList(
            _Down(widths[level], widths[level + 1], level_blocks[level])
            for level in range(stages)
        )
        self.mid = _blocks(widths[-1], blocks, DILATIONS)
        self.to_stats = nn.Conv2d(widths[-1], latent * stats, kernel_size=3, padding=1)

    def forward(self, image):
        hidden = self.stem(image)
        for stage in self.downs:
            hidden = stage(hidden)
        hidden = self.mid(hidden)
        return self.to_stats(F.silu(hidden))


class Decoder(nn.Module):
    def __init__(self, widths, blocks, latent, level_blocks):
        super().__init__()
        stages = len(widths) - 1
        self.from_latent = nn.Conv2d(latent, widths[-1], kernel_size=3, padding=1)
        self.mid = _blocks(widths[-1], blocks, DILATIONS)
        self.ups = nn.ModuleList(
            _Up(widths[level + 1], widths[level], level_blocks[level])
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

    def __init__(
        self, channels, blocks, latent, factor, beta,
        encoder_blocks=None, decoder_blocks=None, widths=None, decoder_widths=None,
        posterior=None,
    ):
        super().__init__()
        if channels < 1 or blocks < 1 or latent < 1:
            raise SystemExit("channels, blocks, and latent must be positive")
        if factor < 2 or factor > 16 or factor & (factor - 1):
            raise SystemExit("factor must be a power of two from 2 to 16")
        posterior = posterior or "gaussian"
        if posterior not in POSTERIORS:
            raise SystemExit(f"posterior must be gaussian or fixed, not {posterior!r}")
        if posterior == "fixed":
            if beta is not None:
                raise SystemExit("posterior fixed has no KL and takes no beta")
            beta = 0.0
        elif isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) <= 0:
            raise SystemExit("beta must be a positive number")
        self.posterior = posterior
        self.noise_tau = 0.0
        self.in_frames = 1
        self.channels = channels
        self.latent = latent
        self.factor = factor
        self.beta = float(beta)
        self.kl = None
        self.kl_loss = None
        encoder_blocks, decoder_blocks, widths, decoder_widths = layout(
            factor, encoder_blocks, decoder_blocks, widths, decoder_widths,
        )
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.widths = tuple(channels * multiple for multiple in widths)
        self.decoder_widths = tuple(channels * multiple for multiple in decoder_widths)
        stats = 2 if posterior == "gaussian" else 1
        self.encoder = Encoder(self.widths, blocks, latent, encoder_blocks, stats)
        self.decoder = Decoder(self.decoder_widths, blocks, latent, decoder_blocks)
        self.register_buffer("latent_shift", torch.zeros(latent))
        self.register_buffer("latent_scale", torch.ones(latent))
        # Zero until the first training batch sets it.
        self.register_buffer("latent_ema_std", torch.zeros(latent))
        self.mu = None
        self.logvar = None
        self.mu_grad = None
        self.code = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older checkpoints keep one shift and one scale, and no running std.
        for name in ("latent_shift", "latent_scale"):
            key = prefix + name
            value = state_dict.get(key)
            if value is not None and value.dim() == 0:
                state_dict[key] = value.expand(self.latent).clone()
        key = prefix + "latent_ema_std"
        if key not in state_dict:
            state_dict[key] = torch.zeros(self.latent)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def encode(self, image):
        """μ and the log-variance. The log-variance is None for posterior fixed."""
        stats = self.encoder(image)
        if self.posterior == "fixed":
            return stats, None
        mu, logvar = stats.chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def decode(self, latent, clamp=False):
        image = self.decoder(latent)
        return image.clamp(0, 1) if clamp else image

    def last_layer(self):
        """The weight the adaptive GAN weight compares gradients on."""
        return self.decoder.to_rgb.weight

    def set_latent_stats(self, mean, std):
        """Per-channel shift and scale so (μ - shift) * scale has mean 0 and std 1."""
        mean = torch.as_tensor(mean, dtype=self.latent_shift.dtype).to(self.latent_shift.device)
        std = torch.as_tensor(std, dtype=self.latent_scale.dtype).to(self.latent_scale.device)
        self.latent_shift.copy_(mean.expand(self.latent))
        self.latent_scale.copy_(1.0 / std.expand(self.latent).clamp_min(1e-6))

    @torch.no_grad()
    def _track_std(self, mu):
        batch_std = mu.detach().float().transpose(0, 1).reshape(self.latent, -1).std(dim=1)
        if float(self.latent_ema_std.abs().sum()) == 0.0:
            self.latent_ema_std.copy_(batch_std)
        else:
            self.latent_ema_std.mul_(EMA_DECAY).add_((1.0 - EMA_DECAY) * batch_std)

    def add_noise(self, latent, tau, std=None, generator=None):
        """latent + s * std_c * eps with s ~ |N(0, tau^2)| per frame."""
        std = self.latent_ema_std if std is None else std
        count = latent.shape[0]
        draw = torch.randn(count, 1, 1, 1, device=latent.device, dtype=latent.dtype, generator=generator)
        eps = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
        return latent + (draw.abs() * tau) * std.view(1, -1, 1, 1).to(latent.dtype) * eps

    def forward(self, image):
        self.kl = None
        self.kl_loss = None
        padded, height, width = pad_to(image, self.factor)
        mu, logvar = self.encode(padded)
        latent = mu
        if self.training and logvar is not None:
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
        if self.training:
            self._track_std(mu)
            if self.noise_tau > 0:
                latent = self.add_noise(latent, self.noise_tau)
        recon = self.decode(latent)[:, :, :height, :width]
        if logvar is not None:
            mean = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
            self.kl = mean.detach()
            self.kl_loss = self.beta * mean
        self.mu = mu.detach()
        self.logvar = None if logvar is None else logvar.detach()
        # Still in the graph: mu for its L2, the decoded code for scale_eq.
        self.mu_grad = mu
        self.code = latent
        return recon


class DecoderExport(nn.Module):
    """The decoder alone with the output clamped to [0, 1], for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent):
        return self.model.decode(latent, clamp=True)
