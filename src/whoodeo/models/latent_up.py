"""A 2x upscaler that predicts the latent of a frozen vae2 instead of pixels.

The low frame is h x w, the master 2h x 2w. A vae2 at factor 8 codes the
master at 2h/8 x 2w/8 = h/4 x w/4, so this network maps the low RGB frame to
that grid: the normalized code z = (μ - shift) * scale of the vae2, `latent`
channels at 1/4 of each low side. The frozen vae2 decoder turns
z / scale + shift into the 2x picture. Training and the decoder live in
train.py; this module is the trainable part alone.

The layers are the vae2 encoder's: a 3x3 stem, residual blocks of
conv + activation (`activation`, silu or relu as in vae2) whose last convolution starts at zero, a stride-2 convolution to
each coarser level, and `blocks` residual blocks at 1/4 with dilations 1, 2,
4 in turn. A 3x3 head after the activation gives the code. No normalization, no
attention, so it runs on the Neural Engine at a fixed size. `channels` is the
width at the low resolution, `widths` the multiples per level (low, 1/2, 1/4),
`encoder_blocks` the residual count at low and at 1/2. A side that 4 does not
divide is padded on the bottom and the right, as vae2 pads, and the caller
crops the decoded frame back to 2h x 2w.
"""

import torch
import torch.nn as nn

from whoodeo.models.common import pad_to
from whoodeo.models.vae2 import DILATIONS, Encoder, layout

# The network shrinks the low frame by this much: vae2 factor 8 over the 2x scale.
FACTOR = 4
# Low, 1/2, 1/4 as multiples of channels; residual blocks at low and at 1/2.
WIDTHS = (1, 2, 4)
LEVEL_BLOCKS = (2, 4)


def latent_layout(encoder_blocks=None, widths=None, where="latent_up"):
    """Residual blocks per level above 1/4 and width multiples, defaults filled in."""
    blocks = LEVEL_BLOCKS if encoder_blocks is None else encoder_blocks
    multiples = WIDTHS if widths is None else widths
    chosen, _dec, chosen_widths, _dec_widths = layout(FACTOR, blocks, None, multiples, None, where=where)
    return chosen, chosen_widths


class LatentUp(nn.Module):
    """Low RGB in, the normalized vae2 code at 1/4 of the padded low frame out."""

    def __init__(self, channels, blocks, latent, encoder_blocks=None, widths=None, activation=None):
        super().__init__()
        if channels < 1 or blocks < 0 or latent < 1:
            raise SystemExit("channels and latent must be positive, blocks zero or more")
        encoder_blocks, widths = latent_layout(encoder_blocks, widths)
        self.in_frames = 1
        self.factor = FACTOR
        self.channels = channels
        self.blocks = blocks
        self.latent = latent
        self.encoder_blocks = encoder_blocks
        self.widths = tuple(channels * multiple for multiple in widths)
        self.activation = activation or "silu"
        self.encoder = Encoder(self.widths, blocks, latent, encoder_blocks, stats=1, activation=activation)

    def forward(self, low):
        padded, _height, _width = pad_to(low, FACTOR)
        return self.encoder(padded)


@torch.no_grad()
def init_from_vae(model, vae):
    """Copy the vae2 encoder's latent blocks and its head into `model`. Returns a note.

    The vae2 blocks at the latent go into the last blocks at 1/4 here, with the
    same dilations, when the widths agree. The head is the vae2's last
    convolution with the shift and scale folded in, so it gives
    (μ - shift) * scale. Anything that does not fit is left as it was.
    """
    ours = model.encoder
    theirs = vae.encoder
    notes = []
    width = model.widths[-1]
    if width != vae.widths[-1]:
        return f"no warm start: width {width} at 1/4, vae2 latent width {vae.widths[-1]}"
    count = len(theirs.mid)
    offset = len(ours.mid) - count
    if offset >= 0 and offset % len(DILATIONS) == 0:
        for index in range(count):
            ours.mid[offset + index].load_state_dict(theirs.mid[index].state_dict())
        notes.append(f"blocks {offset}..{offset + count - 1} at 1/4 from the vae2 latent blocks")
    else:
        notes.append(
            f"latent blocks left: {len(ours.mid)} here, {count} in the vae2, "
            "the difference must be a multiple of 3"
        )
    head = theirs.to_stats
    if head.weight.shape == ours.to_stats.weight.shape:
        scale = vae.latent_scale.to(head.weight)
        shift = vae.latent_shift.to(head.weight)
        ours.to_stats.weight.copy_(head.weight * scale.view(-1, 1, 1, 1))
        ours.to_stats.bias.copy_((head.bias - shift) * scale)
        notes.append("head from the vae2 head, shift and scale folded in")
    return "warm start: " + ", ".join(notes)


class LatentUpExport(nn.Module):
    """latent_up and the vae2 decoder in one graph, the frame clamped to [0, 1], for tracing.

    The vae2 decoder reads μ = z / scale + shift. Both are per channel and sit
    right behind the head, the last convolution of latent_up, so they fold
    into its weight and bias: no extra op, and exact, since the decoder's zero
    padding sees μ in both cases. The input must already divide by FACTOR;
    the caller pads the bottom and the right as `pad_to` does and crops the
    frame back to 2h x 2w.
    """

    def __init__(self, model, vae):
        super().__init__()
        import copy

        self.encoder = copy.deepcopy(model.encoder)
        self.decoder = copy.deepcopy(vae.decoder)
        head = self.encoder.to_stats
        inverse = vae.latent_scale.reciprocal().to(head.weight)
        shift = vae.latent_shift.to(head.weight)
        with torch.no_grad():
            head.weight.copy_(head.weight * inverse.view(-1, 1, 1, 1))
            head.bias.copy_(head.bias * inverse + shift)

    def forward(self, frames):
        return self.decoder(self.encoder(frames)).clamp(0, 1)


class LatentUpReference(nn.Module):
    """The unfolded path training takes: code, z / scale + shift, decode, crop to 2h x 2w."""

    def __init__(self, model, vae):
        super().__init__()
        self.model = model
        self.vae = vae

    def forward(self, low):
        height, width = low.shape[-2:]
        code = self.model(low)
        shift = self.vae.latent_shift.view(1, -1, 1, 1)
        scale = self.vae.latent_scale.view(1, -1, 1, 1)
        image = self.vae.decode(code / scale + shift, clamp=True)
        return image[:, :, :2 * height, :2 * width]
