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

`condition: codec` adds FiLM. A vector of CONDITION_SIZE values says what
happened to the low frame: one slot per codec of CODECS (h264, mpeg4,
mpeg2), all zero when the codec is unknown, then RESERVE slots that stay
zero for now (deinterlacing, strength, blur later, without a new shape). A
small MLP turns it into a scale γ and a shift β per channel of every
residual block, and the block's output becomes x * (1 + γ) + β. The last
layer of the MLP starts at zero, so a fresh FiLM gives γ = β = 0 and the
network is exactly the one without it: a latent_up checkpoint without
condition loads into it with the FiLM weights left at their start. The
encoder keeps its parameter names. For an export the vector is a second input, and
the FiLM MLP stays in the graph, a multiply and an add per block.

`in_frames: 3` reads the previous, middle, and next low frame, stacked on
the channels as the other multi-frame archs take them (9 channels). Each
frame runs through the encoder's stem and the residual blocks at the low
resolution (with FiLM), so every frame has `channels` features at full low
resolution. `FeatureAlign` then brings the neighbors onto the middle frame:
a frozen SPyNet computes the flow from the middle low frame to each neighbor
(two runs, no symmetry assumed), the neighbor features are warped along it
(bilinear, border), and a fusion reads the middle features, both warped
neighbor features, and one warp-error channel per neighbor (the mean
absolute difference of the warped neighbor low frame and the middle one):
3x3 convolution to `channels`, activation, 3x3 convolution, added onto the
middle features. From the stride-2 convolution to 1/2 on, one stream runs as
for one frame. All of it is inside the model: training, export, and apply
feed the raw 9 channels. SPyNet's weights are part of the state dict and
never train (requires_grad false); a fresh run loads the official weights
(training key spynet_weights). The fusion has PyTorch's default start. The
clip's edges repeat the middle frame, as for every multi-frame arch; the flow
there is about zero. A scene cut makes the warp error large, which the
fusion can learn to read. A one-frame checkpoint does not load into it.
"""

import torch
import torch.nn as nn

from whoodeo.models.common import pad_to
from whoodeo.spynet import SPyNet, flow_warp
from whoodeo.models.vae2 import DILATIONS, Encoder, activation_fn, layout

# The network shrinks the low frame by this much: vae2 factor 8 over the 2x scale.
FACTOR = 4
# Low, 1/2, 1/4 as multiples of channels; residual blocks at low and at 1/2.
WIDTHS = (1, 2, 4)
LEVEL_BLOCKS = (2, 4)
# The one-hot slots of the codec condition, in vector order. No unknown slot:
# unknown is all of them zero.
CODECS = ("h264", "mpeg4", "mpeg2")
# Slots after the codecs that stay zero for now, so later inputs (the kind of
# deinterlacing, strength, blur) fit without changing the vector's shape.
RESERVE = 8
# condition: in the model section, and the length of its vector.
CONDITIONS = {"codec": len(CODECS) + RESERVE}
# Hidden width of the FiLM MLP.
FILM_HIDDEN = 64
# Variant name prefix (before the first '-') -> codec, as make-low names them:
# x264-crf28, x264-500k, mpeg4-q12.
CODEC_PREFIXES = {"x264": "h264", "h264": "h264", "mpeg4": "mpeg4", "mpeg2": "mpeg2"}


def latent_layout(encoder_blocks=None, widths=None, where="latent_up"):
    """Residual blocks per level above 1/4 and width multiples, defaults filled in."""
    blocks = LEVEL_BLOCKS if encoder_blocks is None else encoder_blocks
    multiples = WIDTHS if widths is None else widths
    chosen, _dec, chosen_widths, _dec_widths = layout(FACTOR, blocks, None, multiples, None, where=where)
    return chosen, chosen_widths


def codec_of(variant):
    """The codec of a low variant name such as x264-crf28 or mpeg4-q12, None when unknown."""
    return CODEC_PREFIXES.get(str(variant).split("-", 1)[0])


def condition_vector(codec, size):
    """The condition of one frame: the codec's one-hot slot, all zero for None (unknown)."""
    vector = torch.zeros(size)
    if codec is not None:
        if codec not in CODECS:
            raise SystemExit(f"codec must be one of {', '.join(CODECS)} or unknown, not {codec!r}")
        vector[CODECS.index(codec)] = 1.0
    return vector


def block_widths(encoder):
    """Channels of each residual block of a vae2 Encoder, in the order they run."""
    widths = []
    for stage in encoder.downs:
        widths.extend(block.conv1.in_channels for block in stage.blocks)
    widths.extend(block.conv1.in_channels for block in encoder.mid)
    return widths


def _blocks(hidden, sequence, pairs):
    for block in sequence:
        hidden = block(hidden)
        if pairs is not None:
            gamma, beta = next(pairs)
            hidden = hidden * (1 + gamma) + beta
    return hidden


def run_encoder(encoder, image, modulation=None, align=None, frames=1):
    """Encoder.forward, with x * (1 + γ) + β behind each residual block when modulated.

    `modulation` is one (γ, β) pair of shape (B or 1, C, 1, 1) per block, in
    the order of `block_widths`, or None. With `align` (a FeatureAlign),
    `image` holds `frames` frames on its channels: stem and the blocks at
    low resolution run per frame (as one batch of B * frames), the align
    fuses them onto the middle frame, and the rest runs once.
    """
    pairs = None if modulation is None else iter(modulation)
    stages = list(encoder.downs)
    first = stages[0]
    if align is None:
        hidden = _blocks(encoder.stem(image), first.blocks, pairs)
    else:
        batch, _channels, height, width = image.shape
        images = [image[:, 3 * index:3 * index + 3] for index in range(frames)]
        stacked = torch.cat(images, dim=0)
        front = None
        if modulation is not None:
            front = iter([
                (_per_frame(gamma, frames), _per_frame(beta, frames))
                for gamma, beta in (next(pairs) for _ in range(len(first.blocks)))
            ])
        hidden = _blocks(encoder.stem(stacked), first.blocks, front)
        hidden = align(images, list(hidden.split(batch, dim=0)))
    hidden = first.down(hidden)
    for stage in stages[1:]:
        hidden = stage.down(_blocks(hidden, stage.blocks, pairs))
    hidden = _blocks(hidden, encoder.mid, pairs)
    return encoder.to_stats(encoder.act(hidden))


def _per_frame(value, frames):
    """A (B, C, 1, 1) modulation for the frame-major batch of B * frames; (1, ...) broadcasts."""
    if value.shape[0] == 1:
        return value
    return value.repeat(frames, 1, 1, 1)


class FeatureAlign(nn.Module):
    """Neighbor features warped onto the middle frame by a frozen SPyNet, then fused.

    `images` are the (padded) low frames, `features` their features at the same
    size, both lists in frame order. Returns the middle features plus the
    fusion's correction.
    """

    def __init__(self, width, frames, activation=None):
        super().__init__()
        self.frames = frames
        self.spynet = SPyNet()
        for parameter in self.spynet.parameters():
            parameter.requires_grad_(False)
        neighbors = frames - 1
        self.inner = nn.Conv2d(width * frames + neighbors, width, kernel_size=3, padding=1)
        self.act = activation_fn(activation)
        self.outer = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def train(self, mode=True):
        super().train(mode)
        self.spynet.eval()
        return self

    def flows(self, images):
        """Flow (N, H, W, 2) from the middle frame to each neighbor, neighbors batch-major, no gradient."""
        middle = self.frames // 2
        center = images[middle]
        others = [images[index] for index in range(self.frames) if index != middle]
        with torch.no_grad():
            supp = torch.cat(others, dim=0).float()
            ref = center.float().repeat(len(others), 1, 1, 1)
            flow = self.spynet(ref, supp).permute(0, 2, 3, 1)
            error = (flow_warp(supp, flow) - ref).abs().mean(dim=1, keepdim=True)
        return flow, error

    def forward(self, images, features):
        middle = self.frames // 2
        batch = features[middle].shape[0]
        flow, error = self.flows(images)
        others = [features[index] for index in range(self.frames) if index != middle]
        warped = flow_warp(torch.cat(others, dim=0), flow.to(features[middle].dtype))
        error = error.to(features[middle].dtype)
        parts = [features[middle], *warped.split(batch, dim=0), *error.split(batch, dim=0)]
        return features[middle] + self.outer(self.act(self.inner(torch.cat(parts, dim=1))))


class FiLM(nn.Module):
    """Condition (B, size) -> one (γ, β) per residual block, each (B, C, 1, 1).

    Linear, activation, linear. The last layer starts at zero: γ = β = 0 for
    any condition until training moves it.
    """

    def __init__(self, size, widths, activation=None, hidden=FILM_HIDDEN):
        super().__init__()
        self.size = size
        self.widths = tuple(widths)
        self.act = activation_fn(activation)
        self.inner = nn.Linear(size, hidden)
        self.outer = nn.Linear(hidden, 2 * sum(self.widths))
        nn.init.zeros_(self.outer.weight)
        nn.init.zeros_(self.outer.bias)

    def forward(self, condition):
        out = self.outer(self.act(self.inner(condition)))
        pairs = []
        start = 0
        for width in self.widths:
            gamma = out[:, start:start + width]
            beta = out[:, start + width:start + 2 * width]
            start += 2 * width
            pairs.append((gamma[:, :, None, None], beta[:, :, None, None]))
        return pairs


class LatentUp(nn.Module):
    """Low RGB in, the normalized vae2 code at 1/4 of the padded low frame out."""

    def __init__(
        self, channels, blocks, latent, encoder_blocks=None, widths=None, activation=None, condition=None,
        in_frames=1,
    ):
        super().__init__()
        if channels < 1 or blocks < 0 or latent < 1:
            raise SystemExit("channels and latent must be positive, blocks zero or more")
        if condition is not None and condition not in CONDITIONS:
            raise SystemExit(f"condition must be one of {', '.join(CONDITIONS)}, not {condition!r}")
        if in_frames < 1 or in_frames % 2 != 1:
            raise SystemExit(f"latent_up in_frames must be odd, not {in_frames}")
        encoder_blocks, widths = latent_layout(encoder_blocks, widths)
        self.in_frames = in_frames
        self.factor = FACTOR
        self.channels = channels
        self.blocks = blocks
        self.latent = latent
        self.encoder_blocks = encoder_blocks
        self.widths = tuple(channels * multiple for multiple in widths)
        self.activation = activation or "silu"
        self.encoder = Encoder(self.widths, blocks, latent, encoder_blocks, stats=1, activation=activation)
        self.condition = condition
        self.condition_size = 0 if condition is None else CONDITIONS[condition]
        self.film = None
        if condition is not None:
            self.film = FiLM(self.condition_size, block_widths(self.encoder), activation)
        self.align = None
        if in_frames > 1:
            self.align = FeatureAlign(self.widths[0], in_frames, activation)

    def condition_batch(self, codecs, device=None):
        """(B, condition_size) for a list of codecs, None meaning unknown."""
        rows = [condition_vector(codec, self.condition_size) for codec in codecs]
        return torch.stack(rows).to(device)

    def modulation(self, condition):
        """The (γ, β) pairs for a condition batch, or None without FiLM."""
        if self.film is None:
            return None
        return self.film(condition)

    def forward(self, low, condition=None):
        """`low` (B, 3 * in_frames, H, W); `condition` (B, condition_size), None is the unknown codec."""
        if low.shape[1] != 3 * self.in_frames:
            raise SystemExit(f"latent_up reads {self.in_frames} frame(s), {3 * self.in_frames} channels, not {low.shape[1]}")
        padded, _height, _width = pad_to(low, FACTOR)
        modulation = None
        if self.film is None:
            if condition is not None:
                raise SystemExit("this latent_up has no condition")
        else:
            if condition is None:
                condition = torch.zeros(low.shape[0], self.condition_size, device=low.device, dtype=low.dtype)
            modulation = self.film(condition)
        return run_encoder(self.encoder, padded, modulation, self.align, self.in_frames)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(key.startswith("mix.") for key in state_dict):
            raise SystemExit(
                "a 3-frame latent_up with the old FrameMix (mix.*) does not load into the "
                "feature-aligned one; that code is in git before the feature alignment"
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def neighbor_usage(self):
        """Weight and gradient ratio of the fusion's last layer, as vote reports its vote.

        weight is the mean absolute kernel. grad is its mean absolute
        gradient over the same on the head. None for one frame.
        """
        if self.align is None:
            return None
        kernel = self.align.outer.weight
        stats = {"weight": float(kernel.detach().abs().mean())}
        head_grad = self.encoder.to_stats.weight.grad
        if kernel.grad is not None and head_grad is not None:
            ratio = kernel.grad.detach().abs().mean() / head_grad.detach().abs().mean().clamp_min(1e-12)
            stats["grad"] = float(ratio)
        return stats


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


def fixed_condition(model, codec):
    """The (1, condition_size) vector of one codec for an export, None without FiLM.

    `codec` is a name of CODECS, or None or "unknown" for all zero.
    """
    if codec == "unknown":
        codec = None
    if model.film is None:
        if codec is not None:
            raise SystemExit(f"this latent_up has no condition, so it takes no codec ({codec})")
        return None
    return condition_vector(codec, model.condition_size).unsqueeze(0)


class LatentUpExport(nn.Module):
    """latent_up and the vae2 decoder in one graph, the frame clamped to [0, 1], for tracing.

    The vae2 decoder reads μ = z / scale + shift. Both are per channel and sit
    right behind the head, the last convolution of latent_up, so they fold
    into its weight and bias: no extra op, and exact, since the decoder's zero
    padding sees μ in both cases. The input must already divide by FACTOR;
    the caller pads the bottom and the right as `pad_to` does and crops the
    frame back to 2h x 2w.

    With FiLM the condition vector is a second input of shape (B, condition_size):
    one-hot of CODECS, unknown is all zero, then RESERVE zeros. The FiLM MLP
    stays in the graph, so one package serves every codec, and later reserve
    slots can be filled without a new export.

    One frame only: the 3-frame model goes to the app through the MLX port
    (whoodeo.mlxport.convert).
    """

    def __init__(self, model, vae):
        super().__init__()
        import copy

        self.encoder = copy.deepcopy(model.encoder)
        self.decoder = copy.deepcopy(vae.decoder)
        if model.in_frames != 1:
            raise SystemExit(
                "the Core ML export takes a one-frame latent_up; a 3-frame one goes to the app "
                "with python -m whoodeo.mlxport.convert CHECKPOINT --app"
            )
        self.film = None if model.film is None else copy.deepcopy(model.film)
        self.condition_size = 0 if model.film is None else model.condition_size
        head = self.encoder.to_stats
        inverse = vae.latent_scale.reciprocal().to(head.weight)
        shift = vae.latent_shift.to(head.weight)
        with torch.no_grad():
            head.weight.copy_(head.weight * inverse.view(-1, 1, 1, 1))
            head.bias.copy_(head.bias * inverse + shift)

    def forward(self, frames, condition=None):
        pairs = None
        if self.film is not None:
            if condition is None:
                condition = torch.zeros(
                    frames.shape[0], self.condition_size, device=frames.device, dtype=frames.dtype,
                )
            pairs = self.film(condition)
        return self.decoder(run_encoder(self.encoder, frames, pairs)).clamp(0, 1)


class LatentUpReference(nn.Module):
    """The unfolded path training takes: code, z / scale + shift, decode, crop to 2h x 2w.

    With FiLM, `codec` fixes the condition vector (a name of CODECS, or unknown).
    """

    def __init__(self, model, vae, codec=None):
        super().__init__()
        self.model = model
        self.vae = vae
        self.condition = fixed_condition(model, codec)

    def forward(self, low):
        height, width = low.shape[-2:]
        if self.condition is None:
            code = self.model(low)
        else:
            condition = self.condition.to(low).expand(low.shape[0], -1)
            code = self.model(low, condition)
        shift = self.vae.latent_shift.view(1, -1, 1, 1)
        scale = self.vae.latent_scale.view(1, -1, 1, 1)
        image = self.vae.decode(code / scale + shift, clamp=True)
        return image[:, :, :2 * height, :2 * width]
