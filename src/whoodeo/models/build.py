"""Named networks shared by train and apply.

`vae2` reconstructs one frame through a code of `latent` channels, with
tapered widths and no normalization. `latent_up` maps three low frames to the
code of a frozen vae2, whose decoder gives the 2x picture; `condition: codec`
adds FiLM on the codec of the low frame. The training config sets depth,
width, and the layout.
"""

from whoodeo.models.latent_up import LatentUp, latent_layout
from whoodeo.models.vae2 import VAE2, layout_text

PRESETS = {
    "vae2": {"blocks": 6, "channels": 64},
    "latent_up": {"blocks": 12, "channels": 64},
}


def _activation_text(activation):
    return "" if activation in (None, "silu") else f" activation {activation}"


def build_model(
    arch, blocks=None, channels=None, in_frames=3, *,
    latent=None, factor=None, encoder_blocks=None, decoder_blocks=None, widths=None,
    activation=None, condition=None,
):
    if arch not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise SystemExit(f"unknown arch {arch}. choices: {known}")
    preset = PRESETS[arch]
    depth = preset["blocks"] if blocks is None else blocks
    width = preset["channels"] if channels is None else channels
    if depth < 1 or width < 1:
        raise SystemExit("blocks and channels must be positive")
    if isinstance(latent, bool) or not isinstance(latent, int) or latent < 1:
        raise SystemExit("latent must be a positive integer")
    if arch == "latent_up":
        model = LatentUp(width, depth, latent, encoder_blocks, widths, activation, condition, in_frames)
        count = sum(p.numel() for p in model.parameters())
        chosen_blocks, chosen_widths = latent_layout(encoder_blocks, widths)
        label = (
            f"latent_up {depth}x{width} latent {latent} "
            f"encoder_blocks {'/'.join(str(item) for item in chosen_blocks)} "
            f"widths {'/'.join(str(item) for item in chosen_widths)}"
            f" in_frames {in_frames}"
            f"{_activation_text(activation)}"
            f"{'' if condition is None else f' condition {condition}'} parameters {count}"
        )
        return model, label
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2 or factor > 16 or factor & (factor - 1):
        raise SystemExit("factor must be a power of two from 2 to 16")
    model = VAE2(width, depth, latent, factor, encoder_blocks, decoder_blocks, widths, activation)
    count = sum(p.numel() for p in model.parameters())
    levels = layout_text(factor, encoder_blocks, decoder_blocks, widths)
    label = (
        f"vae2 {depth}x{width} latent {latent} factor {factor}{levels}"
        f"{_activation_text(activation)} parameters {count}"
    )
    return model, label
