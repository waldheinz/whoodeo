"""Named networks shared by train and apply.

`modified` is a residual ESPCN. `deform` keeps that stack and aligns the
neighbor frames with a deformable 3x3. `shift` aligns them with a dense
integer search. `pyramid` aligns them with a coarse-to-fine flow. `vq`
keeps the residual stack, reads one frame, and quantizes the correction.
`espcn` and `edsr` are the other architectures. `swinir`, `swinir_light`,
and `swinir_real` are SwinIR at 2x: one shared trunk, three reconstruction
heads. The training config sets depth, width, and input frames.
"""

from whoodeo.models.deform import DeformESPCN
from whoodeo.models.edsr import EDSR
from whoodeo.models.espcn import ESPCN
from whoodeo.models.modified import ModifiedESPCN
from whoodeo.models.pyramid import PyramidESPCN
from whoodeo.models.shift import ShiftESPCN
from whoodeo.models.swinir import SwinIR
from whoodeo.models.vq import CODES, PATCH, VQESPCN

# Six Swin layers per residual block, six heads. Channels must divide by the heads.
_SWINIR_LAYERS = 6
_SWINIR_HEADS = 6
_SWINIR_UPSAMPLER = {
    "swinir": "pixelshuffle",
    "swinir_light": "pixelshuffledirect",
    "swinir_real": "nearest+conv",
}

PRESETS = {
    "modified": {"blocks": 32, "channels": 64, "frames": None},
    "deform": {"blocks": 32, "channels": 64, "frames": None},
    "shift": {"blocks": 8, "channels": 64, "frames": None},
    "pyramid": {"blocks": 8, "channels": 64, "frames": None},
    "vq": {"blocks": 8, "channels": 64, "frames": 1},
    "espcn": {"blocks": None, "channels": None, "frames": 1},
    "edsr": {"blocks": 32, "channels": 256, "frames": 1},
    "swinir": {"blocks": 6, "channels": 180, "frames": 1},
    "swinir_light": {"blocks": 4, "channels": 60, "frames": 1},
    "swinir_real": {"blocks": 6, "channels": 180, "frames": 1},
}

def build_model(
    arch, blocks=None, channels=None, in_frames=5, *,
    radius=None, stem=None, sharpness=None, reject=None, levels=None,
    bilinear=None, codes=None, patch=None,
):
    if arch not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise SystemExit(f"unknown arch {arch}. choices: {known}")
    preset = PRESETS[arch]
    fixed_frames = preset["frames"]
    if fixed_frames is not None and in_frames != fixed_frames:
        raise SystemExit(f"{arch} takes {fixed_frames} input frame, not {in_frames}")
    if arch == "espcn":
        if blocks is not None or channels is not None:
            raise SystemExit("espcn has a fixed size")
        model = ESPCN()
        label = f"espcn in_frames 1 parameters {sum(p.numel() for p in model.parameters())}"
        return model, label

    depth = preset["blocks"] if blocks is None else blocks
    width = preset["channels"] if channels is None else channels
    if depth < 1 or width < 1:
        raise SystemExit("blocks and channels must be positive")
    if arch == "modified":
        model = ModifiedESPCN(num_res_blocks=depth, num_filters=width, in_frames=in_frames)
    elif arch == "deform":
        model = DeformESPCN(num_res_blocks=depth, num_filters=width, in_frames=in_frames)
    elif arch == "shift":
        _check_shift(radius, stem, sharpness, reject)
        model = ShiftESPCN(width, depth, in_frames, radius, stem, sharpness, reject)
    elif arch == "pyramid":
        _check_pyramid(levels, stem, radius)
        model = PyramidESPCN(width, depth, in_frames, levels, stem, radius)
    elif arch == "vq":
        if bilinear is None:
            bilinear = True
        if codes is None:
            codes = CODES
        if patch is None:
            patch = PATCH
        _check_vq(codes, patch, bilinear)
        model = VQESPCN(
            num_res_blocks=depth, num_filters=width, bilinear=bilinear, codes=codes, patch=patch,
        )
    elif arch in _SWINIR_UPSAMPLER:
        if width % _SWINIR_HEADS != 0:
            raise SystemExit(f"{arch} channels must be divisible by {_SWINIR_HEADS}")
        model = SwinIR(
            embed_dim=width,
            depths=[_SWINIR_LAYERS] * depth,
            num_heads=[_SWINIR_HEADS] * depth,
            upsampler=_SWINIR_UPSAMPLER[arch],
        )
    else:
        model = EDSR({
            "n_resblocks": depth,
            "n_feats": width,
            "res_scale": 0.1,
            "scale": 2,
            "rgb_range": 1,
            "n_colors": 3,
        })
    count = sum(p.numel() for p in model.parameters())
    if arch == "shift":
        label = (
            f"shift {depth}x{width} in_frames {in_frames} "
            f"radius {radius} stem {stem} sharpness {sharpness:g} "
            f"reject {str(reject).lower()} parameters {count}"
        )
    elif arch == "pyramid":
        label = (
            f"pyramid {depth}x{width} in_frames {in_frames} "
            f"levels {levels} stem {stem} radius {radius} parameters {count}"
        )
    elif arch == "vq":
        label = (
            f"vq {depth}x{width} in_frames {in_frames} "
            f"codes {codes} patch {patch} bilinear {str(bilinear).lower()} "
            f"parameters {count}"
        )
    else:
        label = f"{arch} {depth}x{width} in_frames {in_frames} parameters {count}"
    return model, label


def _check_vq(codes, patch, bilinear):
    if isinstance(codes, bool) or not isinstance(codes, int) or codes < 1 or codes > 65536:
        raise SystemExit("codes must be an integer from 1 to 65536")
    if isinstance(patch, bool) or not isinstance(patch, int) or patch < 1 or patch > 16:
        raise SystemExit("patch must be an integer from 1 to 16")
    if not isinstance(bilinear, bool):
        raise SystemExit("bilinear must be true or false")


def _check_shift(radius, stem, sharpness, reject):
    if radius is None or stem is None or sharpness is None or reject is None:
        raise SystemExit("arch shift needs radius, stem, sharpness, and reject")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0 or radius > 16:
        raise SystemExit("radius must be an integer from 0 to 16")
    if isinstance(stem, bool) or not isinstance(stem, int) or stem < 0:
        raise SystemExit("stem must be an integer >= 0")
    if isinstance(sharpness, bool) or not isinstance(sharpness, (int, float)) or float(sharpness) <= 0:
        raise SystemExit("sharpness must be a positive number")
    if not isinstance(reject, bool):
        raise SystemExit("reject must be true or false")


def _check_pyramid(levels, stem, radius):
    if levels is None or stem is None or radius is None:
        raise SystemExit("arch pyramid needs levels, stem, and radius")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1 or levels > 6:
        raise SystemExit("levels must be an integer from 1 to 6")
    if isinstance(stem, bool) or not isinstance(stem, int) or stem < 0:
        raise SystemExit("stem must be an integer >= 0")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0 or radius > 8:
        raise SystemExit("radius must be an integer from 0 to 8")
