"""Named networks shared by train and apply.

`modified` is a residual ESPCN. `deform` keeps that stack and aligns the
neighbor frames with a deformable 3x3. `shift` aligns them with a dense
integer search. `pyramid` aligns them with a coarse-to-fine flow. `espcn`
and `edsr` are the other architectures. The training config sets depth,
width, and input frames.
"""

from whoodeo.models.deform import DeformESPCN
from whoodeo.models.edsr import EDSR
from whoodeo.models.espcn import ESPCN
from whoodeo.models.modified import ModifiedESPCN
from whoodeo.models.pyramid import PyramidESPCN
from whoodeo.models.shift import ShiftESPCN

PRESETS = {
    "modified": {"blocks": 32, "channels": 64, "frames": None},
    "deform": {"blocks": 32, "channels": 64, "frames": None},
    "shift": {"blocks": 8, "channels": 64, "frames": None},
    "pyramid": {"blocks": 8, "channels": 64, "frames": None},
    "espcn": {"blocks": None, "channels": None, "frames": 1},
    "edsr": {"blocks": 32, "channels": 256, "frames": 1},
}

def build_model(
    arch, blocks=None, channels=None, in_frames=5, *,
    radius=None, stem=None, sharpness=None, reject=None, levels=None,
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
    else:
        label = f"{arch} {depth}x{width} in_frames {in_frames} parameters {count}"
    return model, label


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
