"""Named networks shared by train and apply.

`modified` is a residual ESPCN. Its default, 32 blocks and 64 channels, is about
2.4 million parameters. `deform` keeps that stack and aligns the neighbor frames
with a deformable 3x3. `espcn` and `edsr` are the other architectures already
in the repo. `--blocks` and `--channels` override the preset for modified, deform, and edsr.
"""

from whoodeo.model import DeformESPCN, ESPCN, ModifiedESPCN
from whoodeo.model_edsr import EDSR

PRESETS = {
    "modified": {"blocks": 32, "channels": 64, "frames": None},
    "deform": {"blocks": 32, "channels": 64, "frames": None},
    "espcn": {"blocks": None, "channels": None, "frames": 1},
    "edsr": {"blocks": 32, "channels": 256, "frames": 1},
}


def add_model_args(parser):
    parser.add_argument("--arch", default="modified", choices=sorted(PRESETS))
    parser.add_argument("--blocks", type=int, default=None, help="override preset depth")
    parser.add_argument("--channels", type=int, default=None, help="override preset width")
    parser.add_argument("--in-frames", type=int, default=5)


def build_model(arch, blocks=None, channels=None, in_frames=5):
    if arch not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise SystemExit(f"unknown arch {arch}. choices: {known}")
    preset = PRESETS[arch]
    fixed_frames = preset["frames"]
    if fixed_frames is not None and in_frames != fixed_frames:
        raise SystemExit(f"{arch} takes {fixed_frames} input frame, not {in_frames}")
    if arch == "espcn":
        if blocks is not None or channels is not None:
            raise SystemExit("espcn has a fixed size; --blocks and --channels apply to modified, deform, and edsr")
        model = ESPCN()
        label = f"espcn in_frames 1 parameters {sum(p.numel() for p in model.parameters())}"
        return model, label

    depth = preset["blocks"] if blocks is None else blocks
    width = preset["channels"] if channels is None else channels
    if depth < 1 or width < 1:
        raise SystemExit("--blocks and --channels must be positive")
    if arch == "modified":
        model = ModifiedESPCN(num_res_blocks=depth, num_filters=width, in_frames=in_frames)
    elif arch == "deform":
        model = DeformESPCN(num_res_blocks=depth, num_filters=width, in_frames=in_frames)
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
    label = f"{arch} {depth}x{width} in_frames {in_frames} parameters {count}"
    return model, label
