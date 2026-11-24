"""Frozen SPyNet alignment of neighbor low frames onto the middle frame.

The optical flow is computed on the low-resolution inputs. Neighbor frames
are warped into the middle frame's coordinates before the generator sees
them. The generator architecture is unchanged. Weights stay frozen.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp `x` (N,C,H,W) by `flow` (N,H,W,2) in pixel offsets.

    Official SPyNet uses bilinear sampling, border padding, align_corners=True.
    Those are literals here so TorchScript keeps them as strings.
    """
    height = x.size(2)
    width = x.size(3)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=x.dtype),
        torch.arange(width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)
    grid_flow = grid + flow
    # denom stays in the graph so a Core ML export is not locked to one size.
    denom_w = (grid_flow[:, :, :, 0] * 0 + (width - 1)).clamp(min=1)
    denom_h = (grid_flow[:, :, :, 1] * 0 + (height - 1)).clamp(min=1)
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / denom_w - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / denom_h - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(
        x,
        grid_flow,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class _ConvReLU(nn.Module):
    """mmcv ConvModule stand-in: `conv` plus optional ReLU, matching SPyNet keys."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True,
        )
        self.activate = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvReLU(8, 32, 7, 1, 3, relu=True),
            _ConvReLU(32, 64, 7, 1, 3, relu=True),
            _ConvReLU(64, 32, 7, 1, 3, relu=True),
            _ConvReLU(32, 16, 7, 1, 3, relu=True),
            _ConvReLU(16, 2, 7, 1, 3, relu=False),
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    """Optical flow from ref to supp, as in BasicVSR / FTVSR."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, height, width = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _level in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]
        flow = ref[0].new_zeros(n, 2, height // 32, width // 32)
        # enumerate so TorchScript can index the ModuleList.
        for level, block in enumerate(self.basic_module):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + block(
                torch.cat(
                    [
                        ref[level],
                        flow_warp(supp[level], flow_up.permute(0, 2, 3, 1)),
                        flow_up,
                    ],
                    1,
                )
            )
        return flow

    def forward(self, ref, supp):
        height, width = ref.size(2), ref.size(3)
        # Same as "next multiple of 32", one path, friendlier for Core ML.
        h_up = ((height + 31) // 32) * 32
        w_up = ((width + 31) // 32) * 32
        ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref_up, supp_up), size=(height, width), mode="bilinear", align_corners=False)
        # Same numbers as float(width)/float(w_up), kept as tensor math for export.
        scale_w = (width + flow[:, 0:1] * 0) / (w_up + flow[:, 0:1] * 0)
        scale_h = (height + flow[:, 1:2] * 0) / (h_up + flow[:, 1:2] * 0)
        return torch.cat((flow[:, 0:1] * scale_w, flow[:, 1:2] * scale_h), 1)


def load_spynet(weights, device):
    """Load official SPyNet weights, freeze, eval, move to device."""
    path = Path(weights)
    if not path.is_file():
        raise SystemExit(f"align: no SPyNet weights at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if any(key.startswith("generator.spynet.") for key in state):
        state = {
            key[len("generator.spynet."):]: value
            for key, value in state.items()
            if key.startswith("generator.spynet.")
        }
    model = SPyNet()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if key not in {"mean", "std"}]
    if missing or unexpected:
        raise SystemExit(
            f"align: SPyNet weights at {path} do not match: "
            f"missing {missing[:8] or 'none'}, unexpected {unexpected[:8] or 'none'}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model.to(device)


@torch.no_grad()
def align_neighbors(spynet, low, in_frames):
    """Warp non-center frames of `low` (B, 3*in_frames, H, W) onto the middle frame."""
    if in_frames < 3:
        return low
    batch, channels, height, width = low.shape
    if channels != 3 * in_frames:
        raise ValueError(f"expected {3 * in_frames} channels for {in_frames} frames, got {channels}")
    parts = low.reshape(batch, in_frames, 3, height, width)
    center = parts[:, in_frames // 2]
    aligned = [center if index == in_frames // 2 else None for index in range(in_frames)]
    # One SPyNet batch for all neighbors: (B*(F-1), 3, H, W)
    others = [index for index in range(in_frames) if index != in_frames // 2]
    neigh = torch.cat([parts[:, index] for index in others], dim=0)
    ref = center.repeat(len(others), 1, 1, 1)
    flow = spynet(ref, neigh)
    warped = flow_warp(neigh, flow.permute(0, 2, 3, 1))
    warped = warped.clamp(0, 1)
    chunks = warped.split(batch, dim=0)
    for index, frame in zip(others, chunks):
        aligned[index] = frame
    # Clone so the result is not an inference tensor; scene_cut writes in place.
    return torch.cat(aligned, dim=1).clone()


def _run_config(path):
    """config.yaml of the run that holds this checkpoint, or None."""
    path = Path(path)
    for parent in [path, *path.parents]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
        if parent.name == "checkpoints":
            candidate = parent.parent / "config.yaml"
            if candidate.is_file():
                return candidate
    return None


def resolve_align_weights(written):
    """The SPyNet file, remapping a Spark path onto the Mac when needed."""
    if written is None:
        return None
    path = Path(str(written)).expanduser()
    if path.is_file():
        return path
    name = path.name
    roots = [
        Path(__file__).resolve().parents[2] / "weights",
        Path.home() / "ftvsr" / "weights",
        Path("/Users/trem/Arbeitsplatz/Meins/whoodeo/weights"),
        Path("/home/trem/ftvsr/weights"),
    ]
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return path


def detect_align(path):
    """Return (method, weights_path) from the run's config.yaml, or (None, None).

    Alignment is not part of the generator architecture, so a checkpoint
    alone does not record it. The training config next to the run does.
    """
    import yaml

    config = _run_config(path)
    if config is None:
        return None, None
    try:
        data = yaml.safe_load(config.read_text()) or {}
    except yaml.YAMLError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    raw = data.get("align")
    method = None
    written = data.get("align_weights")
    if isinstance(raw, str):
        method = raw.strip() or None
    elif isinstance(raw, dict):
        value = raw.get("method")
        method = value.strip() if isinstance(value, str) else None
        if written is None:
            written = raw.get("weights")
    if method != "spynet":
        return None, None
    weights = resolve_align_weights(written)
    if weights is None or not Path(weights).is_file():
        raise SystemExit(
            f"{config}: align is spynet, but the SPyNet weights were not found"
            + (f" ({written})" if written else "")
        )
    return method, Path(weights)


class AlignedReference(nn.Module):
    """Warp neighbors onto the middle low frame, then run the usual upscaler."""

    def __init__(self, inner, spynet, in_frames):
        super().__init__()
        self.inner = inner
        self.spynet = spynet
        self.in_frames = in_frames

    def forward(self, low):
        return self.inner(align_neighbors(self.spynet, low, self.in_frames))
