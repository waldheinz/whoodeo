"""SPyNet, the optical flow network of BasicVSR / FTVSR, and the warp it uses.

latent_up with in_frames 3 holds one inside its FeatureAlign, frozen, to
warp the neighbor features onto the middle frame (see models/latent_up.py).
`load_weights` fills it with the official weights when a run starts fresh;
after that they travel in the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x, flow):
    """Warp `x` (N, C, H, W) by `flow` (N, H, W, 2) in pixel offsets (dx, dy).

    Bilinear sampling, border padding, align_corners=True, as official SPyNet.
    """
    _n, _c, height, width = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=x.dtype),
        torch.arange(width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2) + flow
    grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(
        x, torch.stack((grid_x, grid_y), dim=3),
        mode="bilinear", padding_mode="border", align_corners=True,
    )


class _ConvReLU(nn.Module):
    """mmcv ConvModule stand-in: `conv` plus optional ReLU, matching SPyNet keys."""

    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.relu else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvReLU(8, 32),
            _ConvReLU(32, 64),
            _ConvReLU(64, 32),
            _ConvReLU(32, 16),
            _ConvReLU(16, 2, relu=False),
        )

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    """Optical flow from ref to supp (N, 2, H, W) in pixels, six levels coarse to fine."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def compute_flow(self, ref, supp):
        n, _, height, width = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _level in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]
        flow = ref[0].new_zeros(n, 2, height // 32, width // 32)
        for level, block in enumerate(self.basic_module):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(supp[level], flow_up.permute(0, 2, 3, 1))
            flow = flow_up + block(torch.cat([ref[level], warped, flow_up], 1))
        return flow

    def forward(self, ref, supp):
        """Any size: resized to the next multiple of 32 and back, the flow scaled to match."""
        height, width = ref.shape[2:]
        h_up = ((height + 31) // 32) * 32
        w_up = ((width + 31) // 32) * 32
        ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(
            self.compute_flow(ref_up, supp_up), size=(height, width), mode="bilinear", align_corners=False,
        )
        return torch.cat((flow[:, 0:1] * (width / w_up), flow[:, 1:2] * (height / h_up)), 1)


def load_weights(spynet, weights):
    """The official SPyNet weights (or FTVSR's generator.spynet.*) into `spynet`."""
    path = Path(weights)
    if not path.is_file():
        raise SystemExit(f"no SPyNet weights at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    prefix = "generator.spynet."
    if any(key.startswith(prefix) for key in state):
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    state = {key: value for key, value in state.items() if key not in ("mean", "std")}
    missing, unexpected = spynet.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"SPyNet weights at {path} do not match: "
            f"missing {missing[:8] or 'none'}, unexpected {unexpected[:8] or 'none'}"
        )
