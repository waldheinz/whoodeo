"""The align model in MLX: SPyNet, warp, latent_up with FiLM, vae2 decoder.

`params` is the flat dict convert.py writes (torch state-dict names, conv
kernels already in MLX layout), `config` its JSON side. The forward mirrors
spynet.AlignedReference around latent_up.LatentUpReference: warp both
neighbors onto the middle low, code, z / scale + shift, decode, clamp, crop.
"""

import json
from pathlib import Path

import mlx.core as mx

from whoodeo.mlxport.ops import (
    avg_pool2, conv2d, flow_warp, pad_reflect_br, resize_bilinear, upsample_nearest2,
)

DILATIONS = (1, 2, 4)
FACTOR = 4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _act(name):
    if name == "relu":
        return lambda x: mx.maximum(x, 0)
    return lambda x: x * mx.sigmoid(x)


class AlignModel:
    """`dtype` is the precision of latent_up and the decoder (SPyNet and the warp stay fp32).

    float16 is what the Core ML export computes in (compute_precision FLOAT16).
    """

    def __init__(self, params, config, dtype=mx.float32):
        self.dtype = dtype
        if dtype != mx.float32:
            params = {k: (v if k.startswith("spynet.") else v.astype(dtype)) for k, v in params.items()}
        self.p = params
        self.c = config
        self.act = _act(config["activation"])
        self.in_frames = config["in_frames"]
        self.condition_size = config["condition_size"]
        self.mean = mx.array(MEAN, dtype=mx.float32)
        self.std = mx.array(STD, dtype=mx.float32)

    @classmethod
    def load(cls, path, dtype=mx.float32):
        path = Path(path)
        if path.suffix == ".mlxmodel":
            weights, meta = path / "model.safetensors", path / "config.json"
        else:
            weights, meta = path.with_suffix(".safetensors"), path.with_suffix(".json")
        params = mx.load(str(weights))
        config = json.loads(meta.read_text())
        return cls(params, config, dtype)

    def conv(self, name, x, stride=1, padding=1, dilation=1):
        return conv2d(x, self.p[name + ".weight"], self.p.get(name + ".bias"), stride, padding, dilation)

    # SPyNet -----------------------------------------------------------------

    def spynet_block(self, level, x):
        for index in range(5):
            x = self.conv(f"spynet.basic_module.{level}.basic_module.{index}.conv", x, padding=3)
            if index < 4:
                x = mx.maximum(x, 0)
        return x

    def compute_flow(self, ref, supp):
        n, height, width, _c = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _level in range(5):
            ref.append(avg_pool2(ref[-1]))
            supp.append(avg_pool2(supp[-1]))
        ref = ref[::-1]
        supp = supp[::-1]
        flow = mx.zeros((n, height // 32, width // 32, 2), dtype=mx.float32)
        for level in range(6):
            if level == 0:
                flow_up = flow
            else:
                h, w = flow.shape[1:3]
                flow_up = resize_bilinear(flow, 2 * h, 2 * w, align_corners=True) * 2.0
            x = mx.concatenate([ref[level], flow_warp(supp[level], flow_up), flow_up], axis=-1)
            flow = flow_up + self.spynet_block(level, x)
        return flow

    def spynet(self, ref, supp):
        """Flow (N, H, W, 2) from ref to supp, any size, as spynet.SPyNet.forward."""
        height, width = ref.shape[1:3]
        h_up = ((height + 31) // 32) * 32
        w_up = ((width + 31) // 32) * 32
        ref_up = resize_bilinear(ref, h_up, w_up, align_corners=False)
        supp_up = resize_bilinear(supp, h_up, w_up, align_corners=False)
        flow = resize_bilinear(self.compute_flow(ref_up, supp_up), height, width, align_corners=False)
        scale = mx.array([width / w_up, height / h_up], dtype=mx.float32)
        return flow * scale

    def align(self, low):
        """`low` (B, H, W, 3 * F) -> same shape, neighbors warped onto the middle and clamped."""
        frames = self.in_frames
        if frames < 3:
            return low, None
        b = low.shape[0]
        parts = [low[..., 3 * i:3 * i + 3] for i in range(frames)]
        mid = frames // 2
        others = [i for i in range(frames) if i != mid]
        neigh = mx.concatenate([parts[i] for i in others], axis=0)
        ref = mx.concatenate([parts[mid]] * len(others), axis=0)
        flow = self.spynet(ref, neigh)
        warped = mx.clip(flow_warp(neigh, flow), 0, 1)
        for k, i in enumerate(others):
            parts[i] = warped[k * b:(k + 1) * b]
        return mx.concatenate(parts, axis=-1), flow

    # latent_up --------------------------------------------------------------

    def film(self, condition):
        """(B, size) -> list of (γ, β), each (B, 1, 1, C), in block order."""
        p = self.p
        hidden = self.act(condition @ p["film.inner.weight"].T + p["film.inner.bias"])
        out = hidden @ p["film.outer.weight"].T + p["film.outer.bias"]
        pairs, start = [], 0
        for width in self.c["block_widths"]:
            gamma = out[:, start:start + width][:, None, None, :]
            beta = out[:, start + width:start + 2 * width][:, None, None, :]
            pairs.append((gamma, beta))
            start += 2 * width
        return pairs

    def block(self, name, x, dilation=1):
        h = self.conv(name + ".conv1", self.act(x), padding=dilation, dilation=dilation)
        return x + self.conv(name + ".conv2", self.act(h))

    def blocks(self, prefix, count, x, pairs, dilations=(1,)):
        for index in range(count):
            x = self.block(f"{prefix}.{index}", x, dilations[index % len(dilations)])
            if pairs is not None:
                gamma, beta = next(pairs)
                x = x * (1 + gamma) + beta
        return x

    def front(self, image, pairs):
        x = self.conv("encoder.stem", image)
        for level, count in enumerate(self.c["encoder_blocks"]):
            x = self.blocks(f"encoder.downs.{level}.blocks", count, x, pairs)
            x = self.conv(f"encoder.downs.{level}.down", x, stride=2)
        return x

    def back(self, x, pairs):
        x = self.blocks("encoder.mid", self.c["blocks"], x, pairs, DILATIONS)
        return self.conv("encoder.to_stats", self.act(x))

    def code(self, padded, condition):
        """Normalized code z of `padded` (B, H, W, 3 * F), H and W multiples of 4."""
        pairs = self.film(condition) if "film.inner.weight" in self.p else None
        count = sum(self.c["encoder_blocks"])
        front_pairs = None if pairs is None else iter(pairs[:count])
        back_pairs = None if pairs is None else iter(pairs[count:])
        frames = self.in_frames
        b = padded.shape[0]
        stacked = mx.concatenate([padded[..., 3 * i:3 * i + 3] for i in range(frames)], axis=0)
        if front_pairs is not None:
            # (B, ...) modulation repeated per frame for the frame-major batch.
            front_list = [(mx.concatenate([g] * frames, 0) if g.shape[0] > 1 else g,
                           mx.concatenate([be] * frames, 0) if be.shape[0] > 1 else be)
                          for g, be in front_pairs]
            front_pairs = iter(front_list)
        hidden = self.front(stacked, front_pairs)
        if frames == 1:
            return self.back(hidden, back_pairs)
        feats = [hidden[i * b:(i + 1) * b] for i in range(frames)]
        mixed = self.conv("mix.inner", mx.concatenate(feats, axis=-1), padding=0)
        mixed = self.conv("mix.outer", self.act(mixed), padding=0)
        return self.back(feats[frames // 2] + mixed, back_pairs)

    # vae2 decoder -----------------------------------------------------------

    def decode(self, latent):
        c = self.c
        x = self.conv("decoder.from_latent", latent)
        x = self.blocks("decoder.mid", c["vae_blocks"], x, None, DILATIONS)
        for index, count in enumerate(c["decoder_blocks_run"]):
            x = upsample_nearest2(x)
            x = self.conv(f"decoder.ups.{index}.mix", x)
            x = self.blocks(f"decoder.ups.{index}.blocks", count, x, None)
        return self.conv("decoder.to_rgb", self.act(x))

    # whole path -------------------------------------------------------------

    def __call__(self, low, condition=None, stages=False):
        """`low` (B, H, W, 3 * F) in [0, 1] -> (B, 2H, 2W, 3) in [0, 1].

        `condition` (B, condition_size) or None for the unknown codec.
        With `stages` also returns a dict of the intermediate results.
        """
        b, height, width, _c = low.shape
        aligned, flow = self.align(low)
        if condition is None:
            condition = mx.zeros((b, self.condition_size), dtype=mx.float32)
        z = self.code(pad_reflect_br(aligned, FACTOR).astype(self.dtype), condition.astype(self.dtype))
        latent = z / self.p["vae.latent_scale"] + self.p["vae.latent_shift"]
        image = mx.clip(self.decode(latent), 0, 1)[:, :2 * height, :2 * width].astype(mx.float32)
        if stages:
            return image, {"flow": flow, "aligned": aligned, "code": z}
        return image
