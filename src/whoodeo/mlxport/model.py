"""The 3-frame latent_up in MLX: feature alignment (SPyNet, warp, fusion), FiLM, vae2 decoder.

`params` is the flat dict convert.py writes (torch state-dict names, conv
kernels already in MLX layout), `config` its JSON side. The forward mirrors
latent_up.LatentUpReference: pad, stem and low-resolution blocks per frame,
SPyNet flow middle -> each neighbor, warp the neighbor features, fuse, the
rest of the encoder once, z / scale + shift, decode, clamp, crop.
"""

import json
from pathlib import Path

import mlx.core as mx

from whoodeo.mlxport.ops import (
    avg_pool2, conv2d, flow_warp, pad_reflect_br, resize_bilinear, upsample_nearest2,
)

DILATIONS = (1, 2, 4)
# State-dict prefix of the frozen SPyNet inside latent_up's FeatureAlign.
SPYNET = "align.spynet."
FACTOR = 4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _act(name):
    if name == "relu":
        return lambda x: mx.maximum(x, 0)
    return lambda x: x * mx.sigmoid(x)


class AlignModel:
    """`dtype` is the precision of latent_up and the decoder (SPyNet and its flow stay fp32).

    float16 is what the Core ML export computes in (compute_precision FLOAT16).
    """

    def __init__(self, params, config, dtype=mx.float32):
        self.dtype = dtype
        if dtype != mx.float32:
            params = {k: (v if k.startswith(SPYNET) else v.astype(dtype)) for k, v in params.items()}
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
            x = self.conv(f"{SPYNET}basic_module.{level}.basic_module.{index}.conv", x, padding=3)
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

    def fuse(self, images, feats, stages=None):
        """FeatureAlign: neighbor features warped onto the middle one, fused, added to it.

        `images` the padded low frames in fp32, `feats` their features, frame order.
        """
        frames = self.in_frames
        mid = frames // 2
        b = feats[mid].shape[0]
        others = [i for i in range(frames) if i != mid]
        supp = mx.concatenate([images[i] for i in others], axis=0)
        ref = mx.concatenate([images[mid]] * len(others), axis=0)
        flow = self.spynet(ref, supp)
        error = mx.abs(flow_warp(supp, flow) - ref).mean(axis=-1, keepdims=True)
        warped = flow_warp(mx.concatenate([feats[i] for i in others], axis=0), flow).astype(self.dtype)
        error = error.astype(self.dtype)
        parts = [feats[mid]]
        parts += [warped[k * b:(k + 1) * b] for k in range(len(others))]
        parts += [error[k * b:(k + 1) * b] for k in range(len(others))]
        hidden = self.conv("align.inner", mx.concatenate(parts, axis=-1))
        fused = feats[mid] + self.conv("align.outer", self.act(hidden))
        if stages is not None:
            stages.update(flow=flow, warped=warped, error=error, fused=fused)
        return fused

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

    def code(self, padded, condition, images=None, stages=None):
        """Normalized code z of `padded` (B, H, W, 3 * F), H and W multiples of 4.

        `images` the same frames in fp32 for SPyNet, `padded` when left out.
        """
        pairs = iter(self.film(condition)) if "film.inner.weight" in self.p else None
        frames = self.in_frames
        first = self.c["encoder_blocks"][0]
        front = [next(pairs) for _ in range(first)] if pairs is not None else None
        # Stem and the low-resolution blocks one frame at a time: as one batch they held
        # far more memory at 720p, for the same result.
        feats = [
            self.blocks("encoder.downs.0.blocks", first,
                        self.conv("encoder.stem", padded[..., 3 * i:3 * i + 3]),
                        iter(front) if front is not None else None)
            for i in range(frames)
        ]
        x = feats[0]
        if frames > 1:
            source = padded if images is None else images
            images = [source[..., 3 * i:3 * i + 3].astype(mx.float32) for i in range(frames)]
            x = self.fuse(images, feats, stages)
        x = self.conv("encoder.downs.0.down", x, stride=2)
        for level, count in enumerate(self.c["encoder_blocks"][1:], start=1):
            x = self.blocks(f"encoder.downs.{level}.blocks", count, x, pairs)
            x = self.conv(f"encoder.downs.{level}.down", x, stride=2)
        x = self.blocks("encoder.mid", self.c["blocks"], x, pairs, DILATIONS)
        return self.conv("encoder.to_stats", self.act(x))

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
        if condition is None:
            condition = mx.zeros((b, self.condition_size), dtype=mx.float32)
        padded = pad_reflect_br(low, FACTOR)
        found = {} if stages else None
        z = self.code(padded.astype(self.dtype), condition.astype(self.dtype), padded, found)
        latent = z / self.p["vae.latent_scale"] + self.p["vae.latent_shift"]
        image = mx.clip(self.decode(latent), 0, 1)[:, :2 * height, :2 * width].astype(mx.float32)
        if stages:
            found["code"] = z
            return image, found
        return image
