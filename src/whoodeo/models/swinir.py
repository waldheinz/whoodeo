# SwinIR: Image Restoration Using Swin Transformer
# Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte
# https://github.com/jingyunliang/swinir
# https://arxiv.org/abs/2108.10257
# Originally written by Ze Liu, modified by Jingyun Liang.
#
# Adapted for whoodeo. The trunk and the three 2x super-resolution heads follow
# the official network, except every 3x3 uses reflection padding instead of
# zeros. Denoising, JPEG restoration, and the large real-world model are not
# included.
#
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SwinIR at 2x, with the classical, lightweight, and real-world heads."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        return (value[0], value[1])
    return (value, value)


class DropPath(nn.Module):
    """Stochastic depth. Replaces timm's DropPath."""

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask.div_(keep)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch, height // window_size, window_size, width // window_size, window_size, channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def window_reverse(windows, window_size, height, width):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)."""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch, height // window_size, width // window_size, window_size, window_size, -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Window attention with a relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size[0] - 1
        relative[:, :, 1] += window_size[1] - 1
        relative[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(batch // windows, windows, self.num_heads, length, length)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, length, length)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ value).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, mlp_ratio, drop_path):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, _to_2tuple(self.window_size), num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

        if self.shift_size > 0:
            mask = self.calculate_mask(self.input_resolution)
        else:
            mask = None
        self.register_buffer("attn_mask", mask)
        self._mask_key = None

    def calculate_mask(self, x_size, device=None):
        height, width = x_size
        img_mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in height_slices:
            for w_slice in width_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        masks = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = masks.unsqueeze(1) - masks.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _attention_mask(self, x_size, device):
        # Only shifted windows are masked. The cached mask matches the nominal
        # resolution; any other frame size is built once and kept.
        if self.shift_size == 0:
            return None
        if x_size == self.input_resolution:
            return self.attn_mask
        if self._mask_key == (x_size, device):
            return self._runtime_mask
        mask = self.calculate_mask(x_size, device=device)
        self._runtime_mask = mask
        self._mask_key = (x_size, device)
        return mask

    def forward(self, x, x_size):
        height, width = x_size
        batch, _, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        windows = self.attn(windows, mask=self._attention_mask(x_size, x.device))
        windows = windows.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        x = shortcut + self.drop_path(x.view(batch, height * width, channels))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio, drop_path):
        super().__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (index % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[index],
            )
            for index in range(depth)
        ])

    def forward(self, x, x_size):
        for block in self.blocks:
            x = block(x, x_size)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patches_resolution = [
            _to_2tuple(img_size)[0] // _to_2tuple(patch_size)[0],
            _to_2tuple(img_size)[1] // _to_2tuple(patch_size)[1],
        ]
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


def _conv2d(in_channels, out_channels):
    """3x3. Reference SwinIR zero-pads; this one reflects.

    JingyunLiang/SwinIR leaves `padding_mode` at "zeros". On a full video
    frame those missing neighbors become the outer two pixels after the
    pixel shuffle, a pale rim. Reflection padding is the departure.
    """
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode="reflect")


class RSTB(nn.Module):
    """Residual Swin Transformer block: several Swin layers and a 3x3."""

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio, drop_path, img_size,
    ):
        super().__init__()
        self.residual_group = BasicLayer(
            dim, input_resolution, depth, num_heads, window_size, mlp_ratio, drop_path,
        )
        self.conv = _conv2d(dim, dim)
        self.patch_embed = PatchEmbed(img_size, patch_size=1, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(dim)

    def forward(self, x, x_size):
        group = self.residual_group(x, x_size)
        return self.patch_embed(self.conv(self.patch_unembed(group, x_size))) + x


class Upsample(nn.Sequential):
    """Pixel-shuffle tail used by classical SwinIR. `num_feat` is 64 there."""

    def __init__(self, scale, num_feat):
        layers = []
        if scale & (scale - 1):
            raise ValueError(f"scale {scale} is not a power of two")
        for _ in range(int(math.log(scale, 2))):
            layers.append(_conv2d(num_feat, 4 * num_feat))
            layers.append(nn.PixelShuffle(2))
        super().__init__(*layers)


class UpsampleOneStep(nn.Sequential):
    """One convolution into the pixel shuffle, used by lightweight SwinIR."""

    def __init__(self, scale, num_feat, num_out):
        super().__init__(
            _conv2d(num_feat, (scale ** 2) * num_out),
            nn.PixelShuffle(scale),
        )


class SwinIR(nn.Module):
    """2x SwinIR. `upsampler` selects the official reconstruction head.

    `pixelshuffle` is classical SR, `pixelshuffledirect` is lightweight SR,
    and `nearest+conv` is real-world SR. At 2x the real-world head takes one
    nearest-neighbor step.
    """

    def __init__(
        self, embed_dim, depths, num_heads, upsampler, *,
        window_size=8, mlp_ratio=2.0, drop_path_rate=0.1, upscale=2, img_size=64, img_range=1.0,
    ):
        super().__init__()
        if upsampler not in ("pixelshuffle", "pixelshuffledirect", "nearest+conv"):
            raise ValueError(f"unknown upsampler {upsampler}")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        for heads in num_heads:
            if embed_dim % heads != 0:
                raise ValueError(f"embed_dim {embed_dim} is not divisible by {heads} heads")

        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.img_range = img_range
        # Frozen DIV2K mean. Left out of checkpoints, same as the official net.
        mean = torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)

        self.conv_first = _conv2d(3, embed_dim)
        self.patch_embed = PatchEmbed(img_size, patch_size=1, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        resolution = tuple(self.patch_embed.patches_resolution)
        self.pos_drop = nn.Dropout(p=0.0)

        drop_path = [rate.item() for rate in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for index, depth in enumerate(depths):
            start = sum(depths[:index])
            self.layers.append(RSTB(
                dim=embed_dim,
                input_resolution=resolution,
                depth=depth,
                num_heads=num_heads[index],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[start:start + depth],
                img_size=img_size,
            ))
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.conv_after_body = _conv2d(embed_dim, embed_dim)

        num_feat = 64
        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                _conv2d(embed_dim, num_feat),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = _conv2d(num_feat, 3)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(upscale, embed_dim, 3)
        else:
            self.conv_before_upsample = nn.Sequential(
                _conv2d(embed_dim, num_feat),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_up1 = _conv2d(num_feat, num_feat)
            self.conv_hr = _conv2d(num_feat, num_feat)
            self.conv_last = _conv2d(num_feat, 3)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def check_image_size(self, x):
        _, _, height, width = x.size()
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x):
        height, width = x.shape[2:]
        x = self.check_image_size(x)
        mean = self.mean.to(dtype=x.dtype)
        x = (x - mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        if self.upsampler == "pixelshuffle":
            x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        elif self.upsampler == "pixelshuffledirect":
            x = self.upsample(x)
        else:
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))

        x = x / self.img_range + mean
        return x[:, :, :height * self.upscale, :width * self.upscale]
