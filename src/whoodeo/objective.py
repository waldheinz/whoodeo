"""Pixel, VGG, color, Oklab, MS-SSIM, and GAN terms. The loss is the sum of weight * raw.

A weight multiplies its raw term. Nothing is normalized or calibrated, and
the weights need not add up to anything. The VGG term weighs each layer: its
part is the sum of layer weight * L1 on that layer. A vae2 gan term is not
in that sum. The training step weighs it.

oklab, ms_ssim, and vgg take an optional per-pixel map (B, 1, H, W) at full
resolution, the faces weight of whoodeo.faces with a mean of 1 per frame.
oklab multiplies its L1 by the map. ms_ssim takes the weighted mean of the
SSIM and contrast-structure maps at each scale, the map pooled along with the
frames and cut to the valid window centers. vgg multiplies each layer's L1 by
the map area-pooled to that layer's size. None is the plain mean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.config import VGG_LAYERS

# BT.709 RGB to Y, Cb, Cr, one row each.
_YCBCR = (
    (0.2126, 0.7152, 0.0722),
    (-0.1146, -0.3854, 0.5),
    (0.5, -0.4542, -0.0458),
)
# Cb and Cr count twice as much as Y: the term is there for color drift.
_YCBCR_WEIGHT = (1.0, 2.0, 2.0)
# Oklab (Ottosson 2020, https://bottosson.github.io/posts/oklab/): linear sRGB
# to LMS, and the cube-rooted LMS to L, a, b.
_OKLAB_M1 = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_OKLAB_M2 = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
# a and b count twice as much as L, as Cb and Cr do: the term is there for
# color drift. Oklab scales a and b so that a step is about as visible as the
# same step in L. On vae2c (step 2416, 1/16 pool) the mean |dL|, |da|, |db|
# came out about equal, 0.0079, 0.0079, 0.0093, so a and b carry 4/5 of it.
_OKLAB_WEIGHT = (1.0, 2.0, 2.0)
# Added under the cube root, so its slope at 0 stays finite (about 3e3).
_OKLAB_EPS = 1e-6
# The full-resolution oklab term takes the cube root of CIE L*a*b*: x^(1/3)
# above (6/29)^3, the tangent line below. Its slope stays at 7.8 or less, and
# negative LMS (an overshoot below black) keeps a gradient. About sRGB 0.1 and
# darker is linear in LMS then; black lands at L 0.14, which a difference
# does not see.
_CBRT_ROOT = 6.0 / 29.0
_CBRT_KNEE = _CBRT_ROOT ** 3
# MS-SSIM (Wang, Simoncelli, Bovik 2003): the weight of each scale, the
# Gaussian window, and the constants for a range of 1.
_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2


def pixel_loss(pred, high, kind):
    if kind == "l1":
        return F.l1_loss(pred, high)
    return F.mse_loss(pred, high)


def color_loss(pred, high, space, pool):
    """L1 on 1/pool average-pooled color, the chroma channels weighted twice.

    `space` is ycbcr or oklab. ycbcr pools the gamma sRGB frames and compares
    BT.709 Y, Cb, Cr. oklab pools in linear light and compares Oklab L, a, b.
    Pooling first leaves texture to the other terms, so this reads the slow
    color of flat areas. A remainder below `pool` on an edge is left out.
    """
    if space == "oklab":
        diff = oklab(F.avg_pool2d(srgb_linear(pred), pool)) - oklab(
            F.avg_pool2d(srgb_linear(high), pool)
        )
        weights = _OKLAB_WEIGHT
    else:
        matrix = pred.new_tensor(_YCBCR)
        pooled = F.avg_pool2d(pred, pool) - F.avg_pool2d(high, pool)
        diff = torch.einsum("ij,bjhw->bihw", matrix, pooled)
        weights = _YCBCR_WEIGHT
    weight = pred.new_tensor(weights).view(1, 3, 1, 1)
    return (diff.abs() * weight).mean() * 3.0 / float(sum(weights))


def srgb_linear(image):
    """sRGB to linear light, the piecewise sRGB curve.

    The decoder output is not clamped, so the curve is mirrored below 0 and
    runs on above 1. A pooled block then stays right when a few of its
    pixels overshoot, and their gradient is kept.
    """
    size = image.abs()
    curve = ((size.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4
    return torch.where(size <= 0.04045, image / 12.92, torch.sign(image) * curve)


def oklab(linear):
    """Linear sRGB to Oklab L, a, b, one channel each.

    LMS is clamped at 0 before the cube root: only a pooled block outside the
    gamut gets there. The root is taken of LMS + eps, so the gradient at 0
    stays finite. That lifts black to L 0.01, which a difference does not see.
    """
    lms = torch.einsum("ij,bjhw->bihw", linear.new_tensor(_OKLAB_M1), linear)
    root = (lms.clamp_min(0) + _OKLAB_EPS) ** (1.0 / 3.0)
    return torch.einsum("ij,bjhw->bihw", linear.new_tensor(_OKLAB_M2), root)


def safe_cbrt(value):
    """x^(1/3) above (6/29)^3, its tangent below, as CIE L*a*b* does."""
    root = value.clamp_min(_CBRT_KNEE) ** (1.0 / 3.0)
    line = _CBRT_ROOT + (value - _CBRT_KNEE) / (3.0 * _CBRT_ROOT * _CBRT_ROOT)
    return torch.where(value > _CBRT_KNEE, root, line)


def oklab_full(image):
    """gamma sRGB to Oklab L, a, b with the safe cube root, for full-resolution terms."""
    linear = srgb_linear(image)
    lms = torch.einsum("ij,bjhw->bihw", linear.new_tensor(_OKLAB_M1), linear)
    return torch.einsum("ij,bjhw->bihw", linear.new_tensor(_OKLAB_M2), safe_cbrt(lms))


def oklab_loss(pred_lab, high_lab, channels, pixel_weight=None):
    """L1 on Oklab L, a, b at full resolution, the mean weighted by `channels`.

    `pixel_weight` (B, 1, H, W), mean 1 per frame, multiplies each pixel.
    """
    weight = pred_lab.new_tensor(channels).view(1, 3, 1, 1)
    diff = (pred_lab - high_lab).abs() * weight
    if pixel_weight is not None:
        diff = diff * pixel_weight
    return diff.mean() * 3.0 / float(sum(channels))


def _ssim_window(like):
    coords = torch.arange(_SSIM_WINDOW, dtype=like.dtype, device=like.device) - _SSIM_WINDOW // 2
    gauss = torch.exp(-coords.square() / (2 * _SSIM_SIGMA ** 2))
    return gauss / gauss.sum()


def _blur(image, window):
    size = window.numel()
    image = F.conv2d(image, window.view(1, 1, 1, size))
    return F.conv2d(image, window.view(1, 1, size, 1))


def _weighted_mean(values, weight):
    """Mean per frame, weighted by `weight` (B, 1, h, w) when given."""
    if weight is None:
        return values.flatten(1).mean(1)
    return (values * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1e-12)


def _ssim_parts(x, y, window, weight=None):
    """Mean SSIM and mean contrast-structure per frame, valid Gaussian windows.

    `weight` is at the size of x; it is cut to the window centers.
    """
    mu_x = _blur(x, window)
    mu_y = _blur(y, window)
    var_x = _blur(x * x, window) - mu_x.square()
    var_y = _blur(y * y, window) - mu_y.square()
    cov = _blur(x * y, window) - mu_x * mu_y
    cs = (2 * cov + _SSIM_C2) / (var_x + var_y + _SSIM_C2)
    luminance = (2 * mu_x * mu_y + _SSIM_C1) / (mu_x.square() + mu_y.square() + _SSIM_C1)
    if weight is not None:
        margin = window.numel() // 2
        weight = weight[..., margin:margin + cs.shape[-2], margin:margin + cs.shape[-1]]
    return _weighted_mean(luminance * cs, weight), _weighted_mean(cs, weight)


def ms_ssim(x, y, scales, weight=None):
    """MS-SSIM of one-channel frames with a range of 1, the mean over the batch.

    Scales whose shorter side falls below the 11-pixel window are dropped and
    the weights of the rest renormalized, so a pooled scale_eq frame works too.
    Each factor is clamped at 1e-4 before its power: the power of 0 has no slope.
    `weight` (B, 1, H, W) weighs the mean at each scale, pooled like the frames.
    """
    window = _ssim_window(x)
    height, width = x.shape[-2:]
    usable = 0
    while usable < scales and min(height, width) >> usable >= _SSIM_WINDOW:
        usable += 1
    if usable == 0:
        raise ValueError(f"ms_ssim: {width}x{height} is smaller than the {_SSIM_WINDOW}-pixel window")
    weights = x.new_tensor(_MS_SSIM_WEIGHTS[:usable])
    weights = weights / weights.sum()
    factors = []
    for level in range(usable):
        ssim, cs = _ssim_parts(x, y, window, weight)
        if level == usable - 1:
            factors.append(ssim)
        else:
            factors.append(cs)
            pad = (0, x.shape[-1] % 2, 0, x.shape[-2] % 2)
            x = F.avg_pool2d(F.pad(x, pad, mode="replicate"), 2)
            y = F.avg_pool2d(F.pad(y, pad, mode="replicate"), 2)
            if weight is not None:
                weight = F.avg_pool2d(F.pad(weight, pad, mode="replicate"), 2)
    stacked = torch.stack(factors, dim=1).clamp_min(1e-4)
    return (stacked ** weights.view(1, -1)).prod(dim=1).mean()


def ms_ssim_loss(pred_lab, high_lab, scales, pixel_weight=None):
    """1 - MS-SSIM on Oklab L, optionally weighted per pixel."""
    return 1.0 - ms_ssim(pred_lab[:, :1], high_lab[:, :1], scales, pixel_weight)


class VGGLoss(nn.Module):
    """L1 on frozen VGG19 layers. The classifier head is dropped."""

    def __init__(self, layers):
        super().__init__()
        self.layer_names = tuple(layers)
        indexes = [VGG_LAYERS[name] for name in self.layer_names]
        self.indexes = tuple(indexes)
        from torchvision.models import VGG19_Weights, vgg19

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(indexes) + 1]
        for layer in features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.features = features.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def layer_losses(self, pred, target, pixel_weight=None):
        pred_maps = self._maps(pred.clamp(0, 1))
        with torch.no_grad():
            target_maps = self._maps(target.clamp(0, 1))
        if pixel_weight is None:
            return {
                name: F.l1_loss(pred_maps[name], target_maps[name])
                for name in self.layer_names
            }
        found = {}
        for name in self.layer_names:
            diff = (pred_maps[name] - target_maps[name]).abs()
            weight = F.interpolate(pixel_weight, size=diff.shape[-2:], mode="area")
            found[name] = (diff * weight).mean()
        return found

    def _maps(self, image):
        hidden = (image - self.mean) / self.std
        found = {}
        wanted = set(self.indexes)
        for index, layer in enumerate(self.features):
            hidden = layer(hidden)
            if index in wanted:
                found[index] = hidden
        return {name: found[VGG_LAYERS[name]] for name in self.layer_names}


class Objective:
    """Weights from the recipe. `combine` is the only place they apply."""

    def __init__(self, cfg, device):
        terms = cfg.objective_terms()
        self.names = tuple(term.name for term in terms)
        self.weights = {term.name: term.weight for term in terms if term.name != "vgg"}
        self.layer_weights = {}
        self.vgg = None
        vgg = cfg.term("vgg")
        if vgg is not None:
            self.layer_weights = dict(vgg.layers)
            self.vgg = VGGLoss(tuple(self.layer_weights)).to(device)

    def vgg_layers(self, pred, high, pixel_weight=None):
        """Raw L1 per VGG layer, still in the graph."""
        return self.vgg.layer_losses(pred, high, pixel_weight)

    def combine(self, raw, layer_raw):
        """Sum of weight * raw over the terms. vgg sums its weighted layers.

        Returns the total and the weighted part of each term.
        """
        total = None
        weighted = {}
        for name in self.names:
            if name == "vgg":
                value = sum(
                    weight * layer_raw[layer]
                    for layer, weight in self.layer_weights.items()
                )
            else:
                value = self.weights[name] * raw[name]
            weighted[name] = value
            total = value if total is None else total + value
        return total, weighted


def raw_scalars(prefix, raw, layer_raw):
    """<prefix>_raw/<term> and <prefix>_raw/vgg_<layer>, the unweighted values."""
    found = {f"{prefix}_raw/{name}": _number(value) for name, value in raw.items()}
    for name, value in layer_raw.items():
        found[f"{prefix}_raw/vgg_{name}"] = _number(value)
    return found


def _number(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)
