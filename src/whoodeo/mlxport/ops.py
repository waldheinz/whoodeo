"""PyTorch ops the align path needs, in MLX on NHWC arrays.

Each function follows the torch op it names, including its coordinate
conventions, so a converted checkpoint gives the same numbers.
"""

import mlx.core as mx


def conv2d(x, weight, bias, stride=1, padding=0, dilation=1):
    """torch Conv2d; `weight` already in MLX layout (O, kH, kW, I)."""
    y = mx.conv2d(x, weight, stride=stride, padding=padding, dilation=dilation)
    return y if bias is None else y + bias


def avg_pool2(x):
    """avg_pool2d(kernel 2, stride 2) on even sides."""
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))


def upsample_nearest2(x):
    """interpolate(scale_factor=2, mode='nearest')."""
    n, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, 2, w, 2, c))
    return x.reshape(n, 2 * h, 2 * w, c)


def _source_index(size_in, size_out, align_corners):
    """Indices and weights of 1-D bilinear sampling, as torch upsample_bilinear2d computes them."""
    dst = mx.arange(size_out, dtype=mx.float32)
    if align_corners:
        scale = (size_in - 1) / (size_out - 1) if size_out > 1 else 0.0
        src = dst * scale
    else:
        scale = size_in / size_out
        src = mx.maximum((dst + 0.5) * scale - 0.5, 0.0)
    i0 = mx.minimum(mx.floor(src).astype(mx.int32), size_in - 1)
    i1 = mx.minimum(i0 + 1, size_in - 1)
    frac = src - i0.astype(mx.float32)
    return i0, i1, frac


def resize_bilinear(x, height, width, align_corners):
    """interpolate(size=(height, width), mode='bilinear') on NHWC."""
    _n, h, w, _c = x.shape
    if (h, w) == (height, width) and not align_corners:
        return x
    y0, y1, fy = _source_index(h, height, align_corners)
    x0, x1, fx = _source_index(w, width, align_corners)
    fy = fy[None, :, None, None]
    fx = fx[None, None, :, None]
    # torch interpolates along w first, then h; same order here.
    top = mx.take(x, y0, axis=1)
    bottom = mx.take(x, y1, axis=1)
    top = mx.take(top, x0, axis=2) * (1 - fx) + mx.take(top, x1, axis=2) * fx
    bottom = mx.take(bottom, x0, axis=2) * (1 - fx) + mx.take(bottom, x1, axis=2) * fx
    return top * (1 - fy) + bottom * fy


def pad_reflect_br(x, factor):
    """common.pad_to: pad bottom and right to a multiple of `factor`, reflect when it fits."""
    _n, h, w, _c = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x
    reflect = h > pad_h and w > pad_w
    rows = list(range(h)) + [(h - 2 - i) if reflect else h - 1 for i in range(pad_h)]
    cols = list(range(w)) + [(w - 2 - i) if reflect else w - 1 for i in range(pad_w)]
    x = mx.take(x, mx.array(rows, dtype=mx.int32), axis=1)
    return mx.take(x, mx.array(cols, dtype=mx.int32), axis=2)


def flow_warp(x, flow):
    """spynet.flow_warp: grid_sample(bilinear, border, align_corners=True), flow in pixels.

    `x` (N, H, W, C), `flow` (N, H, W, 2) with (dx, dy). The normalize and
    unnormalize round trip of the torch path is kept for the same rounding.
    """
    n, h, w, c = x.shape
    gx = mx.arange(w, dtype=mx.float32)[None, None, :] + flow[..., 0]
    gy = mx.arange(h, dtype=mx.float32)[None, :, None] + flow[..., 1]
    denom_w = float(max(w - 1, 1))
    denom_h = float(max(h - 1, 1))
    gx = 2.0 * gx / denom_w - 1.0
    gy = 2.0 * gy / denom_h - 1.0
    # grid_sampler_unnormalize with align_corners, then border clip.
    ix = mx.clip((gx + 1) / 2 * (w - 1), 0, w - 1)
    iy = mx.clip((gy + 1) / 2 * (h - 1), 0, h - 1)
    x0f = mx.floor(ix)
    y0f = mx.floor(iy)
    # Coordinates stay fp32; the weights take the type of x (fp16 features interpolate in fp16).
    fx = (ix - x0f)[..., None].astype(x.dtype)
    fy = (iy - y0f)[..., None].astype(x.dtype)
    x0 = x0f.astype(mx.int32)
    y0 = y0f.astype(mx.int32)
    x1 = mx.minimum(x0 + 1, w - 1)
    y1 = mx.minimum(y0 + 1, h - 1)
    flat = x.reshape(n * h * w, c)
    base = (mx.arange(n, dtype=mx.int32) * (h * w))[:, None, None]

    def gather(yi, xi):
        index = (base + yi * w + xi).reshape(-1)
        return mx.take(flat, index, axis=0).reshape(n, h, w, c)

    return (
        gather(y0, x0) * ((1 - fx) * (1 - fy))
        + gather(y0, x1) * (fx * (1 - fy))
        + gather(y1, x0) * ((1 - fx) * fy)
        + gather(y1, x1) * (fx * fy)
    )
