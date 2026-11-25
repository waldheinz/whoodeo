"""Compare the MLX 3-frame latent_up with the PyTorch reference, stage by stage. No timing.

    python -m whoodeo.mlxport.check RUN/checkpoints/STEP weights/mlx/align-STEP \
        --sizes 128x96 256x192 200x150 --device cpu

The PyTorch side runs on the CPU in fp32. Inputs are a smooth random
picture; the neighbors are the same picture shifted by a few pixels, so
SPyNet has real motion to find. Each stage is also checked with the torch
result of the stage before as input, so an error does not hide in the next.
"""

import argparse

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F


def _nhwc(tensor):
    return mx.array(np.ascontiguousarray(tensor.detach().float().numpy().transpose(0, 2, 3, 1)))


def _np(array):
    return np.array(array, dtype=np.float64)


def compare(name, ours, theirs, peak=None):
    """ours MLX NHWC, theirs torch NCHW (or already NHWC numpy)."""
    a = _np(ours)
    b = theirs.detach().double().numpy() if isinstance(theirs, torch.Tensor) else np.asarray(theirs, np.float64)
    if b.ndim == 4 and b.shape != a.shape:
        b = b.transpose(0, 2, 3, 1)
    err = a - b
    mse = float(np.mean(err ** 2))
    peak = float(np.abs(b).max()) if peak is None else peak
    psnr = float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)
    print(f"  {name:28s} max|err| {np.abs(err).max():.3e}  PSNR {psnr:6.1f} dB  (peak {peak:.3g})")
    return psnr


def test_frames(height, width, seed=42, shifts=((3, -2), (-4, 1))):
    """(1, 9, H, W): prev, middle, next; prev and next are the middle shifted by (dx, dy)."""
    generator = torch.Generator().manual_seed(seed)
    pad = 8
    big_h, big_w = height + 2 * pad, width + 2 * pad
    coarse = torch.rand(1, 3, max(big_h // 6, 2), max(big_w // 6, 2), generator=generator)
    fine = torch.rand(1, 3, big_h, big_w, generator=generator) * 0.08
    big = (F.interpolate(coarse, size=(big_h, big_w), mode="bicubic", align_corners=False) + fine).clamp(0, 1)

    def crop(dx, dy):
        return big[:, :, pad + dy:pad + dy + height, pad + dx:pad + dx + width]

    return torch.cat([crop(*shifts[0]), crop(0, 0), crop(*shifts[1])], dim=1)


def torch_stages(model, padded, condition):
    """The PyTorch path of LatentUp.forward on an already padded input, with its stages."""
    from whoodeo.models.latent_up import _blocks, _per_frame

    encoder, align, frames = model.encoder, model.align, model.in_frames
    batch = padded.shape[0]
    pairs = iter(model.film(condition))
    first = encoder.downs[0]
    images = [padded[:, 3 * i:3 * i + 3] for i in range(frames)]
    front = iter([(_per_frame(g, frames), _per_frame(b, frames)) for g, b in (next(pairs) for _ in first.blocks)])
    feats = list(_blocks(encoder.stem(torch.cat(images, 0)), first.blocks, front).split(batch, 0))
    flow, error = align.flows(images)
    fused = align(images, feats)
    from whoodeo.spynet import flow_warp

    others = [feats[i] for i in range(frames) if i != frames // 2]
    warped = flow_warp(torch.cat(others, 0), flow)
    return {"feats": feats, "flow": flow, "error": error, "warped": warped, "fused": fused, "images": images}


def check_size(reference, ours, height, width, codec_index):
    from whoodeo.models.common import pad_to

    print(f"{width}x{height}  codec slot {codec_index}")
    low = test_frames(height, width)
    model = reference.model
    vae = reference.vae
    condition = torch.zeros(1, model.condition_size)
    if codec_index is not None:
        condition[0, codec_index] = 1.0
    cond_mx = mx.array(condition.numpy())

    padded, _h, _w = pad_to(low, 4)
    t = torch_stages(model, padded, condition)
    t_code = model(low, condition)
    scale = vae.latent_scale.view(1, -1, 1, 1)
    shift = vae.latent_shift.view(1, -1, 1, 1)
    t_image = vae.decode(t_code / scale + shift, clamp=True)[:, :, :2 * height, :2 * width]
    # The reference module end to end, to be sure the stages above are the same path.
    reference.condition = condition
    assert torch.equal(reference(low), t_image), "stage-by-stage torch path differs from LatentUpReference"

    image, stages = ours(_nhwc(low), cond_mx, stages=True)
    mid = t["images"][1]
    neigh = torch.cat([t["images"][0], t["images"][2]], 0)
    flow_iso = ours.spynet(_nhwc(mid.repeat(2, 1, 1, 1)), _nhwc(neigh))
    compare("flow (SPyNet)", flow_iso, t["flow"].permute(0, 3, 1, 2))
    compare("flow (end to end)", stages["flow"], t["flow"].permute(0, 3, 1, 2))
    compare("warp error (end to end)", stages["error"], t["error"], peak=1.0)
    compare("warped feats (end to end)", stages["warped"], t["warped"])
    compare("fused feats (end to end)", stages["fused"], t["fused"])
    compare("code z (end to end)", stages["code"], t_code)
    latent = _nhwc(t_code / scale + shift)
    decoded_iso = mx.clip(ours.decode(latent.astype(ours.dtype)), 0, 1)[:, :2 * height, :2 * width]
    compare("decoder (torch code in)", decoded_iso, t_image, peak=1.0)
    return compare("image (end to end)", image, t_image, peak=1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("weights", help="convert.py output, without suffix")
    parser.add_argument("--sizes", nargs="+", default=["128x96", "256x192", "200x150"])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--codec", type=int, default=0, help="one-hot slot of the condition, -1 for unknown")
    parser.add_argument("--fp16", action="store_true", help="latent_up and decoder in fp16, as the app")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    torch.set_grad_enabled(False)
    torch.manual_seed(42)

    from whoodeo.mlxport.convert import load_reference
    from whoodeo.mlxport.model import AlignModel

    reference, *_rest = load_reference(args.checkpoint)
    ours = AlignModel.load(args.weights, mx.float16 if args.fp16 else mx.float32)
    print(f"MLX {mx.__version__} on {mx.default_device()}")
    codec = None if args.codec < 0 else args.codec
    for size in args.sizes:
        width, height = (int(v) for v in size.lower().split("x"))
        check_size(reference, ours, height, width, codec)


if __name__ == "__main__":
    main()
