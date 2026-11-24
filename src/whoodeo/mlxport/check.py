"""Compare the MLX align model with the PyTorch reference, stage by stage. No timing.

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


def check_size(reference, ours, height, width, codec_index):
    from whoodeo.spynet import flow_warp as torch_warp

    print(f"{width}x{height}  codec slot {codec_index}")
    low = test_frames(height, width)
    model = reference.inner.model
    vae = reference.inner.vae
    spynet = reference.spynet
    condition = torch.zeros(1, model.condition_size)
    if codec_index is not None:
        condition[0, codec_index] = 1.0
    cond_mx = mx.array(condition.numpy())

    # Torch stages, as AlignedReference + LatentUpReference run them.
    parts = low.reshape(1, 3, 3, height, width)
    neigh = torch.cat([parts[:, 0], parts[:, 2]], 0)
    ref = parts[:, 1].repeat(2, 1, 1, 1)
    t_flow = spynet(ref, neigh)
    t_warped = torch_warp(neigh, t_flow.permute(0, 2, 3, 1)).clamp(0, 1)
    t_aligned = torch.cat([t_warped[0:1], parts[:, 1], t_warped[1:2]], 1)
    from whoodeo.models.common import pad_to

    t_code = model(t_aligned, condition)
    t_image = vae.decode(t_code / vae.latent_scale.view(1, -1, 1, 1) + vae.latent_shift.view(1, -1, 1, 1), clamp=True)
    t_image = t_image[:, :, :2 * height, :2 * width]
    # The reference module end to end, to be sure the stages above are the same path.
    with torch.no_grad():
        inner_ref = type(reference.inner)(model, vae, None)
        inner_ref.condition = condition
        t_full = type(reference)(inner_ref, spynet, model.in_frames)(low)
    assert torch.equal(t_full, t_image), "stage-by-stage torch path differs from AlignedReference"

    low_mx = _nhwc(low)
    flow_mx = ours.spynet(_nhwc(ref), _nhwc(neigh))
    compare("flow (SPyNet)", flow_mx, t_flow)
    from whoodeo.mlxport.ops import flow_warp, pad_reflect_br

    warped_iso = mx.clip(flow_warp(_nhwc(neigh), _nhwc(t_flow)), 0, 1)
    compare("warp (torch flow in)", warped_iso, t_warped, peak=1.0)
    image, stages = ours(low_mx, cond_mx, stages=True)
    compare("aligned (end to end)", stages["aligned"], t_aligned, peak=1.0)
    code_iso = ours.code(pad_reflect_br(_nhwc(t_aligned), 4), cond_mx)
    compare("code z (torch aligned in)", code_iso, t_code)
    compare("code z (end to end)", stages["code"], t_code)
    latent = _nhwc(t_code / vae.latent_scale.view(1, -1, 1, 1) + vae.latent_shift.view(1, -1, 1, 1))
    decoded_iso = mx.clip(ours.decode(latent), 0, 1)[:, :2 * height, :2 * width]
    compare("decoder (torch code in)", decoded_iso, t_image, peak=1.0)
    return compare("image (end to end)", image, t_image, peak=1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("weights", help="convert.py output, without suffix")
    parser.add_argument("--sizes", nargs="+", default=["128x96", "256x192", "200x150"])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--codec", type=int, default=0, help="one-hot slot of the condition, -1 for unknown")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    torch.set_grad_enabled(False)
    torch.manual_seed(42)

    from whoodeo.mlxport.convert import load_reference
    from whoodeo.mlxport.model import AlignModel

    reference, *_rest = load_reference(args.checkpoint)
    ours = AlignModel.load(args.weights)
    print(f"MLX {mx.__version__} on {mx.default_device()}")
    codec = None if args.codec < 0 else args.codec
    for size in args.sizes:
        width, height = (int(v) for v in size.lower().split("x"))
        check_size(reference, ours, height, width, codec)


if __name__ == "__main__":
    main()
