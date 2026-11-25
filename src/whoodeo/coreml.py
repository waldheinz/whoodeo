"""Convert a vae2 checkpoint into Core ML packages for the app's VAE playground.

--part decoder (the default) gives the decoder alone: a latent in, the frame
clamped to [0, 1] out, both fp16. The input takes the enumerated latent sizes
of --latent, 60x107, 90x160, and 135x240 by default (480p, 720p, and 1080p at
factor 8). One size gives a fixed shape. The metadata carries the latent
shift and scale, one value per channel. --part encoder gives the encoder
instead: the frame in (sides 8 times the latent sizes), μ out, both fp16, the
same metadata. --part pair writes both. Without -o everything goes to
~/Library/Application Support/Whoodeo/VAE, where the playground looks.
The 3-frame latent_up goes to the app through MLX (whoodeo.mlxport.convert).
--conv-scale k (16 by default) multiplies each convolution's weight and bias
by k and puts a separate multiply by 1/k behind it. The pass that would fold
that multiply back into the weights is off, so the products stay large
(residual convolutions that start at zero would otherwise go very small). In
fp32 the picture does not change.
coremltools stays out of the training dependencies. On the Mac:

    uv pip install --python .venv/bin/python 'coremltools==9.1.dev1'
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from whoodeo.checkpoint import build_generator, load_generator

# Latent sides of 480p (856x480), 720p, and 1080p at factor 8.
LATENT_SIZES = ((60, 107), (90, 160), (135, 240))
# Convolution weights and bias times this, with a multiply by its inverse behind.
CONV_SCALE = 16.0


def _coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit(
            "coremltools is not installed. It is not a training dependency. "
            "On the Mac: uv pip install --python .venv/bin/python 'coremltools==9.1.dev1'"
        ) from exc
    return ct


def _prepare(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return model


class ScaledConv(torch.nn.Module):
    """A convolution with weight and bias times k, its output times 1/k: the same picture,
    larger intermediate products."""

    def __init__(self, conv, k):
        super().__init__()
        import copy

        self.conv = copy.deepcopy(conv)
        with torch.no_grad():
            self.conv.weight.mul_(k)
            if self.conv.bias is not None:
                self.conv.bias.mul_(k)
        self.inverse = 1.0 / k

    def forward(self, x):
        return self.conv(x) * self.inverse


def scale_convs(module, k):
    """A copy of `module` with every Conv2d wrapped in ScaledConv. k of 1 returns it as is."""
    if k is None or k == 1:
        return module
    import copy

    module = copy.deepcopy(module)

    def walk(parent):
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.Conv2d):
                setattr(parent, name, ScaledConv(child, k))
            else:
                walk(child)

    walk(module)
    return module.eval()


def _pipeline(ct, k):
    """The default passes, without the conv-scale fusion when the convolutions are scaled."""
    import copy

    pipeline = copy.deepcopy(ct.PassPipeline.DEFAULT)
    if k is not None and k != 1:
        pipeline.remove_passes(["common::fuse_conv_scale"])
    return pipeline


def _check_scaled(plain, scaled, example, label):
    """The scaled copy must give the same output in fp32."""
    with torch.no_grad():
        want = plain(example)
        got = scaled(example)
    diff = float((want - got).abs().max())
    print(f"conv scale check {label}  max abs {diff:.3g}", flush=True)
    if diff > 1e-3:
        raise SystemExit(f"scaling the convolutions changed the output by {diff}")


def convert(checkpoint, output, latent_sizes=LATENT_SIZES, conv_scale=None, part="decoder"):
    """Write the vae2 decoder, the encoder, or both (`part` pair, `output` a folder)."""
    ct = _coremltools()
    spec = load_generator(checkpoint)
    arch = spec.architecture.get("arch")
    if arch != "vae2":
        raise SystemExit(f"{checkpoint}: whoodeo-coreml converts a vae2, not {arch}")
    if conv_scale is None:
        conv_scale = CONV_SCALE
    if part == "encoder":
        return convert_encoder(ct, spec, output, latent_sizes, conv_scale)
    if part == "pair":
        folder = Path(output)
        convert_encoder(ct, spec, folder / package_filename(checkpoint, spec, part="encoder"),
                        latent_sizes, conv_scale)
        return convert_decoder(ct, spec, folder / package_filename(checkpoint, spec),
                               latent_sizes, conv_scale)
    return convert_decoder(ct, spec, output, latent_sizes, conv_scale)


def convert_decoder(ct, spec, output, latent_sizes, conv_scale=CONV_SCALE):
    """Write the vae2 decoder with enumerated latent sizes, or one fixed size."""
    from whoodeo.models.vae2 import DecoderExport

    model, label, _in_frames, step, _path = build_generator(spec, torch.device("cpu"))
    _prepare(model)
    latent = spec.architecture["latent"]
    sizes = [tuple(size) for size in latent_sizes]
    trace_h, trace_w = sizes[len(sizes) // 2]
    example = torch.zeros(1, latent, trace_h, trace_w)
    plain = DecoderExport(model).eval()
    export = scale_convs(plain, conv_scale)
    if conv_scale != 1:
        _check_scaled(plain, export, torch.randn(1, latent, 12, 20), "decoder")
    with torch.no_grad():
        traced = torch.jit.trace(export, example)
    if len(sizes) == 1:
        shape = (1, latent, trace_h, trace_w)
    else:
        shape = ct.EnumeratedShapes(
            shapes=[(1, latent, height, width) for height, width in sizes],
            default=(1, latent, trace_h, trace_w),
        )
    print(f"convert {label}  step {step}  decoder -> {output}", flush=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name="latent", shape=shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        pass_pipeline=_pipeline(ct, conv_scale),
    )
    mlmodel.author = "whoodeo"
    mlmodel.short_description = f"{label} decoder"
    _assign_metadata(mlmodel, _vae2_metadata(spec, model, "decoder", sizes, conv_scale))
    return _save_vae2_part(mlmodel, output, spec)


def _vae2_metadata(spec, model, part, sizes, conv_scale):
    return {
        "whoodeo.arch": "vae2",
        "whoodeo.part": part,
        "whoodeo.blocks": str(spec.architecture.get("blocks")),
        "whoodeo.channels": str(spec.architecture.get("channels")),
        "whoodeo.latent": str(spec.architecture["latent"]),
        "whoodeo.factor": str(spec.architecture.get("factor")),
        "whoodeo.activation": spec.architecture.get("activation", "silu"),
        "whoodeo.step": "" if spec.step is None else str(spec.step),
        "whoodeo.input": "fixed" if len(sizes) == 1 else "enumerated",
        "whoodeo.conv_scale": f"{conv_scale:g}",
        "whoodeo.latent_sizes": ",".join(f"{height}x{width}" for height, width in sizes),
        # One value per latent channel, comma separated.
        "whoodeo.latent_shift": ",".join(f"{float(v):.8g}" for v in model.latent_shift.reshape(-1)),
        "whoodeo.latent_scale": ",".join(f"{float(v):.8g}" for v in model.latent_scale.reshape(-1)),
    }


def _save_vae2_part(mlmodel, output, spec):
    output = Path(output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"wrote {output}", flush=True)
    return output, spec


class EncoderExport(torch.nn.Module):
    """The vae2 encoder alone: the frame in, μ out , for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode(image)


def convert_encoder(ct, spec, output, latent_sizes, conv_scale=CONV_SCALE):
    """Write the vae2 encoder: frames 8x the latent sizes in, μ out, enumerated or one fixed size."""
    model, label, _in_frames, step, _path = build_generator(spec, torch.device("cpu"))
    _prepare(model)
    factor = int(spec.architecture.get("factor") or model.factor)
    sizes = [tuple(size) for size in latent_sizes]
    frames = [(height * factor, width * factor) for height, width in sizes]
    trace_h, trace_w = frames[len(frames) // 2]
    example = torch.rand(1, 3, trace_h, trace_w)
    plain = EncoderExport(model).eval()
    export = scale_convs(plain, conv_scale)
    if conv_scale != 1:
        _check_scaled(plain, export, torch.rand(1, 3, 96, 160), "encoder")
    with torch.no_grad():
        traced = torch.jit.trace(export, example)
    if len(frames) == 1:
        shape = (1, 3, trace_h, trace_w)
    else:
        shape = ct.EnumeratedShapes(
            shapes=[(1, 3, height, width) for height, width in frames],
            default=(1, 3, trace_h, trace_w),
        )
    print(f"convert {label}  step {step}  encoder -> {output}", flush=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name="image", shape=shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="mu", dtype=np.float16)],
        pass_pipeline=_pipeline(ct, conv_scale),
    )
    mlmodel.author = "whoodeo"
    mlmodel.short_description = f"{label} encoder"
    _assign_metadata(mlmodel, _vae2_metadata(spec, model, "encoder", sizes, conv_scale))
    return _save_vae2_part(mlmodel, output, spec)


def playground_directory():
    """Where the app's VAE playground looks for an encoder and decoder pair. Created on demand."""
    path = Path.home() / "Library" / "Application Support" / "Whoodeo" / "VAE"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_slug(checkpoint):
    path = Path(checkpoint)
    for parent in [path, *path.parents]:
        if parent.parent.name == "checkpoints" and parent.name.isdigit():
            return _safe_name(parent.parent.parent.name)
        if parent.name == "checkpoints":
            return _safe_name(parent.parent.name)
    return _safe_name(path.stem if path.suffix else path.name)


def _safe_name(text):
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(text))
    cleaned = cleaned.strip("-")
    return cleaned or "model"


def package_filename(checkpoint, spec, part="decoder"):
    """Stable name the VAE playground reads without opening the package: …-decoder or …-encoder."""
    blocks = spec.architecture.get("blocks")
    channels = spec.architecture.get("channels")
    shape = f"{blocks}x{channels}" if blocks and channels else "model"
    step = spec.step
    step_part = f"step{int(step):06d}" if step is not None else "step"
    kind = "encoder" if part == "encoder" else "decoder"
    return f"{_run_slug(checkpoint)}-vae2-{shape}-{step_part}-{kind}.mlpackage"


def _default_output(checkpoint, spec, part="decoder"):
    if part == "pair":
        return playground_directory()
    return playground_directory() / package_filename(checkpoint, spec, part)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="convert a vae2 decoder or encoder to Core ML")
    parser.add_argument("model", type=Path, help="checkpoint, run directory, or model.pt")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="mlpackage to write (a folder for --part pair). "
             "Default: ~/Library/Application Support/Whoodeo/VAE",
    )
    parser.add_argument(
        "--latent",
        type=_latent_size,
        action="append",
        default=None,
        help="latent size HxW, repeatable. Default: 60x107, 90x160, 135x240",
    )
    parser.add_argument(
        "--conv-scale",
        type=float,
        default=None,
        help="multiply conv weights by k and the outputs by 1/k. 1 turns it off. Default: 16",
    )
    parser.add_argument(
        "--part",
        choices=("decoder", "encoder", "pair"),
        default="decoder",
        help="the decoder, the encoder, or both for the app's VAE playground",
    )
    return parser.parse_args(argv)


def _latent_size(text):
    try:
        height, width = (int(part) for part in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected HxW, got {text!r}") from exc
    if height < 1 or width < 1:
        raise argparse.ArgumentTypeError(f"expected positive sides, got {text!r}")
    return height, width


def _assign_metadata(model, mapping):
    defined = model.user_defined_metadata
    for key, value in mapping.items():
        defined[key] = value


def main(argv=None):
    args = parse_args(argv)
    spec = load_generator(args.model)
    if args.out is not None:
        output = args.out
    else:
        output = _default_output(args.model, spec, args.part)
    if args.conv_scale is not None and args.conv_scale <= 0:
        raise SystemExit("--conv-scale must be positive")
    convert(args.model, output, args.latent or LATENT_SIZES, args.conv_scale, args.part)


if __name__ == "__main__":
    main()
