"""Convert a vote or edsr checkpoint into a Core ML program, a vae2 decoder, or latent_up.

The package is what the macOS app loads. Without -o it goes to
~/Library/Application Support/Whoodeo/Models, which is the app's collection.
Three low frames are stacked on the channel axis, previous then current then
next, the same order as apply.
A vae2 checkpoint gives its decoder alone: a latent in, the frame clamped
to [0, 1] out, both fp16. The input takes the enumerated latent sizes of
--latent, 60x107, 90x160, and 135x240 by default (480p, 720p, and 1080p at
factor 8). One size gives a fixed shape.
The metadata carries the latent shift and scale, one value per channel.
--part encoder gives the vae2 encoder instead: the frame in (sides 8 times the
latent sizes), μ out, both fp16, the same metadata. --part pair writes both
into ~/Library/Application Support/Whoodeo/VAE (or the folder -o names), for
the app's VAE playground; the app's model list does not see that folder.
A latent_up checkpoint gives one program with its vae2 decoder behind it: the
low frame in, the 2x frame clamped to [0, 1] out, both fp16. The latent shift
and scale fold into the head of latent_up, so the graph hands the decoder μ
directly. The input takes any size from 64x64 to --max-low, 1280x720 by
default, so a 720p low frame fits. The app runs it on the GPU. --low gives enumerated sizes
instead. The app pads a frame on the bottom and the right
(reflect) to the next multiple of 4 and crops the result to 2h x 2w. The vae2
is the one the checkpoint names; a path under /mnt/home-storinator-trem is
looked up under /Volumes/home, then ~/storinator/home, and --vae overrides it.
A latent_up with condition (FiLM) is one package: the frames plus a
condition vector (whoodeo.condition_size, 11 for codec: one-hot h264/mpeg4/mpeg2,
unknown is all zero, then 8 reserve slots). The FiLM MLP stays in the graph, so
the app sets the vector per film and later reserve slots need no new export.
The package name ends in -cond.
A latent_up with in_frames 3 takes the previous, middle, and next low frame
on 9 channels, in that order (whoodeo.in_frames in the metadata); the clip's
edges repeat the middle frame.
--conv-scale k (16 by default for vae2 and latent_up, off for vote and edsr)
multiplies each convolution's weight and bias by k and puts a separate
multiply by 1/k behind it. The pass that would fold that multiply back into
the weights is off, so the products stay large (residual convolutions that
start at zero would otherwise go very small). In fp32 the picture does not
change.
coremltools stays out of the training dependencies. On the Mac:

    uv pip install --python .venv/bin/python 'coremltools==9.1.dev1'
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from whoodeo.checkpoint import build_generator, load_generator

# One 10-bit step. fp16 on the GPU stayed inside this against fp32.
STEP = 1 / 1023
_FLEX_H = (64, 1080, 360)
_FLEX_W = (64, 1920, 640)
# Latent sides of 480p (856x480), 720p, and 1080p at factor 8.
LATENT_SIZES = ((60, 107), (90, 160), (135, 240))
LOW_PAD = 4
# latent_up's flexible input: 64 up to this, height x width. The app runs it on the GPU.
LOW_MIN = 64
LOW_MAX = (720, 1280)
# Convolution weights and bias times this, with a multiply by its inverse behind.
CONV_SCALE = 16.0
# Where the Spark's runs mount lives on the Mac, in the order they are tried.
VAE_MOUNTS = (
    ("/mnt/home-storinator-trem", "/Volumes/home"),
    ("/mnt/home-storinator-trem", str(Path.home() / "storinator" / "home")),
)


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


def _shape(ct, in_frames):
    channels = in_frames * 3
    return ct.Shape(shape=(
        1,
        channels,
        ct.RangeDim(lower_bound=_FLEX_H[0], upper_bound=_FLEX_H[1], default=_FLEX_H[2]),
        ct.RangeDim(lower_bound=_FLEX_W[0], upper_bound=_FLEX_W[1], default=_FLEX_W[2]),
    ))


def _metadata(spec, arch, blocks, channels, in_frames, res_scale, stem, vote):
    meta = {
        "whoodeo.arch": arch,
        "whoodeo.blocks": str(blocks),
        "whoodeo.channels": str(channels),
        "whoodeo.in_frames": str(in_frames),
        "whoodeo.scale": "2",
        "whoodeo.step": "" if spec.step is None else str(spec.step),
        "whoodeo.res_scale": "" if res_scale is None else f"{float(res_scale):g}",
        "whoodeo.stem": "" if stem is None else str(stem),
        "whoodeo.vote": "" if vote is None else str(vote),
        "whoodeo.input": "flex",
        "whoodeo.width_min": str(_FLEX_W[0]),
        "whoodeo.width_max": str(_FLEX_W[1]),
        "whoodeo.height_min": str(_FLEX_H[0]),
        "whoodeo.height_max": str(_FLEX_H[1]),
    }
    return meta


def convert(
    checkpoint, output, latent_sizes=LATENT_SIZES, low_sizes=None, vae=None,
    conv_scale=None, max_low=LOW_MAX, part="decoder",
):
    """Write an mlprogram with a flexible input, a vae2 decoder, or latent_up with its decoder.

    `conv_scale` None picks CONV_SCALE for vae2 and latent_up and 1 for vote and edsr.
    `low_sizes` None gives latent_up a flexible input up to `max_low`.
    """
    ct = _coremltools()
    spec = load_generator(checkpoint)
    arch = spec.architecture.get("arch")
    if conv_scale is None:
        conv_scale = CONV_SCALE if arch in ("vae2", "latent_up") else 1.0
    if part != "decoder" and arch != "vae2":
        raise SystemExit(f"--part {part} is for a vae2, not {arch}")
    if arch == "vae2":
        if part == "encoder":
            return convert_encoder(ct, spec, output, latent_sizes, conv_scale)
        if part == "pair":
            folder = Path(output)
            convert_encoder(ct, spec, folder / package_filename(checkpoint, spec, part="encoder"),
                            latent_sizes, conv_scale)
            return convert_decoder(ct, spec, folder / package_filename(checkpoint, spec),
                                   latent_sizes, conv_scale)
        return convert_decoder(ct, spec, output, latent_sizes, conv_scale)
    if arch == "latent_up":
        return convert_latent_up(ct, spec, output, low_sizes, vae, conv_scale, max_low)
    if arch not in ("vote", "edsr"):
        raise SystemExit(
            f"{checkpoint}: whoodeo-coreml converts vote, edsr, vae2, and latent_up, not {arch}"
        )
    model, label, in_frames, step, _path = build_generator(spec, torch.device("cpu"))
    _prepare(model)
    if conv_scale != 1:
        scaled = scale_convs(model, conv_scale)
        _check_scaled(model, scaled, torch.rand(1, in_frames * 3, 64, 96), label)
        model = scaled
    blocks = spec.architecture.get("blocks")
    channels = spec.architecture.get("channels")
    res_scale = spec.architecture.get("res_scale")
    stem = spec.architecture.get("stem")
    vote = spec.architecture.get("vote")
    trace_h = _FLEX_H[2]
    trace_w = _FLEX_W[2]
    example = torch.zeros(1, in_frames * 3, trace_h, trace_w)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    # iOS18 is the spec version Core ML shares with macOS 15.
    target = ct.target.iOS18
    print(f"convert {label}  step {step}  -> {output}", flush=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=target,
        inputs=[ct.TensorType(name="frames", shape=_shape(ct, in_frames), dtype=np.float32)],
        outputs=[ct.TensorType(name="image", dtype=np.float32)],
        pass_pipeline=_pipeline(ct, conv_scale),
    )
    mlmodel.author = "whoodeo"
    mlmodel.short_description = label
    # The property is the protobuf map. It has no setter in coremltools 9.
    meta = _metadata(spec, arch, blocks, channels, in_frames, res_scale, stem, vote)
    meta["whoodeo.conv_scale"] = f"{conv_scale:g}"
    _assign_metadata(mlmodel, meta)
    output = Path(output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"wrote {output}", flush=True)
    return output, spec


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
    """The vae2 encoder alone: the frame in, μ out (no log-variance, no sample), for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        mu, _logvar = self.model.encode(image)
        return mu


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


def resolve_vae(path, override=None):
    """The vae2 checkpoint latent_up decodes with, mapped onto this machine's mounts."""
    if override is not None:
        candidate = Path(override).expanduser()
        if not candidate.exists():
            raise SystemExit(f"--vae {override}: not found")
        return candidate
    if not path:
        raise SystemExit("the latent_up checkpoint names no vae2. Pass --vae")
    candidate = Path(path)
    if candidate.exists():
        return candidate
    text = str(path)
    for remote, local in VAE_MOUNTS:
        if text == remote or text.startswith(remote + "/"):
            mapped = Path(local + text[len(remote):])
            if mapped.exists():
                return mapped
    raise SystemExit(f"vae2 not found: {path}. Pass --vae")


def load_latent_up(spec, device, vae_override=None):
    """latent_up, its frozen vae2, and the vae2 spec, all in eval mode on `device`."""
    from whoodeo.models.latent_up import FACTOR

    model, label, _in_frames, step, _path = build_generator(spec, device)
    _prepare(model)
    vae_path = resolve_vae(spec.architecture.get("vae"), vae_override)
    vae_spec = load_generator(vae_path)
    if vae_spec.architecture.get("arch") != "vae2":
        raise SystemExit(f"{vae_path}: latent_up decodes with a vae2, not {vae_spec.architecture.get('arch')}")
    vae, vae_label, _frames, _vae_step, _vae_file = build_generator(vae_spec, device)
    _prepare(vae)
    if vae.factor != 2 * FACTOR:
        raise SystemExit(f"{vae_path}: vae2 factor {vae.factor}, latent_up needs {2 * FACTOR}")
    if vae.latent != model.latent:
        raise SystemExit(f"{vae_path}: vae2 has {vae.latent} latent channels, latent_up {model.latent}")
    if bool((vae.latent_scale == 1).all()) and bool((vae.latent_shift == 0).all()):
        raise SystemExit(f"{vae_path}: the vae2 has no latent shift and scale")
    return model, label, step, vae, vae_label, vae_spec, vae_path


def _smooth_frames(height, width, seed=0, frames=1):
    """A picture-like test input: coarse noise, upsampled, in [0, 1]. `frames` stacks that many."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(1, 3 * frames, max(height // 8, 1), max(width // 8, 1), generator=generator)
    return torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)


def convert_latent_up(
    ct, spec, output, low_sizes, vae_override, conv_scale=CONV_SCALE, max_low=LOW_MAX,
):
    """Write latent_up and its vae2 decoder as one program, flexible up to `max_low` or enumerated.

    With condition, the package takes a condition vector as a second input.
    """
    from whoodeo.models.latent_up import (
        CODECS, CONDITIONS, FACTOR, RESERVE, LatentUpExport, LatentUpReference, fixed_condition,
    )

    model, label, step, vae, vae_label, vae_spec, vae_path = load_latent_up(
        spec, torch.device("cpu"), vae_override,
    )
    channels = 3 * model.in_frames
    sizes = None if low_sizes is None else [tuple(size) for size in low_sizes]
    for height, width in sizes or [max_low]:
        if height % FACTOR or width % FACTOR:
            raise SystemExit(f"low size {width}x{height}: both sides must divide by {FACTOR}")
    condition = spec.architecture.get("condition")
    condition_size = 0 if condition is None else CONDITIONS[condition]
    if condition is not None:
        print(
            f"condition {condition}: vector of {condition_size} as input "
            f"(codecs {', '.join(CODECS)}, then {RESERVE} reserve)",
            flush=True,
        )
    folded = LatentUpExport(model, vae).eval()
    export = scale_convs(folded, conv_scale)
    # Neither the fold nor the conv scale may change the picture. fp32 on the CPU, padded and unpadded.
    with torch.no_grad():
        for height, width in ((96, 128), (90, 130)):
            frames = _smooth_frames(height, width, frames=model.in_frames)
            padded, _h, _w = _pad(frames)
            if condition is None:
                want = LatentUpReference(model, vae).eval()(frames)
                got = export(padded)[:, :, :2 * height, :2 * width]
                fold = float((want - got).abs().max())
                print(f"fold and conv scale {conv_scale:g} check {width}x{height}  max abs {fold:.3g}", flush=True)
                if fold > 1e-4:
                    raise SystemExit(
                        f"folding the latent shift and scale or the conv scale changed the frame by {fold}"
                    )
            else:
                for name in ("unknown", "h264", "mpeg4"):
                    vector = fixed_condition(model, name)
                    want = LatentUpReference(model, vae, name).eval()(frames)
                    got = export(padded, vector)[:, :, :2 * height, :2 * width]
                    fold = float((want - got).abs().max())
                    print(
                        f"fold and conv scale {conv_scale:g} check {width}x{height} {name}  max abs {fold:.3g}",
                        flush=True,
                    )
                    if fold > 1e-4:
                        raise SystemExit(
                            f"folding the latent shift and scale or the conv scale changed the frame by {fold}"
                        )
    trace_h, trace_w = 360, 640
    if sizes is not None and (trace_h, trace_w) not in sizes:
        trace_h, trace_w = sizes[len(sizes) // 2]
    example = torch.zeros(1, channels, trace_h, trace_w)
    with torch.no_grad():
        if condition is None:
            traced = torch.jit.trace(export, example)
        else:
            example_cond = torch.zeros(1, condition_size)
            example_cond[0, 0] = 1
            traced = torch.jit.trace(export, (example, example_cond))
    if sizes is None:
        max_h, max_w = max_low
        shape = ct.Shape(shape=(
            1,
            channels,
            ct.RangeDim(lower_bound=LOW_MIN, upper_bound=max_h, default=min(trace_h, max_h)),
            ct.RangeDim(lower_bound=LOW_MIN, upper_bound=max_w, default=min(trace_w, max_w)),
        ))
        kind = "range"
        widths, heights = (LOW_MIN, max_w), (LOW_MIN, max_h)
    elif len(sizes) == 1:
        shape = (1, channels, trace_h, trace_w)
        kind = "fixed"
    else:
        shape = ct.EnumeratedShapes(
            shapes=[(1, channels, height, width) for height, width in sizes],
            default=(1, channels, trace_h, trace_w),
        )
        kind = "enumerated"
    if sizes is not None:
        widths = [width for _height, width in sizes]
        heights = [height for height, _width in sizes]
    print(f"convert {label}  step {step}  + {vae_label} decoder  {kind} -> {output}", flush=True)
    cond_shape = (1, condition_size) if condition is not None else None
    mlmodel = _convert_latent_up_traced(ct, traced, shape, conv_scale, cond_shape)
    mlmodel.author = "whoodeo"
    mlmodel.short_description = f"{label} + vae2 decoder"
    meta = {
        "whoodeo.arch": "latent_up",
        "whoodeo.part": "upscaler+decoder",
        "whoodeo.blocks": str(spec.architecture.get("blocks")),
        "whoodeo.channels": str(spec.architecture.get("channels")),
        "whoodeo.latent": str(spec.architecture.get("latent")),
        "whoodeo.in_frames": str(model.in_frames),
        "whoodeo.scale": "2",
        "whoodeo.step": "" if spec.step is None else str(spec.step),
        "whoodeo.vae": str(vae_path),
        "whoodeo.vae_step": "" if vae_spec.step is None else str(vae_spec.step),
        "whoodeo.activation": spec.architecture.get("activation", "silu"),
        "whoodeo.vae_activation": vae_spec.architecture.get("activation", "silu"),
        "whoodeo.input": kind,
        "whoodeo.conv_scale": f"{conv_scale:g}",
        "whoodeo.pad": str(FACTOR),
        "whoodeo.pad_mode": "reflect",
        # The app checks the padded size against these.
        "whoodeo.width_min": str(min(widths)),
        "whoodeo.width_max": str(max(widths)),
        "whoodeo.height_min": str(min(heights)),
        "whoodeo.height_max": str(max(heights)),
    }
    if sizes is not None:
        # Width x height, the order the app reads.
        meta["whoodeo.sizes"] = ",".join(f"{width}x{height}" for height, width in sizes)
    if condition is not None:
        meta["whoodeo.condition"] = condition
        meta["whoodeo.condition_size"] = str(condition_size)
        meta["whoodeo.codecs"] = ",".join(CODECS)
        meta["whoodeo.reserve"] = str(RESERVE)
    _assign_metadata(mlmodel, meta)
    output = Path(output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"wrote {output}", flush=True)
    return output, spec


def _convert_latent_up_traced(ct, traced, shape, conv_scale=CONV_SCALE, condition_shape=None):
    inputs = [ct.TensorType(name="frames", shape=shape, dtype=np.float16)]
    if condition_shape is not None:
        inputs.append(ct.TensorType(name="condition", shape=condition_shape, dtype=np.float16))
    return ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        inputs=inputs,
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        pass_pipeline=_pipeline(ct, conv_scale),
    )


def _pad(frames):
    from whoodeo.models.common import pad_to
    return pad_to(frames, LOW_PAD)


def library_directory():
    """The folder the macOS app lists. Created on demand."""
    path = Path.home() / "Library" / "Application Support" / "Whoodeo" / "Models"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def package_filename(checkpoint, spec, part=None):
    """Stable name the sidebar can read without opening the package. A FiLM latent_up ends in -cond.

    A model on more than one frame carries the count behind the step, "…-step003000-3f-hd-cond".
    """
    arch = spec.architecture.get("arch") or "model"
    blocks = spec.architecture.get("blocks")
    channels = spec.architecture.get("channels")
    shape = f"{blocks}x{channels}" if blocks and channels else "model"
    step = spec.step
    step_part = f"step{int(step):06d}" if step is not None else "step"
    # More than one input frame, e.g. "-3f": the sidebar tells it from a one-frame model.
    in_frames = int(spec.architecture.get("in_frames") or 1)
    if in_frames > 1:
        step_part += f"-{in_frames}f"
    # latent_up ends in "hd": flexible up to 1280x720, for the GPU. The app reads the old names too.
    kind = {"vae2": "decoder", "latent_up": "hd"}.get(arch, "flex")
    if arch == "vae2" and part == "encoder":
        kind = "encoder"
    if spec.architecture.get("condition") is not None:
        kind += "-cond"
    return f"{_run_slug(checkpoint)}-{arch}-{shape}-{step_part}-{kind}.mlpackage"


def _default_output(checkpoint, spec, part="decoder"):
    if part == "pair":
        return playground_directory()
    return library_directory() / package_filename(checkpoint, spec, part)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="convert a vote or edsr checkpoint, a vae2 decoder, or latent_up to Core ML",
    )
    parser.add_argument("model", type=Path, help="checkpoint, run directory, or model.pt")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="mlpackage to write. Default: ~/Library/Application Support/Whoodeo/Models",
    )
    parser.add_argument(
        "--latent",
        type=_latent_size,
        action="append",
        default=None,
        help="vae2 latent size HxW, repeatable. Default: 60x107, 90x160, 135x240",
    )
    parser.add_argument(
        "--low",
        type=_latent_size,
        action="append",
        default=None,
        help="latent_up: enumerated low sizes HxW instead of the flexible input, repeatable, "
             "sides divisible by 4",
    )
    parser.add_argument(
        "--max-low",
        type=_latent_size,
        default=LOW_MAX,
        help="latent_up: largest low frame HxW of the flexible input, sides divisible by 4. "
             "Default: 720x1280",
    )
    parser.add_argument(
        "--conv-scale",
        type=float,
        default=None,
        help="multiply conv weights by k and the outputs by 1/k. "
             "1 turns it off. Default: 16 for vae2 and latent_up, 1 for vote and edsr",
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=None,
        help="latent_up: the vae2 checkpoint. Default: the one the checkpoint names, "
             "/mnt/home-storinator-trem mapped to /Volumes/home",
    )
    parser.add_argument(
        "--part",
        choices=("decoder", "encoder", "pair"),
        default="decoder",
        help="vae2: the decoder, the encoder, or both for the app's VAE playground "
             "(-o is then a folder, default ~/Library/Application Support/Whoodeo/VAE)",
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
    convert(
        args.model, output, args.latent or LATENT_SIZES, args.low, args.vae,
        args.conv_scale, args.max_low, args.part,
    )


if __name__ == "__main__":
    main()
