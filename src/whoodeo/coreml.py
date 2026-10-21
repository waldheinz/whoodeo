"""Convert a vote or edsr checkpoint into a Core ML program, a vae2 decoder, or latent_up.

The package is what the macOS app loads. Without -o it goes to
~/Library/Application Support/Whoodeo/Models, which is the app's collection.
Three low frames are stacked on the channel axis, previous then current then
next, the same order as apply.
A vae2 checkpoint gives its decoder alone: a latent in, the frame clamped
to [0, 1] out, both fp16. The input takes the enumerated latent sizes of
--latent, 60x107, 90x160, and 135x240 by default (480p, 720p, and 1080p at
factor 8), so the Neural Engine can run it. One size gives a fixed shape.
The metadata carries the latent shift and scale, one value per channel.
A latent_up checkpoint gives one program with its vae2 decoder behind it: the
low frame in, the 2x frame clamped to [0, 1] out, both fp16. The latent shift
and scale fold into the head of latent_up, so the graph hands the decoder μ
directly. The input takes the enumerated low sizes of --low, 428x240,
640x360, 720x480, 720x576, and 960x540 by default, all divisible by 4. The app
pads a frame on the bottom and the right (reflect) to the next multiple of 4
and crops the result to 2h x 2w. The vae2 is the one the checkpoint names;
a path under /mnt/home-storinator-trem is looked up under /Volumes/home, and
--vae overrides it.
coremltools stays out of the training dependencies. On the Mac:

    uv pip install --python .venv/bin/python 'coremltools==9.1.dev1'
"""

import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch

from whoodeo.checkpoint import build_generator, load_generator
from whoodeo.video import frame_tensor

# One 10-bit step. fp16 on the GPU stayed inside this against fp32.
STEP = 1 / 1023
# The bench accepts a compute unit when its worst PSNR against PyTorch fp32 is at
# least this. One 8-bit step of uniform rounding noise is about 59 dB.
PSNR_MIN = 50.0
_FLEX_H = (64, 1080, 360)
_FLEX_W = (64, 1920, 640)
# Latent sides of 480p (856x480), 720p, and 1080p at factor 8.
LATENT_SIZES = ((60, 107), (90, 160), (135, 240))
# Low sides latent_up takes, height x width: 480p (426x240 padded), 360p, NTSC,
# PAL, and 540p. Each divides by 4.
LOW_SIZES = ((240, 428), (360, 640), (480, 720), (576, 720), (540, 960))
LOW_PAD = 4
# Where the Spark's runs mount lives on the Mac.
VAE_MOUNTS = (("/mnt/home-storinator-trem", "/Volumes/home"),)


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


def convert(checkpoint, output, latent_sizes=LATENT_SIZES, low_sizes=LOW_SIZES, vae=None, plan=False):
    """Write an mlprogram with a flexible input, a vae2 decoder, or latent_up with its decoder."""
    ct = _coremltools()
    spec = load_generator(checkpoint)
    arch = spec.architecture.get("arch")
    if arch == "vae2":
        return convert_decoder(ct, spec, output, latent_sizes)
    if arch == "latent_up":
        return convert_latent_up(ct, spec, output, low_sizes, vae, plan)
    if arch not in ("vote", "edsr"):
        raise SystemExit(
            f"{checkpoint}: whoodeo-coreml converts vote, edsr, vae2, and latent_up, not {arch}"
        )
    model, label, in_frames, step, _path = build_generator(spec, torch.device("cpu"))
    _prepare(model)
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
    )
    mlmodel.author = "whoodeo"
    mlmodel.short_description = label
    # The property is the protobuf map. It has no setter in coremltools 9.
    _assign_metadata(mlmodel, _metadata(
        spec, arch, blocks, channels, in_frames, res_scale, stem, vote,
    ))
    output = Path(output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"wrote {output}", flush=True)
    return output, spec


def convert_decoder(ct, spec, output, latent_sizes):
    """Write the vae2 decoder with enumerated latent sizes, or one fixed size."""
    from whoodeo.models.vae2 import DecoderExport

    model, label, _in_frames, step, _path = build_generator(spec, torch.device("cpu"))
    _prepare(model)
    latent = spec.architecture["latent"]
    sizes = [tuple(size) for size in latent_sizes]
    trace_h, trace_w = sizes[len(sizes) // 2]
    example = torch.zeros(1, latent, trace_h, trace_w)
    with torch.no_grad():
        traced = torch.jit.trace(DecoderExport(model).eval(), example)
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
    )
    mlmodel.author = "whoodeo"
    mlmodel.short_description = f"{label} decoder"
    _assign_metadata(mlmodel, {
        "whoodeo.arch": "vae2",
        "whoodeo.part": "decoder",
        "whoodeo.blocks": str(spec.architecture.get("blocks")),
        "whoodeo.channels": str(spec.architecture.get("channels")),
        "whoodeo.latent": str(latent),
        "whoodeo.factor": str(spec.architecture.get("factor")),
        "whoodeo.step": "" if spec.step is None else str(spec.step),
        "whoodeo.input": "fixed" if len(sizes) == 1 else "enumerated",
        "whoodeo.latent_sizes": ",".join(f"{height}x{width}" for height, width in sizes),
        # One value per latent channel, comma separated.
        "whoodeo.latent_shift": ",".join(f"{float(v):.8g}" for v in model.latent_shift.reshape(-1)),
        "whoodeo.latent_scale": ",".join(f"{float(v):.8g}" for v in model.latent_scale.reshape(-1)),
    })
    output = Path(output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"wrote {output}", flush=True)
    try:
        counts = _plan_counts(mlmodel, ct.ComputeUnit.CPU_AND_NE)
    except Exception as exc:
        counts = {"plan": str(exc)}
    print(f"plan ane {trace_h}x{trace_w} {counts}", flush=True)
    return output, spec


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


def _smooth_frames(height, width, seed=0):
    """A picture-like test input: coarse noise, upsampled, in [0, 1]."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(1, 3, max(height // 8, 1), max(width // 8, 1), generator=generator)
    return torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)


def convert_latent_up(ct, spec, output, low_sizes, vae_override, plan):
    """Write latent_up and its vae2 decoder as one program with enumerated low sizes."""
    from whoodeo.models.latent_up import FACTOR, LatentUpExport, LatentUpReference

    model, label, step, vae, vae_label, vae_spec, vae_path = load_latent_up(
        spec, torch.device("cpu"), vae_override,
    )
    sizes = [tuple(size) for size in low_sizes]
    for height, width in sizes:
        if height % FACTOR or width % FACTOR:
            raise SystemExit(f"low size {width}x{height}: both sides must divide by {FACTOR}")
    export = LatentUpExport(model, vae).eval()
    reference = LatentUpReference(model, vae).eval()
    # The fold must not change the picture. fp32 on the CPU, padded and unpadded.
    with torch.no_grad():
        for height, width in ((96, 128), (90, 130)):
            frames = _smooth_frames(height, width)
            want = reference(frames)
            padded, _h, _w = _pad(frames)
            got = export(padded)[:, :, :2 * height, :2 * width]
            fold = float((want - got).abs().max())
            print(f"fold check {width}x{height}  max abs {fold:.3g}", flush=True)
            if fold > 1e-4:
                raise SystemExit(f"folding the latent shift and scale changed the frame by {fold}")
    trace_h, trace_w = sizes[len(sizes) // 2] if len(sizes) > 1 else sizes[0]
    if (360, 640) in sizes:
        trace_h, trace_w = 360, 640
    example = torch.zeros(1, 3, trace_h, trace_w)
    with torch.no_grad():
        traced = torch.jit.trace(export, example)
    if len(sizes) == 1:
        shape = (1, 3, trace_h, trace_w)
    else:
        shape = ct.EnumeratedShapes(
            shapes=[(1, 3, height, width) for height, width in sizes],
            default=(1, 3, trace_h, trace_w),
        )
    print(f"convert {label}  step {step}  + {vae_label} decoder -> {output}", flush=True)
    mlmodel = _convert_latent_up_traced(ct, traced, shape)
    mlmodel.author = "whoodeo"
    mlmodel.short_description = f"{label} + vae2 decoder"
    widths = [width for _height, width in sizes]
    heights = [height for height, _width in sizes]
    _assign_metadata(mlmodel, {
        "whoodeo.arch": "latent_up",
        "whoodeo.part": "upscaler+decoder",
        "whoodeo.blocks": str(spec.architecture.get("blocks")),
        "whoodeo.channels": str(spec.architecture.get("channels")),
        "whoodeo.latent": str(spec.architecture.get("latent")),
        "whoodeo.in_frames": "1",
        "whoodeo.scale": "2",
        "whoodeo.step": "" if spec.step is None else str(spec.step),
        "whoodeo.vae": str(vae_path),
        "whoodeo.vae_step": "" if vae_spec.step is None else str(vae_spec.step),
        "whoodeo.input": "fixed" if len(sizes) == 1 else "enumerated",
        # Width x height, the order the app reads.
        "whoodeo.sizes": ",".join(f"{width}x{height}" for height, width in sizes),
        "whoodeo.pad": str(FACTOR),
        "whoodeo.pad_mode": "reflect",
        "whoodeo.width_min": str(min(widths)),
        "whoodeo.width_max": str(max(widths)),
        "whoodeo.height_min": str(min(heights)),
        "whoodeo.height_max": str(max(heights)),
    })
    output = Path(output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"wrote {output}", flush=True)
    if plan:
        _plan_sizes(ct, traced, sizes)
    return output, spec


def _convert_latent_up_traced(ct, traced, shape):
    return ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name="frames", shape=shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
    )


def _plan_sizes(ct, traced, sizes):
    """The Neural Engine plan of each size, from a fixed-shape copy of the program."""
    import tempfile

    for height, width in sizes:
        fixed = _convert_latent_up_traced(ct, traced, (1, 3, height, width))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.mlpackage"
            fixed.save(str(path))
            fixed = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            try:
                counts = _plan_counts(fixed, ct.ComputeUnit.CPU_AND_NE)
            except Exception as exc:
                counts = {"plan": str(exc)}
        print(f"plan ane {width}x{height} {_plan_text(counts)}", flush=True)


def _plan_text(counts):
    if "plan" in counts:
        return counts["plan"]
    ops = sum(value for key, value in counts.items() if not key.startswith("conv_"))
    convs = sum(value for key, value in counts.items() if key.startswith("conv_"))
    parts = [f"ops {counts.get('ane', 0)}/{ops} ane"]
    parts.append(f"conv {counts.get('conv_ane', 0)}/{convs} ane")
    others = {key: value for key, value in counts.items() if key not in ("ane", "conv_ane")}
    if others:
        parts.append(str(others))
    return "  ".join(parts)


def _pad(frames):
    from whoodeo.models.common import pad_to
    return pad_to(frames, LOW_PAD)


def _read_window(path, index, radius):
    """Low frames around `index`, display order. The clip's edges repeat."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for number, frame in enumerate(container.decode(stream)):
            frames.append(frame_tensor(frame))
            if number >= index + radius:
                break
    if not frames:
        raise SystemExit(f"no frames: {path}")
    center = min(index, len(frames) - 1)
    picked = []
    for dt in range(-radius, radius + 1):
        slot = min(max(center + dt, 0), len(frames) - 1)
        picked.append(frames[slot])
    stack = torch.cat(picked, dim=0)
    return stack, center


def _error(reference, got):
    diff = np.abs(reference.astype(np.float64) - got.astype(np.float64))
    max_abs = float(diff.max())
    mse = float(np.mean(np.square(diff)))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    return max_abs, psnr


def _time_predict(predict, repeats=8):
    predict()
    predict()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        predict()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def _device_kind(device):
    from coremltools.models.compute_device import (
        MLCPUComputeDevice,
        MLGPUComputeDevice,
        MLNeuralEngineComputeDevice,
    )
    if isinstance(device, MLNeuralEngineComputeDevice):
        return "ane"
    if isinstance(device, MLGPUComputeDevice):
        return "gpu"
    if isinstance(device, MLCPUComputeDevice):
        return "cpu"
    return type(device).__name__


def _assign_metadata(model, mapping):
    defined = model.user_defined_metadata
    for key, value in mapping.items():
        defined[key] = value


def _plan_counts(mlmodel, unit):
    from coremltools.models.compute_plan import MLComputePlan
    compiled = mlmodel.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(compiled, compute_units=unit)
    program = plan.model_structure.program
    if program is None:
        return {}
    counts = {}

    def walk(block):
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is not None:
                name = _device_kind(usage.preferred_compute_device)
                counts[name] = counts.get(name, 0) + 1
                if operation.operator_name.split(".")[-1] == "conv":
                    key = f"conv_{name}"
                    counts[key] = counts.get(key, 0) + 1
            for nested in operation.blocks:
                walk(nested)

    main = program["main"] if isinstance(program, dict) else program.functions["main"]
    walk(main.block)
    return counts


def _predict_array(mlmodel, batch):
    result = mlmodel.predict({"frames": batch})
    image = result.get("image")
    if image is None:
        raise SystemExit(f"Core ML output names: {', '.join(result)}")
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    return array


def _cases_upscaler(videos, radius):
    """Frame 100 of each video at its own size, with its neighbors."""
    for video in videos:
        stack, center = _read_window(video, 100, radius)
        height, width = stack.shape[-2], stack.shape[-1]
        yield f"{Path(video).name} #{center}", stack.unsqueeze(0), width, height


def _cases_latent_up(videos, sizes):
    """Frame 100 of each video, resized to each enumerated low size."""
    frames = []
    for video in videos:
        stack, center = _read_window(video, 100, 0)
        frames.append((f"{Path(video).name} #{center}", stack.unsqueeze(0)))
    for height, width in sizes:
        for name, frame in frames:
            if frame.shape[-2:] == (height, width):
                sized = frame
            else:
                sized = torch.nn.functional.interpolate(
                    frame, size=(height, width), mode="bilinear", antialias=True, align_corners=False,
                ).clamp(0, 1)
            yield name, sized, width, height


def _package_sizes(package, ct):
    from coremltools.models.utils import load_spec
    spec = load_spec(str(package))
    text = spec.description.metadata.userDefined.get("whoodeo.sizes", "")
    sizes = []
    for part in text.split(","):
        if "x" in part:
            width, height = (int(item) for item in part.split("x"))
            sizes.append((height, width))
    return sizes


def bench(package, checkpoint, videos, vae=None):
    """Compare the package with PyTorch fp32 on real frames. PSNR sets the default unit."""
    ct = _coremltools()
    spec = load_generator(checkpoint)
    arch = spec.architecture.get("arch")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if arch == "latent_up":
        from whoodeo.models.latent_up import LatentUpReference
        model, label, _step, vae_model, _vae_label, _vae_spec, _vae_path = load_latent_up(spec, device, vae)
        model = LatentUpReference(model, vae_model).eval()
        sizes = _package_sizes(package, ct)
        cases = list(_cases_latent_up(videos, sizes))
        dtype = np.float16
        repeats = 5
    else:
        model, label, in_frames, _step, _path = build_generator(spec, torch.device("cpu"))
        _prepare(model)
        model.to(device)
        cases = list(_cases_upscaler(videos, in_frames // 2))
        dtype = np.float32
        repeats = 8
    units = (
        ("gpu", ct.ComputeUnit.CPU_AND_GPU),
        ("ane", ct.ComputeUnit.CPU_AND_NE),
    )
    loaded = {}
    for name, unit in units:
        loaded[name] = ct.models.MLModel(
            str(package),
            compute_units=unit,
            optimization_hints={"reshapeFrequency": ct.ReshapeFrequency.Infrequent},
        )

    rows = []
    timed = {}
    for case, batch_t, width, height in cases:
        print(f"frame {case}  {width}x{height}", flush=True)
        with torch.inference_mode():
            reference = model(batch_t.to(device)).squeeze(0).detach().float().cpu().numpy()
            if device.type == "mps":
                torch.mps.synchronize()
        batch = batch_t.numpy().astype(dtype, copy=False)
        for name, _unit in units:
            mlmodel = loaded[name]

            def predict(mlmodel=mlmodel, batch=batch):
                return _predict_array(mlmodel, batch)

            # The first shape pays for the compiler. Keep it out of the median.
            started = time.perf_counter()
            got = predict()
            compile_s = time.perf_counter() - started
            max_abs, psnr = _error(reference, got)
            shape_key = (name, width, height)
            if shape_key not in timed:
                timed[shape_key] = _time_predict(predict, repeats)
                note = f"  compile {compile_s:.1f}s"
            else:
                note = ""
            median = timed[shape_key]
            print(
                f"  {name}  max {max_abs:.6f}  psnr {psnr:.2f}  {median:.1f} ms{note}",
                flush=True,
            )
            rows.append({
                "name": name,
                "width": width,
                "height": height,
                "max_abs": max_abs,
                "psnr": psnr,
                "ms": median,
            })

    summary = {}
    for name, _unit in units:
        mine = [row for row in rows if row["name"] == name]
        if not mine:
            continue
        worst = min(mine, key=lambda row: row["psnr"])
        sizes_ms = {}
        for row in mine:
            sizes_ms[(row["width"], row["height"])] = row["ms"]
        summary[name] = {
            "psnr": worst["psnr"],
            "max_abs": max(row["max_abs"] for row in mine),
            "width": worst["width"],
            "height": worst["height"],
            "ms": sum(sizes_ms.values()) / len(sizes_ms),
            "sizes_ms": sizes_ms,
        }
    passing = [name for name, row in summary.items() if row["psnr"] >= PSNR_MIN]
    if passing:
        prefer = min(passing, key=lambda name: summary[name]["ms"])
    else:
        prefer = "gpu"
    parts = []
    for name, title in (("gpu", "GPU"), ("ane", "Neural Engine")):
        row = summary.get(name)
        if row is None:
            continue
        mark = "ok" if row["psnr"] >= PSNR_MIN else f"unter {PSNR_MIN:g} dB"
        step = "innerhalb" if row["max_abs"] <= STEP else "über"
        parts.append(
            f"{title} PSNR min {row['psnr']:.1f} dB ({mark}), max {row['max_abs']:.6f} "
            f"({step} einem 10-bit-Schritt), im Mittel {row['ms']:.0f} ms"
        )
    note = ". ".join(parts)
    note += ". Voreinstellung GPU." if prefer == "gpu" else ". Voreinstellung Neural Engine."
    print(note, flush=True)
    for width, height in sorted({(row["width"], row["height"]) for row in rows}):
        cells = "  ".join(
            f"{name} {summary[name]['sizes_ms'][(width, height)]:.1f} ms"
            for name in summary if (width, height) in summary[name]["sizes_ms"]
        )
        print(f"  {width}x{height}  {cells}", flush=True)
    _write_bench(package, summary.get("gpu"), summary.get("ane"), prefer, note)
    return prefer


def _write_bench(package, gpu, ane, prefer, note):
    """Patch the spec in place. Saving the whole package would delete its weights."""
    from coremltools.models.utils import _get_model_spec_path, load_spec, save_spec
    spec = load_spec(str(package))
    defined = spec.description.metadata.userDefined
    for name, row in (("gpu", gpu), ("ane", ane)):
        if row is None:
            continue
        defined[f"whoodeo.bench.{name}_max_abs"] = f"{row['max_abs']:.6g}"
        defined[f"whoodeo.bench.{name}_psnr"] = f"{row['psnr']:.2f}"
        defined[f"whoodeo.bench.{name}_ms"] = f"{row['ms']:.1f}"
        defined[f"whoodeo.bench.{name}_ok"] = "1" if row["psnr"] >= PSNR_MIN else "0"
        defined[f"whoodeo.bench.{name}_sizes_ms"] = ",".join(
            f"{width}x{height}:{ms:.1f}" for (width, height), ms in sorted(row["sizes_ms"].items())
        )
    defined["whoodeo.bench.prefer"] = prefer
    defined["whoodeo.bench.note"] = note
    save_spec(spec, _get_model_spec_path(str(package)))
    print(f"bench metadata -> {package}", flush=True)


def library_directory():
    """The folder the macOS app lists. Created on demand."""
    path = Path.home() / "Library" / "Application Support" / "Whoodeo" / "Models"
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


def package_filename(checkpoint, spec):
    """Stable name the sidebar can read without opening the package."""
    arch = spec.architecture.get("arch") or "model"
    blocks = spec.architecture.get("blocks")
    channels = spec.architecture.get("channels")
    shape = f"{blocks}x{channels}" if blocks and channels else "model"
    step = spec.step
    step_part = f"step{int(step):06d}" if step is not None else "step"
    kind = {"vae2": "decoder", "latent_up": "enum"}.get(arch, "flex")
    return f"{_run_slug(checkpoint)}-{arch}-{shape}-{step_part}-{kind}.mlpackage"


def _default_output(checkpoint, spec):
    return library_directory() / package_filename(checkpoint, spec)


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
        "--bench",
        type=Path,
        nargs="+",
        default=None,
        help="compare with PyTorch on frame 100 of each video",
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
        help="latent_up low size HxW, repeatable, sides divisible by 4. "
             "Default: 240x428, 360x640, 480x720, 576x720, 540x960",
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=None,
        help="latent_up: the vae2 checkpoint. Default: the one the checkpoint names, "
             "/mnt/home-storinator-trem mapped to /Volumes/home",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="latent_up: print the Neural Engine plan of each low size",
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


def main(argv=None):
    args = parse_args(argv)
    spec = load_generator(args.model)
    if args.bench and spec.architecture.get("arch") == "vae2":
        raise SystemExit("--bench compares upscalers. A vae2 decoder has no low frames")
    if args.out is not None:
        output = args.out
    else:
        output = _default_output(args.model, spec)
    package, _spec = convert(
        args.model, output, args.latent or LATENT_SIZES, args.low or LOW_SIZES, args.vae, args.plan,
    )
    if args.bench:
        bench(package, args.model, args.bench, args.vae)


if __name__ == "__main__":
    main()
