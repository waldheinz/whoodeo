"""Convert a vote or edsr checkpoint into a Core ML program, or a vae2 decoder.

The package is what the macOS app loads. Without -o it goes to
~/Library/Application Support/Whoodeo/Models, which is the app's collection.
Three low frames are stacked on the channel axis, previous then current then
next, the same order as apply.
A vae2 checkpoint gives its decoder alone: a latent in, the frame clamped
to [0, 1] out, both fp16. The input takes the enumerated latent sizes of
--latent, 60x107, 90x160, and 135x240 by default (480p, 720p, and 1080p at
factor 8), so the Neural Engine can run it. One size gives a fixed shape.
The metadata carries the latent shift and scale.
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
_FLEX_H = (64, 1080, 360)
_FLEX_W = (64, 1920, 640)
# Latent sides of 480p (856x480), 720p, and 1080p at factor 8.
LATENT_SIZES = ((60, 107), (90, 160), (135, 240))


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


def convert(checkpoint, output, latent_sizes=LATENT_SIZES):
    """Write an mlprogram with a flexible input, or a vae2 decoder."""
    ct = _coremltools()
    spec = load_generator(checkpoint)
    arch = spec.architecture.get("arch")
    if arch == "vae2":
        return convert_decoder(ct, spec, output, latent_sizes)
    if arch not in ("vote", "edsr"):
        raise SystemExit(
            f"{checkpoint}: whoodeo-coreml converts vote, edsr, and vae2, not {arch}"
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
        "whoodeo.latent_shift": f"{float(model.latent_shift):.8g}",
        "whoodeo.latent_scale": f"{float(model.latent_scale):.8g}",
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
                if operation.operator_name == "conv":
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


def bench(package, checkpoint, videos):
    """Compare the package with PyTorch on real frames. The worst case sets the default."""
    ct = _coremltools()
    spec = load_generator(checkpoint)
    model, label, in_frames, _step, _path = build_generator(spec, torch.device("cpu"))
    _prepare(model)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    radius = in_frames // 2
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
    planned = set()
    for video in videos:
        stack, center = _read_window(video, 100, radius)
        height, width = stack.shape[-2], stack.shape[-1]
        print(f"frame {video}  {width}x{height}  index {center}", flush=True)
        batch_t = stack.unsqueeze(0)
        with torch.inference_mode():
            if device.type == "mps":
                torch.mps.synchronize()
            reference = model(batch_t.to(device)).squeeze(0).detach().cpu().numpy()
        batch = batch_t.numpy().astype(np.float32, copy=False)
        for name, _unit in units:
            mlmodel = loaded[name]

            def predict(mlmodel=mlmodel, batch=batch):
                return _predict_array(mlmodel, batch)

            # The first shape pays for the compiler. Keep it out of the median.
            started = time.perf_counter()
            got = predict()
            compile_s = time.perf_counter() - started
            shape_key = (name, width, height)
            if shape_key not in planned:
                planned.add(shape_key)
                try:
                    counts = _plan_counts(mlmodel, _unit)
                except Exception as exc:
                    counts = {"plan": str(exc)}
                print(f"plan {name} {width}x{height} {counts}", flush=True)
            max_abs, psnr = _error(reference, got)
            median = _time_predict(predict)
            print(
                f"  {name}  max {max_abs:.6f}  psnr {psnr:.2f}  "
                f"{median:.1f} ms  compile {compile_s:.1f}s",
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

    worst = {}
    for row in rows:
        held = worst.get(row["name"])
        if held is None or row["max_abs"] > held["max_abs"]:
            worst[row["name"]] = row
    gpu = worst.get("gpu")
    ane = worst.get("ane")
    gpu_ok = gpu is not None and gpu["max_abs"] <= STEP
    ane_ok = ane is not None and ane["max_abs"] <= STEP
    if gpu_ok or not ane_ok:
        prefer = "gpu"
    else:
        prefer = "ane"
    parts = []
    for name, row in (("GPU", gpu), ("Neural Engine", ane)):
        if row is None:
            continue
        if row["max_abs"] <= STEP:
            mark = "innerhalb eines 10-bit-Schritts"
        else:
            mark = "über einem 10-bit-Schritt"
        parts.append(
            f"{name} max {row['max_abs']:.6f} ({mark}) "
            f"bei {row['width']}×{row['height']} in {row['ms']:.0f} ms"
        )
    note = ". ".join(parts)
    if prefer == "gpu":
        note += ". Voreinstellung GPU."
    else:
        note += ". Voreinstellung Neural Engine."
    print(note, flush=True)
    _write_bench(package, gpu, ane, prefer, note)
    return prefer


def _write_bench(package, gpu, ane, prefer, note):
    """Patch the spec in place. Saving the whole package would delete its weights."""
    from coremltools.models.utils import _get_model_spec_path, load_spec, save_spec
    spec = load_spec(str(package))
    defined = spec.description.metadata.userDefined
    if gpu is not None:
        defined["whoodeo.bench.gpu_max_abs"] = f"{gpu['max_abs']:.6g}"
        defined["whoodeo.bench.gpu_ms"] = f"{gpu['ms']:.1f}"
    if ane is not None:
        defined["whoodeo.bench.ane_max_abs"] = f"{ane['max_abs']:.6g}"
        defined["whoodeo.bench.ane_ms"] = f"{ane['ms']:.1f}"
        defined["whoodeo.bench.ane_ok"] = "1" if ane["max_abs"] <= STEP else "0"
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
    kind = "decoder" if arch == "vae2" else "flex"
    return f"{_run_slug(checkpoint)}-{arch}-{shape}-{step_part}-{kind}.mlpackage"


def _default_output(checkpoint, spec):
    return library_directory() / package_filename(checkpoint, spec)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="convert a vote or edsr checkpoint, or a vae2 decoder, to Core ML",
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
    package, _spec = convert(args.model, output, args.latent or LATENT_SIZES)
    if args.bench:
        bench(package, args.model, args.bench)


if __name__ == "__main__":
    main()
