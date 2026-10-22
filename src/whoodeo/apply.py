"""Reconstruct a video with a generator written by train.

With one path, the video is reconstructed by latest in the runs directory.
With two paths, the first picks a checkpoint and the second is the video.
A run directory opens its newest checkpoint. The reconstruction is written
into the run directory as <stem>-recon-<step>.mkv.

model.pt holds the generator and its architecture. An older model.pt that
also holds the optimizer still loads; the optimizer is ignored.

The network runs on each full frame. A 5-frame model sees the center frame
plus two neighbors on either side, repeating the first or last frame at the
ends. The live page shows the input and the reconstruction. Audio
streams are copied packet for packet, without re-encoding.

A latent_up checkpoint runs with the vae2 its config names (a path under
/mnt/home-storinator-trem is looked up under /Volumes/home, --vae overrides
it): latent_up predicts the code, the vae2 decodes it, the frame is clamped
and cropped to 2h x 2w, exactly as training validates. PyTorch picks MPS,
then CUDA, then the CPU.

--coreml runs a package from whoodeo-coreml instead of PyTorch, on the unit
--unit names: gpu (CPU_AND_GPU, the default), ane (CPU_AND_NE), cpu, or all.
A .mlpackage as the model does the same. The frame is padded on the bottom
and the right to the package's whoodeo.pad (reflect) and must fit its size
range; the result is cropped to 2h x 2w, as in the macOS app. Without --out a
package writes <stem>-recon-<package>-<unit>.mkv into the current directory.
"""

import argparse
import os
import time
import unicodedata
from pathlib import Path

import av
import numpy as np
import torch
from tqdm import tqdm

from whoodeo.checkpoint import build_generator, latest_link, load_generator, reconstruction_path
from whoodeo.live import add_live_args, open_live
from whoodeo.video import read_video_frames, rgb_image


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path, device, vae=None):
    spec = load_generator(path)
    if spec.architecture.get("arch") in ("vae", "vae2"):
        raise SystemExit(
            f"{path}: a {spec.architecture['arch']} checkpoint reconstructs frames. "
            "whoodeo-apply upscales video"
        )
    if spec.architecture.get("arch") == "latent_up":
        # coreml imports coremltools only when it converts or benches.
        from whoodeo.coreml import load_latent_up
        from whoodeo.models.latent_up import LatentUpReference

        model, label, step, vae_model, vae_label, _vae_spec, vae_path = load_latent_up(spec, device, vae)
        print(f"vae2 {vae_path}", flush=True)
        reference = LatentUpReference(model, vae_model).eval()
        return reference, f"{label} + {vae_label} decoder", 1, step, spec.path
    if vae is not None:
        raise SystemExit("--vae is for a latent_up checkpoint")
    return build_generator(spec, device)


COREML_UNITS = ("gpu", "ane", "cpu", "all")


class CoreMLModel:
    """A whoodeo-coreml package as a callable on NCHW float tensors in [0, 1].

    Pads to the package's whoodeo.pad (reflect, as pad_to), checks the size
    against its range or enumerated sizes, feeds the input dtype the package
    declares, and crops the result to 2h x 2w.
    """

    def __init__(self, package, unit="gpu"):
        try:
            import coremltools as ct
        except ImportError as exc:
            raise SystemExit(
                "coremltools is not installed. On the Mac: "
                "uv pip install --python .venv/bin/python 'coremltools==9.1.dev1'"
            ) from exc
        from coremltools.models.utils import load_spec

        package = Path(package)
        if not package.exists():
            raise SystemExit(f"no package: {package}")
        units = {
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
            "ane": ct.ComputeUnit.CPU_AND_NE,
            "cpu": ct.ComputeUnit.CPU_ONLY,
            "all": ct.ComputeUnit.ALL,
        }
        spec = load_spec(str(package))
        meta = dict(spec.description.metadata.userDefined)
        inputs = {item.name: item for item in spec.description.input}
        if "frames" not in inputs:
            raise SystemExit(f"{package}: no input named frames ({', '.join(inputs)})")
        array_type = inputs["frames"].type.multiArrayType
        # ArrayDataType FLOAT16 is 65552, FLOAT32 65568.
        self.dtype = np.float16 if array_type.dataType == 65552 else np.float32
        self.package = package
        self.unit = unit
        self.meta = meta
        self.arch = meta.get("whoodeo.arch", "")
        self.step = int(meta["whoodeo.step"]) if meta.get("whoodeo.step", "").isdigit() else None
        self.in_frames = int(meta.get("whoodeo.in_frames", "1") or 1)
        self.pad = int(meta.get("whoodeo.pad", "1") or 1)
        self.sizes = []
        for part in meta.get("whoodeo.sizes", "").split(","):
            if "x" in part:
                width, height = (int(item) for item in part.split("x"))
                self.sizes.append((height, width))
        self.range = None
        if not self.sizes and "whoodeo.width_max" in meta:
            self.range = (
                int(meta["whoodeo.height_min"]), int(meta["whoodeo.height_max"]),
                int(meta["whoodeo.width_min"]), int(meta["whoodeo.width_max"]),
            )
        self.label = (
            f"{package.name}  {self.arch or 'model'}  step {self.step}  unit {unit}  "
            f"conv scale {meta.get('whoodeo.conv_scale', '?')}"
        )
        self.model = ct.models.MLModel(
            str(package),
            compute_units=units[unit],
            optimization_hints={"reshapeFrequency": ct.ReshapeFrequency.Infrequent},
        )

    def check_size(self, height, width):
        """The padded size, or SystemExit with what the package takes."""
        pad_h = height + (-height) % self.pad
        pad_w = width + (-width) % self.pad
        if self.sizes and (pad_h, pad_w) not in self.sizes:
            known = ", ".join(f"{w}x{h}" for h, w in self.sizes)
            raise SystemExit(f"{self.package.name}: {pad_w}x{pad_h} is not one of its sizes: {known}")
        if self.range is not None:
            min_h, max_h, min_w, max_w = self.range
            if not (min_h <= pad_h <= max_h and min_w <= pad_w <= max_w):
                raise SystemExit(
                    f"{self.package.name}: {pad_w}x{pad_h} is outside its range "
                    f"{min_w}x{min_h} to {max_w}x{max_h}. Scale the video down or export with --max-low"
                )
        return pad_h, pad_w

    def eval(self):
        return self

    def __call__(self, frames):
        from whoodeo.models.common import pad_to

        height, width = frames.shape[-2:]
        self.check_size(height, width)
        padded, _h, _w = pad_to(frames.float().cpu(), self.pad) if self.pad > 1 else (frames, height, width)
        batch = padded.numpy().astype(self.dtype, copy=False)
        result = self.model.predict({"frames": batch})
        image = result.get("image")
        if image is None:
            raise SystemExit(f"{self.package.name}: output names {', '.join(result)}")
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3:
            array = array[None]
        return torch.from_numpy(array[:, :, :2 * height, :2 * width].copy())


def is_package(path):
    return path is not None and Path(path).suffix == ".mlpackage"


def iter_reconstructions(model, frames, radius, device, timings=None):
    """Yield (low, recon) for every input frame. Neighbors outside the clip repeat the edge frame.

    `timings`, a list, gets the seconds each model call took, the copy back included.
    """
    buf = []
    abs_start = 0
    emitted = 0

    def produce(final):
        nonlocal abs_start, emitted
        while buf:
            last_abs = abs_start + len(buf) - 1
            if emitted > last_abs:
                return
            if not final and last_abs < emitted + radius:
                return
            center_local = emitted - abs_start
            n = len(buf)
            stacked = torch.cat([
                buf[min(max(center_local + dt, 0), n - 1)]
                for dt in range(-radius, radius + 1)
            ], dim=0)
            low = buf[center_local]
            started = time.perf_counter()
            with torch.inference_mode():
                recon = model(stacked.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
            if timings is not None:
                timings.append(time.perf_counter() - started)
            emitted += 1
            while emitted - radius > abs_start:
                buf.pop(0)
                abs_start += 1
            yield low, recon

    for frame in frames:
        buf.append(frame)
        yield from produce(False)
    yield from produce(True)


def source_rate(stream):
    rate = stream.average_rate
    if rate is None or float(rate) <= 0:
        return 30
    return rate


def attach_audio(container, source):
    """Add output audio streams that will carry the source packets unchanged."""
    mapping = {}
    for stream in source.streams.audio:
        try:
            copied = container.add_stream_from_template(stream, opaque=False)
            if stream.time_base is not None:
                copied.time_base = stream.time_base
        except ValueError:
            # Names like mp3float are decoders. Matroska still accepts the packets.
            copied = container.add_stream_from_template(stream, opaque=True)
        mapping[stream.index] = copied
    return mapping


def copy_audio(container, source, mapping, limit_sec):
    """Mux the source audio packets. limit_sec drops packets that start after the written video."""
    if not mapping:
        return 0
    streams = [stream for stream in source.streams.audio if stream.index in mapping]
    copied = 0
    finished = set()
    for packet in source.demux(streams):
        if packet.dts is None:
            continue
        if limit_sec is not None and packet.pts is not None:
            if float(packet.pts * packet.time_base) >= limit_sec:
                finished.add(packet.stream.index)
                if len(finished) >= len(streams):
                    break
                continue
        packet.stream = mapping[packet.stream.index]
        container.mux(packet)
        copied += 1
    return copied


def output_frame_count(container, stream, fps, limit):
    """How many frames will be written, or None when the input does not say.

    Matroska often leaves the track frame count and duration empty. The
    length is then only on the container, in microseconds.
    """
    total = int(stream.frames or 0)
    if total <= 0:
        seconds = None
        if stream.duration is not None and stream.time_base:
            seconds = float(stream.duration * stream.time_base)
        if (seconds is None or seconds <= 0) and container.duration:
            seconds = container.duration / av.time_base
        if seconds and seconds > 0 and float(fps) > 0:
            total = int(round(seconds * float(fps)))
    if limit:
        return min(total, limit) if total else limit
    return total or None


def encode_frame(container, stream, tensor, configured):
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise RuntimeError(f"reconstruction has shape {tuple(tensor.shape)}, expected 3,H,W")
    frame_np = tensor.permute(1, 2, 0).contiguous().numpy()
    # 16-bit RGB, so x265 quantizes to 10-bit instead of receiving an 8-bit rounding.
    frame_np = np.rint(np.clip(frame_np, 0, 1) * 65535.0).astype(np.uint16)
    height, width = frame_np.shape[:2]
    if width % 2 or height % 2:
        raise RuntimeError(f"reconstruction is {width}x{height}; hevc needs even sizes")
    # libx265 starts at 640x480. Set the real size before the first packet.
    if not configured[0]:
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p10le"
        configured[0] = True
    frame = av.VideoFrame.from_ndarray(frame_np, format="rgb48le")
    for packet in stream.encode(frame):
        container.mux(packet)


def resolve_video(path):
    """Find `path` when the directory stores another Unicode normalization.

    A precomposed name and the same name with a combining mark differ as
    bytes. Compare the normalized form.
    """
    if path.is_file() or not path.parent.is_dir():
        return path
    want = unicodedata.normalize("NFC", path.name)
    for name in os.listdir(path.parent):
        if unicodedata.normalize("NFC", name) == want:
            return path.parent / name
    return path


def apply(args):
    args.input = resolve_video(args.input)
    if not args.input.is_file():
        raise SystemExit(f"no video: {args.input}")
    package = args.coreml
    if package is None and is_package(args.model):
        package = args.model
    if package is not None:
        if args.vae is not None:
            raise SystemExit("--vae is for a PyTorch latent_up checkpoint; a package has its decoder built in")
        device = torch.device("cpu")
        model = CoreMLModel(package, args.unit or "gpu")
        label, in_frames, step = model.label, model.in_frames, model.step
        checkpoint_path = None
    else:
        if args.unit is not None:
            raise SystemExit("--unit picks the Core ML unit. Pass --coreml PACKAGE (whoodeo-coreml writes one)")
        device = get_device()
        model, label, in_frames, step, checkpoint_path = load_checkpoint(args.model, device, args.vae)
    if in_frames % 2 != 1:
        raise SystemExit(f"checkpoint uses {in_frames} input frames; that count has to be odd")
    source = av.open(str(args.input))
    video_in = source.streams.video[0]
    fps = source_rate(video_in)
    width = video_in.codec_context.width
    height = video_in.codec_context.height
    if package is not None:
        model.check_size(height, width)
    out = args.out
    if out is None and package is not None:
        out = Path.cwd() / f"{args.input.stem}-recon-{Path(package).stem}-{model.unit}.mkv"
    elif out is None:
        out = reconstruction_path(checkpoint_path, step, args.input)
    out.parent.mkdir(parents=True, exist_ok=True)

    if package is not None:
        print(f"core ml {label}", flush=True)
    else:
        print(f"device {device}", flush=True)
        print(f"checkpoint {checkpoint_path}  step {step}  {label}", flush=True)
    print(f"input {args.input}  {width}x{height}  {float(fps):g} fps", flush=True)
    if source.streams.audio:
        codecs = " ".join(
            stream.codec_context.codec.canonical_name for stream in source.streams.audio
        )
        print(f"audio {codecs}  copy", flush=True)
    else:
        print("audio none", flush=True)
    print(f"output {out}  libx265 main10 yuv420p10le crf 25", flush=True)

    live = open_live(args.live_bind, not args.no_preview)
    try:
        container = av.open(str(out), "w")
        stream = container.add_stream("libx265", rate=fps)
        stream.options = {"crf": "25", "profile": "main10", "x265-params": "log-level=error"}
        try:
            audio_map = attach_audio(container, source)
        except ValueError as exc:
            container.close()
            source.close()
            out.unlink(missing_ok=True)
            raise SystemExit(f"cannot copy audio into {out.name}: {exc}") from exc
        configured = [False]
        timings = []
        frames = iter_reconstructions(
            model,
            read_video_frames(args.input),
            in_frames // 2,
            device,
            timings,
        )
        count = 0
        interrupted = False
        total = output_frame_count(source, video_in, fps, args.frames)
        bar = tqdm(
            total=total,
            desc=out.name,
            unit="frame",
        )
        try:
            for low, recon in frames:
                count += 1
                if live is not None:
                    progress = f"{count}/{total}" if total else str(count)
                    live.status(f"{out.name}  {progress}")
                    live.show({
                        "low": rgb_image(low),
                        "recon": rgb_image(recon),
                    })
                encode_frame(container, stream, recon, configured)
                bar.update(1)
                if args.frames and count >= args.frames:
                    break
        except KeyboardInterrupt:
            interrupted = True
            bar.close()
            print(f"interrupted after {count} frames", flush=True)
        finally:
            frames.close()
            bar.close()
            try:
                if configured[0]:
                    for packet in stream.encode():
                        container.mux(packet)
                    limit = count / float(fps) if args.frames or interrupted else None
                    copy_audio(container, source, audio_map, limit)
            finally:
                container.close()
                source.close()
        print(f"wrote {count} frames  {out}", flush=True)
        if len(timings) > 1:
            # The first call pays for the compiler (Core ML) or the kernels (MPS).
            steady = sorted(timings[1:])
            print(
                f"model {steady[len(steady) // 2] * 1000:.1f} ms per frame (median), "
                f"first {timings[0] * 1000:.0f} ms",
                flush=True,
            )
    finally:
        if live is not None:
            live.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="reconstruct a video with a saved train model",
        usage="%(prog)s [checkpoint] video",
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        help="checkpoint, run directory, model.pt, train.pt, or a .mlpackage "
             "(default: latest in the runs directory)",
    )
    parser.add_argument("input", nargs="?", type=Path, help="video to reconstruct")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="mkv to write (default: <stem>-recon-<step>.mkv in the run directory)",
    )
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames; 0 means the whole video")
    parser.add_argument(
        "--vae",
        type=Path,
        default=None,
        help="latent_up: the vae2 checkpoint. Default: the one the checkpoint names, "
             "/mnt/home-storinator-trem mapped to /Volumes/home",
    )
    parser.add_argument(
        "--coreml",
        type=Path,
        default=None,
        help="run this whoodeo-coreml package instead of PyTorch",
    )
    parser.add_argument(
        "--unit",
        choices=COREML_UNITS,
        default=None,
        help="Core ML compute unit: gpu (default), ane (Neural Engine), cpu, or all",
    )
    add_live_args(parser)
    args = parser.parse_args(argv)
    if args.input is None and args.model is not None:
        args.input = args.model
        args.model = None
    if args.input is None:
        parser.error("pass the video to reconstruct")
    if args.model is None and args.coreml is None:
        args.model = latest_link()
        if not args.model.exists():
            raise SystemExit(
                f"no checkpoint at {args.model}. Pass a checkpoint on the command line."
            )
    return args


def main():
    apply(parse_args())


if __name__ == "__main__":
    main()
