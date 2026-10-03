"""Reconstruct a video with a generator written by train.

With one path, the video is reconstructed by runs/latest. With two paths,
the first picks a checkpoint and the second is the video. A run directory
opens its newest checkpoint. The reconstruction is written into the run
directory as <stem>-recon-<step>.mkv.

model.pt holds the generator and its architecture. An older model.pt that
also holds the optimizer still loads; the optimizer is ignored.

The network runs on each full frame. A 5-frame model sees the center frame
plus two neighbors on either side, repeating the first or last frame at the
ends. The live page shows the input and the reconstruction. Audio
streams are copied packet for packet, without re-encoding.
"""

import argparse
import os
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


def load_checkpoint(path, device):
    return build_generator(load_generator(path), device)


def iter_reconstructions(model, frames, radius, device):
    """Yield (low, recon) for every input frame. Neighbors outside the clip repeat the edge frame."""
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
            with torch.inference_mode():
                recon = model(stacked.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
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
    device = get_device()
    model, label, in_frames, step, checkpoint_path = load_checkpoint(args.model, device)
    if in_frames % 2 != 1:
        raise SystemExit(f"checkpoint uses {in_frames} input frames; that count has to be odd")
    source = av.open(str(args.input))
    video_in = source.streams.video[0]
    fps = source_rate(video_in)
    width = video_in.codec_context.width
    height = video_in.codec_context.height
    out = args.out
    if out is None:
        out = reconstruction_path(checkpoint_path, step, args.input)
    out.parent.mkdir(parents=True, exist_ok=True)

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
        frames = iter_reconstructions(
            model,
            read_video_frames(args.input),
            in_frames // 2,
            device,
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
        help="checkpoint, run directory, model.pt, or train.pt (default: runs/latest)",
    )
    parser.add_argument("input", nargs="?", type=Path, help="video to reconstruct")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="mkv to write (default: <stem>-recon-<step>.mkv in the run directory)",
    )
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames; 0 means the whole video")
    add_live_args(parser)
    args = parser.parse_args(argv)
    if args.input is None and args.model is not None:
        args.input = args.model
        args.model = None
    if args.input is None:
        parser.error("pass the video to reconstruct")
    if args.model is None:
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
