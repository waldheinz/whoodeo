"""Write half-resolution training pairs for videos in the orig folder.

Variant directories live under the low root. Each one keeps the source
filename. Codec, quantizer, and bitrate come from a JSON config: the short
name is the directory, and the value is extra ffmpeg arguments.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from whoodeo.catalog import EnvPath, resolve_data

CONFIG = Path(__file__).resolve().parent / "degrade.json"
VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
WEBM_CODECS = {"vp8", "vp9", "libvpx", "libvpx-vp9", "av1", "libaom-av1", "libsvtav1"}


def load_config(path):
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no config: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: {exc}")
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: expected an object")
    scale = config.get("scale", 2)
    flags = config.get("scale_flags", "bicubic")
    pix_fmt = config.get("pix_fmt", "yuv420p")
    variants = config.get("variants")
    if not isinstance(scale, int) or scale < 2:
        raise SystemExit(f"{path}: scale must be an integer >= 2")
    if not isinstance(flags, str) or not flags:
        raise SystemExit(f"{path}: scale_flags must be a string")
    if not isinstance(pix_fmt, str) or not pix_fmt:
        raise SystemExit(f"{path}: pix_fmt must be a string")
    if not isinstance(variants, dict) or not variants:
        raise SystemExit(f"{path}: variants must be a non-empty object")
    for name, args in variants.items():
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name:
            raise SystemExit(f"{path}: bad variant name {name!r}")
        if not isinstance(args, str) or not args.strip():
            raise SystemExit(f"{path}: variant {name} must be an ffmpeg argument string")
    return scale, flags, pix_fmt, variants


def select_variants(variants, text):
    if text is None:
        return list(variants.items())
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise SystemExit("--variants is empty")
    missing = [name for name in names if name not in variants]
    if missing:
        raise SystemExit("unknown variant: " + ", ".join(missing))
    return [(name, variants[name]) for name in names]


def codec_of(args):
    tokens = shlex.split(args)
    for flag in ("-c:v", "-codec:v"):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                return tokens[index + 1]
    return None


def run_probe(args):
    result = subprocess.run(
        ["ffprobe", "-v", "error", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def size_of(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text or "," not in text:
        raise RuntimeError(f"no video size: {path}")
    width, height = text.split(",", 1)
    return int(width), int(height)


def sample_aspect(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=sample_aspect_ratio",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text or text == "N/A":
        return None
    return text


def square_pixels(sar):
    if sar is None:
        return True
    if ":" not in sar:
        return False
    num, den = sar.split(":", 1)
    try:
        width, height = int(num), int(den)
    except ValueError:
        return False
    return width > 0 and height > 0 and width == height


def packets_of(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no packet count: {path}")
    return int(text)


def has_audio(path):
    text = run_probe([
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ])
    return bool(text)


def encode_one(src, low_dir, variant, ffmpeg_args, scale, scale_flags, pix_fmt, force, strict):
    name = src.name
    label = f"{variant}/{name}"
    if src.suffix.lower() not in VIDEO_EXTS:
        if strict:
            print(f"skip {label}: train does not read this extension", file=sys.stderr)
            return False
        return True
    codec = codec_of(ffmpeg_args)
    if src.suffix.lower() == ".webm" and codec not in WEBM_CODECS:
        print(
            f"skip {label}: webm cannot hold codec {codec or '(unset)'}",
            file=sys.stderr,
        )
        return not strict
    sar = sample_aspect(src)
    if not square_pixels(sar):
        print(
            f"{label} has sample aspect ratio {sar}; stored pixels are not square",
            file=sys.stderr,
        )
        return False

    out_dir = low_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    if out.exists() and not force:
        print(f"keep {label}", flush=True)
        return True

    width, height = size_of(src)
    if width % (scale * 2) or height % (scale * 2):
        print(
            f"{label} is {width}x{height}. Both sides must be divisible by "
            f"{scale * 2} so the 1/{scale} frame is even.",
            file=sys.stderr,
        )
        return False
    low_w = width // scale
    low_h = height // scale
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    print(f"encode {label}  {width}x{height} -> {low_w}x{low_h}  {ffmpeg_args}", flush=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-map", "0:v:0", "-an",
        "-vf", f"scale={low_w}:{low_h}:flags={scale_flags}",
        *shlex.split(ffmpeg_args),
        "-pix_fmt", pix_fmt,
        "-fps_mode", "passthrough",
        str(tmp),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"{label}: ffmpeg failed", file=sys.stderr)
        return False

    try:
        out_size = size_of(tmp)
        in_packets = packets_of(src)
        out_packets = packets_of(tmp)
        audio = has_audio(tmp)
    except RuntimeError as exc:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {exc}", file=sys.stderr)
        return False
    if out_size != (low_w, low_h):
        tmp.unlink(missing_ok=True)
        print(f"{label}: output is {out_size[0]}x{out_size[1]}, expected {low_w}x{low_h}", file=sys.stderr)
        return False
    if in_packets != out_packets:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {in_packets} input frames, {out_packets} output frames", file=sys.stderr)
        return False
    if audio:
        tmp.unlink(missing_ok=True)
        print(f"{label}: output still has audio", file=sys.stderr)
        return False
    tmp.replace(out)
    print(f"wrote {out}  {low_w}x{low_h}  {out_packets} frames", flush=True)
    return True


def sources_from_args(orig_dir, names):
    if not names:
        found = sorted(
            path for path in orig_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        )
        if not found:
            raise SystemExit(f"no videos in {orig_dir}")
        return found, False
    sources = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = orig_dir / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"no video: {path}")
        if path.parent != orig_dir:
            raise SystemExit(f"{path} is not in {orig_dir}")
        sources.append(path)
    return sources, True


def parse_args():
    parser = argparse.ArgumentParser(description="encode half-resolution variants of orig videos")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--variants", help="comma-separated variant names; default is all of them")
    parser.add_argument("-o", "--orig", type=Path, default=EnvPath("orig"), help="master videos (default: $WHOODEO_DATA/orig)")
    parser.add_argument("-l", "--low", type=Path, default=EnvPath("low"), help="degraded variants (default: $WHOODEO_DATA/low)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing low video")
    parser.add_argument("videos", nargs="*", help="filenames in orig; default encodes every missing partner")
    return parser.parse_args()


def main():
    args = parse_args()
    args.orig = resolve_data(args.orig)
    args.low = resolve_data(args.low)
    scale, scale_flags, pix_fmt, variants = load_config(args.config)
    selected = select_variants(variants, args.variants)
    orig_dir = args.orig.expanduser().resolve()
    low_dir = args.low.expanduser().resolve()
    if not orig_dir.is_dir():
        raise SystemExit(f"no orig folder: {orig_dir}")
    low_dir.mkdir(parents=True, exist_ok=True)
    sources, strict = sources_from_args(orig_dir, args.videos)
    failed = False
    for src in sources:
        for variant, ffmpeg_args in selected:
            ok = encode_one(
                src, low_dir, variant, ffmpeg_args,
                scale, scale_flags, pix_fmt, args.force, strict,
            )
            failed = failed or not ok
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
