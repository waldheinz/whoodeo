"""Write training videos.

`whoodeo-make-low import` writes a source into the orig folder. A picture
about Full HD or larger is encoded near 1280x720. That master keeps the
picture's aspect and is 10-bit 4:4:4 HEVC with a short closed GOP, so
training seeks stay cheap. A smaller picture is remuxed as it is: the video
bitstream stays, and the audio is dropped. Low variants are written unless
`--no-degrade` is set.

The plain command writes half-resolution variants of videos already in orig.
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
MASTER_PIXELS = 1280 * 720
FULLHD_PIXELS = 1920 * 1080
MASTER_PIX_FMT = "yuv444p10le"
# No preset: x265 medium. CRF 12 is the quality knob. psy stays at x265's default.
# bframes=0 keeps encode order equal to display order. It does not repair
# timestamps: passthrough copies a duplicate source pts straight through.
X265_PARAMS = (
    "keyint=12:min-keyint=12:open-gop=0:repeat-headers=1:"
    "scenecut=40:bframes=0:aq-mode=1:rc-lookahead=12:"
    "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:"
    "log-level=error"
)
HDR_TRANSFER = {"smpte2084", "arib-std-b67"}
HDR_PRIMARIES = {"bt2020"}
HDR_MATRIX = {"bt2020nc", "bt2020c", "bt2020ncl"}
INTERLACED = {"tt", "bb", "tb", "bt"}
# ffprobe matrix name → colorspace filter `all`/`iall` value.
MATRIX_ALL = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "bt470m": "bt470m",
    "smpte240m": "smpte240m",
}


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


def probe_video(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,sample_aspect_ratio,"
        "field_order,color_space,color_transfer,color_primaries,color_range,"
        "has_b_frames",
        "-of", "json",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no video stream: {path}")
    payload = json.loads(text)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream: {path}")
    return streams[0]


def stream_text(stream, key):
    value = stream.get(key)
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"N/A", "unknown", "unspecified", "none"}:
        return ""
    return text


def sar_factors(sar):
    if not sar:
        return 1, 1
    if ":" not in sar:
        return None
    num, den = sar.split(":", 1)
    try:
        num, den = int(num), int(den)
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num, den


def reaches_full_hd(width, height, sar):
    """True when the display picture is about Full HD or larger.

    None means the sample aspect ratio cannot be read, so the display size
    is unknown.
    """
    factors = sar_factors(sar)
    if factors is None:
        return None
    num, den = factors
    return width * num * height >= FULLHD_PIXELS * den


def snap4(value):
    snapped = int(round(value / 4.0)) * 4
    if snapped < 4:
        return 4
    return snapped


def master_dimensions(width, height, sar):
    """Square-pixel size near 1280x720, aspect kept, both sides a multiple of 4.

    A picture that already has fewer pixels stays at its display size. The
    multiple of 4 keeps the half-resolution low frame even.
    """
    factors = sar_factors(sar)
    if factors is None:
        return None
    num, den = factors
    disp_w = width * num / den
    disp_h = float(height)
    scale = 1.0
    if disp_w * disp_h > MASTER_PIXELS:
        scale = (MASTER_PIXELS / (disp_w * disp_h)) ** 0.5
    return snap4(disp_w * scale), snap4(disp_h * scale)


def input_matrix(color_space):
    if not color_space:
        return "bt709"
    return MATRIX_ALL.get(color_space)


def master_filter(width, height, matrix, color_range):
    irange = "pc" if color_range in {"pc", "jpeg"} else "tv"
    fast = "1" if matrix == "bt709" else "0"
    # setpts numbers frames in display order. A source can name two pictures
    # with one timestamp; passthrough would keep that, and mpeg4 then refuses it.
    return (
        "setpts=N/(FRAME_RATE*TB),"
        f"colorspace=iall={matrix}:all=bt709:irange={irange}:range=tv:"
        f"format=yuv444p12:dither=none:fast={fast},"
        f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp:"
        "sws_dither=none,setsar=1,format=yuv444p10le,"
        "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709"
    )


def _ratio(text):
    if not text or "/" not in text:
        return None
    num, den = text.split("/", 1)
    try:
        num, den = int(num), int(den)
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num, den


def video_timing(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,time_base",
        "-of", "json",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no video timing: {path}")
    streams = json.loads(text).get("streams") or []
    if not streams:
        raise RuntimeError(f"no video timing: {path}")
    rate = _ratio(str(streams[0].get("r_frame_rate") or ""))
    base = _ratio(str(streams[0].get("time_base") or ""))
    if rate is None or base is None:
        raise RuntimeError(f"no frame rate: {path}")
    return rate, base


def packet_pts(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "packet=pts",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no timestamps: {path}")
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "N/A":
            raise RuntimeError(f"missing timestamp: {path}")
        pts.append(int(line))
    if not pts:
        raise RuntimeError(f"no timestamps: {path}")
    return pts


def packet_times(path):
    """Presentation and decode timestamps for each video packet, in file order."""
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "packet=pts,dts",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no timestamps: {path}")
    times = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2 or parts[0] in {"", "N/A"}:
            raise RuntimeError(f"missing timestamp: {path}")
        # Matroska does not store the leading negative decode timestamps.
        # Those packets come back without a DTS; a later hole is a broken file.
        dts = None if parts[1] in {"", "N/A"} else int(parts[1])
        times.append((int(parts[0]), dts))
    if not times:
        raise RuntimeError(f"no timestamps: {path}")
    return times


def frame_tick(pts, rate, base):
    """Nearest frame index at this rate. base is the timestamp timebase."""
    rate_num, rate_den = rate
    tb_num, tb_den = base
    return (pts * tb_num * rate_num + (tb_den * rate_den) // 2) // (tb_den * rate_den)


def timestamp_problem(path):
    """None when each next packet is a later frame tick.

    mpeg4 turns the master timestamps into one tick per frame. Two packets on
    the same tick fail the encode. Packet order is display order for these
    masters because the import writes no B-frames.
    """
    rate, base = video_timing(path)
    pts = packet_pts(path)
    if any(b < a for a, b in zip(pts, pts[1:])):
        return "timestamps go backwards"
    last = None
    for value in pts:
        tick = frame_tick(value, rate, base)
        if last is not None and tick <= last:
            return "two frames share one timestamp"
        last = tick
    return None


def restamp_timestamps(src, dst):
    """Stream-copy src to dst on a constant frame-rate grid. Return an error or None.

    setts numbers packets in file order. That matches display order while
    timestamps never run backwards, which is true for a master without B-frames.
    """
    rate, base = video_timing(src)
    pts = packet_pts(src)
    if any(b < a for a, b in zip(pts, pts[1:])):
        return "timestamps go backwards"
    rate_num, rate_den = rate
    tb_num, tb_den = base
    expr = f"floor(N*{rate_den}*{tb_den}/({rate_num}*{tb_num})+0.5)"
    dst.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-i", str(src),
            "-map", "0:v:0", "-an", "-c", "copy",
            "-bsf:v", f"setts=pts={expr}:dts={expr}",
            str(dst),
        ],
        check=False,
    )
    if result.returncode != 0:
        dst.unlink(missing_ok=True)
        return "restamp failed"
    if packets_of(dst) != len(pts):
        dst.unlink(missing_ok=True)
        return "restamp changed the frame count"
    problem = timestamp_problem(dst)
    if problem:
        dst.unlink(missing_ok=True)
        return problem
    return None


def time_order_problem(times, rate, base, delay):
    """None when display ticks advance and packet DTS strictly increases.

    PTS is checked in presentation order. A source with B-frames stores a
    later picture before an earlier one, so packet PTS may step backwards
    while the display time still advances. Matroska leaves the leading
    decode timestamps empty, at most `delay` of them (the reorder delay).
    """
    last = None
    for value in sorted(item[0] for item in times):
        tick = frame_tick(value, rate, base)
        if last is not None and tick <= last:
            return "two frames share one timestamp"
        last = tick
    seen = False
    leading = 0
    prev = None
    for _pts, dts in times:
        if dts is None:
            if seen or leading >= delay:
                return "missing timestamp"
            leading += 1
            continue
        seen = True
        if prev is not None and dts <= prev:
            return "dts does not increase"
        prev = dts
    return None


def b_frame_delay(path):
    """Reorder delay. Unknown or unreadable counts as B-frames, so no restamp."""
    try:
        delay = int(probe_video(path).get("has_b_frames") or 0)
    except (TypeError, ValueError):
        return 1
    if delay < 0:
        return 1
    return delay


def settle_remux_time(path, label):
    """Accept path, or restamp it when packet order is display order.

    setts numbers packets in file order. That is wrong once B-frames have
    stored a later picture first, and it is wrong when PTS already runs
    backwards. Those files are left untouched and rejected.
    """
    rate, base = video_timing(path)
    times = packet_times(path)
    delay = b_frame_delay(path)
    err = time_order_problem(times, rate, base, delay)
    if err is None:
        return None
    pts = [item[0] for item in times]
    backwards = any(b < a for a, b in zip(pts, pts[1:]))
    # setts follows file order. Only a colliding display clock with no
    # B-frames and no backwards PTS can be renumbered that way.
    if err != "two frames share one timestamp" or delay != 0 or backwards:
        return err
    print(f"{label}: {err}; restamping timestamps", flush=True)
    stamped = path.with_name(f"{path.stem}.stamped{path.suffix}")
    restamp_error = restamp_timestamps(path, stamped)
    if restamp_error:
        stamped.unlink(missing_ok=True)
        return restamp_error
    stamped.replace(path)
    rate, base = video_timing(path)
    return time_order_problem(packet_times(path), rate, base, b_frame_delay(path))


def remux_problem(path, source, packets):
    """None when the remux kept the picture and dropped the audio."""
    stream = probe_video(path)
    if stream.get("codec_name") != source.get("codec_name"):
        return f"codec is {stream.get('codec_name')}"
    width = int(source.get("width") or 0)
    height = int(source.get("height") or 0)
    if int(stream.get("width") or 0) != width or int(stream.get("height") or 0) != height:
        return (
            f"output is {stream.get('width')}x{stream.get('height')}, "
            f"expected {width}x{height}"
        )
    pix = source.get("pix_fmt")
    if stream.get("pix_fmt") != pix:
        return f"pixel format is {stream.get('pix_fmt')}, expected {pix}"
    out_packets = packets_of(path)
    if out_packets != packets:
        return f"{packets} input frames, {out_packets} output frames"
    if has_audio(path):
        return "output still has audio"
    return None


def master_problem(path, width, height, packets):
    stream = probe_video(path)
    if stream.get("codec_name") != "hevc":
        return f"codec is {stream.get('codec_name')}"
    if int(stream.get("width") or 0) != width or int(stream.get("height") or 0) != height:
        return f"output is {stream.get('width')}x{stream.get('height')}, expected {width}x{height}"
    if stream.get("pix_fmt") != MASTER_PIX_FMT:
        return f"pixel format is {stream.get('pix_fmt')}, expected {MASTER_PIX_FMT}"
    sar = stream_text(stream, "sample_aspect_ratio") or None
    if not square_pixels(sar):
        return f"sample aspect ratio {sar}"
    tags = {
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "color_range": "tv",
    }
    for key, want in tags.items():
        got = stream_text(stream, key)
        if got != want:
            return f"{key} is {got or 'unset'}, expected {want}"
    out_packets = packets_of(path)
    if out_packets != packets:
        return f"{packets} input frames, {out_packets} output frames"
    if has_audio(path):
        return "output still has audio"
    return timestamp_problem(path)


def encode_master(src, label, out, stream, in_packets, width, height, sar, matrix_name):
    """Encode src to a 10-bit 4:4:4 HEVC master near 1280x720."""
    matrix = input_matrix(matrix_name)
    if matrix is None:
        print(f"{label}: unsupported color matrix {matrix_name}", file=sys.stderr)
        return None
    size = master_dimensions(width, height, sar)
    if size is None:
        print(f"{label}: bad sample aspect ratio {sar or 'unset'}", file=sys.stderr)
        return None
    out_w, out_h = size
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    print(
        f"import {label}  {width}x{height} -> {out_w}x{out_h}  {MASTER_PIX_FMT} crf 12",
        flush=True,
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-map", "0:v:0", "-an",
        "-vf", master_filter(out_w, out_h, matrix, stream_text(stream, "color_range")),
        "-c:v", "libx265",
        "-profile:v", "main444-10",
        "-pix_fmt", MASTER_PIX_FMT,
        "-crf", "12",
        "-fps_mode", "passthrough",
        "-x265-params", X265_PARAMS,
        str(tmp),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"{label}: ffmpeg failed", file=sys.stderr)
        return None
    try:
        problem = timestamp_problem(tmp)
        if problem:
            print(f"{label}: {problem}; restamping timestamps", flush=True)
            stamped = tmp.with_name(f"{tmp.stem}.stamped{tmp.suffix}")
            restamp_error = restamp_timestamps(tmp, stamped)
            if restamp_error:
                tmp.unlink(missing_ok=True)
                stamped.unlink(missing_ok=True)
                print(f"{label}: {restamp_error}", file=sys.stderr)
                return None
            stamped.replace(tmp)
        problem = master_problem(tmp, out_w, out_h, in_packets)
    except (RuntimeError, json.JSONDecodeError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    if problem:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {problem}", file=sys.stderr)
        return None
    tmp.replace(out)
    print(f"wrote {out}  {out_w}x{out_h}  {in_packets} frames", flush=True)
    return out.name


def remux_master(src, label, out, stream, in_packets, width, height):
    """Copy the video bitstream into orig and drop the audio."""
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    print(f"import {label}  {width}x{height}  remux", flush=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-map", "0:v:0", "-an", "-c", "copy",
        str(tmp),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"{label}: ffmpeg failed", file=sys.stderr)
        return None
    try:
        problem = remux_problem(tmp, stream, in_packets)
        if problem is None:
            problem = settle_remux_time(tmp, label)
            if problem is None:
                problem = remux_problem(tmp, stream, in_packets)
    except (RuntimeError, json.JSONDecodeError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    if problem:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {problem}", file=sys.stderr)
        return None
    tmp.replace(out)
    print(f"wrote {out}  {width}x{height}  {in_packets} frames", flush=True)
    return out.name


def import_one(src, orig_dir, force):
    """Write src into orig. Return the new filename, or None on failure.

    About Full HD and larger is encoded near 1280x720. A smaller picture is
    remuxed without scaling or re-encoding.
    """
    label = src.name
    if src.suffix.lower() not in VIDEO_EXTS:
        print(f"skip {label}: train does not read this extension", file=sys.stderr)
        return None
    if not src.is_file():
        print(f"no video: {src}", file=sys.stderr)
        return None
    if src.parent == orig_dir:
        print(f"{src} is already in {orig_dir}", file=sys.stderr)
        return None
    out = orig_dir / f"{src.stem}.mkv"
    if out.exists() and not force:
        print(f"{out.name}: already in {orig_dir}", file=sys.stderr)
        return None
    try:
        stream = probe_video(src)
        in_packets = packets_of(src)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width < 2 or height < 2:
        print(f"{label}: no video size", file=sys.stderr)
        return None
    field_order = stream_text(stream, "field_order")
    if field_order in INTERLACED:
        print(f"{label}: interlaced ({field_order})", file=sys.stderr)
        return None
    transfer = stream_text(stream, "color_transfer")
    primaries = stream_text(stream, "color_primaries")
    matrix_name = stream_text(stream, "color_space")
    if transfer in HDR_TRANSFER or primaries in HDR_PRIMARIES or matrix_name in HDR_MATRIX:
        print(f"{label}: HDR or wide-gamut source", file=sys.stderr)
        return None
    sar = stream_text(stream, "sample_aspect_ratio")
    full_hd = reaches_full_hd(width, height, sar)
    if full_hd is None and width * height >= FULLHD_PIXELS:
        print(f"{label}: bad sample aspect ratio {sar or 'unset'}", file=sys.stderr)
        return None
    if full_hd:
        return encode_master(
            src, label, out, stream, in_packets, width, height, sar, matrix_name,
        )
    return remux_master(src, label, out, stream, in_packets, width, height)


def degrade_files(orig_dir, low_dir, names, force):
    scale, flags, pix_fmt, variants = load_config(CONFIG)
    failed = False
    for name in names:
        src = orig_dir / name
        for variant, ffmpeg_args in variants.items():
            ok = encode_one(
                src, low_dir, variant, ffmpeg_args,
                scale, flags, pix_fmt, force, True,
            )
            failed = failed or not ok
    return not failed


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


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="encode half-resolution variants of orig videos. "
        "`whoodeo-make-low import` writes a source into orig first",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--variants", help="comma-separated variant names; default is all of them")
    parser.add_argument("-o", "--orig", type=Path, default=EnvPath("orig"), help="master videos (default: $WHOODEO_DATA/orig)")
    parser.add_argument("-l", "--low", type=Path, default=EnvPath("low"), help="degraded variants (default: $WHOODEO_DATA/low)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing low video")
    parser.add_argument("videos", nargs="*", help="filenames in orig; default encodes every missing partner")
    return parser.parse_args(argv)


def parse_import_args(argv):
    parser = argparse.ArgumentParser(
        prog="whoodeo-make-low import",
        description="write sources into orig. About Full HD and larger become "
        "10-bit 4:4:4 HEVC masters near 1280x720. Smaller sources are remuxed "
        "without re-encoding. Low variants are written unless --no-degrade",
    )
    parser.add_argument("-o", "--orig", type=Path, default=EnvPath("orig"), help="master videos (default: $WHOODEO_DATA/orig)")
    parser.add_argument("-l", "--low", type=Path, default=EnvPath("low"), help="low variants, written unless --no-degrade (default: $WHOODEO_DATA/low)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing master")
    parser.add_argument("--no-degrade", action="store_true", help="do not write the low variants of each new master")
    parser.add_argument("videos", nargs="+", help="source videos to write into orig")
    return parser.parse_args(argv)


def import_main(argv):
    args = parse_import_args(argv)
    args.orig = resolve_data(args.orig)
    orig_dir = args.orig.expanduser().resolve()
    orig_dir.mkdir(parents=True, exist_ok=True)
    low_dir = None
    if not args.no_degrade:
        args.low = resolve_data(args.low)
        low_dir = args.low.expanduser().resolve()
        low_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    written = []
    for name in args.videos:
        src = Path(name).expanduser().resolve()
        out_name = import_one(src, orig_dir, args.force)
        if out_name is None:
            failed = True
            continue
        written.append(out_name)
    if not args.no_degrade and written:
        if not degrade_files(orig_dir, low_dir, written, args.force):
            failed = True
    if failed:
        raise SystemExit(1)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "import":
        import_main(argv[1:])
        return
    args = parse_args(argv)
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
